"""
Frequency composition transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

"""
Based on Diffusion Transformer paper and some downsampling.
"""
class RealtimeDenoisingTransformer_ViT(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = [16, 32]
        self.unet_channels = [self.num_internal_channels, 24, 24, 24, 24, 24, 24]

        self.true_num_input_channels = 9
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.encoder_net = nn.Sequential(
            FastUNet(channels=self.unet_channels, per_level_outputs=False)
        )
        
        self.patch_proj = nn.ModuleList([
            nn.Sequential(
                ImageLayerNorm(self.num_internal_channels),
                nn.Conv2d(self.num_internal_channels, self.num_transformer_channels[0], kernel_size=1)
            ),
            nn.Sequential(
                ImageLayerNorm(self.num_internal_channels),
                nn.Conv2d(self.num_internal_channels, self.num_transformer_channels[1], kernel_size=20, stride=16, padding=2)
            )
        ])

        self.hfqpos_emb = nn.Parameter(torch.randn(256, self.num_transformer_channels[0]))
        self.hfkpos_emb = nn.Parameter(torch.randn(400, self.num_transformer_channels[0]))
        self.lffpos_emb = nn.Parameter(torch.randn(256, self.num_transformer_channels[1]))

    def run_frame(self, frame_input : torch.Tensor, temporal_state):
        N = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        input = frame_input[:, :9, :, :]
        motionvec = frame_input[:, 13:, :, :]
    
        features = self.projector(input)

        latent = checkpoint.checkpoint_sequential(self.encoder_net, segments=1, input=features, use_reentrant=False)

        filtered = self.filter_image(latent, color)

        remod = albedo.pow(1.0 / 2.2) * filtered

        return (remod, None)

    def filter_image(self, latent : torch.Tensor, color : torch.Tensor):
        # color is formed via subtracting consecutive average pools
        # each patch is 4x4 and we repeat the process 3 times
        # 

        N, _, H, W = latent.shape



        HFPDIM = (H // 16, W // 16)
        LFPDIM = (H // 256, W // 256)

        lfc = F.avg_pool2d(color, kernel_size=16, stride=16)
        lff = self.patch_proj[1](latent)

        LFC_SHAPE = lfc.shape[2:]
        HFC_SHAPE = color.shape[2:]

        lfc = F.unfold(lfc, kernel_size=16, stride=16).unflatten(1, (-1, 256)).permute(0, 3, 2, 1).flatten(0, 1)
        lff = F.unfold(lff, kernel_size=16, stride=16).unflatten(1, (-1, 256)).permute(0, 3, 2, 1).flatten(0, 1)

        lff = lff + self.lffpos_emb

        dnlf = F.scaled_dot_product_attention(
            query=lff,
            key=lff,
            value=lfc
        )

        # obtain our denoised image now via folding
        dnlf = dnlf.unflatten(0, (N, -1)).permute(0, 3, 2, 1).flatten(1, 2)
        dnlf = F.fold(dnlf, LFC_SHAPE, kernel_size=16, stride=16)

        # now it is (N, 3, H, W)
        # notably H * W = L for high frequency
        # so we will flatten now rather than later
        dnlf = dnlf.flatten(2, 3).unsqueeze(2)

        # unflatten hfv

        hfc = F.unfold(color, kernel_size=20, stride=16, padding=2).unflatten(1, (-1, 400))


        hfc = hfc - dnlf

        hfc = hfc.permute(0, 3, 2, 1).flatten(0, 1)


        hff = self.patch_proj[0](latent)
        hfq = F.unfold(hff, kernel_size=16, stride=16).unflatten(1, (-1, 256)).permute(0, 3, 2, 1).flatten(0, 1)
        hfk = F.unfold(hff, kernel_size=20, stride=16, padding=2).unflatten(1, (-1, 400)).permute(0, 3, 2, 1).flatten(0, 1)

        hfq = hfq + self.hfqpos_emb
        hfk = hfk + self.hfkpos_emb

        dnhf = F.scaled_dot_product_attention(
            query=hfq,
            key=hfk,
            value=hfc
        )


        # upscale now
        dnhf = dnhf.unflatten(0, (N, -1)).permute(0, 3, 2, 1)

        dnhf = dnhf + dnlf

        dnhf = F.fold(dnhf.flatten(1, 2), HFC_SHAPE, kernel_size=16, stride=16)
        

        return dnhf


        """

        # isolate everything into low and high frequency components

        # (N, 3, k x k, H, W)
        hfp = F.unfold(color, kernel_size=18, stride=16, padding=2).unflatten(2, HFPDIM).unflatten(1, (3, -1))
        lfp = torch.mean(hfp, dim=2)
        hfp = hfp - lfp.unsqueeze(2)

        # denoise high frequency patches
        # for this we need high frequency features
        # (N, C, k x k, H, W)
        hff = F.unfold(self.patch_proj[0](latent), kernel_size=18, stride=16, padding=2).unflatten(2, HFPDIM).unflatten(1, (self.num_transformer_channels[0], -1))

        # permute so we can run SPDA
        # (N x H x W, k x k, C/3)
        hfp = hfp.permute(0, 3, 4, 2, 1).flatten(0, 2)
        hff = hff.permute(0, 3, 4, 2, 1).flatten(0, 2)

        # self attention
        dn_hfp = F.scaled_dot_product_attention(
            query=hff, 
            key=hff, 
            value=hfp
        )


        # now do low frequency features
        # (N, L, C)
        lff = F.unfold(latent, kernel_size=18, stride=16, padding=2).transpose(1, 2)
        lff : torch.Tensor = self.patch_proj[1](lff)

        # (N, H, W, C) -> (N, C, H, W)
        lff = lff.unflatten(1, LFPDIM).permute(0, 3, 1, 2)

        # (N, C, H, W) -> (N, C * k * k, L) -> same old
        lfp = F.unfold(lfp, kernel_size=16, stride=16).unflatten(2, LFPDIM).unflatten(1, (3, -1))
        lff = F.unfold(lff, kernel_size=16, stride=16).unflatten(2, LFPDIM).unflatten(1, (self.num_transformer_channels[1], -1))

        lfp = lfp.permute(0, 3, 4, 2, 1).flatten(0, 2)
        lff = lff.permute(0, 3, 4, 2, 1).flatten(0, 2)

        # self attention
        dn_lfp = F.scaled_dot_product_attention(
            query=lff, 
            key=lff, 
            value=lfp
        )

        # depermute everything and add it back in again
        hfp = hfp.unflatten(0, (N, -1)).unflatten(1, HFPDIM)

        

        pass
        """
    def patch_and_filter(self, qk, v, border):
        N, C, H, W = qk.shape

        FSHAPE = (H, W)
        TSHAPE = (H // 16, W // 16)

        # set unfold params
        quf = F.unfold(qk, stride=16, kernel_size=16, padding=0)
        kuf = F.unfold(qk, stride=16, kernel_size=20 if border else 16, padding=2 if border else 0)
        vuf = F.unfold(v,  stride=16, kernel_size=20 if border else 16, padding=2 if border else 0)

        # (N, C * L, P)
        # (N, P, C, L)
        # (N * P, C, L)
        # (N * P, L, C)
        quf = quf.permute(0, 2, 1).unflatten(2, (C, -1)).flatten(0, 1).transpose(1, 2)
        kuf = kuf.permute(0, 2, 1).unflatten(2, (C, -1)).flatten(0, 1).transpose(1, 2)
        vuf = vuf.permute(0, 2, 1).unflatten(2, (C, -1)).flatten(0, 1).transpose(1, 2)

        auf = F.scaled_dot_product_attention(
            query=quf,
            key=kuf,
            value=vuf
        )

        # utilize folding to get our shape back
        # (N * P, L, C)
        # P = H * W of the patches
        # L = k * k of the patches

        auf = auf.transpose(1, 2).flatten(1, 2).unflatten(0, (N, -1)).permute(0, 2, 1)

        auf = F.fold(auf, output_size=FSHAPE, kernel_size=16, stride=16)

        return auf
