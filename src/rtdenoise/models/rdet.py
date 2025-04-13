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
class RealtimeDenoisingTransformer(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 16
        self.unet_channels = [self.num_internal_channels, 24, 24, 32, 32, 48, 64]
        self.num_filtering_scales = len(self.unet_channels) - 1 # analyze the last level but don't filter it

        self.window_size = 3
        self.skip_center = False

        self.true_num_input_channels = 9
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.encoder_net = nn.Sequential(
            FastUNet(channels=self.unet_channels, per_level_outputs=True)
        )

        self.logit_extractor = nn.Sequential(
            nn.BatchNorm2d(self.num_internal_channels),
            nn.Conv2d(self.num_internal_channels, self.num_filtering_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.key_extractors = nn.ModuleList([ 
            nn.Sequential(
                ImageLayerNorm(self.unet_channels[i]),
                nn.Conv2d(self.unet_channels[i], self.num_transformer_channels + 1, kernel_size=1)
            )
            for i in range(self.num_filtering_scales)
        ])

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

        latents = checkpoint.checkpoint_sequential(self.encoder_net, segments=1, input=features, use_reentrant=False)

        filtered = self.filter_image(latents, color)

        remod = albedo.pow(1.0 / 2.2) * filtered

        return (remod, None)

    def filter_image(self, latents : list[torch.Tensor], color : torch.Tensor):

        # (N, L, 1, H, W)
        level_weights = self.logit_extractor(latents[0]).unsqueeze(2)

        # (N, L, 3, H, W)
        levels = color.unsqueeze(1) * level_weights

        keys = [
            self.key_extractors[i](latents[i])
            for i in range(self.num_filtering_scales)
        ]

        # filter each partition indepedently
        # upscale using attention 
        dnaccum = None
        lvaccum = None
        for i in reversed(range(self.num_filtering_scales)):
            lvaccum = levels[:, i:].sum(1)

            dnaccum = checkpoint.checkpoint(
                self.filter_level, 
                keys[i - 1] if i > 0 else None, 
                keys[i],
                levels[:, i],
                dnaccum,
                lvaccum,
                i,
                use_reentrant=False
            )


        return dnaccum


    def filter_level(self, query : torch.Tensor, key : torch.Tensor, level : torch.Tensor, dnaccum : torch.Tensor, lvaccum : torch.Tensor, i : int):
        texel_size = 2 ** i

        # ignore first channel 
        csource = query if query is not None else key
        cweight = csource[:, :1].sigmoid()

        query = query[:, 1:] if query is not None else None
        key = key[:, 1:]

        dslevel = F.avg_pool2d(level, kernel_size=texel_size, stride=texel_size) if i != 0 else level


        # filter level
        dnlevel = kernel_attn(key, dslevel, None, None, None, None, window_size=5, skip_center=False)

        # add accumulated after the fact
        if dnaccum is not None:
            dnlevel = dnlevel + dnaccum


        if i != 0:
            rslevel = F.avg_pool2d(lvaccum, kernel_size=texel_size // 2, stride=texel_size // 2)
            return checkpoint.checkpoint(resid_upsample_attn, query, key, dnlevel, rslevel, use_reentrant=False)

        # upsample to next level
        # if we have a large receptive field use kernel attn instead
        if i != 0:
            # use bilinear super sampling first
            bnlevel = F.interpolate(dnlevel, size=query.shape[2:], mode="bilinear", align_corners=False)
            rslevel = F.avg_pool2d(lvaccum, kernel_size=texel_size // 2, stride=texel_size // 2) - bnlevel

            drlevel = kernel_attn(query, rslevel, qk1=None, v1=None, qk2=None, v2=None, window_size=3, skip_center=False) 
            fslevel = drlevel + bnlevel

            ualevel = upscale_attn(query, key, dnlevel, b = torch.zeros_like(query[:, :9]), kernel_size=3, scale_power=1)

            bulevel = fslevel * cweight + ualevel * (1 - cweight)

            return bulevel
        else:
            return dnlevel


    

