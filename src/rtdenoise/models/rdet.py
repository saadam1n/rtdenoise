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
        self.num_filtering_scales = len(self.unet_channels) - 1

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
                nn.Conv2d(self.unet_channels[i], self.num_transformer_channels, kernel_size=1)
            )
            for i in range(self.num_filtering_scales)
        ])

        self.query_extractor = nn.Sequential(
            ImageLayerNorm(self.unet_channels[0]),
            nn.Conv2d(self.unet_channels[0], self.num_transformer_channels, kernel_size=1)
        )

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
        accum = None
        for i in reversed(range(self.num_filtering_scales)):
            accum = checkpoint.checkpoint(
                self.filter_level, 
                keys[i - 1] if i > 0 else None, 
                keys[i],
                levels[:, i],
                accum,
                i,
                use_reentrant=False
            )

        return accum


    def filter_level(self, query : torch.Tensor, key : torch.Tensor, level : torch.Tensor, accum : torch.Tensor, i : int):
        texel_size = 2 ** i

        dslevel = F.avg_pool2d(level, kernel_size=texel_size, stride=texel_size) if i != 0 else level


        # filter level
        dnlevel = kernel_attn(key, dslevel, None, None, None, None, window_size=3, skip_center=False)

        # add accumulated after the fact
        if accum is not None:
            dnlevel = dnlevel + accum


        # upsample to next level
        if i != 0:
            fslevel = upscale_attn(query, key, dnlevel, b = torch.zeros_like(query[:, :9]), kernel_size=3, scale_power=1)

            return fslevel
        else:
            return dnlevel


    

