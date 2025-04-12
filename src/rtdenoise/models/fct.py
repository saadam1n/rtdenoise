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
Actual channels_out is channels_in * channels_out
"""
class ColorFrequencyExtractor(nn.Module):
    def __init__(self, channels_out, weight : torch.Tensor = None):
        super(ColorFrequencyExtractor, self).__init__()

        self.weight = nn.Parameter(torch.randn(channels_out, 1, 3, 3) if weight is None else weight.clone())

        self.channels_out = channels_out

    def forward(self, x):

        rpw = self.weight.repeat(3, 1, 1, 1)

        conv = F.conv2d(x, weight=rpw, groups=3, padding=1)

        return conv

"""
Soon to have an actual transformer (TM)
"""
class FrequencyCompositionTransformer(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 16
        self.num_cfe_channels = 32
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

        self.cfe_weight = torch.randn(self.num_cfe_channels, 1, 3, 3)
        self.cfe_blocks = nn.ModuleList([ 
            ColorFrequencyExtractor(self.num_cfe_channels, self.cfe_weight)
            for i in range(self.num_filtering_scales)
        ])

        self.logit_extractor = nn.Sequential(
            nn.BatchNorm2d(self.num_internal_channels),
            nn.Conv2d(self.num_internal_channels, self.num_filtering_scales, kernel_size=1)
        )

        self.kernel_predictors = nn.ModuleList([
            nn.Sequential(
                FeedForwardGELU(self.unet_channels[i], 9, channel_multiplier=2),
                nn.Softmax(dim=1)
            )
            for i in range(self.num_filtering_scales)
        ])
        self.rct_weights = nn.Parameter(torch.randn(1, self.num_cfe_channels, 1, 1))

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

        remod = albedo * filtered

        return (remod, None)

    def filter_image(self, latents, radiance):

        cfe = []

        current_ds = None
        for i in range(self.num_filtering_scales):
            current_ds = radiance if i == 0 else F.avg_pool2d(current_ds, kernel_size=2, stride=2)
            


            kernel = self.kernel_predictors[i](latents[i])

            ppc = checkpoint.checkpoint(op_per_pixel_conv, self.cfe_blocks[i](current_ds), kernel, kernel_size=3, use_reentrant=False)

            ppc = F.interpolate(
                ppc,
                size=radiance.shape[2:],
                mode="bilinear",
                align_corners=False
            ) if i != 0 else ppc

            cfe.append(ppc)

        # (N, L, C, H, W)
        cfe = torch.stack(cfe, dim=1)

        # (N, L, 1, H, W)
        logits = self.logit_extractor(latents[0])
        weights = F.softmax(logits, dim=1).unsqueeze(2)

        # (N, C, H, W)
        agg = (cfe * weights).sum(1)

        rct = F.conv2d(agg, weight=self.rct_weights.expand(3, -1, -1, -1), groups=3)

        return rct
        




    

