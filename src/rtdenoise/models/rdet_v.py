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
        self.num_transformer_channels = 16
        self.unet_channels = [self.num_internal_channels, 24, 24, 24, 24, 24, 24]
        self.num_filtering_scales = 3

        self.true_num_input_channels = 9
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.encoder_net = nn.Sequential(
            FastUNet(channels=self.unet_channels, per_level_outputs=False)
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

        latent = checkpoint.checkpoint_sequential(self.encoder_net, segments=1, input=features, use_reentrant=False)

        filtered = self.filter_image(latent, color)

        remod = albedo.pow(1.0 / 2.2) * filtered

        return (remod, None)

    def filter_image(self, latent : torch.Tensor, color : torch.Tensor):
        # color is formed via subtracting consecutive average pools
        # each patch is 4x4 and we repeat the process 3 times
        # e.g. for a 256 x 256 image, we would do the following:
        # 4x4 grid, 256 x 256 image, 64 x 64 patches
        # 4x4 grid, 64 x 64 image, 16 x 16 patches
        # 4x4 grid, 16 x 16 image, 4 x 4 patches

        # 

        pass

    

