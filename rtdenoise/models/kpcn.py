# implementation of the KPCN model from that disney research paper, with some modifications for efficiency 

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_denoiser import BaseDenoiser
from .components import DepthWiseConvFormer

"""
A version of KPCN that utilizes depth-wise convolutions and the MetaFormer architecture to predict weights
"""
class FastKPCN(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32
        self.num_kernels = 13
        self.kernel_dim = 7

        self.kernel_predictor = nn.Sequential(
            # skip batch norm for the first layer since I've found it messes with the denoiser for some reason
            nn.Conv2d(self.num_input_channels, self.num_internal_channels, kernel_size=1), # project raw features to a different dimension
            DepthWiseConvFormer(self.num_internal_channels),
            DepthWiseConvFormer(self.num_internal_channels),
            DepthWiseConvFormer(self.num_internal_channels),
            DepthWiseConvFormer(self.num_internal_channels),
            DepthWiseConvFormer(self.num_internal_channels),
            DepthWiseConvFormer(self.num_internal_channels),
            nn.Conv2d(self.num_internal_channels, self.num_kernels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.num_kernels, self.kernel_dim ** 2, kernel_size=1),
            nn.Softmax2d()
        )
    
        self.unfold = nn.Unfold(kernel_size=self.kernel_dim, padding=self.kernel_dim // 2)

    def run_frame(self, frame_input, temporal_state):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        kernels = self.kernel_predictor(frame_input).unsqueeze(1)
        patches = self.unfold(frame_input[:, :3, :, :]).view(B, 3, self.kernel_dim ** 2, H, W)

        accum = (kernels * patches).sum(2)

        denoised_output = albedo * accum

        return (denoised_output, temporal_state)