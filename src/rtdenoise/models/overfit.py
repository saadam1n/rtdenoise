import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *


class OverfitNet(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 16
        self.unet_channels = [self.num_internal_channels, 24, 24, 32, 32, 48, 64]
        self.num_filtering_scales = len(self.unet_channels) - 1

        self.nz_scales = [5, 11, 17, 29]

        self.window_size = 3
        self.skip_center = False

        self.true_num_input_channels = (9) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            RestormerConvolutionBlock(self.true_num_input_channels, self.num_internal_channels)
        )

        self.encoder_net = nn.Sequential(
            GatedFormerBlock(self.num_internal_channels, 128),
            GatedFormerBlock(128, 128),
            GatedFormerBlock(128, 128),
            GatedFormerBlock(128, 128),
            FeedForwardGELU(128, 3, channel_multiplier=2)
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
    
        if temporal_state is None or True:
            prev_input = torch.zeros_like(input)
            hidden_state = [None] * self.num_filtering_scales
        else:
            prev_input, hidden_state = temporal_state
            prev_input = op_warp_tensor(prev_input, motionvec)
            hidden_state = [
                (op_warp_tensor(state[0], motionvec),
                 op_warp_tensor(state[1], motionvec))
                for state in hidden_state
            ]


        features = self.projector(
            torch.cat((
                input, 
                prev_input, 
                op_extract_nz_features(color, scales=self.nz_scales)
            ), dim=1)
        )

        filtered = checkpoint.checkpoint_sequential(
            self.encoder_net,
            segments=2,
            input=features,
            use_reentrant=False
        )

        # apply gamma correction to albedo before modulating it back in
        denoised = albedo.pow(1.0 / 2.2) * filtered
        next_temporal_state = None

        return (denoised, next_temporal_state)
    
