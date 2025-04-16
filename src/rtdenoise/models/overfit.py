import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

from .utils import *

class ResNeXtBlock(nn.Module):
    def __init__(self, num_input_channels, num_intermediate_channels, num_groups, dilation):
        super().__init__()

        self.num_groups = num_groups

        self.encode = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_intermediate_channels * num_groups, kernel_size=1),
            nn.GELU(),
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_intermediate_channels * num_groups),
            nn.Conv2d(num_intermediate_channels * num_groups, num_intermediate_channels * num_groups, kernel_size=5, padding=2 * dilation, dilation=dilation, groups=self.num_groups),
            nn.GELU(),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm2d(num_intermediate_channels * num_groups),
            nn.Conv2d(num_intermediate_channels * num_groups, num_input_channels, kernel_size=1),
        )

    def forward(self, input):
        skip = input

        enc = self.encode(input)
        conv = self.conv(enc)
        dec = self.decode(conv)

        output = dec + skip

        return output

class OverfitNet(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 48

        self.nz_scales = [5, 11, 17, 29]

        self.true_num_input_channels = (9) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.encoder_net = nn.Sequential(
            ResNeXtBlock(num_input_channels=self.num_internal_channels, num_intermediate_channels=6, num_groups=16, dilation=1),
            ResNeXtBlock(num_input_channels=self.num_internal_channels, num_intermediate_channels=6, num_groups=16, dilation=1),
            ResNeXtBlock(num_input_channels=self.num_internal_channels, num_intermediate_channels=6, num_groups=16, dilation=1),
            ResNeXtBlock(num_input_channels=self.num_internal_channels, num_intermediate_channels=6, num_groups=16, dilation=1),
            FeedForwardGELU(self.num_internal_channels, 3, channel_multiplier=2, has_skip=True)
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
        else:
            prev_input, hidden_state = temporal_state
            prev_input = op_warp_tensor(prev_input, motionvec)
            hidden_state = [
                (op_warp_tensor(state[0], motionvec),
                 op_warp_tensor(state[1], motionvec))
                for state in hidden_state
            ]

        quick_save_img("/tmp/color1.exr", color)
        quick_save_img("/tmp/color2.exr", F.avg_pool2d(color, kernel_size=2, stride=2))
        quick_save_img("/tmp/color4.exr", F.avg_pool2d(color, kernel_size=4, stride=4))
        quick_save_img("/tmp/color8.exr", F.avg_pool2d(color, kernel_size=8, stride=8))
        quick_save_img("/tmp/color16.exr", F.avg_pool2d(color, kernel_size=16, stride=16))


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
    
