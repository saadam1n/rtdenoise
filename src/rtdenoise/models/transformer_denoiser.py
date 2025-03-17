import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

class TransformerDenoiser(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.nz_scales = [5, 11, 17, 29]

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.t_unet = TransformerUNet(channels=[self.num_internal_channels, 32, 32, 48, 48, 64, 96])


    def run_frame(self, frame_input : torch.Tensor, temporal_state):
        N = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        input = frame_input[:, :9, :, :]
        motionvec = frame_input[:, 9:, :, :]
    
        if temporal_state is None:
            prev_input = torch.zeros_like(input)
            hidden_state = self.t_unet.create_empty_hidden_state()
        else:
            prev_input, hidden_state = temporal_state
            prev_input = op_warp_tensor(prev_input, motionvec)


        latent_features = self.projector(
            torch.cat((
                input, 
                prev_input, 
                op_extract_nz_features(color, scales=self.nz_scales)
            ), dim=1)
        )


        filtered, hidden_state = checkpoint.checkpoint(
            self.t_unet, 
            color, 
            latent_features, 
            motionvec, 
            *hidden_state, 
            use_reentrant=False
        )

        denoised = albedo * filtered
        temporal_state = (torch.cat((filtered, input[:, 3:, :, :]), dim=1), hidden_state)

        return (denoised, temporal_state)