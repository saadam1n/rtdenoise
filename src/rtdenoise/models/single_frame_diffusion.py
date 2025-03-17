import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

"""
This class will have two implementations:
- A non-latent diffusion method which is color stable and hopefully is easier to train.
- A latent diffusion method which is not color stable and will use a CFM-like architecture for gather operations
"""

class SingleFrameDiffusion(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.nz_scales = [5, 11, 17, 29]

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.num_diffusion_steps = 3
        self.l_unets = nn.ModuleList([
            LaplacianUNet(channels=[self.num_internal_channels, 32, 32, 32, 48, 48, 64]) for _ in range(self.num_diffusion_steps)
        ])


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
            hidden_state = [
                l_unet.create_empty_hidden_state() for l_unet in self.l_unets
            ]
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


        filtered = color
        for i in range(len(self.l_unets)):
            filtered, latent_features, hidden_state[i] = checkpoint.checkpoint(
                self.l_unets[i], 
                filtered, 
                latent_features, 
                motionvec, 
                *hidden_state[i], 
                use_reentrant=False
            )

        denoised = albedo * filtered
        temporal_state = (torch.cat((filtered, input[:, 3:, :, :]), dim=1), hidden_state)

        return (denoised, temporal_state)

# latent diffusion has proved to be highly unstable
# we may need larger batch sizes or something idk
class LatentSingleFrameDiffusion(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24

        self.nz_scales = [5, 11, 17, 29]

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        # a small number of channels all the way to make this fast
        self.ld_unet = LatentDiffusionNet(channels=[self.num_internal_channels, 24, 24, 24, 24, 24, 24])

        self.latent_decoder = nn.Sequential(
            nn.BatchNorm2d(self.num_internal_channels),
            nn.Conv2d(self.num_internal_channels, 3, kernel_size=1)
        )

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
            hidden_state = self.ld_unet.create_empty_hidden_state()
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


        latent_filtered = self.ld_unet(latent_features, motionvec, hidden_state)
        filtered = self.latent_decoder(latent_filtered)

        denoised = albedo * filtered
        temporal_state = (torch.cat((filtered, input[:, 3:, :, :]), dim=1), hidden_state)

        return (denoised, temporal_state)