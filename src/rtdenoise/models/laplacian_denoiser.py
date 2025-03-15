"""
This file describes a Laplacian UNet Pyramid Denoiser (LPUNet). The denoiser is more of a skeleton for any multiscale denoising,
but I am using it to determine whether a lightweight MetaFormer architecture might be viable for denoising.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

class UNetConvolutionBlock(nn.Module):
    """A resolution level of a LPU Net. is_bottleneck changes input and output behavior"""
    def __init__(self, channels_in, channels_out, is_bottleneck):
        super(UNetConvolutionBlock, self).__init__()

        self.encoder = GatedFormerBlock(channels_in, channels_out)
        self.decoder = GatedFormerBlock(channels_out if is_bottleneck else 2 * channels_out, channels_in)

        self.kernel_alpha_predictor = FeedForwardReLU(channels_in, 10 if is_bottleneck else 11, channel_multiplier=2)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        dec = self.decoder(x)
        kernel_alpha = self.kernel_alpha_predictor(dec)

        # expect user to apply kernel prediction externally
        return dec, kernel_alpha

    def clear_memory(self):
        pass

class LaplacianDenoiser(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.nz_scales = [5, 11, 17, 29]

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.l_unet = LaplacianUNet(channels=[32, 32, 64, 64, 96, 128, 128])


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
            hidden_state = self.l_unet.create_empty_hidden_state()
        else:
            prev_input, hidden_state = temporal_state


        latent_features = self.projector(
            torch.cat((
                input, 
                prev_input, 
                op_extract_nz_features(color, scales=self.nz_scales)
            ), dim=1)
        )


        denoised = albedo * self.l_unet(color, latent_features, motionvec, hidden_state)
        temporal_state = (input, hidden_state)

        return (denoised, temporal_state)