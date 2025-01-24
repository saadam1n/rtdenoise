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

# includes the encoder and decoder for a particular U-Net layer
class UNetStackableLayer(nn.Module):
    # bottleneck doesn't have to be a bottleneck; it could actually be another stackable layer
    def __init__(self, channels_in, channels_out, bottleneck):
        super(UNetStackableLayer, self).__init__()

        self.bottleneck = bottleneck

        self.encoder = KernelMatchingFormer(channels_in, channels_out) 
        self.decoder = KernelMatchingFormer(channels_out * 2, channels_in) 

        self.kernel_alpha_predictor = ChannelMlp(channels_in, 10, 2)

        self.per_pixel_conv = PerPixelConv(3)

    def forward(self, input, radiance):
        upscale_size = (input.size(2), input.size(3))

        enc = checkpoint.checkpoint(self.encoder, input, use_reentrant=False)

        binput = F.max_pool2d(enc, kernel_size=2, stride=2)
        bradiance = F.avg_pool2d(radiance, kernel_size=2, stride=2)

        bdec, bfiltered = checkpoint.checkpoint(self.bottleneck, binput, bradiance, use_reentrant=False)

        bdec = F.interpolate(bdec, size=upscale_size, mode="bilinear", align_corners=True)
        bfiltered = F.interpolate(bfiltered, size=upscale_size, mode="bilinear", align_corners=True)

        dinput = torch.cat((enc, bdec), dim=1)

        dec = checkpoint.checkpoint(self.decoder, dinput, use_reentrant=False)

        # get kernel and alpha
        kernel_alpha = self.kernel_alpha_predictor(dec)
        kernel = F.softmax(kernel_alpha[:, :9, :, :], dim=1)
        alpha = F.sigmoid(kernel_alpha[:, 9:, :, :])

        filtered = self.per_pixel_conv(radiance, kernel)

        # combine via a laplacian pyramid
        low_freq = F.interpolate(F.avg_pool2d(filtered, kernel_size=2, stride=2), size=upscale_size, mode="bilinear", align_corners=True)
        filtered = filtered - alpha * low_freq + alpha * bfiltered

        return (dec, filtered)

class UNetBottleneck(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UNetBottleneck, self).__init__()

        # use two cuz why not
        self.bottleneck = nn.Sequential(
            KernelMatchingFormer(channels_in, channels_in),
            KernelMatchingFormer(channels_in, channels_out),
        )
        self.per_pixel_conv = PerPixelConv(3)

        self.kernel_predictor = ChannelMlp(channels_out, 9, 2)

    def forward(self, input, radiance):
        conv = self.bottleneck(input)

        kernel = self.kernel_predictor(conv)
        kernel = F.softmax(kernel, dim=1)

        filtered = self.per_pixel_conv(radiance, kernel)

        return (conv, filtered)
    

class LaplacianPyramidUNet(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.num_input_channels),
            nn.Conv2d(self.num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.unet5 = UNetBottleneck(128, 128)
        self.unet4 = UNetStackableLayer(96, 128, self.unet5)
        self.unet3 = UNetStackableLayer(64, 96, self.unet4)
        self.unet2 = UNetStackableLayer(64, 64, self.unet3)
        self.unet1 = UNetStackableLayer(32, 64, self.unet2)
        self.unet0 = UNetStackableLayer(32, 32, self.unet1)



    def run_frame(self, frame_input, temporal_state):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        proj_input = self.projector(frame_input)

        ignore, filtered = self.unet0(proj_input, color)

        denoised_output = albedo * filtered

        return (denoised_output, temporal_state)