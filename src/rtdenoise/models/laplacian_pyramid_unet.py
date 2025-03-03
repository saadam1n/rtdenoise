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

def simple_warp(image, motionvec):
    with torch.no_grad():
        i = torch.arange(image.shape[2], device=image.device)
        j = torch.arange(image.shape[3], device=image.device)

        ii, jj = torch.meshgrid((i, j), indexing="ij")

        ii = torch.clamp(ii + motionvec[:, 1:, :, :], min=0, max=image.shape[2] - 1).to(torch.int32).view(-1, image.shape[2], image.shape[3])
        jj = torch.clamp(jj + motionvec[:, :1, :, :], min=0, max=image.shape[3] - 1).to(torch.int32).view(-1, image.shape[2], image.shape[3])
        batch_idx = torch.arange(image.shape[0], device=image.device).view(-1, 1, 1).expand_as(ii)

    warped = image[batch_idx, :, ii, jj].permute(0, 3, 1, 2)
    
    return warped


# includes the encoder and decoder for a particular U-Net layer
class UNetStackableLayer(nn.Module):
    # bottleneck doesn't have to be a bottleneck; it could actually be another stackable layer
    def __init__(self, channels_in, channels_out, bottleneck):
        super(UNetStackableLayer, self).__init__()

        self.bottleneck = bottleneck

        self.encoder = KernelMatchingFormer(channels_in, channels_out) 
        self.decoder = KernelMatchingFormer(channels_out * 2, channels_in) 

        self.kernel_alpha_predictor = ChannelMlp(channels_in, 11, 2)

        self.per_pixel_conv = PerPixelConv(3)

        self.prev_filtered = None

    def forward(self, input, radiance, temporal_state, motionvec):
        upscale_size = (input.size(2), input.size(3))

        enc = checkpoint.checkpoint(self.encoder, input, use_reentrant=False)

        binput = F.max_pool2d(enc, kernel_size=2, stride=2)
        bradiance = F.avg_pool2d(radiance, kernel_size=2, stride=2)
        bmotionvec = F.avg_pool2d(motionvec, kernel_size=2, stride=2)

        bdec, bfiltered = checkpoint.checkpoint(self.bottleneck, binput, bradiance, temporal_state, bmotionvec, use_reentrant=False)

        bdec = F.interpolate(bdec, size=upscale_size, mode="bilinear", align_corners=True)

        bfiltered = F.interpolate(bfiltered, size=upscale_size, mode="bilinear", align_corners=True)

        dinput = torch.cat((enc, bdec), dim=1)

        dec = checkpoint.checkpoint(self.decoder, dinput, use_reentrant=False)

        # get kernel and alpha
        kernel_alpha = self.kernel_alpha_predictor(dec)
        kernel = F.softmax(kernel_alpha[:, :9, :, :], dim=1)
        alpha = F.sigmoid(kernel_alpha[:, 9:, :, :])

        laplacian_alpha = alpha[:, :1, :, :]
        temporal_alpha = alpha[:, 1:, :, :]

        prev_filtered = simple_warp(self.prev_filtered, motionvec) if self.prev_filtered is not None else torch.zeros_like(radiance)

        filtered = self.per_pixel_conv(radiance, kernel) * (1.0 - temporal_alpha) + prev_filtered * temporal_alpha

        # combine via a laplacian pyramid
        low_freq = F.interpolate(F.avg_pool2d(filtered, kernel_size=2, stride=2), size=upscale_size, mode="bilinear", align_corners=True)
        filtered = filtered - laplacian_alpha * low_freq + laplacian_alpha * bfiltered

        self.prev_filtered = filtered

        return (dec, filtered)
    
    def clear_memory(self):
        self.prev_filtered = None
        self.bottleneck.clear_memory()

class UNetBottleneck(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(UNetBottleneck, self).__init__()

        # use two cuz why not
        self.bottleneck = nn.Sequential(
            KernelMatchingFormer(channels_in, channels_in),
            KernelMatchingFormer(channels_in, channels_out),
        )
        self.per_pixel_conv = PerPixelConv(3)

        self.kernel_predictor = ChannelMlp(channels_out, 10, 2)

        self.prev_filtered = None

    def forward(self, input, radiance, temporal_state, motionvec):
        conv = self.bottleneck(input)

        kernel_alpha = self.kernel_predictor(conv)
        kernel = F.softmax(kernel_alpha[:, :9, :, :], dim=1)
        temporal_alpha = F.sigmoid(kernel_alpha[:, 9:, :, :])

        prev_filtered = simple_warp(self.prev_filtered, motionvec) if self.prev_filtered is not None else torch.zeros_like(radiance)
        filtered = self.per_pixel_conv(radiance, kernel) * (1.0 - temporal_alpha) + prev_filtered * temporal_alpha

        self.prev_filtered = filtered

        return (conv, filtered)
    
    def clear_memory(self):
        self.prev_filtered = None
    

class LaplacianPyramidUNet(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.true_num_input_channels = (self.num_input_channels - 2) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.unet5 = UNetBottleneck(128, 128)
        self.unet4 = UNetStackableLayer(96, 128, self.unet5)
        self.unet3 = UNetStackableLayer(64, 96, self.unet4)
        self.unet2 = UNetStackableLayer(64, 64, self.unet3)
        self.unet1 = UNetStackableLayer(32, 64, self.unet2)
        self.unet0 = UNetStackableLayer(32, 32, self.unet1)

        self.prev_frame_input = None

    def run_frame(self, frame_input, temporal_state):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        trunc_input = frame_input[:, :9, :, :]
        motionvec = frame_input[:, 9:, :, :]
    
        prev_frame_input = self.prev_frame_input if temporal_state is not None else torch.zeros_like(trunc_input)

        # transform features
        combined_temporal_input = torch.cat((trunc_input, prev_frame_input), dim=1)
        proj_input = self.projector(combined_temporal_input)

        ignore, filtered = self.unet0(proj_input, color, temporal_state, motionvec)

        # finalize output
        denoised_output = albedo * filtered
        self.prev_frame_input = torch.cat((denoised_output, frame_input[:, 3:9, :, :]), dim=1)

        # value doesn't matter, this is more of a marker
        temporal_state = self.prev_frame_input

        return (denoised_output, temporal_state)