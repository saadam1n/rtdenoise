import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint

import openexr_numpy as exr
import os
import sys

"""
Custom "operators"
"""

"""
A generic operator for applying different convolution kernels per pixel.
Useful for using the results of a kernel prediction block
"""
def op_per_pixel_conv(image: torch.Tensor, kernel: torch.Tensor, kernel_size: int):
    N, _, H, W = image.shape

    unfolded = F.unfold(image, kernel_size=kernel_size, padding=kernel_size // 2)

    unfolded = unfolded.view(N, 3, 9, H, W)

    filtered = unfolded * kernel.unsqueeze(1)
    filtered = filtered.sum(2)

    return filtered

"""
Warping operator. Expects full resolution motion vectors. Image can be at any scale.
"""
def op_warp_tensor(image : torch.Tensor, motionvec : torch.Tensor):
    N, _, H, W = motionvec.shape
    _, _, H2, W2 = image.shape

    x, y = torch.meshgrid(
        torch.linspace(-1, 1, W, device=image.device),
        torch.linspace(-1, 1, H, device=image.device),
        indexing="xy"
    )

    sample_positons = torch.stack(
        [
            motionvec[:, 0, :, :] * 2.0 / W + x,
            y - motionvec[:, 1, :, :] * 2.0 / H
        ],
        dim=3
    )

    warped_image = F.grid_sample(image, sample_positons, mode="bilinear", padding_mode="border", align_corners=True)

    resized_warped_image = F.interpolate(warped_image, size=(H2, W2), mode="bilinear", align_corners=True)

    return resized_warped_image

"""This operator extracts more useful information on a global context from the image."""
def op_extract_nz_features(image : torch.Tensor, scales=list[int]):
    # we disable gradient calculation because this is performed directly on the inputs and has no parameters
    # we don't want the backward pass keeping track of unnessary tensors
    with torch.no_grad():
        lum = F.conv2d(image, weight=torch.ones(1, 3, 1, 1, device=image.device) / 3.0)
        nonzero = (lum != 0).float()

        nz_percentages = []
        nz_averages = []

        for scale in scales:
            percentage = F.avg_pool2d(nonzero, kernel_size=scale, stride=1, padding=scale // 2)
            average = F.avg_pool2d(lum, kernel_size=scale, stride=1, padding=scale // 2) / percentage.clamp_min(min=1.0 / (scale ** 2))

            nz_percentages.append(percentage)
            nz_averages.append(average)

        features = torch.cat(nz_percentages + nz_averages, dim=1)

    return features


def op_dbg_channel_stats(image: torch.Tensor):
    with torch.no_grad():
        var, mean = torch.var_mean(image, dim=[0, 2, 3], keepdim=False, unbiased=False)
    return var, mean

"""
Modules
"""

class FeedForwardReLU(nn.Module):
    """
    A very basic Channel MLP that does two 1x1 convolutions. Skip connection not included.
    """
    def __init__(self, channels_in, channels_out, channel_multiplier):
        super(FeedForwardReLU, self).__init__()

        self.channel_mlp = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channel_multiplier * channels_in, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(channel_multiplier * channels_in),
            nn.Conv2d(channel_multiplier * channels_in, channels_out, kernel_size=1),
        )

    def forward(self, input):
        return self.channel_mlp(input)

class GatedFormerBlock(nn.Module):
    """
    Building block for the LPU and other U-Nets.
    To improve performance and numerical stability, the post-FFN skip connection has been removed.
    """
    def __init__(self, channels_in, channels_out):
        super(GatedFormerBlock, self).__init__()

        self.dw_conv = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1, groups=channels_in),
        )

        # we utilize sigmoid instead of ReLU here for increased numerical stability
        self.pw_conv = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=1),
            nn.Sigmoid() 
        )

        self.channel_mlp = FeedForwardReLU(channels_in, channels_out, channel_multiplier=2)


    """
    Refer to "GLU Variants Improve Transformer"
    https://arxiv.org/pdf/2002.05202v1 

    We notice that our DW x PW combination is analogous to a ReGLU.
    Since the PW convolution will have more expressive power, we put the ReLU on that.
    """
    def forward(self, x):
        """
        We need to be careful about weight initialization to prevent exploding gradients.
        - Assume x is value with mean 0 and var 1
        - DW conv produces value with mean 0 and var 1
        - PW conv produces value with mean 0 and var 1
        - DW * PW produces value with mean 0 and var 1
        - DW * PW + x produces value with mean 0 and var 2
        """
        glu = self.dw_conv(x) * self.pw_conv(x) + x

        mlp = self.channel_mlp(glu)

        return mlp
    
class DenseConvBlock(nn.Module):
    """
    Two back-to-back dense convolutions.
    Meant to be used for debugging.
    """
    def __init__(self, channels_in, channels_out):
        super(DenseConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(channels_out),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class UNetConvolutionBlock(nn.Module):
    """
    A general convolution block for U-Nets. Contains an encoder and decoder pair built on DepthPointwiseBlock.
    If is_bottleneck is true, the decoder block will not expect a skip connection.
    """
    def __init__(self, channels_in, channels_out, channels_extra, is_bottleneck):
        super(UNetConvolutionBlock, self).__init__()

        self.encoder = GatedFormerBlock(channels_in, channels_out)
        self.decoder = GatedFormerBlock(channels_extra + (channels_out if is_bottleneck else 2 * channels_out), channels_in)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


class LaplacianFilter(nn.Module):
    """
    Laplacian filter does non-latent filtering via the laplacian pyramid.
    It has built-in kernel prediction.
    """
    def __init__(self, channels_in, is_bottleneck):
        super(LaplacianFilter, self).__init__()

        self.num_ka_channels = 10 if is_bottleneck else 11

        self.ka_predictor = FeedForwardReLU(channels_in, self.num_ka_channels, channel_multiplier=2)

    def forward(self, latent_ka, radiance, hidden_state, prev_level, motionvec):
        # channel format: (kernel, temporal alpha, composite alpha)
        ka = self.ka_predictor(latent_ka)

        # like the "Attention is All You Need" paper we divide our logits 
        # by a constant value to increase gradient flow early in training
        kernel = F.softmax(ka[:, :9, :, :] / 9.0, dim=1)
        alpha_t = F.sigmoid(ka[:, 9:10, :, :] / 9.0)

        if hidden_state is None:
            hidden_state = torch.zeros_like(radiance)
        else:
            hidden_state = op_warp_tensor(hidden_state, motionvec)

        filtered = op_per_pixel_conv(image=radiance, kernel=kernel, kernel_size=3) * alpha_t + hidden_state * (1.0 - alpha_t)

        # laplacian composition involves swapping out the low frequencies 
        # of a signal with its downsampled counterpart.
        # this can be viewed as a gated linear unit that modulates both images
        # to combine them.
        if prev_level is not None:
            alpha_c = F.sigmoid(ka[:, 10:, :, :] / 9.0)

            delta_bands = F.interpolate(
                prev_level - F.avg_pool2d(filtered, kernel_size=2, stride=2),
                size=filtered.shape[2:4]
            )

            filtered = filtered + delta_bands * (1.0 - alpha_c)

        return filtered


class LaplacianUNet(nn.Module):
    def __init__(self, channels):
        super(LaplacianUNet, self).__init__()

        # self.channels[0] corresponds to the input to the entire network
        # self.channels[1...n] corresponds to the output of each encoder block
        self.channels = channels

        self.levels = nn.ModuleList([
            UNetConvolutionBlock(
                channels_in=self.channels[i - 1], 
                channels_out=self.channels[i], 
                channels_extra=0,
                is_bottleneck=(i == len(self.channels) - 1)
            ) for i in range(1, len(self.channels))
        ])

        self.filters = nn.ModuleList([
            LaplacianFilter(
                channels_in=self.channels[i - 1], 
                is_bottleneck=(i == len(self.channels) - 1)
            ) for i in range(1, len(self.channels))
        ])

    def forward(self, radiance, latent, motionvec, hidden_states):
        # encoder pass 
        ds_color = []
        skip_tensors = []

        for i, level in enumerate(self.levels):
            ds_color.append(
                radiance if i == 0 
                else F.avg_pool2d(ds_color[i - 1], kernel_size=2, stride=2)
            )

            skip_tensors.append(
                checkpoint.checkpoint(
                    level.encode,
                    latent if i == 0 
                    else F.max_pool2d(skip_tensors[i - 1], kernel_size=2, stride=2),
                    use_reentrant=False
                )
            )

        # decoder pass
        filtered = None
        for i, level in reversed(list(enumerate(self.levels))):
            decoded_latent = checkpoint.checkpoint(
                level.decode,
                skip_tensors[i] if i == len(self.levels) - 1 
                else torch.cat((skip_tensors[i], F.interpolate(decoded_latent, size=skip_tensors[i].shape[2:4], mode="bilinear", align_corners=True)), dim=1),
                use_reentrant=False
            )

            filtered = checkpoint.checkpoint(
                self.filters[i],
                decoded_latent,
                ds_color[i], 
                hidden_states[i],
                filtered,
                motionvec,
                use_reentrant=False
            )

            hidden_states[i] = filtered

        return filtered
        
    def create_empty_hidden_state(self):
        return [None] * len(self.filters)
    

class LatentDiffusionNet(nn.Module):
    def __init__(self, channels):
        super(LatentDiffusionNet, self).__init__()

        # self.channels[0] corresponds to the input to the entire network
        # self.channels[1...n] corresponds to the output of each encoder block
        self.channels = channels

        self.levels = nn.ModuleList([
            UNetConvolutionBlock(
                channels_in=self.channels[i - 1], 
                channels_out=self.channels[i], 
                channels_extra=self.channels[i],
                is_bottleneck=(i == len(self.channels) - 1)
            ) for i in range(1, len(self.channels))
        ])

    def forward(self, latent, motionvec, hidden_states):
        # encoder pass 
        skip_tensors = []
        for i, level in enumerate(self.levels):
            skip_tensors.append(
                checkpoint.checkpoint(
                    level.encode,
                    latent if i == 0 
                    else F.max_pool2d(skip_tensors[i - 1], kernel_size=2, stride=2),
                    use_reentrant=False
                )
            )

        # decoder pass
        for i, level in reversed(list(enumerate(self.levels))):
            decoder_inputs = [
                skip_tensors[i],
                op_warp_tensor(hidden_states[i], motionvec) if hidden_states[i] is not None else torch.zeros_like(skip_tensors[i])
            ]

            if i != len(self.levels) - 1:
                decoder_inputs.append(
                    F.interpolate(decoded_latent, size=skip_tensors[i].shape[2:4], mode="bilinear", align_corners=True)
                )

            decoded_latent = checkpoint.checkpoint(
                level.decode,
                torch.cat(decoder_inputs, dim=1),
                use_reentrant=False
            )

            hidden_states[i] = decoded_latent

        # add skip connection
        return decoded_latent + latent
        
    def create_empty_hidden_state(self):
        return [None] * len(self.levels)