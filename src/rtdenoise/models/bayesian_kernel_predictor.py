# implementation of some ideas I had
# Utilize 3x3 kernels everywhere
# Combine depth-wise convolution with shallow embedded U-nets in encoder, CFM, and channel MLP from metaformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_denoiser import BaseDenoiser
from .components import *


"""
A graphical model might help a bit more

Encoder -> Skip
        -> Pool -> Encoder -> Skip
                           -> Pool -> Encoder -> Skip                                                       -> Upscale + Concat -> Decoder -> 
                                              -> Pool -> Bottlneck -> Decoder -> KP Conv + Upscale features -> Convolved Image

We have a function to abstract away upscale + concat 
Decoder should output convolved image directly to prevent stuff from staying alive for very long 

"""

class AdaptivePool(nn.Module):
    def __init__(self, num_input_channels):
        super().__init__()

        self.mixer = nn.Parameter(torch.randn(1, num_input_channels, 1, 1))

    def forward(self, input):
        avg_pool = F.avg_pool2d(input, kernel_size=2, stride=2)
        max_pool = F.max_pool2d(input, kernel_size=2, stride=2)

        return avg_pool + max_pool * self.mixer

class BayesianReinforcer(nn.Module):
    def __init__(self, num_input_channels, num_bayesian_channels):
        super().__init__()

        self.num_bayesian_channels = num_bayesian_channels

        self.projector = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_bayesian_channels, kernel_size=1),
        )

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(num_bayesian_channels),
            nn.Conv2d(num_bayesian_channels, num_bayesian_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_bayesian_channels),
            nn.Conv2d(num_bayesian_channels, num_bayesian_channels, kernel_size=3, padding=1),
        )

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(num_bayesian_channels),
            nn.Conv2d(num_bayesian_channels, num_bayesian_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_bayesian_channels),
            nn.Conv2d(num_bayesian_channels * 2, num_bayesian_channels, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(num_bayesian_channels * 2),
            nn.Conv2d(num_bayesian_channels * 2, num_bayesian_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_bayesian_channels),
            nn.Conv2d(num_bayesian_channels * 2, num_bayesian_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, input):
        proj = self.projector(input)

        enc = self.encoder(proj) + proj

        pooled = F.max_pool2d(enc)

        processed = F.relu(self.bottleneck(pooled) + pooled)

        upscaled = F.upsample_bilinear(processed, scale_factor=2)

        skipped = torch.cat((enc, upscaled), dim=1)

        dec = self.decoder(skipped)

        cfm = dec[:, :self.num_bayesian_channels, :, :] * dec[:, self.num_bayesian_channels:, :, :] 
        
        return cfm

"""
Encoder structure:
- N input channels
- Pass all N input channels through 3x3 depth-wise convolution and add residual connection (cause why not)
- Push channels through Bayesian Reinforcement Block and concantenates new M channels with original N channels
- Residual MLP that goes from M -> 2N -> 2N
- Adaptive Pool

Bayesian Reinforcement Block (this I am not very clear on yet):
- Project N input channels to M channels
- 3x3 Conv X2 with residual connection, end with 2M channels
- MaxPool
- 3x3 Conv x2
- Bilinear Upsample
- Concatenate with skip connection
- Conv3x3, Conv1x1, CFM to go from 2M channels to M channels
"""
class KernelMatchingEncoder(nn.Module):
    def __init__(self, num_input_channels, num_bayesian_channels, num_output_channels):
        super().__init__()

        self.dw_conv = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1, groups=num_input_channels)
        )

        self.bayesian_reinforcer = nn.Sequential()

        self.channel_mlp = nn.Sequential(
            nn.BatchNorm2d(num_input_channels + num_bayesian_channels),
            nn.Conv2d(num_input_channels + num_bayesian_channels, 2 * num_input_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * num_input_channels),
            nn.Conv2d(2 * num_input_channels, num_output_channels, kernel_size=1),
        )

        self.res_projection = nn.Sequential(
            nn.BatchNorm2d(num_input_channels + num_bayesian_channels),
            nn.Conv2d(num_input_channels + num_bayesian_channels, num_output_channels, kernel_size=1)
        )


    def forward(self, input):
        features = self.dw_conv(input) + input

        bayesian = self.bayesian_reinforcer(features)

        combined = torch.cat((features, bayesian), dim=1)

        output = self.channel_mlp(combined) + self.res_projection(combined)

        return output

# incorporates kernel prediction
class KernelMatchingDecoderKP(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.kernel_dim = 3

        # placeholder for now
        self.decoder = nn.Sequential(
            DepthWiseConvFormer(num_input_channels),
            DepthWiseConvFormer(num_input_channels),
            nn.Conv2d(num_input_channels, num_output_channels + self.kernel_dim ** 2, kernel_size=3, padding=1)
        )

        self.unfold = nn.Unfold(kernel_size=3, padding=1)

    def forward(self, input, radiance):
        dec = self.decoder(input)

        features = dec[:, :self.num_output_channels, :, :]
        kernels = dec[:, self.num_output_channels:, :, :].view(input.size(0), 1, self.kernel_dim ** 2, input.size(2), input.size(3))

        patches = self.unfold(radiance).view(input.size(0), 3, self.kernel_dim ** 2, input.size(2), input.size(3))

        conv = kernels * patches

        accum = conv.sum(2)

        return (features, accum)