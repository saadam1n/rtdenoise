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

"""
Architecture of the encoder block:
- Hadamard product of 3x3 depth conv and 1x1 conv
- Skip connection (add)
- Channel MLP w/ skip connection

What should we name this block? Well, what does this block do? 
According to our theory, it takes prior information, information of nearby pixels, 
and then updates that information. It creates a clearer picture of the local context.
"""

class LPUBlock(nn.Module):
    """Building block for the LPU"""
    def __init__(self, channels_in, channels_out):
        super(LPUBlock, self).__init__()

        self.pre_mlp_skip = nn.BatchNorm2d(channels_in)

        self.dw_conv = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1, groups=channels_in)
        )
        self.pw_conv = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=1)
        )
        self.channel_mlp = ChannelMlp(channels_in, channels_out, channel_multiplier=2)

        self.skip_transform = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=1)
        ) if channels_in != channels_out else None

    def forward(self, x):

        x_updated = self.dw_conv(x) + self.pre_mlp_skip(x)

        x_mlp = self.channel_mlp(x_updated) + (self.skip_transform(x_updated) if self.skip_transform is not None else x_updated)

        return x_mlp



class LPULevel(nn.Module):
    """A resolution level of a LPU Net. is_bottleneck changes input and output behavior"""
    def __init__(self, channels_in, channels_out, is_bottleneck):
        super(LPULevel, self).__init__()

        self.encoder = LPUBlock(channels_in, channels_out)
        self.decoder = LPUBlock(channels_out if is_bottleneck else 2 * channels_out, channels_in)

        self.kernel_alpha_predictor = ChannelMlp(channels_in, 10 if is_bottleneck else 11, channel_multiplier=2)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        dec = self.decoder(x)
        kernel_alpha = self.kernel_alpha_predictor(dec)

        # expect user to apply kernel prediction externally
        return dec, kernel_alpha

    def clear_memory(self):
        pass

class LaplacianPyramidUNet(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        self.feature_scales = [5, 11, 17, 29]
        self.feature_extractor = NonZeroFeatureExtractor(self.feature_scales)

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.feature_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.num_internal_channels, kernel_size=1)
        )

        self.levels = nn.ModuleList([
            LPULevel(32, 32, False),
            LPULevel(32, 64, False),
            LPULevel(64, 64, False),
            LPULevel(64, 96, False),
            LPULevel(96, 128, False),
            LPULevel(128, 128, True)
        ])

        self.per_pixel_conv = PerPixelConv(3)

        self.warp = TensorWarp()

    def run_frame(self, frame_input, temporal_state):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        trunc_input = frame_input[:, :9, :, :]
        motionvec = frame_input[:, 9:, :, :]
    
        if temporal_state is None:
            self.prev_trunc_input = torch.zeros_like(trunc_input)
            self.prev_filtered = [None] * len(self.levels)

            temporal_state = True # placeholder value
        else:
            self.prev_trunc_input = self.warp(self.prev_trunc_input, motionvec)

            self.prev_filtered = [
                self.warp(prev_filtered, motionvec) for prev_filtered in self.prev_filtered
            ]

        # transform features
        combined_temporal_input = torch.cat((trunc_input, self.prev_trunc_input, self.feature_extractor(frame_input[:, :3, :, :])), dim=1) 
        proj_input = self.projector(combined_temporal_input)

        
        # encoder pass 
        downsampled_color = []
        skip_tensors = []
        for i, level in enumerate(self.levels):
            downsampled_color.append(
                color if i == 0 else F.avg_pool2d(downsampled_color[i - 1], kernel_size=2, stride=2)
            )

            skip_tensors.append(
                checkpoint.checkpoint(
                    level.encode,
                    proj_input if i == 0 else F.max_pool2d(skip_tensors[i - 1], kernel_size=2, stride=2),
                    use_reentrant=False
                )
            )

        # decoder pass
        for i, level in reversed(list(enumerate(self.levels))):
            upsample_shape = (skip_tensors[i].shape[2], skip_tensors[i].shape[3])

            # input: upsampled decoder output + block 
            # calculate kernel and temporal alpha (and possible compositing alpha)
            dec, kernel_alpha = checkpoint.checkpoint(
                level.decode,
                skip_tensors[i] if i == len(self.levels) - 1 else torch.cat((skip_tensors[i], F.interpolate(dec, size=upsample_shape)), dim=1),
                use_reentrant=False
            )

            kernel = F.softmax(kernel_alpha[:, :9, :, :] / 3.0, dim=1)

            current_filtered = self.per_pixel_conv(downsampled_color[i], kernel)

            alpha_components = F.sigmoid(kernel_alpha[:, 9:, :, :])
            temporal_alpha = alpha_components[:, :1, :, :]

            if self.prev_filtered[i] is not None:
                current_filtered = current_filtered * temporal_alpha + self.warp(self.prev_filtered[i], motionvec) * (1.0 - temporal_alpha)

            if i == len(self.levels) - 1:
                filtered = current_filtered
            else:
                composite_alpha = alpha_components[:, 1:, :, :]

                freq_replacement = F.interpolate(filtered - F.avg_pool2d(current_filtered, kernel_size=2, stride=2), size=upsample_shape)
                filtered = current_filtered + composite_alpha * freq_replacement

            self.prev_filtered[i] = filtered

        # finalize output
        denoised_output = albedo * filtered

        # add the denoised value to the feedback
        self.prev_trunc_input = torch.cat((filtered, trunc_input[:, 3:, :, :]), dim=1)
        

        return (denoised_output, temporal_state)