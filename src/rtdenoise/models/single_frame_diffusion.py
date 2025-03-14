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

class DiffusionEncoderDecoder(nn.Module):
    """Some Information about DiffusionEncoderDecoder"""
    def __init__(self, channels_in: int, channels_out:int, bottleneck : bool, next_block):
        super(DiffusionEncoderDecoder, self).__init__()

        self.bottleneck = bottleneck
        self.next_block = next_block

        if self.bottleneck and self.next_block is not None:
            raise ValueError("Bottleneck and next block cannot both be defined for an encoder-decoder pair")
        elif not self.bottleneck and self.next_block is None:
            raise ValueError("An encoder-decoder pair must have a next_block or be a bottleneck")

        self.encoder = DepthPointwiseBlock(channels_in, channels_out)
        self.decoder = DepthPointwiseBlock(channels_out * 2 if not self.bottleneck else channels_out, channels_in)

        self.kernel_alpha_extractor = ChannelMlp(channels_in, 11 if not self.bottleneck else 10, channel_multiplier=2)

        self.ppc = PerPixelConv(3)


    def forward(self, radiance, temporal_radiance, latent):
        encoded_latent = checkpoint.checkpoint(self.encoder, latent, use_reentrant=False)
        filtered_downsampled, next_latent = checkpoint.checkpoint(self.call_next, radiance, temporal_radiance, encoded_latent, use_reentrant=False)
        decoded_latent = checkpoint.checkpoint(self.decode_latent, encoded_latent, next_latent, use_reentrant=False)
        filtered = checkpoint.checkpoint(self.filter_image, radiance, temporal_radiance, filtered_downsampled, decoded_latent, use_reentrant=False)

        return filtered, decoded_latent
        
    def filter_image(self, radiance, temporal_radiance, filtered_downsampled, decoded_latent):
        ka = self.kernel_alpha_extractor(decoded_latent)

        # do not apply softmax reduction for more expressive power
        kernel = ka[:, :9, :, :]
        temporal_alpha = ka[:, 9:10, :, :]

        filtered = self.ppc(radiance, kernel) * temporal_alpha + temporal_radiance * (1.0 - temporal_alpha)

        if not self.bottleneck:
            composite_alpha = ka[:, 10:11, :, :]
            low_frequency = F.interpolate(
                filtered_downsampled - F.avg_pool2d(filtered, kernel_size=2, stride=2),
                size=radiance.shape[2:]
            )
            filtered = filtered * composite_alpha + low_frequency * (1.0 - composite_alpha)     

        return filtered   
    
    def call_next(self, radiance, temporal_radiance, encoded_latent):
        # call the next level down
        if not self.bottleneck:
            filtered_downsampled, next_latent = checkpoint.checkpoint(
                self.next_block, 
                F.avg_pool2d(radiance, kernel_size=2, stride=2), 
                F.avg_pool2d(temporal_radiance, kernel_size=2, stride=2), 
                F.max_pool2d(encoded_latent, kernel_size=2, stride=2),
                use_reentrant=False 
            )
        else:
            filtered_downsampled = None
            next_latent = None

        return (filtered_downsampled, next_latent)

    def decode_latent(self, encoded_latent, next_latent):
        if not self.bottleneck:
            upsampled_latent = F.interpolate(
                next_latent, 
                size=encoded_latent.shape[2:],
                mode="bilinear", align_corners=True
            )

        decoded_latent = checkpoint.checkpoint(
            self.decoder,
            encoded_latent if self.bottleneck else torch.cat((encoded_latent, upsampled_latent), dim=1),
            use_reentrant=False
        )

        return decoded_latent

class DiffusionUNet(nn.Module):
    def __init__(self, channels):
        super(DiffusionUNet, self).__init__()

        self.encoder_decoders = nn.ModuleList([])

        prev_block = None
        for i in range(len(channels) - 1, 0, -1):
            current_block = DiffusionEncoderDecoder(
                channels_in=channels[i - 1],
                channels_out=channels[i],
                bottleneck=True if prev_block is None else False,
                next_block=prev_block
            )

            self.encoder_decoders.append(current_block)
            prev_block = current_block

    def forward(self, packed):
        color = packed[:, :3, :, :]
        temporal = packed[:, 3:6, :, :]
        latent = packed[:, 6:, :, :]

        filtered, decoded_latent = checkpoint.checkpoint(self.encoder_decoders[-1], color, temporal, latent, use_reentrant=False)

        # add skip connection for latent
        repacked = torch.cat((filtered, temporal, decoded_latent), dim=1)

        return repacked

"""
Goal of this class is to implement a lightweight diffusion network that runs several times over the input in a frame
"""
class SingleFrameDiffusion(BaseDenoiser):
    def init_components(self):
        self.feature_scales = [5, 11, 17, 29]
        self.sfd_channels = [32, 32, 48, 48, 64, 64]

        self.feature_extractor = NonZeroFeatureExtractor(self.feature_scales)

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.feature_scales) * 2
        print(f"True num input channels is {self.true_num_input_channels}")

        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            nn.Conv2d(self.true_num_input_channels, self.sfd_channels[0], kernel_size=1)
        )

        self.sfd_net = nn.Sequential(
            DiffusionUNet(self.sfd_channels),
            DiffusionUNet(self.sfd_channels),
            DiffusionUNet(self.sfd_channels),
        )

        self.warp = TensorWarp()

    def run_frame(self, frame_input, temporal_state):
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        motionvec = frame_input[:, 9:, :, :]

        """
        Temporal state is a tuple of (prev filtered, prev input)
        """
        if temporal_state is None:
            prev_filtered = torch.zeros(B, 3, H, W, device=frame_input.device)
            prev_input = torch.zeros(B, self.num_input_channels - 2 - 3, H, W, device=frame_input.device)
        else:
            prev_filtered, prev_input = temporal_state

            prev_filtered = self.warp(prev_filtered, motionvec)
            prev_input = self.warp(prev_input, motionvec)

        # feature extraction does not need gradient calculation
        with torch.no_grad():
            nz_features = self.feature_extractor(color)

        proj_input = self.projector(torch.cat(
            (
                frame_input[:, :9, :, :],
                nz_features,
                prev_filtered,
                prev_input
            ),
            dim=1
        ))
        

        packed = checkpoint.checkpoint_sequential(
            self.sfd_net,
            segments=1,
            input=torch.cat(
                (color, prev_filtered, proj_input),
                dim=1
            ),
            use_reentrant=False
        )

        filtered = packed[:, :3, :, :]

        next_temporal_state = (filtered, frame_input[:, 3:9, :, :])
        remodulated = albedo * filtered

        return (remodulated, next_temporal_state)
