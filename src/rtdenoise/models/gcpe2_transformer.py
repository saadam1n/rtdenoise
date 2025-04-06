import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

# we take some inspiration from restormer for this one
class QueryKeyExtractor2(nn.Module):
    def __init__(self, channels_in, transformer_channels):
        super(QueryKeyExtractor2, self).__init__()

        self.qke = nn.Sequential(
            ImageLayerNorm(channels_in),
            nn.Conv2d(channels_in, transformer_channels, kernel_size=1, bias=False)
        )


    def forward(self, x):
        return self.qke(x)


class HybridSoftmax(nn.Module):
    """
    Allows neural network to compute offset for softmax.
    """
    def __init__(self, num_channels):
        super(HybridSoftmax, self).__init__()

        self.num_channels = num_channels

    def forward(self, x):
        logits = x[:, :self.num_channels]
        offset = x[:, self.num_channels:]

        softmax = F.softmax(logits, dim=1)

        hybrid = softmax + offset


        return hybrid

class PartialSoftmax(nn.Module):
    def __init__(self, num_channels):
        super(PartialSoftmax, self).__init__()

        self.num_channels = num_channels

    def forward(self, x):
        logits = x[:, :self.num_channels]
        offset = x[:, self.num_channels:]

        softmax = F.softmax(logits, dim=1)

        hybrid = torch.cat((softmax, offset), dim=1)


        return hybrid

"""
Variant of GCPE that does composition differently. It does a softmax linear combination of various filtering levels. 
"""
class GCPE2(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 16
        self.unet_channels = [self.num_internal_channels, 24, 24, 32, 32, 48, 64]
        self.num_filtering_scales = len(self.unet_channels) - 1

        self.nz_scales = [5, 11, 17, 29]

        self.window_size = 3
        self.skip_center = False

        self.true_num_input_channels = (9) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            RestormerConvolutionBlock(self.true_num_input_channels, self.num_internal_channels)
        )

        self.encoder_net = nn.Sequential(
            FastUNet(channels=self.unet_channels, per_level_outputs=True)
        )

        self.qk_extractors = nn.ModuleList([ 
            QueryKeyExtractor2(
                channels_in=self.unet_channels[i], 
                transformer_channels=self.num_transformer_channels + (1 if self.skip_center else 0),
            )
            for i in range(self.num_filtering_scales)
        ])

        self.weight_extractor = nn.Sequential(
            FeedForwardGELU(self.unet_channels[0], 2 * self.num_filtering_scales, channel_multiplier=2),
            PartialSoftmax(self.num_filtering_scales)
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
            hidden_state = [None] * self.num_filtering_scales
        else:
            prev_input, hidden_state = temporal_state
            prev_input = op_warp_tensor(prev_input, motionvec)
            hidden_state = [
                (op_warp_tensor(state[0], motionvec),
                 op_warp_tensor(state[1], motionvec))
                for state in hidden_state
            ]


        features = self.projector(
            torch.cat((
                input, 
                prev_input, 
                op_extract_nz_features(color, scales=self.nz_scales)
            ), dim=1)
        )

        latents = checkpoint.checkpoint_sequential(
            self.encoder_net,
            segments=1,
            input=features,
            use_reentrant=False
        )

        filter_outputs = checkpoint.checkpoint(self.filter_images, latents, color, use_reentrant=False)

        filtered = filter_outputs

        # apply gamma correction to albedo before modulating it back in
        denoised = albedo.pow(1.0 / 2.2) * filtered
        next_temporal_state = (torch.cat((filtered, input[:, 3:, :, :]), dim=1), filter_outputs)

        return (denoised, next_temporal_state)
    
    def filter_images(self, latents, radiance):

        filtered_scales = []
        filtered_scales_v2 = []
        for i in range(self.num_filtering_scales):
            current_ds = radiance if i == 0 else F.avg_pool2d(current_ds, kernel_size=2, stride=2)

            filtered_ds = checkpoint.checkpoint(
                self.filter_scale,
                latents[i],
                current_ds,
                i,
                use_reentrant=False
            )            

            filtered_scales.append(
                F.interpolate(
                    input=filtered_ds,
                    size=radiance.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            )

            filtered_scales_v2.append(
                F.interpolate(
                    input=F.avg_pool2d(filtered_ds, kernel_size=2, stride=2),
                    size=radiance.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            )

        stacked_scales = torch.stack(filtered_scales + filtered_scales_v2, dim=1)

        weights = self.weight_extractor(latents[0]).unsqueeze(2)

        composition = (stacked_scales * weights).sum(dim=1)

        return composition

    def filter_scale(self, latent, radiance, i):
        cwqk = self.qk_extractors[i](latent)

        if not self.skip_center:
            center_weight = cwqk[:, :1].sigmoid()
            qk = cwqk[:, 1:]
        else:
            qk = cwqk



        center_filtered = kernel_attn(
            qk0=qk, v0=radiance,
            qk1=None, v1=None,
            qk2=None, v2=None,
            window_size=self.window_size,
            skip_center=self.skip_center
        )    

        if not self.skip_center:
            center_filtered = center_weight * radiance + (1.0 - center_weight) * center_filtered

        return center_filtered
    
