import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

import math

class GlobalContextPreEncoderTransformer(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 24
        self.unet_channels = [self.num_internal_channels, 24, 24, 32, 32, 48, 64]
        self.num_filtering_scales = len(self.unet_channels) - 1

        self.nz_scales = [5, 11, 17, 29]

        # analagous to a 3x3 kernel
        self.window_size = 1
        self.band_size = 1
        self.manual_spda = False

        # concate previous frame input and ignore motion vectors
        self.true_num_input_channels = (self.num_input_channels - 2) * 2 + len(self.nz_scales) * 2
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.true_num_input_channels),
            RestormerConvolutionBlock(self.true_num_input_channels, self.num_internal_channels)
        )

        self.encoder_net = nn.Sequential(
            FastUNet(channels=self.unet_channels, per_level_outputs=False),
            FastUNet(channels=self.unet_channels, per_level_outputs=False),
            FastUNet(channels=self.unet_channels, per_level_outputs=True)
        )

        self.qk_extractors = nn.ModuleList([ 
            nn.Sequential(
                ImageLayerNorm(self.unet_channels[i]),
                nn.Conv2d(self.unet_channels[i], self.num_transformer_channels, kernel_size=1, bias=False)
            ) for i in range(self.num_filtering_scales)
        ])

        with torch.no_grad():
            for seq in self.qk_extractors:
                seq[1].weight.mul_(0.05)


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

        qk_latent = checkpoint.checkpoint_sequential(
            self.encoder_net,
            segments=1,
            input=features,
            use_reentrant=False
        )

        filter_outputs = checkpoint.checkpoint(self.hierarchical_filter, color, qk_latent, hidden_state, use_reentrant=False)

        filtered = filter_outputs[0][0]

        denoised = albedo * filtered
        next_temporal_state = (torch.cat((filtered, input[:, 3:, :, :]), dim=1), filter_outputs)

        return (denoised, next_temporal_state)
    
    def hierarchical_filter(self, radiance, qk_latent, hidden_state):
        # build downsample list
        ds_color = []
        for i in range(self.num_filtering_scales):
            ds_color.append(
                radiance if i == 0 else F.avg_pool2d(ds_color[i - 1], kernel_size=2, stride=2)
            )

        outputs = [None] * self.num_filtering_scales

        color_ds = None
        qk_ds = None
        for i in range(self.num_filtering_scales - 1, -1, -1):
            color_ds, qk_ds = checkpoint.checkpoint(
                self.transformer_filter,
                ds_color[i],
                qk_latent[i],
                hidden_state[i][0] if hidden_state[i] else None,
                hidden_state[i][1] if hidden_state[i] else None,
                i,
                color_ds,
                qk_ds,
                use_reentrant=False
            )

            outputs[i] = (color_ds, qk_ds)

        return outputs

    def transformer_filter(
        self, 
        color_fr, 
        qk_latent_fr, 
        color_t,
        qk_t,
        extractor_index, 
        color_ds, 
        qk_ds
    ):
        qk_fr = checkpoint.checkpoint_sequential(
            self.qk_extractors[extractor_index], 
            input=qk_latent_fr, 
            use_reentrant=False,
            segments=1
        )

        if color_ds is not None:
            color_ds = F.interpolate(
                color_ds,
                size=color_fr.shape[2:],
                mode="bilinear",
                align_corners=True
            )

            qk_ds = F.interpolate(
                qk_ds,
                size=qk_fr.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        filtered = kernel_attn(
            qk0=qk_fr, v0=color_fr, 
            qk1=qk_t, v1=color_t, 
            qk2=qk_ds, v2=color_ds,
            window_size=self.window_size
        )

        return filtered, qk_fr
