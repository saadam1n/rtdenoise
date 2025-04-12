import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

# we take some inspiration from restormer for this one
class QueryKeyExtractor3(nn.Module):
    def __init__(self, channels_in, transformer_channels):
        super(QueryKeyExtractor3, self).__init__()

        self.qke = nn.Sequential(
            ImageLayerNorm(channels_in),
            nn.Conv2d(channels_in, transformer_channels, kernel_size=1, bias=False)
        )


    def forward(self, x):
        return self.qke(x)


"""
Variant of GCPE that does composition differently. It does a softmax linear combination of various filtering levels. 
"""
class GCPE3(BaseDenoiser):
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
            QueryKeyExtractor3(
                channels_in=self.unet_channels[i], 
                transformer_channels=self.num_transformer_channels + (1 if self.skip_center else 0),
            )
            for i in range(self.num_filtering_scales)
        ])

        self.bias_extractors = nn.ModuleList([
            FeedForwardGELU(
                channels_in=2, 
                channels_out=9, 
                channel_multiplier=9,
                has_skip=True
            )
            for _ in range(self.num_filtering_scales)
        ])

        self.weight_extractor = nn.Sequential(
            FeedForwardGELU(self.unet_channels[0], self.num_filtering_scales, channel_multiplier=2),
            nn.Softmax(dim=1)
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

        q = self.qk_extractors[0](latents[0])

        for i in range(self.num_filtering_scales):
            current_ds = radiance if i == 0 else F.avg_pool2d(current_ds, kernel_size=2, stride=2)

            # you know what... let's just use raw average pooling
            # let upscaling do the rest of the job
            
            filtered_scales.append(
                checkpoint.checkpoint(self.attn_upscale, q, latents[i], current_ds, i, use_reentrant=False)
                if i != 0 else 
                current_ds
            )

        stacked_scales = torch.stack(filtered_scales, dim=1)

        weights = self.weight_extractor(latents[0]).unsqueeze(2)

        composition = (stacked_scales * weights).sum(dim=1)

        return composition
    
    def attn_upscale(self, q, latent, v, i):
        k = self.qk_extractors[i](latent)

        with torch.no_grad():
            bi = self.bias_input(q, i)

        b = self.bias_extractors[i](bi)

        b = torch.zeros_like(b)

        #op = upscale_attn_pytorch(q, k, v, b, self.window_size, i)
        o = upscale_attn(q, k, v, b, self.window_size, i)



        return o

    """
    IDNAF
    NBG
    Weight sharing
    JDSS
    ODIN

    For graphics paper:
    - Make figures better
    - People like figures
    - Illustatrator 
    """ 

    def bias_input(self, fr, i):
        shape = fr.shape[2:]

        H, W = shape

        TS = 2 ** i

        HR = TS * (H // TS)
        WR = TS * (W // TS)

        # create mesh grid
        x, y = torch.meshgrid(
            torch.arange(0, W, 1, device=fr.device, dtype=torch.int16),
            torch.arange(0, H, 1, device=fr.device, dtype=torch.int16),
            indexing="xy"
        )

        x = torch.min(x, 
            other=torch.tensor([WR - 1], device=x.device)
        )

        y = torch.min(y, 
            other=torch.tensor([HR - 1], device=x.device)
        )

        x = torch.remainder(x, TS).to(fr.dtype) / TS
        y = torch.remainder(y, TS).to(fr.dtype) / TS

        x = x / 2 - 1
        y = y / 2 - 1

        xy = torch.stack((x, y), dim=0).unsqueeze(0)

        return xy