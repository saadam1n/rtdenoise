import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

class GlobalContextPreEncoderTransformer(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 24
        self.num_transformer_channels = 24
        self.unet_channels = [self.num_internal_channels, 24, 24, 32, 32, 48, 64]
        self.num_filtering_scales = len(self.unet_channels) - 1

        self.nz_scales = [5, 11, 17, 29]

        # analagous to a 5x5 kernel
        self.window_size = 1
        self.band_size = 2
        self.kernel_mode = False # runs energy conserving kernels

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

        if torch.isnan(features).any():
            print(f"NaN features!")

        qk_latent = checkpoint.checkpoint_sequential(
            self.encoder_net,
            segments=1,
            input=features,
            use_reentrant=False
        )


        for i, qkl in enumerate(qk_latent):
            if torch.isnan(qkl).any():
                print(f"NaNinng in {i}")

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
        qk_fr = self.qk_extractors[extractor_index](qk_latent_fr)

        q = self.tokenize(qk_fr, banding=False)
        k = self.tokenize(qk_fr, banding=True)

        # in kernel mode we do a splat rather than a gather operation
        # so we don't actually want a window, we just want the pixel itself
        v = self.tokenize(color_fr, banding=not self.kernel_mode) 

        if color_t is not None:
            qk_t_tok = self.tokenize(qk_t, banding=True)
            color_t_tok = self.tokenize(color_t, banding=not self.kernel_mode)

            if not self.kernel_mode:
                k = torch.cat((k, qk_t_tok), dim=2)
                v = torch.cat((v, color_t_tok), dim=2)

        if color_ds is not None:
            # upsample color_ds and qk_ds
            color_ds = F.interpolate(color_ds, size=color_fr.shape[2:], mode="bilinear", align_corners=True)
            qk_ds = F.interpolate(qk_ds, size=qk_fr.shape[2:], mode="bilinear", align_corners=True)

            qk_ds_tok = self.tokenize(qk_ds, banding=True)
            color_ds_tok = self.tokenize(color_ds, banding=not self.kernel_mode)

            if not self.kernel_mode:
                k = torch.cat(
                    (k, qk_ds_tok), 
                    dim=2
                )

                v = torch.cat(
                    (v, color_ds_tok), 
                    dim=2
                )

        # I theorize that torch's SPDA is optimized for the constraints of LLMs 
        # (i.e. large embedding sizes, many heads, many tokens)
        # For use we have small embedding sizes, a single head, and a few tokens
        # If we are in kernel mode where this is especially true, we want to use our
        # own implementation of attention
        if self.kernel_mode:
            num_divisors = 1

            filtered = checkpoint.checkpoint(
                self.energy_conserving_kernels, 
                q, 
                k, 
                v, 
                color_fr.shape[2:], 
                use_reentrant=False
            )

            if color_t is not None:
                filtered = filtered + checkpoint.checkpoint(
                    self.energy_conserving_kernels, 
                    qk_t_tok, 
                    k, 
                    color_t_tok, 
                    color_fr.shape[2:], 
                    use_reentrant=False
                )
                num_divisors += 1

            if color_ds is not None:
                filtered = filtered + checkpoint.checkpoint(
                    self.energy_conserving_kernels, 
                    qk_ds_tok, 
                    k, 
                    color_ds_tok, 
                    color_fr.shape[2:], 
                    use_reentrant=False
                )
                num_divisors += 1

            if num_divisors > 1:
                filtered = filtered / num_divisors
        else:
            filtered = checkpoint.checkpoint(self.perform_spda, q, k, v, use_reentrant=False)

            filtered = checkpoint.checkpoint(self.reformt_results, filtered, color_fr.shape[2:], use_reentrant=False)


        return filtered, qk_fr

    # technical debt goes hard
    # I copy pasted this instead of taking time to genearlize my code
    def tokenize(self, image : torch.Tensor, banding : bool):
        bs = self.band_size if banding else 0

        # use matrix notation to refer to dimensions
        ipad = self.pd_size(image.shape[2]) - image.shape[2]
        jpad = self.pd_size(image.shape[3]) - image.shape[3]

        padded_image = F.pad(image, pad=(0, jpad, 0, ipad), mode="constant", value=0)

        # (N, C * k * k, L) ->
        # (N, C, k * k, L) ->
        # (N, L, k * k, C) ->
        # (N, L, k * k, C)
        tiles = F.unfold(
            padded_image, 
            kernel_size=self.window_size + 2 * bs, 
            stride=self.window_size, 
            padding=bs
        ).unflatten(
            1, 
            (-1, (self.window_size + 2 * bs) ** 2)
        ).permute(
            0, 3, 2, 1
        )

        return tiles

    def perform_spda(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
        N, _, _, _ = q.shape

        # we need to flatten the tensors 
        # torch's spda implementation does some weird things with attention heads
        # rather than viewing attention heads as something independent (like the batch dim), torch does some 
        # weird optimization that breaks when you have many heads (>60k) 
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)

        attn = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v
        )



        # skip connection in latent mode
        attn = attn.unflatten(0, (N, -1))

        return attn
    
    def energy_conserving_kernels(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, shape):
        # if we're in kernel mode, we can take this as an 
        # oppurtunity to have energy-convserving kernels
        # easiest way to do this is to flip the script and 
        # make this a splatting operation instead of a gather one
        # Q is    (N, L, 1, C)
        # K is    (N, L, k * k, C)
        # QK^T is (N, L, 1, k * k)
        # To obtain logits, we need to transpose K
        logits = torch.matmul(q, k.transpose(2, 3))

        # to make this a splat operation, we need to make use of folding
        # folding adds up values when combining windows
        # we assume v is in the shape (N, L, 1, C)
        # if we do multiplication we need weights to be in the shape (N, L, k * k, 1)
        weights = F.softmax(logits.transpose(2, 3), dim=2)

        # now our values are in the format (N, L, k * k, C)
        # we need to permute some stuff before folding
        splatted = F.fold(
            (weights * v).permute(0, 3, 2, 1).flatten(1, 2),
            output_size=shape,
            kernel_size=self.window_size + 2 * self.band_size,
            stride=self.window_size,
            padding=self.band_size
        )

        return splatted


    def reformt_results(self, attn, shape):
        # convert attn back to an image using fold
        # (N, L, k * k, C) ->
        # (N, C, k * k, L) ->
        # (N, C * k * k, L)
        img_attn = F.fold(
            attn.permute(0, 3, 2, 1).flatten(1, 2), 
            output_size=(
                self.pd_size(shape[0]),
                self.pd_size(shape[1])
            ),
            kernel_size=self.window_size,
            stride=self.window_size
        )

        img_attn = img_attn[:, :, :shape[0], :shape[1]]

        return img_attn
    
    def pd_size(self, dim):
        return self.window_size * ((dim - 1) // self.window_size + 1)
