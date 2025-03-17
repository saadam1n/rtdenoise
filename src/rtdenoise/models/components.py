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


class GatedConvolutionUnit(nn.Module):
    def __init__(self, channels_in):
        super(GatedConvolutionUnit, self).__init__()

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

    def forward(self, x):
        return self.dw_conv(x) * self.pw_conv(x) + x

class GatedFormerBlock(nn.Module):
    """
    Building block for the LPU and other U-Nets.
    To improve performance and numerical stability, the post-FFN skip connection has been removed.
    """
    def __init__(self, channels_in, channels_out):
        super(GatedFormerBlock, self).__init__()

        self.gcu = GatedConvolutionUnit(channels_in)

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
        glu = self.gcu(x)

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
    def __init__(self, channels_in, channels_out, channels_in_extra, channels_out_extra, is_bottleneck):
        super(UNetConvolutionBlock, self).__init__()

        self.encoder = DenseConvBlock(channels_in, channels_out)
        self.decoder = DenseConvBlock(channels_in_extra + (channels_out if is_bottleneck else 2 * channels_out), channels_in + channels_out_extra)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


class LaplacianFilter(nn.Module):
    """
    Laplacian filter does non-latent filtering via the laplacian pyramid.
    It has built-in kernel prediction which can be turned off via the predictor argument.
    If kernel prediction is turned off, the logits for the kernel prediction are expected to be supplied externally
    """
    def __init__(self, channels_in, predictor, is_bottleneck):
        super(LaplacianFilter, self).__init__()

        self.num_ka_channels = 10 if is_bottleneck else 11
        self.predictor = predictor

        self.ka_predictor = FeedForwardReLU(channels_in, self.num_ka_channels, channel_multiplier=2) if self.predictor else None

    def forward(self, latent_ka, radiance, hidden_state, prev_level, motionvec):
        # channel format: (kernel, temporal alpha, composite alpha)
        ka = self.ka_predictor(latent_ka) if self.predictor else latent_ka[:, -self.num_ka_channels:, :, :]

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

class UNetTransformerBlock(nn.Module):
    """
    Utilize attention to filter tokens.
    - window_size is the size of the window on which attention is performed
    - band_size is padding added to the window to allow the attention to "see" beyond the edge of the window. However, these values will not be used in query computation. 

    Our general architecture is the following:
    - Utilize a single attention head. For each value vector, the first 3 elements of a value vector are the albedo-modulated RGB radiances. 
    - We take the output of the previous decoder layer and upsample it. Concatenate with skip connection and utilize 1x1 convolutions to create QKV
    - Take the previous decoder layer and perform 1x1 convolutions to create KV
    - Take previous frame's latent vectors to create KV
    - Concatenate all QKV into one tensor and perform scaled dot-product attention.
    - Push V through a FFN to get our decoded latent. Concatenate the RGB back in. This vector now becomes input for next decoder layer and for next time step. 
    """
    def __init__(self, channels_in, channels_out, is_bottleneck, window_size, band_size):
        super(UNetTransformerBlock, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.is_bottleneck = is_bottleneck
        self.window_size = window_size
        self.band_size = band_size

        # encoder block
        self.encoder = DenseConvBlock(channels_in, channels_out)


        # take current skip + upsampled and create new latent vector
        su_in = channels_out if self.is_bottleneck else channels_out * 2
        self.su_proj = nn.Sequential(
            nn.BatchNorm2d(su_in),
            nn.Conv2d(su_in, channels_in * 3 - 3, kernel_size=1) # subtract 3 because we will append radiance
        )

        # take in downsampled and create latent tokens
        self.ds_proj = nn.Sequential(
            nn.BatchNorm2d(channels_out),
            nn.Conv2d(channels_out, channels_in * 2 - 3, kernel_size=1)
        ) if not self.is_bottleneck else None

        self.qln = nn.LayerNorm(channels_in)
        self.kln = nn.LayerNorm(channels_in)
        self.vln = nn.LayerNorm(channels_in - 3)
        
        self.ffn = FeedForwardReLU(channels_in, channels_in - 3, channel_multiplier=2)

    def encode(self, x):
        return self.encoder(x)
        


    def decode(self, radiance : torch.Tensor, skip_latent : torch.Tensor, ds_latent : torch.Tensor, temporal_kv : torch.Tensor, motionvec : torch.Tensor):
        q, kv, kv_image = checkpoint.checkpoint(self.extract_su_tokens, radiance, skip_latent, ds_latent, use_reentrant=False)

        kv_list = [kv]

        if temporal_kv is not None:
            temporal_kv = op_warp_tensor(temporal_kv, motionvec)
            kv_list.append(
                self.tokenize(
                    temporal_kv, 
                    banding=False,
                    half_res=False
                )
            )


        if not self.is_bottleneck:
            ds_filtered = ds_latent[:, -3:, :, :]
            ds_embeddings = F.interpolate(
                torch.cat((self.ds_proj(ds_latent), ds_filtered), dim=1),
                size=skip_latent.shape[2:],
                mode="bilinear",
                align_corners=True
            )

            ds_kv = self.tokenize(
                ds_embeddings,
                banding=False,
                half_res=False
            )

            kv_list.append(ds_kv)


        kv = torch.cat(kv_list, dim=2)

        attn = checkpoint.checkpoint(self.perform_spda, q, kv, use_reentrant=False)

        return checkpoint.checkpoint(self.reformt_results, attn, kv_image, use_reentrant=False)

    def tokenize(self, image : torch.Tensor, banding : bool, half_res : bool):
        adj_window_size = self.window_size if not half_res else self.window_size // 2
        adj_band_size = self.band_size if not half_res else self.band_size // 2

        adj_band_size = adj_band_size if banding else 0

        # use matrix notation to refer to dimensions
        ipad = self.pd_size(adj_window_size, image.shape[2]) - image.shape[2]
        jpad = self.pd_size(adj_window_size, image.shape[3]) - image.shape[3]

        padded_image = F.pad(image, pad=(0, jpad, 0, ipad), mode="constant", value=0)

        # (N, C * k * k, L) ->
        # (N, C, k * k, L) ->
        # (N, L, k * k, C) ->
        # (N, L, k * k, C)
        tiles = F.unfold(
            padded_image, 
            kernel_size=adj_window_size + 2 * adj_band_size, 
            stride=adj_window_size, 
            padding=adj_band_size
        ).unflatten(
            1, 
            (-1, (adj_window_size + 2 * adj_band_size) ** 2)
        ).permute(
            0, 3, 2, 1
        )

        return tiles
    
    def pd_size(self, adj_window_size, dim):
        return adj_window_size * ((dim - 1) // adj_window_size + 1)

    def perform_spda(self, q, kv):
        N, _, _, _ = q.shape


        q = q.flatten(0, 1)
        k = kv[:, :, :, :self.channels_in].flatten(0, 1)
        v = kv[:, :, :, self.channels_in:].flatten(0, 1)

        attn = F.scaled_dot_product_attention(
            query=self.qln(q),
            key=self.kln(k),
            value=torch.cat((self.vln(v[:, :, :-3]), v[:, :, -3:]), dim=2)
        ).unflatten(0, (N, -1))

        return attn
    
    def reformt_results(self, attn, kv_image):
        # convert attn back to an image using fold
        # (N, L, k * k, C) ->
        # (N, C, k * k, L) ->
        # (N, C * k * k, L)
        img_attn = F.fold(
            attn.permute(0, 3, 2, 1).flatten(1, 2), 
            output_size=(
                self.pd_size(self.window_size, kv_image.shape[2]),
                self.pd_size(self.window_size, kv_image.shape[3])
            ),
            kernel_size=self.window_size,
            stride=self.window_size
        )

        img_attn = img_attn[:, :, :kv_image.shape[2], :kv_image.shape[3]]

        # feedback loop without banding 
        temporal_kv = torch.cat((kv_image[:, :-3, :, :], img_attn[:, -3:, :, :]), dim=1)
        decoded_latent = torch.cat((self.ffn(img_attn), img_attn[:, -3:, :, :]), dim=1)

        return decoded_latent, temporal_kv
    
    def extract_su_tokens(self, radiance, skip_latent, ds_latent):
        us_latent = F.interpolate(
            ds_latent, 
            size=skip_latent.shape[2:], 
            mode="bilinear", 
            align_corners=True
        ) if not self.is_bottleneck else None

        su_embeddings = self.su_proj(
            skip_latent if self.is_bottleneck else
            torch.cat((skip_latent, us_latent), dim=1)
        )

        # tokenize everything
        q = self.tokenize(
            su_embeddings[:, :self.channels_in, :, :],
            banding=False,
            half_res=False
        )

        kv_image = torch.cat((su_embeddings[:, self.channels_in:, :, :], radiance), dim=1)
        kv = self.tokenize(
            kv_image,
            banding=True,
            half_res=False
        )

        return q, kv, kv_image


class LaplacianUNet(nn.Module):
    def __init__(self, channels):
        super(LaplacianUNet, self).__init__()

        # self.channels[0] corresponds to the input to the entire network
        # self.channels[1...n] corresponds to the output of each encoder block
        self.channels = channels

        self.filters = nn.ModuleList([
            LaplacianFilter(
                channels_in=self.channels[i - 1], 
                predictor=False,
                is_bottleneck=(i == len(self.channels) - 1)
            ) for i in range(1, len(self.channels))
        ])

        self.levels = nn.ModuleList([
            UNetConvolutionBlock(
                channels_in=self.channels[i - 1], 
                channels_out=self.channels[i], 
                channels_in_extra=0,
                channels_out_extra=self.filters[i - 1].num_ka_channels,
                is_bottleneck=(i == len(self.channels) - 1)
            ) for i in range(1, len(self.channels))
        ])



    def forward(self, radiance, latent, motionvec, *hidden_states):
        hidden_states = list(hidden_states)

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

            # if we are using external prediction, we want to ditch the prediciton channels
            if not self.filters[i].predictor:
                decoded_latent = decoded_latent[:, :-self.filters[i].num_ka_channels, :, :]

            hidden_states[i] = filtered

        return filtered, decoded_latent, hidden_states
        
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
                channels_in_extra=self.channels[i],
                channels_out_extra=0,
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
    
class TransformerUNet(nn.Module):
    def __init__(self, channels):
        super(TransformerUNet, self).__init__()

        # self.channels[0] corresponds to the input to the entire network
        # self.channels[1...n] corresponds to the output of each encoder block
        self.channels = channels

        self.levels = nn.ModuleList([
            UNetTransformerBlock(
                channels_in=self.channels[i - 1], 
                channels_out=self.channels[i], 
                is_bottleneck=(i == len(self.channels) - 1),
                window_size=4,
                band_size=1
            ) for i in range(1, len(self.channels))
        ])



    def forward(self, radiance, latent, motionvec, *hidden_states):
        hidden_states = list(hidden_states)

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
        decoded_latent = None
        for i, level in reversed(list(enumerate(self.levels))):
            decoded_latent, temporal_kv = checkpoint.checkpoint(
                level.decode,
                ds_color[i],
                skip_tensors[i],
                decoded_latent,
                hidden_states[i],
                motionvec,
                use_reentrant=False
            )

            hidden_states[i] = temporal_kv

        return decoded_latent[:, -3:, :, :], hidden_states
        
        
    def create_empty_hidden_state(self):
        return [None] * len(self.levels)