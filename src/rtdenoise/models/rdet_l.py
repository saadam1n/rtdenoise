"""
Frequency composition transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base_denoiser import BaseDenoiser
from .components import *

from ..kernels import *

from .utils import *

class TokenMLP(nn.Module):
    """
    A double layer perceptron network with a skip connection. Automatically performs layer norm on the input.
    """
    def __init__(self, embedding_dim):
        super(TokenMLP, self).__init__()

        self.embedding_dim = embedding_dim

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

    def forward(self, x):
        return self.ffn(x) + x

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim):
        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.total_head_dim = num_attention_heads * head_embedding_dim

        # pre-normalization
        self.pre_norm = nn.LayerNorm(self.embedding_dim)

        # linear layers to project our tokens to Q, K, and V
        self.q_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.k_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.v_linear = nn.Linear(self.embedding_dim, self.total_head_dim, bias=False)
        self.attention_linear = nn.Linear(self.total_head_dim, self.embedding_dim, bias=False)

    """
    Returns (transformed tokens, K, V). Does not maintain a KV cache.
    """
    def forward(self, tokens):
        # nn.Linear gives us (N, L, D_embedding_dim) -> (N, L, D_total_head_dim)
        # we want to reformat it as (N, num_attention_heads, L, D_head_embedding_dim)

        N, L, _ = tokens.shape

        ln_tokens = self.pre_norm(tokens)

        q = self.q_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))
        k = self.v_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))
        v = self.k_linear(ln_tokens).view(N, L, self.num_attention_heads, self.head_embedding_dim).permute((0, 2, 1, 3))

        weighted_values = F.scaled_dot_product_attention(q, k, v)
            
        aggregated_tokens = weighted_values.permute((0, 2, 1, 3)).reshape(N, L, self.total_head_dim)
        attention_output = self.attention_linear(aggregated_tokens) + tokens

        return (attention_output, k, v)
    
class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, head_embedding_dim, aggresive_checkpointing=False):
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_embedding_dim = head_embedding_dim

        self.attention = MultiHeadAttention(
            embedding_dim=self.embedding_dim, 
            num_attention_heads=self.num_attention_heads, 
            head_embedding_dim=self.head_embedding_dim
        )

        self.token_mlp = TokenMLP(embedding_dim=self.embedding_dim)

        self.aggresive_checkpointing = aggresive_checkpointing

    def forward(self, tokens):
        attention_output, _, _ = checkpoint.checkpoint(self.attention, tokens, use_reentrant=False) if self.aggresive_checkpointing else self.attention(tokens)

        token_mlp_output = checkpoint.checkpoint(self.token_mlp, attention_output, use_reentrant=False) if self.aggresive_checkpointing else self.token_mlp(attention_output)

        return token_mlp_output


class AutoEncoderBlock(nn.Module):
    """Some Information about AutoEncoderBlock"""
    def __init__(self, num_channels):
        super(AutoEncoderBlock, self).__init__()

        self.encoder = RestormerConvolutionBlock(num_channels, num_channels)
        self.decoder = RestormerConvolutionBlock(2 * num_channels, num_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

"""
Based on Diffusion Transformer paper and some downsampling.
"""
class RealtimeDenoisingTransformer_ViT_Latent(BaseDenoiser):
    def init_components(self):
        self.num_internal_channels = 32

        # project features
        self.num_input_channels = 9
        self.projector = nn.Sequential(
            nn.BatchNorm2d(self.num_input_channels),
            nn.Conv2d(self.num_input_channels, self.num_internal_channels, kernel_size=1),
            SqueezeExcitation(self.num_internal_channels),
        )

        self.autoencoder_blocks = nn.ModuleList([
            AutoEncoderBlock(self.num_internal_channels)
            for i in range(3)
        ])

        self.transformer = nn.Sequential(
            Transformer(embedding_dim=self.num_internal_channels, num_attention_heads=16, head_embedding_dim=4, aggresive_checkpointing=True),
            Transformer(embedding_dim=self.num_internal_channels, num_attention_heads=16, head_embedding_dim=4, aggresive_checkpointing=True),
            Transformer(embedding_dim=self.num_internal_channels, num_attention_heads=16, head_embedding_dim=4, aggresive_checkpointing=True),
            Transformer(embedding_dim=self.num_internal_channels, num_attention_heads=16, head_embedding_dim=4, aggresive_checkpointing=True)
        )
        
        self.deprojector = nn.Sequential(
            nn.BatchNorm2d(self.num_internal_channels),
            nn.Conv2d(self.num_internal_channels, 3, kernel_size=1)
        ) if False else FeedForwardGELU(self.num_internal_channels, 3, channel_multiplier=2, has_skip=True)

        self.posemb = nn.Embedding(num_embeddings=256, embedding_dim=self.num_internal_channels)


    def run_frame(self, frame_input : torch.Tensor, temporal_state):
        N = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        color = frame_input[:, :3, :, :]
        albedo = frame_input[:, 3:6, :, :]

        input = frame_input[:, :9, :, :]
        motionvec = frame_input[:, 13:, :, :]
    
        features = self.projector(input)

        latents = checkpoint.checkpoint(self.encode_image, features, use_reentrant=False)

        filtered = checkpoint.checkpoint(self.filter_image, latents[-1], use_reentrant=False)

        rgb = checkpoint.checkpoint(self.decode_image, filtered, latents, use_reentrant=False)

        quick_save_img("/tmp/rgb.exr", rgb)

        remod = albedo.pow(1.0 / 2.2) * rgb

        # BE CAREFUL!!!
        remod = rgb

        return (remod, None)

    def encode_image(self, features):
        latents = []

        for i in range(3):
            ds_features = features if i == 0 else F.max_pool2d(features, kernel_size=2, stride=2)

            enc_ds_features = self.autoencoder_blocks[i].encode(ds_features)

            latents.append(enc_ds_features)

        latents.append(F.max_pool2d(enc_ds_features, kernel_size=2, stride=2))

        return latents
    
    def decode_image(self, filtered, latents):
        for i in reversed(range(3)):
            prev_decoder_out = F.interpolate(
                filtered if i == 2 else prev_decoder_out,
                size=latents[i].shape[2:],
                mode="bilinear"
            )

            # concat
            cur_decoder_in = torch.cat((latents[i], prev_decoder_out), dim=1)

            prev_decoder_out = self.autoencoder_blocks[i].decode(cur_decoder_in)
        
        rgb = self.deprojector(prev_decoder_out)

        return rgb

    
    def filter_image(self, image):
        N, _, _, _ = image.shape

        # unfold 16x16 patches

        # (N, C * k * K, L)
        patches = F.unfold(image, kernel_size=16, stride=16)

        # (N * L, k * k, C)
        patches = patches.unflatten(1, (-1, 256)).permute(0, 3, 2, 1).flatten(0, 1)

        # add pos emb
        patches = patches + self.posemb.weight

        # denoise
        denoised : torch.Tensor = self.transformer(patches)

        # de-unfold
        denoised = denoised.unflatten(0, (N, -1)).permute(0, 3, 2, 1).flatten(1, 2)

        denoised = F.fold(denoised, output_size=image.shape[2:], kernel_size=16, stride=16)

        return denoised

