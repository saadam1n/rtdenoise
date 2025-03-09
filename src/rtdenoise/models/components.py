import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is based on the MetaFormer paper. To mix tokens, we utilize a large depth-wise convolution .
This component automatically applies ReLU at the end
"""
class DepthWiseConvFormer(nn.Module):
    def __init__(self, num_input_channels, kernel_size=3, expansion_ratio=2):
        super().__init__()

        self.token_mixer = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=num_input_channels),
        )

        self.channel_mixer = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_input_channels * expansion_ratio, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_input_channels * 2),
            nn.Conv2d(num_input_channels * expansion_ratio, num_input_channels, kernel_size=1),
        )

    def forward(self, input):
        mixed_tokens = F.relu(self.token_mixer(input) + input)

        mixed_channels = F.relu(self.channel_mixer(mixed_tokens) + mixed_tokens)

        return mixed_channels
    

"""
A very basic Channel MLP that does two 1x1 convolutions. Skip connection not included.
"""
class ChannelMlp(nn.Module):
    def __init__(self, channels_in, channels_out, channel_multiplier):
        super(ChannelMlp, self).__init__()

        self.channel_mlp = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channel_multiplier * channels_in, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(channel_multiplier * channels_in),
            nn.Conv2d(channel_multiplier * channels_in, channels_out, kernel_size=1),
        )


    def forward(self, input):
        return self.channel_mlp(input)

# No fancy modifications, just DW Conv and pure metaformer
class KernelMatchingFormer(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(KernelMatchingFormer, self).__init__()

        self.dw_conv = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1, groups=num_input_channels)
        )

        self.channel_mlp = ChannelMlp(num_input_channels, num_output_channels, 2)

        # the original ResNet paper uses projections if in dim != out dim
        self.res_projection = nn.Sequential(
            nn.BatchNorm2d(num_input_channels),
            nn.Conv2d(num_input_channels, num_output_channels, kernel_size=1)
        ) if num_input_channels != num_output_channels else None

    def forward(self, input):
        features = self.dw_conv(input)

        residual_connection = self.res_projection(features) if self.res_projection is not None else features

        output = self.channel_mlp(features) + residual_connection

        return output
    
# generic module for using the results of kernel prediction
class PerPixelConv(nn.Module):
    def __init__(self, kernel_size):
        super(PerPixelConv, self).__init__()

        self.kernel_size = kernel_size

    def forward(self, input, kernel):
        unfolded = F.unfold(input, kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        unfolded = unfolded.view(input.size(0), 3, 9, input.size(2), input.size(3))

        filtered = unfolded * kernel.unsqueeze(1)
        filtered = filtered.sum(2)

        return filtered
    
class TensorWarp(nn.Module):
    def __init__(self):
        super(TensorWarp, self).__init__()

    """
    Please note that for warping to work optimally, you should give the full-resolution warped vectors. 
    This is because average motion vectors over an area doesn't work as perfectly as you would expect.
    """
    def forward(self, image : torch.Tensor, motionvec : torch.Tensor):
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