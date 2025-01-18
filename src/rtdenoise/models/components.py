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