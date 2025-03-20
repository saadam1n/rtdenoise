# Basic implementation of a recurrent denoiser that utilizes checkpointing across frames. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class BaseDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # ideally we want to fix the number of auxiliary features globally 
        self.num_aux_channels = 8
        self.num_input_channels = self.num_aux_channels + 3

        self.init_components()

    def init_components(self):
        raise RuntimeError("init_components method for denoise is not implemented!")

    def forward(self, input : torch.Tensor):
        B = input.size(0)
        num_frames = input.size(1) // self.num_input_channels
        H = input.size(2)
        W = input.size(3)

        output = torch.empty(B, 3 * num_frames, H, W, device=input.device, dtype=input.dtype)
        temporal_state = None

        for i in range(num_frames):

            base_channel_index = i * self.num_input_channels

            frame_input = input[:, base_channel_index:base_channel_index + self.num_input_channels, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, use_reentrant=False)

            oidx = i * 3
            output[:, oidx:oidx + 3, :, :] = frame_output

            temporal_state = next_temporal_state

        return output



    """
    Below is the framework for implementing your own denoiser:
        B = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        # (denoising code)

        return (denoised_output, next_temporal_state)
    """
    def run_frame(self, frame_input, temporal_state):
        raise RuntimeError("run_frame method for denoiser is not implemented!")


    
