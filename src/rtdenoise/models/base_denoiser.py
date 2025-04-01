# Basic implementation of a recurrent denoiser that utilizes checkpointing across frames. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class BaseDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_components()

    def init_components(self):
        raise RuntimeError("init_components method for denoise is not implemented!")

    def forward(self, input : torch.Tensor):
        # Input shape is (N, L, C, H, W)

        L = input.size(1)

        output = []
        temporal_state = None

        for i in range(L):
            # take a slice out of input
            # receive (N, C, H, W) tensor in return
            frame_input = input[:, i, :, :, :]

            (frame_output, next_temporal_state) = checkpoint.checkpoint(self.run_frame, frame_input, temporal_state, use_reentrant=False)

            output.append(frame_output)

            temporal_state = next_temporal_state

        # output is list of (N, 3, H, W) tensor
        # stack it and produce (N, L, 3, H, W)
        output = torch.stack(output, dim=1)

        return output



    """
    run_frame must not modify any of the arguments! 

    Frame input formats:
        0:3     raw color
        3:6     albedo
        6:9     normal
        9:12    position
        12:13   depth
        13:15   motionvec

    Below is the framework for implementing your own denoiser:
        N = frame_input.size(0)
        H = frame_input.size(2)
        W = frame_input.size(3)

        # albedo is channels 3..5
        albedo = frame_input[:, 3:6, :, :]

        # (denoising code)

        return (denoised_output, next_temporal_state)
    """
    def run_frame(self, frame_input, temporal_state):
        raise RuntimeError("run_frame method for denoiser is not implemented!")


    
