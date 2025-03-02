import torch
from torch.utils.data import Dataset

import numpy as np
import openexr_numpy as exr

import os

class FrameDataset(Dataset):
    def __init__(self, dataset_folder, device, seq_len):
        self.dataset_dir = dataset_folder + "/"
        self.seq_len=seq_len

        # keep all sequences with sequence length > self.sequence_length
        self.samples = []
        all_folders = os.listdir(dataset_folder)
        for folder in all_folders:
            sample_folder = self.dataset_dir + folder + "/"

            valid_sample = True
            for i in range(self.seq_len):
                if not os.path.exists(sample_folder + f"reference{i}.exr"):
                    valid_sample = False
                    break

                if valid_sample:
                    self.samples.append(sample_folder)

        print(f"Dataset at {self.dataset_dir} has {len(self.samples)} samples")

        self.device=device


    def __len__(self):
        # length here is defined by the number of frame sequneces we have,
        # not the number of frames
        return len(self.samples)

    # this needs to manually convert things to a tensor
    def __getitem__(self, idx):
        # basically, I want to output it in this format for each element of each batch
        # (B, N, C, H, W)
        # B = batch size
        # N = number of frames
        # C = channels for each frame (color, albedo, world pos, world norm)

        sequence_inputs = []
        sequence_refs = []

        sample_folder = self.samples[idx]
        for i in range(self.seq_len):
            (frame_input, frame_reference) = self.read_frame(sample_folder, i)

            sequence_inputs += frame_input
            sequence_refs += frame_reference

        input = torch.cat(sequence_inputs, dim=0)
        reference = torch.cat(sequence_refs, dim=0)

        return input, reference

    def read_frame(self, sample_folder, i):
        color = self.read_exr(sample_folder, i, "color")
        albedo = self.read_exr(sample_folder, i, "albedo")

        # some stuff in the dataset is a bit broken, so here is a fix
        color = torch.clamp_max(color, max=128.0)  
        
        # position is not in use currently
        #pos = self.read_exr(sample_folder, i, "position")
        norm = self.read_exr(sample_folder, i, "normal")

        ref = self.read_exr(sample_folder, i, "reference")

        return [color, albedo, norm], [ref]


    def read_exr(self, sample_folder, idx, bufname):
        filename = sample_folder + bufname + str(idx) + ".exr"

        img = exr.imread(filename)
        img = torch.tensor(img, device=self.device).permute(2, 0, 1)


        return img

