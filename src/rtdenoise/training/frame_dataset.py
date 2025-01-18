import torch
import numpy as np
import random
import subprocess
from torch.utils.data import Dataset

# We need this so OpenCV imports exr files
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2

class FrameDataset(Dataset):
    def __init__(self, base_dir, dataset_folder, device, seq_len):
        self.dataset_dir = base_dir + "/" + dataset_folder
        self.cache_dir = base_dir + "/CacheV2"
        self.seq_len=seq_len

        self.num_frames=0
        while True:
            
            if os.path.exists(self.dataset_dir + "/" + str(self.num_frames) + "-Reference.exr"):
                self.num_frames+=1
            else:
                break

        print(f"Dataset at {self.dataset_dir} has {self.num_frames} images")

        self.device=device

        self.frame_cache = {}
        self.ref_cache = {}
        self.loaded = {}

        self.patching = True

    def __len__(self):
        # length here is defined by the number of frame sequneces we have,
        # not the number of frames
        return self.num_frames - self.seq_len + 1

    # this needs to manually convert things to a tensor
    def __getitem__(self, idx):
        # basically, I want to output it in this format for each element of each batch
        # (B, N, C, H, W)
        # B = batch size
        # N = number of frames
        # C = channels for each frame (color, albedo, world pos, world norm)

        if True:
            if False:
                yoff = random.randint(0, 720)
                xoff = random.randint(0, 1280)
            else:
                yoff = int(random.gauss(450, 300))
                xoff = int(random.gauss(450, 500))

                yoff = 0 if yoff < 0 else 719 if yoff >= 720 else yoff
                xoff = 0 if xoff < 0 else 1279 if xoff >= 1280 else xoff

        else:
            if False:
                yoff = 475
                xoff = 420
            else:
                yoff = 0
                xoff = 1280

        frame_inputs = []
        frame_references = []
        for i in range(self.seq_len):
            (frame_input, frame_reference) = self.read_frame(idx + i)

            if self.patching:
                frame_input = frame_input[:, yoff:yoff+360, xoff:xoff+640]
                frame_reference = frame_reference[:, yoff:yoff+360, xoff:xoff+640]

            frame_inputs.append(frame_input)
            frame_references.append(frame_reference)

        input = torch.cat(tuple(frame_inputs), dim=0)
        reference = torch.cat(tuple(frame_references), dim=0)

        return input, reference

    def read_frame(self, i):
        if i not in self.loaded:
            self.loaded[i] = True

            color = self.read_exr(i, "Color")
            albedo = self.read_exr(i, "Albedo")

            albedo[albedo < 0.001] = 1.0
            color = color / albedo

            worldpos = self.read_exr(i, "WorldPosition") * 0.05
            worldnorm = self.read_exr(i, "WorldNormal")

            ref = self.read_exr(i, "Reference").permute(2, 0, 1)

            frame_inputs = torch.cat((color, albedo, worldnorm, worldpos), dim=2).permute(2, 0, 1)

            self.frame_cache[i] = frame_inputs
            self.ref_cache[i] = ref

            return (frame_inputs.to(self.device), ref.to(self.device))
        else:
            return (self.frame_cache[i].to(self.device), self.ref_cache[i].to(self.device))


    def read_exr(self, idx, ext):
        filename = str(idx) + "-" + ext + ".exr"

        cache_path = self.cache_dir + "/" + filename + ".npy"

        if os.path.exists(cache_path):
            img = np.load(cache_path)
        else:
            img = cv2.imread(self.dataset_dir + "/" + filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            np.save(cache_path, img)

        return torch.tensor(img, device="cpu", dtype=torch.float32)

    def get_full_img(self):
        self.patching = False
        input, target = self.__getitem__(0)
        self.patching = True
        return input, target

    def print_shape(name, img):
        print(f"{name}\tshape is {img.shape}")
