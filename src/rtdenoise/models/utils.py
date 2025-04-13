import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import openexr_numpy as exr

import os

def quick_save_img(path : str, img : torch.Tensor):
    if not os.path.exists("/tmp/rtsave.txt"):
        return

    if img.dim() > 3:
        if img.shape[0] > 1:
            print("QUICK SAVE: multiple batches detected, saving first one only")

        img = img[0]

    img = img.detach().cpu().permute(1, 2, 0).numpy()

    exr.imwrite(path, img)