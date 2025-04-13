# I am legally obliged to put this comment as a preface to this file named "loss.py":
# |  ||  ||  |_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def flatten_frames(image : torch.Tensor):
    if image.dim() > 4:
        return image.flatten(1, 2)
    elif image.dim() < 4:
        return image.unsqueeze(0)
    else:
        return image

# See "Neural Denoising with Layer Embeddings"
def smape(input : torch.Tensor, target : torch.Tensor, eps=0.01):
    numer = (input - target).abs()
    denom = input.abs() + target.abs() + eps
    per_pixel_loss = numer / denom
    
    reduced = per_pixel_loss.mean()

    return reduced

def smape_antinegative(input : torch.Tensor, target : torch.Tensor, eps=0.01):
    input = flatten_frames(input)
    target = flatten_frames(target)

    # antinegative/positive
    antinegative = torch.where(input < 0, -input, 0)
    antipositive = torch.where(input > 3, input, 0)

    # gradient loss
    kernel = torch.tensor(
        [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]
    ).unsqueeze(0).unsqueeze(0).expand(input.shape[1], -1, -1, -1).to(input.device).to(input.dtype)

    igrad = F.conv2d(input, kernel, padding=1, groups=input.shape[1])
    tgrad = F.conv2d(target, kernel, padding=1, groups=input.shape[1])
    gradloss = (igrad - tgrad).abs()

    # smape
    numer = (input - target).abs()
    denom = input.abs() + target.abs() + eps
    per_pixel_loss = numer / denom + antinegative + antipositive + gradloss

    reduced = per_pixel_loss.mean()

    return reduced

def l1_grad(input : torch.Tensor, target : torch.Tensor):
    input = flatten_frames(input)
    target = flatten_frames(target)

    kernel = torch.tensor(
        [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]
    ).unsqueeze(0).unsqueeze(0).expand(input.shape[1], -1, -1, -1).to(input.device).to(input.dtype)

    igrad = F.conv2d(input, kernel, padding=1, groups=input.shape[1])
    tgrad = F.conv2d(target, kernel, padding=1, groups=input.shape[1])
    gradloss = (igrad - tgrad).abs()

    return F.l1_loss(input, target) + torch.mean(gradloss)

def smape_l2(input : torch.Tensor, target : torch.Tensor):
    return smape(input, target) + F.mse_loss(input, target)

def mix_l1_l2(input : torch.Tensor, target : torch.Tensor):
    return F.l1_loss(input, target) + F.mse_loss(input, target)