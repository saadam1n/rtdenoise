import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
WARNING: not a fused kernel!
This is a rough implementation using plain PyTorch. Expect high memory usage and slow performance.
"""
def upscale_attn(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, bias : torch.Tensor | None = None):
    N, C, HU, WU = q.shape
    _, _, HD, WD = k.shape

    USHAPE = (HU, WU)

    # unfold k and v into 3x3 patches
    kuf = F.unfold(k, kernel_size=3, padding=1).unflatten(2, (HD, WD))
    vuf = F.unfold(v, kernel_size=3, padding=1).unflatten(2, (HD, WD))

    # upscale both into full resolution
    # utilize nearest neighbor sampling
    kuf = F.interpolate(kuf, size=USHAPE, mode="nearest")
    vuf = F.interpolate(vuf, size=USHAPE, mode="nearest")

    # reshape for torch's SPDA
    # (N, C * 3 * 3, H, W) is the input format for KV
    # we need to expand into number of locations, i.e. 
    # (N, C, 3 * 3, H, W)

    kuf = kuf.unflatten(1, (C, 9))
    vuf = vuf.unflatten(1, (3, 9))

    # now for the queries
    # query is in format (N, C, H, W)
    # we want toe format to be the same format as kuf and vuf
    # unflatten for length of 1
    quf = q.unsqueeze(2)

    # for bias, torch expects it in a (N, L, S) format
    # here L = 1 because we have only one query
    # S = 9 because we have 9 biases
    # input is (N, 9, H, W) so we have to swap a few dimensions
    buf = bias.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(1) if bias is not None else None

    # torch SPDA expects input format (N, H, L, D)
    # current format is (N, D, L, H, W)
    # we need to collapse N, H, W into N for SPDA
    # because torch's SPDA implementation is shit
    # we need to do a lot of permuation

    quf = quf.permute(0, 3, 4, 2, 1).flatten(0, 2)
    kuf = kuf.permute(0, 3, 4, 2, 1).flatten(0, 2)
    vuf = vuf.permute(0, 3, 4, 2, 1).flatten(0, 2)


    auf = F.scaled_dot_product_attention(quf, kuf, vuf, attn_mask=buf)

    # output format is (N * H * W, 1, 3)
    # first change format to (N, H, W, 3)
    auf = auf.view(N, HU, WU, 3).permute(0, 3, 1, 2)

    # return answer
    return auf





    
