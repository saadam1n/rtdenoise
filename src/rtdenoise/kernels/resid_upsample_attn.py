import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

"""
WARNING: not a fused kernel!
This is a rough implementation using plain PyTorch. Expect high memory usage and slow performance.
"""
def resid_upsample_attn(
        q : torch.Tensor, 
        k : torch.Tensor, 
        dn : torch.Tensor,
        ns : torch.Tensor, 
        kernel_size : int = 3, 
    ):
    N, C, HU, WU = q.shape
    _, _, HD, WD = k.shape

    USHAPE = (HU, WU)

    logit_scale = 1 / math.sqrt(C)

    # now for the queries
    # query is in format (N, C, H, W)
    # we want toe format to be the same format as kuf and vuf
    # unflatten for length of 1
    # final format is (N, C, 1, H, W)
    quf = q.unsqueeze(2)

    quf_kxk = F.unfold(q, kernel_size=kernel_size, padding=1).unflatten(2, (HU, WU)).unflatten(1, (C, -1))

    # for usample attn, v is just dn
    v = dn



    # unfold k and v into 3x3 patches
    kuf = F.unfold(k, kernel_size=kernel_size, padding=1).unflatten(2, (HD, WD))
    vuf = F.unfold(v, kernel_size=kernel_size, padding=1).unflatten(2, (HD, WD))

    # upscale both into full resolution
    # utilize nearest neighbor sampling
    kuf : torch.Tensor = F.interpolate(kuf, size=USHAPE, mode="nearest").unflatten(1, (C, -1))
    vuf : torch.Tensor = F.interpolate(vuf, size=USHAPE, mode="nearest").unflatten(1, (3, -1))

    # for residual attn, v is ns - usample(dn)

    us_dn : torch.Tensor = F.interpolate(dn, size=USHAPE, mode="bilinear", align_corners=False)

    resid = ns - us_dn

    # unfold and add to kv cache
    ruf = F.unfold(resid, kernel_size=kernel_size, padding=1).unflatten(2, (HU, WU)).unflatten(1, (3, -1))

    kuf = torch.cat((kuf, quf_kxk), dim=2)
    vuf = torch.cat((vuf, ruf), dim=2)





    # torch SPDA expects input format (N, H, L, D)
    # current format is (N, D, L, H, W)
    # we need to collapse N, H, W into N for SPDA
    # because torch's SPDA implementation is shit
    # we need to do a lot of permuation

    quf = quf.permute(0, 3, 4, 2, 1).flatten(0, 2) # (N, 1, D)
    kuf = kuf.permute(0, 3, 4, 2, 1).flatten(0, 2) # (N, L, D)
    vuf = vuf.permute(0, 3, 4, 2, 1).flatten(0, 2) # (N, L, 3)

    # I lied, we aren't using SPDA because we need to implement a custom operator
    softmax = F.softmax(torch.matmul(quf, kuf.transpose(1, 2)) * logit_scale, dim=2)

    readd_factor = softmax[:, :, 9:].sum(dim=2).view(N, HU, WU).unsqueeze(1)

    auf = torch.matmul(softmax, vuf)

    # output format is (N * H * W, 1, 3)
    # first change format to (N, H, W, 3)
    auf = auf.view(N, HU, WU, 3).permute(0, 3, 1, 2) + readd_factor * us_dn

    # return answer
    return auf
