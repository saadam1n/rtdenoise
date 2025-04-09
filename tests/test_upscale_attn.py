import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rtdenoise
import rtdenoise.kernels as K

q = torch.randn(8, 24, 64, 64)
k = torch.randn(8, 24, 16, 16)
v = torch.randn(8, 3, 16, 16)
b = torch.randn(8, 9, 64, 64)

kernel_size = 3
scale_power = 2

ref = K.upscale_attn_pytorch(q, k, v, b, kernel_size)
fsd = K.upscale_attn(q, k, v, b, kernel_size, scale_power)

err = F.l1_loss(fsd, ref)
print(f"L1 loss on output is {err}")

