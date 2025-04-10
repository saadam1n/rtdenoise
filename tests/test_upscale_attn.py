import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rtdenoise
import rtdenoise.kernels as K

qr = torch.randn(8, 24, 64, 64, requires_grad=True)
kr = torch.randn(8, 24, 16, 16, requires_grad=True)
vr = torch.randn(8, 3, 16, 16, requires_grad=True)
br = torch.randn(8, 9, 64, 64, requires_grad=True)

qo = qr.detach().requires_grad_()
ko = kr.detach().requires_grad_()
vo = vr.detach().requires_grad_()
bo = br.detach().requires_grad_()

kernel_size = 3
scale_power = 2

o_ref = K.upscale_attn_pytorch(qr, kr, vr, br, kernel_size)
o_out = K.upscale_attn(qo, ko, vo, bo, kernel_size, scale_power)

err = F.l1_loss(o_ref, o_out)
print(f"L1 loss on output is {err}")

o_ref.sum().backward()
o_out.sum().backward()

print(f"\tL1 loss on Q {F.l1_loss(qr.grad, qo.grad)}")
print(f"\tL1 loss on K {F.l1_loss(kr.grad, ko.grad)}")
print(f"\tL1 loss on V {F.l1_loss(vr.grad, vo.grad)}")
print(f"\tL1 loss on B {F.l1_loss(br.grad, bo.grad)}")

