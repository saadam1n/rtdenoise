import rtdenoise
import rtdenoise.kernels as K

import torch
import torch.nn.functional as F

qk = torch.randn(8, 24, 32, 32)
v = torch.randn(8, 3, 32, 32)
window_size = 3

q_f = F.unfold(qk, kernel_size=1, stride=1).unflatten(1, (-1, 1)).permute(0, 3, 2, 1)
k_f = F.unfold(qk, kernel_size=window_size, stride=1, padding=window_size // 2).unflatten(1, (-1, window_size ** 2)).permute(0, 3, 2, 1)
v_f = F.unfold(v, kernel_size=window_size, stride=1, padding=window_size // 2).unflatten(1, (-1, window_size ** 2)).permute(0, 3, 2, 1)


a_ref = F.scaled_dot_product_attention(q_f, k_f, v_f).permute(0, 3, 2, 1).flatten(1, 2)
a_ref = F.fold(a_ref, output_size=v.shape[2:], kernel_size=1, stride=1)

a_out = K.kernel_attn(qk, v, window_size)

if torch.isnan(a_ref).any():
    print("A Ref has NaN!")

if torch.isnan(a_out).any():
    print("A Out has NaN!")

err = F.l1_loss(a_ref, a_out)
print(f"L1 loss was {err}")