import rtdenoise
import rtdenoise.kernels as K

import torch
import torch.nn.functional as F

qklist = [torch.randn(8, 24, 32, 32, requires_grad=True, device="cuda") for _ in range(3)]
vlist = [torch.randn(8, 3, 32, 32, requires_grad=True, device="cuda") for _ in range(3)]

qkrlist = [qk.detach().requires_grad_() for qk in qklist]
vrlist = [v.detach().requires_grad_() for v in vlist]

window_size = 3

q_f = F.unfold(qkrlist[0], kernel_size=1, stride=1).unflatten(1, (-1, 1)).permute(0, 3, 2, 1)



k_f = torch.cat(
    [F.unfold(qk, kernel_size=window_size, stride=1, padding=window_size // 2).unflatten(1, (-1, window_size ** 2)).permute(0, 3, 2, 1) for qk in qkrlist], 
    dim=2
)


v_f = torch.cat(
    [F.unfold(v, kernel_size=window_size, stride=1, padding=window_size // 2).unflatten(1, (-1, window_size ** 2)).permute(0, 3, 2, 1) for v in vrlist], 
    dim=2
)

a_ref = F.scaled_dot_product_attention(q_f, k_f, v_f).permute(0, 3, 2, 1).flatten(1, 2)
a_ref = F.fold(a_ref, output_size=vlist[0].shape[2:], kernel_size=1, stride=1)

a_out = K.kernel_attn(qklist[0], vlist[0], qklist[1], vlist[1], qklist[2], vlist[2], window_size)

if torch.isnan(a_ref).any():
    print("A Ref has NaN!")

if torch.isnan(a_out).any():
    print("A Out has NaN!")

err = F.l1_loss(a_ref, a_out)
print(f"L1 loss was {err}")

print("\n\nBackwards tests")
a_ref.sum().backward()
a_out.sum().backward()


for i in range(3):
    errqk = F.l1_loss(qklist[i].grad, qkrlist[i].grad)
    print(f"QK L1 loss was {errqk}")

    errv = F.l1_loss(vlist[i].grad, vrlist[i].grad)
    print(f"V L1 loss was {errv}")