import torch

def kernel_attn(qk : torch.Tensor, v : torch.Tensor, window_size : int):
    return torch.ops.rtdenoise.kernel_attn.default(qk, v, window_size)

@torch.library.register_fake("rtdenoise::kernel_attn")
def _(qk, v, window_size):
    torch._check(window_size % 2 == 1)
    torch._check(qk.shape[0] == v.shape[0])
    torch._check(v.shape[1] == 3)
    torch._check(qk.shape[2:] == v.shape[2:])
    torch._check(qk.dtype == torch.float)
    torch._check(v.dtype == torch.float)
    torch._check(qk.device == v.device)
    return torch.empty_like(v) 