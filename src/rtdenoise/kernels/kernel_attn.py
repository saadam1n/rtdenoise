import torch

class KernelAttention(torch.autograd.Function):
    """
    Operates on sliding blocks within an image, like convolution, but applies attention on each block.
    """

    @staticmethod
    def forward(ctx, qk : torch.Tensor, v : torch.Tensor, window_size : int) -> torch.Tensor:

        single_slice = v[:, :1, :, :]
        L = torch.empty_like(single_slice)
        m = torch.empty_like(single_slice)

        a = torch.ops.rtdenoise.kernel_attn.default(qk, v, window_size, L, m)

        ctx.save_for_backward(qk, v, L, m, a)
        ctx.window_size = window_size

        return a

    @staticmethod
    def backward(ctx, dLda : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        qk, v, L, m, a = ctx.saved_tensors
        window_size = ctx.window_size

        dLdqk = torch.zeros_like(qk)
        dLdv = torch.zeros_like(v)

        torch.ops.rtdenoise.kernel_attn_bwd.default(
            qk, v, window_size, L, m, a,
            dLda, dLdqk, dLdv
        )

        return dLdqk, dLdv, None
    
    @staticmethod
    def _(qk : torch.Tensor, v : torch.Tensor, window_size : int):
        torch._check(window_size % 2 == 1)
        torch._check(qk.shape[0] == v.shape[0])
        torch._check(v.shape[1] == 3)
        torch._check(qk.shape[2:] == v.shape[2:])
        torch._check(qk.dtype == torch.float)
        torch._check(v.dtype == torch.float)
        torch._check(qk.device == v.device)
        return torch.empty_like(v) 
    

def kernel_attn(qk : torch.Tensor, v : torch.Tensor, window_size : int) -> torch.Tensor:
    return KernelAttention.apply(qk, v, window_size)