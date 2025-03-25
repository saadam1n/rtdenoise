import torch

class KernelAttention(torch.autograd.Function):
    """
    Operates on sliding blocks within an image, like convolution, but applies attention on each block.
    """

    @staticmethod
    def forward(
        ctx, 
        qk0 : torch.Tensor, v0 : torch.Tensor, 
        qk1 : torch.Tensor, v1 : torch.Tensor, 
        qk2 : torch.Tensor, v2 : torch.Tensor, 
        window_size : int
    ) -> torch.Tensor:

        single_slice = v0[:, :1, :, :]
        L = torch.empty_like(single_slice)
        m = torch.empty_like(single_slice)

        a = torch.ops.rtdenoise.kernel_attn.default(
            qk0, v0, 
            qk1, v1, 
            qk2, v2, 
            window_size, 
            L, m
        )

        ctx.save_for_backward(
            qk0, v0, 
            qk1, v1, 
            qk2, v2, 
            L, m, a
        )
        ctx.window_size = window_size

        return a

    @staticmethod
    def backward(ctx, dLda : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        qk0, v0, qk1, v1, qk2, v2,  L, m, a = ctx.saved_tensors
        window_size = ctx.window_size

        dLdqk0 = torch.zeros_like(qk0) 
        dLdqk1 = torch.zeros_like(qk1) if qk1 is not None else None
        dLdqk2 = torch.zeros_like(qk2) if qk2 is not None else None

        dLdv0 = torch.zeros_like(v0) 
        dLdv1 = torch.zeros_like(v1) if v1 is not None else None
        dLdv2 = torch.zeros_like(v2) if v2 is not None else None

        torch.ops.rtdenoise.kernel_attn_bwd.default(
            qk0, v0, 
            qk1, v1, 
            qk2, v2,  
            window_size, L, m, a,
            dLda, 
            dLdqk0, dLdv0,
            dLdqk1, dLdv1,
            dLdqk2, dLdv2
        )

        return dLdqk0, dLdv0, dLdqk1, dLdv1, dLdqk2, dLdv2, None
    
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
    

def kernel_attn(
    qk0 : torch.Tensor, v0 : torch.Tensor, 
    qk1 : torch.Tensor, v1 : torch.Tensor, 
    qk2 : torch.Tensor, v2 : torch.Tensor, 
    window_size : int
) -> torch.Tensor:
    return KernelAttention.apply(
        qk0, v0, 
        qk1, v1, 
        qk2, v2,  
        window_size
    )