import numpy as np
import torch

torch.manual_seed(0)
bs, sq, ed = 2, 3, 288
x = torch.randn(bs, sq, ed)
w = torch.randn(ed)
out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
np.concatenate(
    [
        x.contiguous().flatten().numpy(),
        w.numpy(),
        out.contiguous().flatten().numpy(),
    ]
).astype(np.float32).tofile("rmsnorm.bin")
