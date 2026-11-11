from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)
x = torch.randn(5, 4)
up = torch.randn(5, 4)
op = torch.nn.SiLU()
x_s = torch.mul(op(x), up)

np.concatenate(
    [
        x.contiguous().flatten().numpy(),
        up.contiguous().flatten().numpy(),
        x_s.contiguous().flatten().numpy(),
    ]
).astype(np.float32).tofile(Path(__file__).parent.parent / "testdata/swiglu.bin")
