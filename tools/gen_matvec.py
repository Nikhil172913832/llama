import numpy as np
import torch

torch.manual_seed(0)
w = torch.randn(512, 288)
x = torch.randn(288)
out = torch.matmul(w, x)
np.concatenate([x.numpy(), w.contiguous().flatten().numpy(), out.numpy()]).astype(
    np.float32
).tofile("../testdata/matvec.bin")
