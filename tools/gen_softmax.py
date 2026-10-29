import random

import numpy as np
import torch

torch.manual_seed(0)
x = torch.randn(4, 288)
for i in range(20):
    num = random.randrange(0, 4 * 288)
    x[num // 288][num % 288] *= 50
softX = torch.softmax(x, dim=-1)
np.concatenate(
    [
        x.contiguous().flatten().numpy(),
        softX.contiguous().flatten().numpy(),
    ]
).astype(np.float32).tofile("softmax.bin")
