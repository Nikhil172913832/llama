from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)
k_cache = torch.randn((1, 2, 16, 8))
v_cache = torch.randn((1, 2, 16, 8))
k_valid = k_cache[:, :, :6, :]
v_valid = v_cache[:, :, :6, :]
q = torch.randn((1, 4, 1, 8))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


k_repeated = repeat_kv(k_valid, 2)
v_repeated = repeat_kv(v_valid, 2)
out = torch.nn.functional.scaled_dot_product_attention(q, k_repeated, v_repeated)
np.concatenate(
    [
        k_cache.contiguous().flatten().numpy(),
        v_cache.contiguous().flatten().numpy(),
        q.contiguous().flatten().numpy(),
        out.contiguous().flatten().numpy(),
    ]
).astype(np.float32).tofile(Path(__file__).parent.parent / "testdata/attention.bin")
