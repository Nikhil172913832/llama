from pathlib import Path

import numpy as np
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def main():
    torch.manual_seed(0)
    freq_cis = precompute_freqs_cis(8, 16)
    xq = torch.randn(1, 16, 4, 8)
    xk = torch.randn(1, 16, 2, 8)
    xq_r, xk_r = apply_rotary_emb(xq, xk, freq_cis)
    np.concatenate(
        [
            xq.contiguous().flatten().numpy(),
            xk.contiguous().flatten().numpy(),
            xq_r.contiguous().flatten().numpy(),
            xk_r.contiguous().flatten().numpy(),
            freq_cis.real.contiguous().flatten().numpy(),
            freq_cis.imag.contiguous().flatten().numpy(),
        ]
    ).astype(np.float32).tofile(Path(__file__).parent.parent / "testdata/rope.bin")


if __name__ == "__main__":
    main()
