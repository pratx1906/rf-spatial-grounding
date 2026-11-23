"""FFTs of arbitrary length on the GPU without per-length cuFFT plans (Task 2b, build speed; world side, torch).

cuFFT builds a new plan for every distinct transform length, and lengths with large prime factors use a slow
Bluestein plan: ~6 ms per call at 0.1-1 M samples, which dominated the per-burst band-pass of the v2 build (every
burst has its own length). fft_any() computes the same DFT with Bluestein's chirp-z identity on power-of-two
transforms, whose few plans stay in torch's plan cache:

    X_k = w_k sum_n (x_n w_n) conj(w_{k-n}),   w_m = exp(-j pi m^2 / L)

as one zero-padded circular convolution of length M = 2^ceil(log2(2L - 1)). The chirp phase is formed from
m^2 mod 2L in integer arithmetic (exact) and evaluated in float64. Results equal torch.fft.fft to float rounding
(relative error ~1e-6 in complex64, ~1e-14 in complex128; tests/unit/test_t2b_batched.py).
Power-of-two lengths go straight to torch.fft (their plans are cached anyway).
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

_CHIRP: OrderedDict = OrderedDict()       # (L, M, dtype, device) -> (w [L], FFT of the conj chirp [M])
_CHIRP_MAX = 16


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _chirp(L: int, M: int, dtype: torch.dtype, device: torch.device):
    key = (L, M, dtype, str(device))
    hit = _CHIRP.get(key)
    if hit is not None:
        _CHIRP.move_to_end(key)
        return hit
    m = torch.arange(L, device=device, dtype=torch.int64)
    ph = (m * m) % (2 * L)                                   # exact: exp(-j pi m^2 / L) has period 2L in m^2
    if dtype == torch.complex128:
        ang = ph.to(torch.float64) * (-math.pi / L)
        w = torch.polar(torch.ones_like(ang), ang)            # w_m, m = 0..L-1
    else:                                                     # complex64: float32 trig on the reduced phase
        cyc = ph.to(torch.float64) / (2 * L)                  # in [0, 1)
        r = ((cyc - torch.round(cyc)) * (-2 * math.pi)).to(torch.float32)
        w = torch.complex(torch.cos(r), torch.sin(r))
    b = torch.zeros(M, dtype=dtype, device=device)
    b[:L] = w.conj()
    if L > 1:
        b[M - L + 1:] = w[1:].conj().flip(0)                 # b[M - m] = conj(w_m)
    B = torch.fft.fft(b)
    _CHIRP[key] = (w, B)
    if len(_CHIRP) > _CHIRP_MAX:
        _CHIRP.popitem(last=False)
    return w, B


def fft_any(x: torch.Tensor, inverse: bool = False, bluestein: bool | None = None) -> torch.Tensor:
    """DFT (or inverse DFT, 1/L normalised like torch.fft.ifft) along the last dimension, any length.
    bluestein: None = on CUDA for non-power-of-two lengths (CPU FFTs have no plan cost); True forces it (tests)."""
    L = x.shape[-1]
    use = (x.device.type == "cuda") if bluestein is None else bluestein
    if L <= 1 or _is_pow2(L) or not use:
        return torch.fft.ifft(x) if inverse else torch.fft.fft(x)
    if not x.is_complex():
        x = x.to(torch.complex64)
    if inverse:                                               # ifft(x) = conj(fft(conj(x))) / L
        return fft_any(x.conj(), bluestein=True).conj() / L
    M = 1 << math.ceil(math.log2(2 * L - 1))
    w, B = _chirp(L, M, x.dtype, x.device)
    a = torch.nn.functional.pad(x * w, (0, M - L))
    conv = torch.fft.ifft(torch.fft.fft(a) * B)[..., :L]
    return conv * w


def ifft_any(x: torch.Tensor, bluestein: bool | None = None) -> torch.Tensor:
    return fft_any(x, inverse=True, bluestein=bluestein)
