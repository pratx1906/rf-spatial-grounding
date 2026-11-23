"""Wideband multipath channel into the AIR7311's 4 elements (RF phase 2, Task 6; world side, torch).

For one source in one receiver dwell:
    y_m(t) = sum_p g_{p,m} s(t - tau_p) e^{j 2 pi nu_p t},
    g_{p,m} = h_p e^{-j 2 pi (f_c - f_band) tau_p} e^{j k_c u_p . r_m}
  h_p      the cached narrowband coefficient at the band centre f_band (it already holds e^{-j 2 pi f_band tau_p});
           the factor corrects the propagation phase to the source's carrier f_c
  u_p      arrival direction at the node (az_deg/el_deg, compass and elevation); r_m the element positions (ENU)
  tau_p    path delay; applied as a frequency-domain phase ramp over the dwell PLUS a pre-roll (pre_roll(n)
           samples of the same waveform before the dwell), and the pre-roll is dropped: the ramp is circular, so
           without it the last tau_p*fs samples would wrap onto the dwell start (audit D Q30)
  nu_p     per-path Doppler (S7.2, audit B Q14): nu_p = v . u_p f_c / c with each path's own drone-end direction
           az_r/el_r, applied as a per-path frequency shift (paths rotate against each other inside the dwell:
           up to 2 v / lambda ~ 770 Hz apart at 5.8 GHz and 20 m/s). A scalar is one shift for all paths.
s(t) is the source waveform resampled to the dwell rate and shifted to f_c - f_window (sim/rf/waveforms.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch

C = 299_792_458.0
DOP_RES_HZ = 2.0          # paths whose Doppler differs by less share one shift (phase error < 1.6 deg / dwell)


def pre_roll(n: int, min_pre: int = 16384) -> int:
    """Samples of waveform history placed before an n-sample dwell so circular delays never wrap into the dwell:
    pads n + pre to a power of two (fast FFT) with at least min_pre (131 us at 125 MSPS, > any cached path delay)."""
    return (1 << math.ceil(math.log2(n + min_pre))) - n


def resample_shift(x: np.ndarray, fs_in: float, fs_out: float, f_shift_hz: float, n_out: int, t0_s: float,
                   device: str = "cpu", ibw_hz: float | None = None) -> torch.Tensor:
    """Band-limited resampling (FFT) of a native-rate waveform to fs_out, frequency shift, placement at t0_s in a
    dwell of n_out samples (zero outside). Returns complex64 [n_out].

    ibw_hz: the receiver's instantaneous bandwidth. Content whose shifted frequency lies outside +-ibw/2 is removed
    (the front end's anti-alias filter) instead of wrapping around fs_out (audit B Q16)."""
    xt = torch.as_tensor(x, dtype=torch.complex64, device=device)
    n_in = xt.shape[0]
    n_new = int(round(n_in * fs_out / fs_in))
    X = torch.fft.fftshift(torch.fft.fft(xt))
    if n_new >= n_in:
        pad = n_new - n_in
        X = torch.nn.functional.pad(X, (pad // 2, pad - pad // 2))
    else:
        cut = (n_in - n_new) // 2
        X = X[cut:cut + n_new]
    if ibw_hz is not None:
        fr = torch.fft.fftshift(torch.fft.fftfreq(n_new, 1 / fs_out, device=device, dtype=torch.float64))
        X = X * ((fr + f_shift_hz).abs() <= ibw_hz / 2).to(X.dtype)
    y = torch.fft.ifft(torch.fft.ifftshift(X)) * (n_new / n_in)
    t = torch.arange(n_new, device=device, dtype=torch.float64) / fs_out
    y = y * torch.exp(1j * 2 * math.pi * f_shift_hz * t).to(torch.complex64)
    out = torch.zeros(n_out, dtype=torch.complex64, device=device)
    i0 = int(round(t0_s * fs_out))
    a, b = max(i0, 0), min(i0 + n_new, n_out)
    if b > a:
        out[a:b] = y[a - i0:b - i0]
    return out


def steering(az_deg, el_deg, pos_m: np.ndarray, f_hz: float) -> np.ndarray:
    """[P, M] plane-wave phases e^{j k u.r} for arrivals (az from +y clockwise, el up) at elements pos_m [M, 3]."""
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)
    return np.exp(1j * 2 * math.pi * f_hz / C * (u @ pos_m.T))


def propagate(s: torch.Tensor, fs: float, h: np.ndarray, tau_s: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray,
              pos_m: np.ndarray, f_carrier_hz: float, f_band_hz: float, doppler_hz=0.0,
              elem_gain=None, f_lo_hz: float | None = None, n_pre: int = 0) -> torch.Tensor:
    """Multipath into M elements: [M, N - n_pre] complex64. Paths with h == 0 are skipped.

    n_pre: the first n_pre samples of s are waveform history before the dwell (see pre_roll); they are dropped after
    the delay, so delays are linear inside the dwell (no wrap) as long as tau < n_pre / fs.

    s is the complex envelope relative to the receiver LO f_lo_hz (default: the carrier, i.e. s centred at 0 Hz).
    The delay ramp exp(-j2pi f tau) over the baseband frequencies already carries the (f_carrier - f_lo) part of the
    carrier phase, so the cached band-centre coefficient is moved to the LO, not to the carrier (counting the offset
    once). Steering uses the carrier."""
    dev = s.device
    keep = np.abs(h) > 0
    dop = np.broadcast_to(np.asarray(doppler_hz, float), np.shape(h))[keep]           # per-path Doppler [P]
    h, tau_s, az_deg, el_deg = h[keep], tau_s[keep], az_deg[keep], el_deg[keep]
    if elem_gain is not None:
        elem_gain = np.asarray(elem_gain)[keep]
    n = s.shape[0]
    if len(h) == 0:
        return torch.zeros((len(pos_m), n - n_pre), dtype=torch.complex64, device=dev)
    if n_pre and float(np.max(tau_s)) * fs >= n_pre:
        raise ValueError(f"path delay {np.max(tau_s) * 1e6:.1f} us exceeds the pre-roll {n_pre / fs * 1e6:.1f} us")
    f_lo = f_carrier_hz if f_lo_hz is None else f_lo_hz
    g = (h * np.exp(-1j * 2 * math.pi * (f_lo - f_band_hz) * tau_s))[:, None] \
        * steering(az_deg, el_deg, pos_m, f_carrier_hz)                                   # [P, M]
    if elem_gain is not None:
        g = g * elem_gain                                                                   # element ripple [P, M]
    S = torch.fft.fft(s)
    f = torch.fft.fftfreq(n, 1 / fs, device=dev, dtype=torch.float64)
    t = None
    y = torch.zeros((len(pos_m), n), dtype=torch.complex64, device=dev)
    grp = np.round(dop / DOP_RES_HZ)
    for gv in np.unique(grp):                   # one IFFT per Doppler group; memory O(M N), not O(P N)
        idx = np.flatnonzero(grp == gv)
        H = torch.zeros((len(pos_m), n), dtype=torch.complex64, device=dev)
        for p in idx:
            ph = (-2 * math.pi * float(tau_s[p])) * f                 # float64 phase (delays up to ~10 us x 62 MHz)
            ramp = torch.polar(torch.ones_like(ph), ph).to(torch.complex64)
            H += torch.as_tensor(g[p], dtype=torch.complex64, device=dev)[:, None] * ramp[None, :]
        yg = torch.fft.ifft(H * S[None, :])
        nu = float(np.mean(dop[idx]))
        if nu:
            if t is None:
                t = torch.arange(n, device=dev, dtype=torch.float64) / fs
            yg = yg * torch.polar(torch.ones_like(t), 2 * math.pi * nu * t).to(torch.complex64)[None, :]
        y += yg
    return y[:, n_pre:]


def doppler_hz(v_xyz: np.ndarray, az_r_deg, el_r_deg, f_hz: float):
    """Doppler of a path (or of each path, for arrays of az_r/el_r): the drone moves along v; az_r/el_r is the
    direction at the drone end pointing back along the path (toward the node or the first interaction), so
    nu = v . u_r f / c (positive when closing)."""
    az, el = np.radians(np.asarray(az_r_deg, float)), np.radians(np.asarray(el_r_deg, float))
    toward = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)
    nu = (toward @ np.asarray(v_xyz, float)) * f_hz / C
    return float(nu) if np.ndim(nu) == 0 else nu


# ---- GPU paths of the v2 build (Task 2b): the same models as resample_shift() / propagate() above (kept as the
# references; tests/unit/test_t2b_batched.py), with the arbitrary-length FFTs on power-of-two plans
# (sim.rf.fastfft) and the per-path loop replaced by one matrix product per Doppler group.

def resample_shift_fast(x, fs_in: float, fs_out: float, f_shift_hz: float, n_out: int, t0_s: float,
                        device: str = "cpu", ibw_hz: float | None = None) -> torch.Tensor:
    from sim.rf.fastfft import fft_any, ifft_any

    xt = torch.as_tensor(x, dtype=torch.complex64, device=device)
    n_in = xt.shape[0]
    n_new = int(round(n_in * fs_out / fs_in))
    X = torch.fft.fftshift(fft_any(xt))
    if n_new >= n_in:
        pad = n_new - n_in
        X = torch.nn.functional.pad(X, (pad // 2, pad - pad // 2))
    else:
        cut = (n_in - n_new) // 2
        X = X[cut:cut + n_new]
    if ibw_hz is not None:
        fr = torch.fft.fftshift(torch.fft.fftfreq(n_new, 1 / fs_out, device=device, dtype=torch.float64))
        X = X * ((fr + f_shift_hz).abs() <= ibw_hz / 2).to(X.dtype)
    y = ifft_any(torch.fft.ifftshift(X)) * (n_new / n_in)
    t = torch.arange(n_new, device=device, dtype=torch.float64) / fs_out
    y = y * torch.exp(1j * 2 * math.pi * f_shift_hz * t).to(torch.complex64)
    out = torch.zeros(n_out, dtype=torch.complex64, device=device)
    i0 = int(round(t0_s * fs_out))
    a, b = max(i0, 0), min(i0 + n_new, n_out)
    if b > a:
        out[a:b] = y[a - i0:b - i0]
    return out


_FREQ: dict = {}


def _freqs(n: int, fs: float, dev) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, float(fs), str(dev))
    if key not in _FREQ:
        if len(_FREQ) > 8:
            _FREQ.clear()
        _FREQ[key] = (torch.fft.fftfreq(n, 1 / fs, device=dev, dtype=torch.float64),
                      torch.arange(n, device=dev, dtype=torch.float64) / fs)
    return _FREQ[key]


def propagate_fast(s: torch.Tensor, fs: float, h: np.ndarray, tau_s: np.ndarray, az_deg: np.ndarray,
                   el_deg: np.ndarray, pos_m: np.ndarray, f_carrier_hz: float, f_band_hz: float, doppler_hz=0.0,
                   elem_gain=None, f_lo_hz: float | None = None, n_pre: int = 0, S: torch.Tensor | None = None,
                   chunk: int = 8) -> torch.Tensor:
    """propagate() with all paths of a Doppler group in one product H = G^T R ([M, P] x [P, N] delay ramps, built
    in float64 phase like the reference). S: the FFT of s when the caller already has it (one source, several
    nodes)."""
    dev = s.device
    keep = np.abs(h) > 0
    dop = np.broadcast_to(np.asarray(doppler_hz, float), np.shape(h))[keep]
    h, tau_s, az_deg, el_deg = h[keep], tau_s[keep], az_deg[keep], el_deg[keep]
    if elem_gain is not None:
        elem_gain = np.asarray(elem_gain)[keep]
    n = s.shape[0]
    if len(h) == 0:
        return torch.zeros((len(pos_m), n - n_pre), dtype=torch.complex64, device=dev)
    if n_pre and float(np.max(tau_s)) * fs >= n_pre:
        raise ValueError(f"path delay {np.max(tau_s) * 1e6:.1f} us exceeds the pre-roll {n_pre / fs * 1e6:.1f} us")
    f_lo = f_carrier_hz if f_lo_hz is None else f_lo_hz
    g = (h * np.exp(-1j * 2 * math.pi * (f_lo - f_band_hz) * tau_s))[:, None] \
        * steering(az_deg, el_deg, pos_m, f_carrier_hz)
    if elem_gain is not None:
        g = g * elem_gain
    grp = np.round(dop / DOP_RES_HZ)
    order = np.argsort(grp, kind="stable")                  # groups as contiguous rows: device slices, no gathers
    gs = grp[order]
    gt = torch.as_tensor(g[order], dtype=torch.complex64, device=dev)                  # [P, M]
    taut = torch.as_tensor(tau_s[order], dtype=torch.float64, device=dev)
    if S is None:
        S = torch.fft.fft(s)
    f, t = _freqs(n, fs, dev)
    y = torch.zeros((len(pos_m), n), dtype=torch.complex64, device=dev)
    for gv in np.unique(gs):
        a0, a1 = np.searchsorted(gs, gv, "left"), np.searchsorted(gs, gv, "right")
        H = torch.zeros((len(pos_m), n), dtype=torch.complex64, device=dev)
        for c0 in range(a0, a1, chunk):
            c1 = min(c0 + chunk, a1)
            H += gt[c0:c1].T @ _cis_cycles(-taut[c0:c1][:, None] * f[None, :])           # exp(-j 2 pi tau f)
        yg = torch.fft.ifft(H * S[None, :])
        nu = float(np.mean(dop[order[a0:a1]]))
        if nu:
            yg = yg * _cis_cycles(nu * t)[None, :]
        y += yg
    return y[:, n_pre:]


def _cis_cycles(c: torch.Tensor) -> torch.Tensor:
    """exp(j 2 pi c) in complex64 for float64 cycles c: the integer part is removed in float64 (exact phase
    reduction) and the trig runs in float32 on |fraction| <= 0.5 (FP64 trig is ~64x slower on consumer GPUs; error
    ~1e-7, the size of the complex64 rounding of the reference's float64 polar)."""
    r = (c - torch.round(c)).to(torch.float32) * (2 * math.pi)
    return torch.complex(torch.cos(r), torch.sin(r))
