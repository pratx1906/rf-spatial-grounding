"""Burst detection, per-burst spatial covariance and DF on a 4-channel dwell (RF phase 2, Task 6).

detect():   non-coherent channel sum of STFT powers, sum_m |STFT(x_m)|^2 (nfft 1024, hop 512, Hann): a plane wave
            gives the same statistic from every azimuth (the unsteered x.sum(0) had a ~4 dB azimuth ripple,
            audit C Q24); noise floor = 20th percentile of all smoothed cells
            (white receiver noise); 3x3-smoothed
            cells above floor + `thr_db` (7 dB) form a mask, closed over 1-bin / 2-frame gaps; connected components (8-neighbour) become bursts
            (t0, dur, f_lo, f_hi, snr_db). Components spanning < 3 frames, < 2 bins or < 6 cells are dropped (noise specks).
burst_cov(): band-pass the burst (FFT mask over its time slice), 4x4 sample covariance with forward-backward
            averaging (same convention as sim.df.air7311.DFEngine.covariance).
Receiver-artefact rejection in detect() (S7.2b; S7.3 found both artefacts detected as bursts in every dwell):
  DC offset   sim.rf.receiver adds a constant DC of dc_dbfs = -50 dBFS x (1 + j) relative to the strongest channel's
              RMS, i.e. -47 dB below the total input power, in one bin at the LO. A strong in-window carrier
              raises it above the threshold (-60 dBm carrier: DC at ~20 dB per-bin SNR). The detector removes it
              the way a zero-IF receiver's DC-tracking loop does: the per-channel dwell mean is subtracted before
              the STFT. Only a component exactly constant over the dwell is lost; any modulated or offset signal
              near DC keeps (almost) all of its power.
  IQ image    the IQ imbalance y' = mu y + nu conj(y) leaves every signal's mirror image about the LO (bin -f) at
              image_rejection_db() = 10 log10 |mu|^2/|nu|^2 below it (38.1 dB for the default 0.2 dB / 1 deg,
              receiver.IQ_DEFAULTS). A cell is active only when it exceeds the noise threshold PLUS the image
              predicted from its mirror cell raised by IMAGE_MARGIN_DB (6 dB):
                  Ps(f, t) > floor 10^(thr/10) + Ps(-f, t) 10^(-(IRR - margin)/10)
              Same time frame, mirror frequency, power ratio near the IRR: exactly the image's signature. A real
              signal at the image frequency more than IRR - margin (32 dB) below the strong one is still detected.
Per-burst DF reuses DFEngine.spectra on that covariance; hops of one emitter are fused by an SNR-weighted circular
mean in the tracker, which averages out frequency-dependent multipath ghosts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from sim.rf.receiver import image_rejection_db

IMAGE_MARGIN_DB = 6.0        # image explained up to 6 dB above the nominal IRR prediction (IRR spread, noise)


@dataclass
class Burst:
    t0: float
    dur: float
    f_lo: float
    f_hi: float
    snr_db: float

    @property
    def fc(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    @property
    def bw(self) -> float:
        return self.f_hi - self.f_lo


def _components(mask: np.ndarray) -> list[np.ndarray]:
    lab = np.zeros(mask.shape, np.int32)
    out, cur = [], 0
    for i, j in zip(*np.nonzero(mask)):
        if lab[i, j]:
            continue
        cur += 1
        stack, cells = [(i, j)], []
        lab[i, j] = cur
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    x, y = a + da, b + db
                    if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and not lab[x, y]:
                        lab[x, y] = cur
                        stack.append((x, y))
        out.append(np.array(cells))
    return out


def _runs(active: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Inclusive index runs of True, bridging gaps of up to `gap` False samples."""
    idx = np.flatnonzero(active)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _segments(dil: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Vectorised burst segmentation (replaces 8-connected flood fill, which was the CPU hot spot for continuous
    wideband emitters): active frequency bands first, then active time runs inside each band; returns the
    original-mask cells of each (band, time-run) box."""
    out = []
    for f0, f1 in _runs(dil.any(axis=1), 1):
        for t0, t1 in _runs(dil[f0:f1 + 1].any(axis=0), 2):
            box = dil[f0:f1 + 1, t0:t1 + 1]
            occ = box.mean(axis=1) > 0.2           # bins active for > 20 % of this run (drops stray specks)
            for g0, g1 in _runs(occ, 1):
                sub = mask[f0 + g0:f0 + g1 + 1, t0:t1 + 1]
                fi, ti = np.nonzero(sub)
                if len(fi):
                    out.append(np.column_stack([fi + f0 + g0, ti + t0]))
    return out


def detect(x: torch.Tensor, fs: float, thr_db: float = 7.0, nfft: int = 1024, irr_db: float | None = None
           ) -> list[Burst]:
    """irr_db: the receiver's image rejection (default receiver.image_rejection_db() of the default IQ values);
    math.inf disables the image test."""
    hop = nfft // 2
    irr = image_rejection_db() if irr_db is None else irr_db
    win = torch.hann_window(nfft, device=x.device, dtype=torch.float32)
    x = x - x.mean(dim=1, keepdim=True)                     # DC-offset removal (zero-IF DC tracking)
    S = torch.stft(x, nfft, hop, window=win, return_complex=True, center=False)            # [M, F, T] two-sided
    P = torch.fft.fftshift(S.abs().pow(2).sum(0), dim=0).cpu().numpy()                    # non-coherent sum
    Ps = sum(np.roll(np.roll(P, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9   # 3x3 smoothing
    # one global floor: receiver noise is white over the window, and a per-bin median over time would take any
    # signal on air > 50 % of the time (OcuSync bursts at ~60 % duty, CCTV, LTE) as its own noise floor
    floor = np.percentile(Ps, 20) + 1e-30
    snr = 10 * np.log10(Ps / floor)
    # IQ image: bin k of the shifted spectrum is f = (k - nfft/2) df, its mirror -f is bin nfft - k
    mirror = np.roll(Ps[::-1], 1, axis=0)
    img = mirror * 10 ** (-(irr - IMAGE_MARGIN_DB) / 10) if math.isfinite(irr) else 0.0
    mask = Ps > floor * 10 ** (thr_db / 10) + img
    # closing: bridge 1-bin / 2-frame gaps so one transmission is one component (fragments at low SNR otherwise)
    dil = mask.copy()
    for i in (-1, 1):
        dil |= np.roll(mask, i, 0)
    for j in (-2, -1, 1, 2):
        dil |= np.roll(mask, j, 1)
    out = []
    for cells in _segments(dil, mask):
        f_i, t_i = cells[:, 0], cells[:, 1]
        if np.ptp(t_i) < 2 or np.ptp(f_i) < 1 or len(cells) < 6:
            continue                                   # chi-square noise specks at 10 dB: ~1e-4 of STFT cells
        df = fs / nfft
        f_lo = (f_i.min() - nfft / 2) * df
        f_hi = (f_i.max() + 1 - nfft / 2) * df
        t0, t1 = t_i.min() * hop / fs, (t_i.max() * hop + nfft) / fs
        out.append(Burst(t0, t1 - t0, f_lo, f_hi, float(np.median(snr[f_i, t_i]))))
    return sorted(out, key=lambda b: b.t0)


def burst_samples(x: torch.Tensor, fs: float, b: Burst) -> torch.Tensor:
    i0, i1 = max(0, int(b.t0 * fs)), min(x.shape[1], int((b.t0 + b.dur) * fs))
    seg = x[:, i0:i1]
    X = torch.fft.fft(seg, dim=1)
    f = torch.fft.fftfreq(seg.shape[1], 1 / fs, device=x.device)
    keep = ((f >= b.f_lo) & (f <= b.f_hi)).to(X.dtype)
    return torch.fft.ifft(X * keep[None, :], dim=1)


def burst_cov(x: torch.Tensor, fs: float, b: Burst, fb: bool = True) -> torch.Tensor:
    seg = burst_samples(x, fs, b)
    R = seg @ seg.conj().T / max(seg.shape[1], 1)
    if fb:
        J = torch.tensor([2, 3, 0, 1], device=x.device)
        R = 0.5 * (R + R[J][:, J].conj())
    return R


def fuse_bearings(az_deg: list[float], snr_db: list[float]) -> float:
    """SNR-weighted circular mean of per-hop bearings."""
    w = np.power(10.0, np.asarray(snr_db) / 10)
    a = np.radians(az_deg)
    return float(math.degrees(math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) % 360)


# ---- GPU paths of the v2 build (Task 2b). The functions above are the reference implementations; the ones below
# return the same results (detect_gpu: identical bursts; band_pass/burst_cov_from: equal to float rounding) with
# the work on the GPU and few host syncs. tests/unit/test_t2b_batched.py checks them against the references.

def percentile_np(t: torch.Tensor, q: float):
    """np.percentile(t.cpu().numpy(), q) (method 'linear') for a float32 tensor, bit-identical, with the order
    statistics taken on the device: numpy's own index / gamma / lerp helpers are applied to the two neighbouring
    order statistics instead of partitioning the array on the CPU."""
    try:
        from numpy.lib import _function_base_impl as fb

        v = t.flatten()
        n = v.numel()
        lin = fb._QuantileMethods["linear"]
        vi = np.asanyarray(lin["get_virtual_index"](n, np.true_divide(q, np.float32(100))))
        prev, nxt = fb._get_indexes(np.empty(0, dtype=np.float32), vi, n)
        k = sorted({int(prev), int(nxt)})
        vals = torch.sort(v).values[k].cpu().numpy()
        a = np.asarray(vals[0], dtype=np.float32)
        b = np.asarray(vals[-1], dtype=np.float32)
        return fb._lerp(a, b, fb._get_gamma(vi, prev, lin))
    except (AttributeError, ImportError, KeyError, TypeError):           # numpy internals moved: exact fallback
        return np.percentile(t.cpu().numpy(), q)


def detect_gpu(x: torch.Tensor, fs: float, thr_db: float = 7.0, nfft: int = 1024, irr_db: float | None = None
               ) -> list[Burst]:
    """detect() with the smoothing, floor, image test and closing on the device (same float32 operations in the
    same order, so the same mask); only the masks and the smoothed power go to the host for the segmentation."""
    hop = nfft // 2
    irr = image_rejection_db() if irr_db is None else irr_db
    win = torch.hann_window(nfft, device=x.device, dtype=torch.float32)
    x = x - x.mean(dim=1, keepdim=True)
    S = torch.stft(x, nfft, hop, window=win, return_complex=True, center=False)
    P = torch.fft.fftshift(S.abs().pow(2).sum(0), dim=0)
    Ps = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            Ps = Ps + torch.roll(torch.roll(P, i, 0), j, 1)
    Ps = Ps / torch.full_like(Ps, 9.0)        # true division (a scalar divisor is a reciprocal multiply on CUDA)
    floor = percentile_np(Ps, 20) + 1e-30                       # numpy float32 scalar, as in detect()
    th = floor * 10 ** (thr_db / 10)                            # float32 (numpy scalar arithmetic)
    if math.isfinite(irr):
        mirror = torch.roll(torch.flip(Ps, [0]), 1, 0)
        img = mirror * 10 ** (-(irr - IMAGE_MARGIN_DB) / 10)
        mask = Ps > (img + float(th))
    else:
        mask = Ps > (float(th) + 0.0)
    dil = mask.clone()
    for i in (-1, 1):
        dil |= torch.roll(mask, i, 0)
    for j in (-2, -1, 1, 2):
        dil |= torch.roll(mask, j, 1)
    packed = (mask.to(torch.uint8) | (dil.to(torch.uint8) << 1)).cpu().numpy()     # one 2 MB transfer
    mask_h, dil_h = (packed & 1).astype(bool), (packed & 2).astype(bool)
    if not dil_h.any():
        return []
    Ps_h = np.zeros(mask_h.shape, np.float32)
    Ps_h[mask_h] = Ps[mask].cpu().numpy()          # values on mask cells only (row-major order on both sides)
    out = []
    for cells in _segments(dil_h, mask_h):
        f_i, t_i = cells[:, 0], cells[:, 1]
        if np.ptp(t_i) < 2 or np.ptp(f_i) < 1 or len(cells) < 6:
            continue
        df = fs / nfft
        f_lo = (f_i.min() - nfft / 2) * df
        f_hi = (f_i.max() + 1 - nfft / 2) * df
        t0, t1 = t_i.min() * hop / fs, (t_i.max() * hop + nfft) / fs
        snr = 10 * np.log10(Ps_h[f_i, t_i] / floor)
        out.append(Burst(t0, t1 - t0, f_lo, f_hi, float(np.median(snr))))
    return sorted(out, key=lambda b: b.t0)


def band_pass(x: torch.Tensor, fs: float, b: Burst) -> torch.Tensor:
    """burst_samples() with the arbitrary-length FFTs done by sim.rf.fastfft (no per-length cuFFT plan)."""
    from sim.rf.fastfft import fft_any, ifft_any

    i0, i1 = max(0, int(b.t0 * fs)), min(x.shape[1], int((b.t0 + b.dur) * fs))
    seg = x[:, i0:i1]
    X = fft_any(seg)
    f = torch.fft.fftfreq(seg.shape[1], 1 / fs, device=x.device)
    keep = ((f >= b.f_lo) & (f <= b.f_hi)).to(X.dtype)
    return ifft_any(X * keep[None, :])


def cov_from(seg: torch.Tensor, fb: bool = True) -> torch.Tensor:
    """burst_cov() on an already band-passed segment [4, L]."""
    R = seg @ seg.conj().T / max(seg.shape[1], 1)
    if fb:
        R = 0.5 * (R + torch.roll(R, (2, 2), (0, 1)).conj())     # = R[J][:, J], J = [2, 3, 0, 1]; no index upload
    return R
