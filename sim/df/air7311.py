"""AIR7311 4-channel DF simulation on the GPU (torch): Sionna paths -> array snapshots -> impairments -> 14-bit
ADC -> covariance -> Bartlett / MUSIC spectra -> peaks.

Physics input per drone cell: Sionna paths with the full narrowband coefficient h, delay tau and arrival direction
(az, el) at the node. Element n receives  x_n(f, t) = sqrt(P) * sum_p h_p exp(j k u_p.r_n) exp(-j 2 pi (f-fc) tau_p)
* s(f, t) + noise, sampled over `subbands` frequencies across the signal bandwidth (wideband multipath
decorrelation) and `snapshots_per_subband` symbols. Validated steering convention: envbuild/air7311_precompute.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

C = 299792458.0


def square_positions(side_m: float) -> np.ndarray:
    h = side_m / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])   # ch 0,1: chip A; 2,3: chip B


def unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)


class DFEngine:
    def __init__(self, hw: dict, device: str = "cuda"):
        self.hw = hw
        self.dev = torch.device(device)
        p = hw["processing"]
        self.az = np.arange(0.0, 360.0, p["az_step_deg"])
        e0, e1, de = p["el_grid_deg"]
        self.el = np.arange(e0, e1 + 1e-9, de)

    def steering_at(self, pos: np.ndarray, f_hz: float, res_hz: float = 0.1e6) -> torch.Tensor:
        """scan_steering cached per array and frequency (rounded to res_hz: 0.1 MHz is < 4e-5 relative at 2.4 GHz).
        Per-burst DF steers at the burst centre frequency, not the tuning-window centre (S7.2, audit A Q10 / D Q26:
        up to 50 MHz off, which biases the elevation)."""
        key = (np.asarray(pos, float).tobytes(), int(round(f_hz / res_hz)))
        cache = self.__dict__.setdefault("_steer_cache", {})
        if key not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[key] = self.scan_steering(pos, key[1] * res_hz)
        return cache[key]

    @staticmethod
    def mdl_snapshots(bw_hz: float, dur_s: float, fs: float, lo: int = 16, hi: int = 4096) -> int:
        """Effective independent snapshots of a burst for MDL: B x T (the fs x T samples are oversampled by fs / B),
        at most fs x T, clipped to [lo, hi]. The cap keeps MDL from fitting calibration / coupling model errors as
        sources on long wideband bursts (B x T reaches ~30,000), where the -20 dB eigenvalue floor alone would let
        any eigenvalue above it through; S7.2, audit D Q26 (was a fixed 256)."""
        n = min(bw_hz, fs) * dur_s
        return int(np.clip(round(n), lo, hi))

    def scan_steering(self, pos: np.ndarray, f_hz: float) -> torch.Tensor:
        """[n_el, n_az, 4] steering vectors over the scan grid."""
        A, E = np.meshgrid(self.az, self.el)
        u = unit(A, E)                                                   # [n_el, n_az, 3]
        st = np.exp(1j * 2 * np.pi * f_hz / C * (u @ pos.T))
        return torch.as_tensor(st, dtype=torch.complex64, device=self.dev)

    def snapshots(self, h, tau_ns, az, el, valid, pos, f_hz, bw_hz, p_rx_scale, noise_pow, rng: torch.Generator,
                  interchip_deg: float, gain_err_db: torch.Tensor, phase_err_deg: torch.Tensor) -> torch.Tensor:
        """Complex baseband snapshots X [B, 4, F*T] for a batch of cells."""
        p = self.hw["processing"]
        F, T = p["subbands"], p["snapshots_per_subband"]
        dev = self.dev
        u = torch.as_tensor(unit(np.nan_to_num(az), np.nan_to_num(el)), dtype=torch.float32, device=dev)  # [B,K,3]
        P = torch.as_tensor(pos, dtype=torch.float32, device=dev)
        k = 2 * math.pi * f_hz / C
        sv = torch.exp(1j * k * (u @ P.T))                                                    # [B,K,4]
        hv = torch.as_tensor(h, dtype=torch.complex64, device=dev) * torch.as_tensor(valid, device=dev)
        df = torch.linspace(-bw_hz / 2, bw_hz / 2, F, device=dev)                              # [F]
        tau = torch.as_tensor(np.nan_to_num(tau_ns) * 1e-9, dtype=torch.float32, device=dev)  # [B,K]
        ph = torch.exp(-1j * 2 * math.pi * df[None, None, :] * tau[..., None])                # [B,K,F]
        H = torch.einsum("bk,bkn,bkf->bnf", hv, sv.to(torch.complex64), ph.to(torch.complex64))  # [B,4,F]
        H = H * torch.as_tensor(np.sqrt(p_rx_scale), dtype=torch.float32, device=dev)
        B = H.shape[0]
        s = (torch.randn(B, 1, F, T, generator=rng, device=dev) +
             1j * torch.randn(B, 1, F, T, generator=rng, device=dev)) / math.sqrt(2)
        X = H[..., None] * s                                                                  # [B,4,F,T]
        n = (torch.randn(B, 4, F, T, generator=rng, device=dev) +
             1j * torch.randn(B, 4, F, T, generator=rng, device=dev)) * math.sqrt(noise_pow / 2)
        X = (X + n).reshape(B, 4, F * T)
        # residual calibration errors: per-channel gain/phase, plus AD9371 A vs B phase offset
        g = 10 ** (gain_err_db / 20) * torch.exp(1j * torch.deg2rad(phase_err_deg))          # [4]
        chip = torch.tensor([0.0, 0.0, 1.0, 1.0], device=dev) * math.radians(interchip_deg)
        X = X * (g * torch.exp(1j * chip))[None, :, None]
        return self.adc(X)

    def adc(self, X: torch.Tensor) -> torch.Tensor:
        """AGC to -12 dBFS RMS per cell, then 14-bit I/Q quantisation with clipping."""
        bits = self.hw["receiver"]["adc_bits"]
        rms = X.abs().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-30)
        fs = rms * 10 ** (12 / 20) * math.sqrt(2)
        q = 2 ** (bits - 1)
        re = torch.clamp(torch.round(X.real / fs * q), -q, q - 1)
        im = torch.clamp(torch.round(X.imag / fs * q), -q, q - 1)
        return (re + 1j * im) * fs / q

    @staticmethod
    def covariance(X: torch.Tensor, fb: bool = True) -> torch.Tensor:
        R = X @ X.conj().transpose(1, 2) / X.shape[-1]                                        # [B,4,4]
        if fb:   # forward-backward averaging; the square array is centro-symmetric: 0<->2, 1<->3
            J = torch.tensor([2, 3, 0, 1], device=X.device)
            R = 0.5 * (R + R[:, J][:, :, J].conj())
        return R

    @staticmethod
    def mdl_order(ev: torch.Tensor, N: int, max_order: int = 2, rel_floor_db: float | None = None) -> torch.Tensor:
        """MDL model order from ascending eigenvalues [B,4], capped at 2 so MUSIC keeps >= 2 noise dimensions
        (D-022: order 3 on a 4-channel array is fragile under calibration errors). With rel_floor_db, eigenvalues
        more than that far below the largest count as noise: residual calibration mismatch under forward-backward
        averaging creates such an eigenvalue (~-29 dB at 0.3 dB gain error; rf_validation.md). None = plain MDL."""
        M = ev.shape[-1]
        lam = torch.flip(ev, [-1]).clamp_min(1e-30)                                           # descending
        best = torch.full(ev.shape[:1], 1, device=ev.device)
        best_val = torch.full(ev.shape[:1], float("inf"), device=ev.device)
        for k in range(0, max_order + 1):
            tail = lam[:, k:]
            g = torch.exp(torch.log(tail).mean(-1))
            a = tail.mean(-1)
            mdl = -N * (M - k) * torch.log(g / a) + 0.5 * k * (2 * M - k) * math.log(N)
            better = mdl < best_val
            best = torch.where(better, torch.full_like(best, k), best)
            best_val = torch.where(better, mdl, best_val)
        if rel_floor_db is not None:
            above = (lam >= lam[:, :1] * 10 ** (rel_floor_db / 10)).sum(-1)
            best = torch.minimum(best, above)
        return best.clamp(1, max_order)

    def spectra(self, R: torch.Tensor, A: torch.Tensor, n_snap: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Bartlett and MUSIC azimuth spectra [B, n_az] (max over the elevation scan), in dB."""
        Af = A.reshape(-1, 4)                                                                  # [D,4]
        bart = torch.einsum("dn,bnm,dm->bd", Af.conj(), R, Af).real.clamp_min(1e-30)
        ev, V = torch.linalg.eigh(R)                                                           # ascending
        order = self.mdl_order(ev, n_snap, rel_floor_db=self.hw["processing"].get("model_error_floor_db"))
        noise_mask = (torch.arange(4, device=R.device)[None, :] < (4 - order)[:, None]).to(V.dtype)
        En = V * noise_mask[:, None, :]
        proj = (Af.conj() @ En).abs().pow(2).sum(-1).clamp_min(1e-12)                          # [B,D]
        music = 1.0 / proj
        ne, na = A.shape[:2]
        bart = 10 * torch.log10(bart.reshape(-1, ne, na).amax(1))
        music = 10 * torch.log10(music.reshape(-1, ne, na).amax(1))
        return bart, music

    def rank_by_power(self, music: torch.Tensor, bart: torch.Tensor, floor_db: float,
                      max_peaks: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """MUSIC peak locations ranked by beamformed (Bartlett) power at each location.

        MUSIC pseudo-spectrum height is not signal power (a weak reflection can give the tallest MUSIC peak),
        so DF systems locate with MUSIC and rank with power. Returns az [B, max_peaks], power dB [B, max_peaks]."""
        left, right = torch.roll(music, 1, -1), torch.roll(music, -1, -1)
        is_pk = (music >= left) & (music > right) & (music >= music.amax(-1, keepdim=True) - 30.0)
        pw = torch.where(is_pk, bart, torch.full_like(bart, -1e9))
        top_v, top_i = torch.topk(pw, max_peaks, dim=-1)
        keep = (top_v > -1e8) & (top_v >= top_v[:, :1] - floor_db)
        az = torch.as_tensor(self.az, device=music.device)[top_i]
        az = torch.where(keep, az, torch.full_like(az, float("nan")))
        return az.cpu().numpy(), torch.where(keep, top_v, torch.full_like(top_v, float("nan"))).cpu().numpy()

    def peaks(self, spec: torch.Tensor, floor_db: float, max_peaks: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Circular local maxima within floor_db of the strongest: az [B, max_peaks] (nan-padded), level dB."""
        left, right = torch.roll(spec, 1, -1), torch.roll(spec, -1, -1)
        is_pk = (spec >= left) & (spec > right) & (spec >= spec.amax(-1, keepdim=True) - floor_db)
        val = torch.where(is_pk, spec, torch.full_like(spec, -1e9))
        top_v, top_i = torch.topk(val, max_peaks, dim=-1)
        az = torch.as_tensor(self.az, device=spec.device)[top_i]
        az = torch.where(top_v > -1e8, az, torch.full_like(az, float("nan")))
        return az.cpu().numpy(), top_v.cpu().numpy()

    @staticmethod
    def mdl_order_n(ev: torch.Tensor, N: torch.Tensor, max_order: int = 2, rel_floor_db: float | None = None
                    ) -> torch.Tensor:
        """mdl_order() with a per-row snapshot count N [B] (batched per-burst DF, Task 2b)."""
        M = ev.shape[-1]
        logn = torch.log(N.to(device=ev.device, dtype=torch.float64))          # math.log(N) of the reference
        N = N.to(device=ev.device, dtype=ev.dtype)
        lam = torch.flip(ev, [-1]).clamp_min(1e-30)
        best = torch.full(ev.shape[:1], 1, device=ev.device)
        best_val = torch.full(ev.shape[:1], float("inf"), device=ev.device)
        for k in range(0, max_order + 1):
            tail = lam[:, k:]
            g = torch.exp(torch.log(tail).mean(-1))
            a = tail.mean(-1)
            mdl = -N * (M - k) * torch.log(g / a) + (0.5 * k * (2 * M - k) * logn).to(ev.dtype)
            better = mdl < best_val
            best = torch.where(better, torch.full_like(best, k), best)
            best_val = torch.where(better, mdl, best_val)
        if rel_floor_db is not None:
            above = (lam >= lam[:, :1] * 10 ** (rel_floor_db / 10)).sum(-1)
            best = torch.minimum(best, above)
        return best.clamp(1, max_order)

    def spectra_batch(self, R: torch.Tensor, A: torch.Tensor, n_snap: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """spectra() with one steering grid per row (A [B, n_el, n_az, 4]: each burst steered at its own centre
        frequency) and per-row MDL snapshot counts. Returns (bartlett dB [B, n_az], music dB [B, n_az], the full
        linear Bartlett grid [B, n_el, n_az] used for the elevation pick)."""
        Bn, ne, na = A.shape[:3]
        Af = A.reshape(Bn, -1, 4)                                                              # [B,D,4]
        full = torch.einsum("bdn,bnm,bdm->bd", Af.conj(), R, Af).real
        bart = full.clamp_min(1e-30)
        ev, V = torch.linalg.eigh(R)
        order = self.mdl_order_n(ev, n_snap, rel_floor_db=self.hw["processing"].get("model_error_floor_db"))
        noise_mask = (torch.arange(4, device=R.device)[None, :] < (4 - order)[:, None]).to(V.dtype)
        En = V * noise_mask[:, None, :]
        proj = (Af.conj() @ En).abs().pow(2).sum(-1).clamp_min(1e-12)                         # [B,D]
        music = 1.0 / proj
        bart_db = 10 * torch.log10(bart.reshape(Bn, ne, na).amax(1))
        music_db = 10 * torch.log10(music.reshape(Bn, ne, na).amax(1))
        return bart_db, music_db, full.reshape(Bn, ne, na)
