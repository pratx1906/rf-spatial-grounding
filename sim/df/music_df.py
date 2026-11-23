#!/usr/bin/env python3
"""music_df.py: four-channel MUSIC direction finder and multi-target bearing tracker, in one file.

Hardware : Deepwave AIR7311 (two AD9371 RFICs, 4 phase-coherent-capable RX; RX1/RX2 share one LO, RX3/RX4 the other)
           with four vertical omnis on a horizontal square, e.g. Gleam Light GLIOM06KGZ (174-6000 MHz, 6 dBi,
           55 deg vertical beamwidth, VSWR <= 2.0, N male).
Simulation: a built-in multi-emitter / multipath / impaired-receiver simulator feeds the identical pipeline.

Pipeline, per block of 4 x N complex samples
  1. STFT channeliser (Hann, nfft bins). Every (frame, bin) cell is one narrowband snapshot y = C a(theta) s + n.
  2. Detector: robust noise floor, cell threshold, dilation + 8-connected components -> bursts (snapshots = the
     detected cells only). Splitting emitters in time-frequency BEFORE DF is what lets a 4-element array handle
     many drones: each burst then holds 1-2 sources (direct path + strongest reflection), inside MUSIC's M - 2 = 2
     source limit.
  3. Per burst: covariance of its cells, removal of the channel gains, forward-backward averaging (nominal
     centro-symmetric square only), eigendecomposition, MDL model order capped at M - 2 and gated by the detector's
     independent noise floor (no spurious second source on a single emitter).
  4. MUSIC on a coarse grid for ALL bursts in one batched product (azimuth only by default; --estimate-el adds an
     elevation scan, low confidence on a horizontal array); peaks LOCATED by MUSIC and RANKED by least-squares
     source power (MUSIC height is not power: a weak reflection can give the tallest MUSIC peak); never more peaks
     than MDL sources; 3-stage zoom + parabolic interpolation (grid bias < 0.01 deg); Cramer-Rao bound (+ a model
     error floor) as each peak's reported std.
  5. Bearing tracker: per-block fusion of repeated detections, constant-rate Kalman filter per track on the circle,
     Hungarian assignment with Mahalanobis and carrier-frequency gates, M-of-N confirmation, multipath-suspect flag
     for same-carrier secondary tracks.

Accuracy is set by calibration and geometry, not by the estimator. On a 0.45-lambda square a 1 mm element error is
3 deg of phase at 2.4 GHz and costs 2-9 deg of bearing (simulated); the self-test reaches 0.13-0.18 deg only with:
  * a measured-manifold TONE TABLE (ManifoldCorrection, `calibrate --tone-az` at ~72 bearings, then `fit-manifold`):
    absorbs coupling, element position / phase-centre errors and mast effects, once per installation and band;
  * per-channel complex gains refreshed by an over-the-air BEACON at a known bearing every block (in-band, no RF
    switch), or a splitter `calibrate` in the same session. Deepwave: "Each calibration is valid only for your
    current configured Rx LO frequency", phasors live only "for the life of the current SoapySDR device object":
    the phase between the two AD9371s is re-randomised on every retune and every device open.
  * geometry: side <= ~0.45 lambda at the HIGHEST frequency used, or bearings alias (`geometry` checks it). The
    omni's diameter (not on its datasheet; measure it) sets the minimum side.
Speed: memory-bandwidth bound in numpy, ~1.4 s of CPU per second of 4 x 31.25 MSPS on a laptop. Above that, blocks
are dropped (a readStream overflow restarts the block); DF needs bursts, not continuous coverage. Use a lower
--fs, or port _stft/_detect to CuPy on the Jetson for full duty.

Usage
  python music_df.py selftest
  python music_df.py sim --seconds 2
  python music_df.py geometry --side 0.054 --fc 2.4835e9 --element-diameter 0.04
  python music_df.py calibrate --fc 2.4415e9 --gain-db 20 --out cal.npz                 # 1:4 splitter connected
  python music_df.py calibrate --fc 2.4415e9 --gain-db 20 --heading 0 --tone-az 35      # repeat per tone bearing
  python music_df.py fit-manifold --side 0.054 --tones tones.npz --out manifold.npz
  python music_df.py hw --fc 2.4415e9 --fs 31.25e6 --side 0.054 --heading 12.5 --gain-db 20 \
        --manifold manifold.npz --beacon-freq 2.4535e9 --beacon-az 300 --beacon-el 0
Output: one JSON line per block on stdout, {"t", "detections": [...], "tracks": [...]}; timing on stderr.
Requires numpy and scipy; SoapySDR (Deepwave AirStack) only for `hw` and `calibrate`.
Conventions: azimuth in degrees clockwise from the array +y axis (array frame) or from true north (outputs:
array az + heading); elevation >= 0, since a horizontal planar array cannot tell +el from -el.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

C0 = 299_792_458.0

# Datasheet values. The omni's diameter is NOT on its datasheet: measure it before choosing the array side.
ANTENNA = {"model": "GLIOM06KGZ", "f_range_hz": (174e6, 6000e6), "gain_dbi": 6.0, "vbw_deg": 55.0, "vswr_max": 2.0}
AIR7311 = {"f_range_hz": (300e6, 6000e6), "ibw_hz": 100e6, "fs_range_hz": (1.875e6, 125e6), "nf_db": 3.0,
           "rx_lo_pairs": ((0, 1), (2, 3)), "p_in_3db_dbm": 15.0, "adc_bits": 14}


# ----------------------------------------------------------------------------------------------------------------
# Geometry and array manifold
# ----------------------------------------------------------------------------------------------------------------

def square_array(side_m: float) -> np.ndarray:
    """Element phase centres [4, 3] (m) in the array frame: x right, y forward (az 0), z up. RX1..RX4 go clockwise
    from the front-left corner, so RX1/RX2 (one AD9371) are the front pair and RX3/RX4 (the other) the back pair."""
    h = side_m / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def unit(az_deg, el_deg) -> np.ndarray:
    """Arrival unit vectors [..., 3] (pointing at the source) for az clockwise from +y and el up from horizontal."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    return np.stack([np.sin(az) * ce, np.cos(az) * ce, np.sin(el) * np.ones_like(az)], -1)


def steering(pos: np.ndarray, f_hz, az_deg, el_deg) -> np.ndarray:
    """Narrowband steering vectors [..., M] = exp(+j 2 pi f / c * u . r_m), phase referenced to the array centroid.
    f_hz broadcasts against az/el (scalar, or one frequency per leading row)."""
    k = 2 * np.pi * np.asarray(f_hz, float) / C0
    return np.exp(1j * np.asarray(k)[..., None] * (unit(az_deg, el_deg) @ pos.T))


def d_steering_daz(pos: np.ndarray, f_hz, az_deg, el_deg) -> np.ndarray:
    """d a / d az (per radian) [..., M], used for the Cramer-Rao bound."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    du = np.stack([np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), np.zeros_like(az)], -1)
    k = 2 * np.pi * np.asarray(f_hz, float) / C0
    return 1j * np.asarray(k)[..., None] * (du @ pos.T) * steering(pos, f_hz, az_deg, el_deg)


def centro_permutation(pos: np.ndarray, tol: float = 1e-6) -> np.ndarray | None:
    """J with pos[J[i]] = -pos[i] (about the centroid), or None when the array is not centro-symmetric. Then
    a(theta)* = J a(theta) and forward-backward averaging R_fb = (R + J R* J) / 2 is valid."""
    c = pos - pos.mean(0)
    d = np.linalg.norm(c[:, None, :] + c[None, :, :], axis=-1)
    J = d.argmin(1)
    ok = np.all(d[np.arange(len(c)), J] < tol * max(1.0, np.abs(c).max())) and len(set(J.tolist())) == len(c)
    return J if ok else None


def element_gain(el_deg, vbw_deg: float = ANTENNA["vbw_deg"]) -> np.ndarray:
    """Relative power pattern of a vertical collinear omni vs elevation: cos(el)^n with the -3 dB points at
    +-vbw/2 (55 deg -> n = 5.8). Identical elements cancel in MUSIC; this only sets SNR (simulation, link budget)."""
    n = math.log(0.5) / math.log(math.cos(math.radians(vbw_deg / 2)))
    return np.cos(np.radians(np.clip(el_deg, -89.9, 89.9))) ** n


def ambiguity(pos: np.ndarray, f_hz: float, min_sep_deg: float = 30.0, el_max: float = 60.0) -> float:
    """Largest normalised manifold correlation |a_i^H a_j| / M between directions at least min_sep_deg apart.
    -> 1.0 means two different bearings give the same array response (aliasing; MUSIC cannot separate them)."""
    az, el = np.meshgrid(np.arange(0, 360, 2.0), np.arange(0, el_max + 1e-9, 15.0))
    u = unit(az.ravel(), el.ravel())
    A = steering(pos - pos.mean(0), f_hz, az.ravel(), el.ravel())
    G = np.abs(A.conj() @ A.T) / pos.shape[0]
    G[(u @ u.T) > math.cos(math.radians(min_sep_deg))] = 0.0
    return float(G.max())


def max_unambiguous_freq(pos: np.ndarray, thr: float = 0.995, f0: float = 100e6, f1: float = 6e9) -> float:
    """Usable upper frequency: where ambiguity() climbs back to thr above its minimum. (At low frequency the
    correlation is also high, but from a small aperture, not aliasing: worse resolution, no wrong bearings.)"""
    fs = np.arange(f0, f1 + 1, 20e6)
    amb = np.array([ambiguity(pos, f) for f in fs])
    above = np.flatnonzero(amb[int(amb.argmin()):] >= thr)
    return float(fs[int(amb.argmin()) + above[0]]) if len(above) else float(f1)


def crb_az_deg(pos: np.ndarray, f_hz: float, az_deg, el_deg, snr_lin, n_snap) -> np.ndarray:
    """Deterministic single-source Cramer-Rao bound on azimuth (deg): var = 1 / (2 N SNR ||P_a^perp da/daz||^2),
    SNR per element per snapshot. The floor any unbiased estimator (MUSIC included) can reach."""
    a = steering(pos, f_hz, az_deg, el_deg)
    d = d_steering_daz(pos, f_hz, az_deg, el_deg)
    M = pos.shape[0]
    dp = d - a * (np.sum(a.conj() * d, -1, keepdims=True) / M)
    g = np.sum(np.abs(dp) ** 2, -1)
    return np.degrees(np.sqrt(1.0 / (2 * np.asarray(n_snap) * np.asarray(snr_lin) * np.maximum(g, 1e-30))))


# ----------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------

def principal_vector(R: np.ndarray) -> np.ndarray:
    """Dominant eigenvector of a Hermitian matrix [M, M]."""
    return np.linalg.eigh(R)[1][:, -1]


def gains_from_reference(R_raw: np.ndarray, a_ref: np.ndarray) -> np.ndarray:
    """Per-channel complex gains g (|g| mean 1, arg g[0] = 0) from the raw covariance of ONE strong reference whose
    array response a_ref is known: splitter with phase-matched cables (a_ref = ones) or a beacon at a known bearing
    (a_ref = the (corrected) steering vector toward it). Model: y = diag(g) a_ref s + n, so the principal
    eigenvector is proportional to g * a_ref."""
    g = principal_vector(R_raw) / a_ref
    g = g * np.exp(-1j * np.angle(g[0]))
    return g / np.mean(np.abs(g))


class ManifoldCorrection:
    """Measured-manifold calibration: per-element complex correction D_m(az) = a_measured,m / a_nominal,m as a
    Fourier series in azimuth (effective-aperture style). One table absorbs mutual coupling, element position and
    phase-centre errors, mast scattering and fixed channel gains, which a single per-channel gain cannot. The
    correction is stored relative to the nominal steering, so the nominal phase still scales with frequency across
    the band; it is measured at one elevation and applied at all. Fit it from tones at known bearings (turntable,
    or a transmitter moved around the array at >= 4 lambda): >= 4H + 2 bearings, 5 deg steps recommended."""

    def __init__(self, coeffs: np.ndarray, f_hz: float):
        self.coeffs = np.asarray(coeffs, complex)                                # [2H+1, M]
        self.H = (self.coeffs.shape[0] - 1) // 2
        self.f_hz = float(f_hz)

    def _basis(self, az_deg) -> np.ndarray:
        return np.exp(1j * np.radians(np.asarray(az_deg, float))[..., None] * np.arange(-self.H, self.H + 1))

    def __call__(self, az_deg) -> np.ndarray:
        """Correction [..., M] at array-frame azimuths."""
        return self._basis(az_deg) @ self.coeffs

    @classmethod
    def fit(cls, R_list, pos: np.ndarray, f_hz: float, az_deg, el_deg: float = 0.0, n_harm: int = 8):
        az = np.asarray(az_deg, float)
        A = steering(pos - pos.mean(0), f_hz, az, np.full_like(az, el_deg))                    # [K, M] nominal
        D = np.stack([principal_vector(R) for R in R_list]) / A                                 # [K, M]
        D = D / D[:, :1]                          # each tone's unknown complex amplitude: element 0 is the reference
        obj = cls(np.zeros((2 * n_harm + 1, A.shape[1])), f_hz)
        obj.coeffs = np.linalg.lstsq(obj._basis(az), D, rcond=None)[0]
        resid = np.abs(obj(az) - D).max()
        if resid > 0.1:
            print(f"[music_df] manifold fit residual {resid:.3f}: noisy tones or too few harmonics", file=sys.stderr)
        return obj

    def save(self, path: str):
        np.savez(path, coeffs=self.coeffs, f_hz=self.f_hz)

    @classmethod
    def load(cls, path: str):
        z = np.load(path)
        return cls(z["coeffs"], float(z["f_hz"]))


# ----------------------------------------------------------------------------------------------------------------
# Direction finder
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class DFConfig:
    positions_m: np.ndarray = field(default_factory=lambda: square_array(0.054))
    heading_deg: float = 0.0          # true bearing of the array +y axis (GNSS compass); output az = array az + this
    fc_hz: float = 2.4415e9           # LO (same on both AD9371s)
    fs_hz: float = 31.25e6            # 125 MSPS / 4
    nfft: int = 256                   # 122 kHz bins at 31.25 MSPS; frames of 8.2 us
    guard_frac: float = 0.10          # band edges excluded (anti-alias filter roll-off)
    cell_snr_db: float = 3.0          # per-cell detection threshold above noise
    min_cells: int = 24               # smallest burst (cells) worth a DF
    max_bursts: int = 64              # per block, strongest first
    max_snap: int = 4096              # strongest cells used per burst covariance
    az_step_deg: float = 1.0          # coarse MUSIC grid (refined to < 0.01 deg afterwards)
    estimate_el: bool = False         # a horizontal 4-element array observes el only through cos(el): mm-level
                                      # element errors move it by 10-40 deg. Azimuth-only (el fixed below) is
                                      # unbiased to first order for a centro-symmetric array and ~13x cheaper.
    el_fixed_deg: float = 0.0
    el_step_deg: float = 5.0
    el_max_deg: float = 60.0
    max_sources: int = 2              # MDL cap: M - 2 keeps 2 noise eigenvectors
    eig_floor_db: float = -20.0       # eigenvalues this far below the largest are noise (calibration residuals)
    peak_window_db: float = 15.0      # MUSIC local maxima within this of the top are candidates
    peak_floor_db: float = 10.0       # report sources within this of a burst's strongest
    max_peaks: int = 3
    merge_deg: float = 3.0            # candidate peaks closer than this within a burst are one source
    fb_average: bool = True
    model_error_deg: float = 0.5      # added in quadrature to the CRB in the reported std (calibration residuals)
    iq_conj: bool = False             # set True if bearings come out mirrored (az -> -az): IQ convention
    beacon_f_hz: float | None = None  # in-band calibration beacon (absolute carrier), true bearing, elevation
    beacon_az_deg: float = 0.0
    beacon_el_deg: float = 0.0
    beacon_alpha: float = 0.3         # EMA weight of each new beacon gain estimate
    beacon_min_snr_db: float = 15.0


def _gamma_median(k: int) -> float:
    """Median of Gamma(k, 1): the median of a sum of k unit exponentials (noise power summed over k channels)."""
    return k - 1 / 3 + 8 / (405 * k)


def mdl_order(ev: np.ndarray, n_snap: np.ndarray, max_order: int, floor_db: float,
              noise: np.ndarray | None = None, margin_db: float = 3.0) -> np.ndarray:
    """MDL source count from ascending eigenvalues [B, M] and per-row snapshot counts, capped at max_order. Any
    eigenvalue more than floor_db below the largest counts as noise, and so does any eigenvalue below margin_db
    above the detector's independent noise floor `noise` [B]: with thousands of snapshots MDL alone takes small
    calibration-residual eigenvalues for sources, and a spurious second source bends the first bearing."""
    B, M = ev.shape
    lam = np.clip(ev[:, ::-1], 1e-30, None)
    n = np.asarray(n_snap, float)
    best, best_val = np.zeros(B, int), np.full(B, np.inf)
    for k in range(max_order + 1):
        tail = lam[:, k:]
        v = -n * (M - k) * np.log(np.exp(np.log(tail).mean(1)) / tail.mean(1)) + 0.5 * k * (2 * M - k) * np.log(n)
        better = v < best_val
        best[better], best_val[better] = k, v[better]
    above = (lam >= lam[:, :1] * 10 ** (floor_db / 10)).sum(1)
    if noise is not None:
        above = np.minimum(above, (lam > np.asarray(noise)[:, None] * 10 ** (margin_db / 10)).sum(1))
    return np.clip(np.minimum(best, above), 1, max_order)


class DirectionFinder:
    """Block-in, detections-out. Stateful only for calibration (beacon EMA)."""

    def __init__(self, cfg: DFConfig, gains: np.ndarray | None = None, correction: ManifoldCorrection | None = None):
        self.cfg = cfg
        self.pos = np.asarray(cfg.positions_m, float) - np.asarray(cfg.positions_m, float).mean(0)
        self.M = len(self.pos)
        self.corr = correction
        # forward-backward averaging needs a(theta)* = J a(theta): true for the nominal square, not for a measured one
        self.J = centro_permutation(self.pos) if cfg.fb_average and correction is None else None
        self.gains = np.ones(self.M, complex) if gains is None else np.asarray(gains, complex)
        self._beacon_init = gains is not None
        self._update_whitener()
        # Hann window times (-1)^n: the FFT output then comes out already fftshifted (LO in bin nfft/2), for free
        self.win = (np.hanning(cfg.nfft) * (-1.0) ** np.arange(cfg.nfft)).astype(np.float32)
        self.az = np.arange(0.0, 360.0, cfg.az_step_deg)
        self.el = np.arange(0.0, cfg.el_max_deg + 1e-9, cfg.el_step_deg) if cfg.estimate_el \
            else np.array([cfg.el_fixed_deg])
        AZ, EL = np.meshgrid(self.az, self.el)                                   # [n_el, n_az]
        self._path = unit(AZ.ravel(), EL.ravel()) @ self.pos.T                   # [D, M] path differences (m)
        self._grid_cache: dict[int, np.ndarray] = {}
        self.beacon_snr_db = float("nan")

    # -- calibration ---------------------------------------------------------------------------------------------
    def _update_whitener(self):
        """W = diag(g)^-1: R_cal = W R W^H removes the per-channel gains (incl. the inter-RFIC phase), leaving the
        nominal or measured manifold."""
        self.W = np.diag(1.0 / self.gains)

    def _steer(self, f_hz, az_deg, el_deg) -> np.ndarray:
        """Steering [..., M]: nominal, times the measured correction when present (then normalised to ||a||^2 = M)."""
        a = steering(self.pos, f_hz, az_deg, el_deg)
        if self.corr is not None:
            a = a * self.corr(az_deg)
            a *= math.sqrt(self.M) / np.linalg.norm(a, axis=-1, keepdims=True)
        return a

    def _beacon_update(self, R_raw: np.ndarray, f_hz: float):
        cfg = self.cfg
        ev = np.linalg.eigvalsh(R_raw)
        noise = max(ev[:-1].mean(), 1e-30)
        self.beacon_snr_db = 10 * math.log10(max((ev[-1] - noise) / self.M / noise, 1e-12))
        if self.beacon_snr_db < cfg.beacon_min_snr_db:
            return
        g = gains_from_reference(R_raw, self._steer(f_hz, cfg.beacon_az_deg - cfg.heading_deg, cfg.beacon_el_deg))
        if self._beacon_init:                                                    # EMA after the first estimate
            g = g * np.exp(1j * np.angle(np.vdot(g, self.gains)))                # align the common phase first
            g = (1 - cfg.beacon_alpha) * self.gains + cfg.beacon_alpha * g
        self.gains, self._beacon_init = g, True
        self._update_whitener()

    # -- front end -----------------------------------------------------------------------------------------------
    def _grid(self, f_hz: float) -> np.ndarray:
        """Coarse-grid steering [D, M] complex64, cached per 1 MHz (phase error < 1e-3 rad for apertures < 0.3 m)."""
        key = int(round(f_hz / 1e6))
        g = self._grid_cache.get(key)
        if g is None:
            if len(self._grid_cache) > 64:
                self._grid_cache.clear()
            g = np.exp(1j * (2 * np.pi * key * 1e6 / C0) * self._path)
            if self.corr is not None:
                g = g * np.tile(self.corr(self.az), (len(self.el), 1))
                g *= math.sqrt(self.M) / np.linalg.norm(g, axis=-1, keepdims=True)
            g = g.astype(np.complex64)
            self._grid_cache[key] = g
        return g

    def _stft(self, X: np.ndarray):
        nf = self.cfg.nfft
        nt = X.shape[1] // nf
        buf = self.__dict__.get("_buf")
        if buf is None or buf.shape != (self.M, nt, nf):
            buf = self._buf = np.empty((self.M, nt, nf), np.complex64)          # reused: no per-block allocation
        np.multiply(X[:, : nt * nf].reshape(self.M, nt, nf), self.win, out=buf)
        Y = sfft.fft(buf, axis=-1, workers=-1)                                  # [M, nt, nf], bin nf/2 = LO
        Yr = Y.view(np.float32)                                                  # interleaved re/im, no copy
        P = np.einsum("mtk,mtk->tk", Yr, Yr).reshape(nt, nf, 2).sum(-1)          # [nt, nf] power over channels
        return Y, P

    def _detect(self, Y: np.ndarray, P: np.ndarray, t0_s: float):
        cfg, M = self.cfg, self.M
        nt, nf = P.shape
        valid = np.ones(nf, bool)
        g = int(round(cfg.guard_frac * nf / 2))
        valid[:g] = False
        valid[nf - g:] = False
        valid[nf // 2 - 1: nf // 2 + 2] = False                                  # DC / LO leakage bins
        # noise per channel per cell: per-bin median over time (robust to bursts), 25th percentile across bins
        # (robust to continuously occupied bins), divided by the Gamma(M) median
        step = max(1, nt // 256)                                                 # 256 frames are plenty for a median
        sigma2 = float(np.percentile(np.median(P[::step], 0)[valid], 25)) / _gamma_median(M)
        hit = (P > M * sigma2 * (1 + 10 ** (cfg.cell_snr_db / 10))) & valid[None, :]
        s3 = np.ones((3, 3), bool)
        # group detected cells that are at most one cell apart (dilate, then label); the snapshots are the
        # detected cells only, so the bridging never adds noise-only cells to a covariance
        lab, n = ndimage.label(ndimage.binary_dilation(hit, s3), s3)
        if n == 0:
            return []
        flat = np.where(hit, lab, 0).ravel()
        idx = np.flatnonzero(flat)
        lbl = flat[idx]
        order = np.argsort(lbl, kind="stable")
        idx, lbl = idx[order], lbl[order]
        groups = np.split(idx, np.flatnonzero(np.diff(lbl)) + 1)
        Pf = P.ravel()
        cand = [(float(Pf[c].sum()), c) for c in groups if len(c) >= cfg.min_cells]
        cand.sort(key=lambda x: -x[0])
        df_bin, dt_frame = cfg.fs_hz / nf, nf / cfg.fs_hz
        bursts = []
        for energy, cells in cand[: cfg.max_bursts]:
            e = Pf[cells]
            if len(cells) > cfg.max_snap:                                        # strongest cells: best SNR per snapshot
                keep = np.argpartition(-e, cfg.max_snap)[: cfg.max_snap]
                cells, e = cells[keep], e[keep]
            ti, fi = np.divmod(cells, nf)
            Z = Y[:, ti, fi].astype(np.complex128)                               # [M, n] snapshots
            bursts.append({
                "R": Z @ Z.conj().T / Z.shape[1], "n": Z.shape[1],
                "f_hz": cfg.fc_hz + (float(np.dot(e, fi)) / float(e.sum()) - nf / 2) * df_bin,
                "bw_hz": (int(fi.max()) - int(fi.min()) + 1) * df_bin,
                "t_s": t0_s + (float(ti.mean()) + 0.5) * dt_frame,
                "dur_s": (int(ti.max()) - int(ti.min()) + 1) * dt_frame,
                "snr_cell_db": 10 * math.log10(max(float(e.mean()) / (M * sigma2) - 1, 1e-6)),
                "noise": sigma2})
        return bursts

    # -- DF core -------------------------------------------------------------------------------------------------
    def _null(self, Pn: np.ndarray, f: np.ndarray, az: np.ndarray, el: np.ndarray) -> np.ndarray:
        """MUSIC null spectrum a^H Pn a [P, G] at arbitrary directions (az, el [P, G]), one Pn and f per row."""
        a = self._steer(f[:, None], az, el)
        return np.einsum("pgm,pmn,pgn->pg", a.conj(), Pn, a).real

    def _refine(self, Pn, f, az, el):
        """3-stage local zoom of the MUSIC null (11 x 7 points per stage, spans 1, 0.2, 0.04 coarse steps), then a
        parabolic fit in azimuth. Vectorised over all peaks of all bursts."""
        cfg = self.cfg
        P = len(az)
        r = np.arange(P)
        n_el = 7 if cfg.estimate_el else 1
        for frac in (1.0, 0.2, 0.04):
            da = np.linspace(-cfg.az_step_deg * frac, cfg.az_step_deg * frac, 11)
            de = np.linspace(-cfg.el_step_deg * frac, cfg.el_step_deg * frac, n_el) if n_el > 1 else np.zeros(1)
            A_ = np.broadcast_to(az[:, None, None] + da[None, None, :], (P, n_el, 11)).reshape(P, -1)
            E_ = np.broadcast_to(np.clip(el[:, None, None] + de[None, :, None], 0, cfg.el_max_deg),
                                 (P, n_el, 11)).reshape(P, -1)
            j = self._null(Pn, f, A_, E_).argmin(1)
            az, el = A_[r, j], E_[r, j]
        h = cfg.az_step_deg * 0.008
        q = self._null(Pn, f, az[:, None] + np.array([-h, 0.0, h]), np.repeat(el[:, None], 3, 1))
        den = q[:, 0] - 2 * q[:, 1] + q[:, 2]
        off = np.where(den > 0, 0.5 * (q[:, 0] - q[:, 2]) / np.where(den > 0, den, 1.0), 0.0) * h
        return (az + np.clip(off, -h, h)) % 360.0, el

    def process(self, X: np.ndarray, t0_ns: int = 0) -> list[dict]:
        """One block X [M, N] (complex) -> detections (true bearings), strongest burst first."""
        cfg, M = self.cfg, self.M
        if cfg.iq_conj:
            X = X.conj()
        Y, Pw = self._stft(np.asarray(X, np.complex64))
        bursts = self._detect(Y, Pw, t0_ns * 1e-9)
        if not bursts:
            return []
        beacon = None
        if cfg.beacon_f_hz is not None:                                          # in-band calibration first
            near = [i for i, b in enumerate(bursts) if abs(b["f_hz"] - cfg.beacon_f_hz) <= max(b["bw_hz"], 2e5)]
            if near:
                beacon = max(near, key=lambda i: bursts[i]["n"])
                self._beacon_update(bursts[beacon]["R"], bursts[beacon]["f_hz"])
        R = np.stack([b["R"] for b in bursts])
        R = R * (self.W.diagonal()[:, None] * self.W.diagonal().conj()[None, :])[None]   # remove channel gains
        if self.J is not None:
            J = self.J
            R = 0.5 * (R + R[:, J][:, :, J].conj())
        f = np.array([b["f_hz"] for b in bursts])
        n = np.array([b["n"] for b in bursts])
        B = len(bursts)
        ev, V = np.linalg.eigh(R)                                                # ascending, batched
        noise = np.array([b["noise"] for b in bursts]) * float(np.mean(np.abs(self.W.diagonal()) ** 2))
        K = mdl_order(ev, n, min(cfg.max_sources, M - 2), cfg.eig_floor_db, noise)
        nmask = np.arange(M)[None, :] < (M - K)[:, None]
        Vn = V * nmask[:, None, :]
        Pn = Vn @ Vn.conj().transpose(0, 2, 1)                                   # noise-subspace projectors [B, M, M]
        sigma2 = (ev * nmask).sum(1) / (M - K)
        # coarse grid, all bursts at once: null[b, d] = a_d^H Pn_b a_d and Bartlett power a_d^H R_b a_d
        A = np.stack([self._grid(x) for x in f])                                 # [B, D, M] complex64
        Ac = A.conj()
        null = np.einsum("bdm,bdm->bd", Ac @ Pn.astype(np.complex64), A).real
        bart = np.einsum("bdm,bdm->bd", Ac @ R.astype(np.complex64), A).real
        ne, na = len(self.el), len(self.az)
        null = null.reshape(B, ne, na)
        eli = null.argmin(1)                                                     # best elevation per azimuth
        mus = -10 * np.log10(np.clip(np.take_along_axis(null, eli[:, None, :], 1)[:, 0], 1e-12, None))
        bpow = bart.reshape(B, ne, na).max(1)
        pk = (mus >= np.roll(mus, 1, 1)) & (mus > np.roll(mus, -1, 1)) & \
             (mus >= mus.max(1, keepdims=True) - cfg.peak_window_db)
        score = np.where(pk, bpow, -np.inf)
        top = np.argsort(-score, 1)[:, : cfg.max_peaks + 1]
        bi, ki = np.nonzero(np.isfinite(np.take_along_axis(score, top, 1)))
        ai = top[bi, ki]
        az, el = self._refine(Pn[bi], f[bi], self.az[ai].astype(float), self.el[eli[bi, ai]].astype(float))
        out = []
        for b in range(B):
            sel = np.flatnonzero(bi == b)
            keep = []                                                            # merge near-duplicates (score order)
            for s in sel:
                if all(abs((az[s] - az[t] + 180) % 360 - 180) > cfg.merge_deg for t in keep):
                    keep.append(s)
            keep = keep[: min(cfg.max_peaks, int(K[b]))]                         # never more sources than MDL found
            if not keep:
                continue
            Ak = self._steer(f[b], az[keep], el[keep])                           # [k, M]
            Pi = np.linalg.pinv(Ak.T)                                            # [k, M]
            pw = np.clip(np.real(np.einsum("km,mn,kn->k", Pi, R[b] - sigma2[b] * np.eye(M), Pi.conj())), 0, None)
            if pw.max() <= 0:
                continue
            snr = pw / max(sigma2[b], 1e-30)
            crb = crb_az_deg(self.pos, f[b], az[keep], el[keep], np.maximum(snr, 1e-6), n[b])
            order = np.argsort(-pw)
            for rank, j in enumerate(order):
                if pw[j] < pw.max() * 10 ** (-cfg.peak_floor_db / 10):
                    continue
                bb = bursts[b]
                out.append({"t_s": round(bb["t_s"], 6), "f_hz": round(bb["f_hz"], 1), "bw_hz": round(bb["bw_hz"], 1),
                            "dur_s": round(bb["dur_s"], 6), "burst": b, "rank": rank,
                            "az_deg": round(float((az[keep[j]] + cfg.heading_deg) % 360), 3),
                            "el_deg": round(float(el[keep[j]]), 2),
                            "snr_db": round(10 * math.log10(max(snr[j], 1e-12)), 2),
                            "std_deg": round(float(math.hypot(crb[j], cfg.model_error_deg)), 3),
                            "order": int(K[b]), "n_snap": int(n[b]), "beacon": b == beacon})
        return out


# ----------------------------------------------------------------------------------------------------------------
# Multi-target bearing tracker
# ----------------------------------------------------------------------------------------------------------------

def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


@dataclass
class Track:
    id: int
    t: float
    x: np.ndarray                     # [az deg (true), az rate deg/s]
    P: np.ndarray
    f_hz: float
    bw_hz: float
    snr_db: float
    hits: int = 1
    t_last: float = 0.0
    t_first: float = 0.0
    confirmed: bool = False


class BearingTracker:
    """Per-node multi-target bearing tracker (global nearest neighbour). Carrier frequency is part of the gate, so
    drones on different channels never swap; a drone's reflection forms its own track, flagged multipath_suspect."""

    def __init__(self, q_deg2_s3: float = 4.0, gate: float = 11.0, f_gate_hz: float = 1e6, confirm_hits: int = 4,
                 coast_s: float = 2.0, tentative_coast_s: float = 0.3, init_rate_std: float = 5.0,
                 merge_floor_deg: float = 0.3):
        self.q, self.gate, self.f_gate = q_deg2_s3, gate, f_gate_hz
        self.confirm_hits, self.coast_s, self.tent_coast = confirm_hits, coast_s, tentative_coast_s
        self.init_rate_std, self.merge_floor = init_rate_std, merge_floor_deg
        self.tracks: list[Track] = []
        self._next = 1

    def merge(self, dets: list[dict]) -> list[dict]:
        """Fuse a block's repeated detections of one source (several bursts, same carrier and bearing) into one
        inverse-variance-weighted bearing, so one track gets one measurement per step."""
        clusters: list[list[dict]] = []
        for d in sorted((d for d in dets if not d.get("beacon")), key=lambda d: -d["snr_db"]):
            for c in clusters:
                r = c[0]
                if abs(d["f_hz"] - r["f_hz"]) <= self.f_gate + 0.5 * max(d["bw_hz"], r["bw_hz"]) and \
                        abs(wrap180(d["az_deg"] - r["az_deg"])) <= 3 * math.hypot(d["std_deg"], r["std_deg"]) + 0.5:
                    c.append(d)
                    break
            else:
                clusters.append([d])
        out = []
        for c in clusters:
            w = np.array([1 / d["std_deg"] ** 2 for d in c])
            ref = c[0]["az_deg"]
            az = (ref + float(np.dot(w, wrap180([d["az_deg"] - ref for d in c]))) / w.sum()) % 360
            out.append({**c[0], "az_deg": az, "std_deg": max(1 / math.sqrt(w.sum()), self.merge_floor),
                        "t_s": float(np.mean([d["t_s"] for d in c])), "n_merged": len(c)})
        return out

    def _predict(self, tr: Track, t: float):
        dt = max(t - tr.t, 0.0)
        if dt == 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        tr.x = F @ tr.x
        tr.x[0] %= 360.0
        tr.P = F @ tr.P @ F.T + Q
        tr.t = t

    def step(self, t: float, dets: list[dict]) -> list[dict]:
        dets = self.merge(dets)
        for tr in self.tracks:
            self._predict(tr, t)
        T, D = len(self.tracks), len(dets)
        BIG = 1e9
        cost = np.full((T, D), BIG)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(dets):
                if abs(d["f_hz"] - tr.f_hz) > self.f_gate + 0.5 * max(tr.bw_hz, d["bw_hz"]):
                    continue
                S = tr.P[0, 0] + d["std_deg"] ** 2
                m = float(wrap180(d["az_deg"] - tr.x[0])) ** 2 / S
                if m < self.gate:
                    cost[i, j] = m + math.log(S)
        used = set()
        if T and D:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= BIG:
                    continue
                tr, d = self.tracks[i], dets[j]
                S = tr.P[0, 0] + d["std_deg"] ** 2
                Kg = tr.P[:, 0] / S
                tr.x = tr.x + Kg * float(wrap180(d["az_deg"] - tr.x[0]))
                tr.x[0] %= 360.0
                tr.P = tr.P - np.outer(Kg, tr.P[0, :])
                tr.f_hz += 0.2 * (d["f_hz"] - tr.f_hz)
                tr.bw_hz = max(tr.bw_hz, d["bw_hz"])
                tr.snr_db += 0.2 * (d["snr_db"] - tr.snr_db)
                tr.hits += 1
                tr.t_last = t
                tr.confirmed |= tr.hits >= self.confirm_hits
                used.add(j)
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(Track(self._next, t, np.array([d["az_deg"], 0.0]),
                                         np.diag([d["std_deg"] ** 2, self.init_rate_std ** 2]),
                                         d["f_hz"], d["bw_hz"], d["snr_db"], t_last=t, t_first=t))
                self._next += 1
        self.tracks = [tr for tr in self.tracks
                       if t - tr.t_last <= (self.coast_s if tr.confirmed else self.tent_coast)]
        return self.report()

    def report(self) -> list[dict]:
        conf = [tr for tr in self.tracks if tr.confirmed]
        out = []
        for tr in conf:
            # same carrier as a stronger confirmed track: most likely a reflection of that emitter
            mp = any(o is not tr and abs(o.f_hz - tr.f_hz) <= 0.5 * max(o.bw_hz, tr.bw_hz) and o.snr_db > tr.snr_db
                     for o in conf)
            out.append({"id": tr.id, "az_deg": round(float(tr.x[0]), 3), "rate_dps": round(float(tr.x[1]), 3),
                        "std_deg": round(math.sqrt(tr.P[0, 0]), 3), "f_hz": round(tr.f_hz, 1),
                        "snr_db": round(tr.snr_db, 1), "hits": tr.hits, "age_s": round(tr.t - tr.t_first, 3),
                        "coasting_s": round(tr.t - tr.t_last, 3), "multipath_suspect": mp})
        return out


def triangulate(xy_nodes, az_true_deg, std_deg) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares intersection of bearings from >= 2 nodes (local east/north metres). Two passes: the
    second weights each line by 1 / (range x sigma)^2, its cross-range variance. Returns xy [2] and covariance."""
    p = np.asarray(xy_nodes, float)
    b = np.radians(np.asarray(az_true_deg, float))
    s = np.radians(np.asarray(std_deg, float))
    nrm = np.stack([np.cos(b), -np.sin(b)], 1)                                   # normal of each bearing line
    w = np.ones(len(p))
    for _ in range(2):
        Hw = nrm * w[:, None]
        Cinv = nrm.T @ Hw
        x = np.linalg.solve(Cinv, Hw.T @ np.sum(nrm * p, 1))
        rng = np.maximum(np.linalg.norm(x - p, axis=1), 1.0)
        w = 1.0 / (rng * s) ** 2
    Hw = nrm * w[:, None]
    return x, np.linalg.inv(nrm.T @ Hw)


# ----------------------------------------------------------------------------------------------------------------
# Sources: simulator and AIR7311
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class Emitter:
    az_deg: float                     # true bearing at t = 0
    el_deg: float
    f_off_hz: float                   # carrier offset from the LO
    bw_hz: float
    snr_db: float                     # in-band SNR per element at el = 0 (element pattern applied on top)
    az_rate_dps: float = 0.0
    duty: float = 1.0                 # on-fraction of each period (bursty links)
    period_s: float = 4e-3
    paths: tuple = ()                 # reflections: (rel_db, d_az_deg, el_deg, extra_delay_ns)
    name: str = ""

    def az_at(self, t: float) -> float:
        return (self.az_deg + self.az_rate_dps * t) % 360


class Simulator:
    """4-channel receiver simulation: band-limited emitters with multipath (true delays), element pattern, mutual
    coupling, per-channel gain/phase errors, a random phase between the two RFICs, element position errors,
    thermal noise, AGC and 14-bit quantisation. Noise power is 1 per channel over fs."""

    def __init__(self, cfg: DFConfig, emitters: list[Emitter], seed: int = 0, gain_sigma_db: float = 0.5,
                 phase_sigma_deg: float = 5.0, interchip_deg: float | None = None, coupling_db: float = -20.0,
                 pos_sigma_m: float = 1e-3, adc_bits: int = 14):
        self.cfg, self.em = cfg, emitters
        self.rng = np.random.default_rng(seed)
        M = len(cfg.positions_m)
        pos = np.asarray(cfg.positions_m, float)
        self.pos = pos - pos.mean(0) + self.rng.normal(0, pos_sigma_m, (M, 3)) * [1, 1, 0]
        chip = math.radians(self.rng.uniform(0, 360) if interchip_deg is None else interchip_deg)
        g = 10 ** (self.rng.normal(0, gain_sigma_db, M) / 20) * np.exp(1j * np.radians(self.rng.normal(0, phase_sigma_deg, M)))
        g[2:] *= np.exp(1j * chip)                                               # RX3/RX4 on the second AD9371
        self.g = g
        dist = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1)
        cpl = 10 ** (coupling_db / 20) * (dist.min(where=dist > 0, initial=1.0) / np.maximum(dist, 1e-9)) * \
            np.exp(1j * self.rng.uniform(0, 2 * np.pi, (M, M)))
        cpl = 0.5 * (cpl + cpl.T)                                                # reciprocal coupling
        np.fill_diagonal(cpl, 1.0)
        self.cpl = cpl
        self.bits = adc_bits
        self.t = 0.0

    def read(self, n: int) -> tuple[np.ndarray, int]:
        cfg, rng = self.cfg, self.rng
        fs = cfg.fs_hz
        f = sfft.fftfreq(n, 1 / fs)
        tt = self.t + np.arange(n) / fs
        t_mid = self.t + n / fs / 2
        X = np.zeros((len(self.pos), n), complex)
        for e in self.em:
            S = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * (np.abs(f - e.f_off_hz) < e.bw_hz / 2)
            s = sfft.ifft(S, workers=-1)
            s /= np.sqrt(np.mean(np.abs(s) ** 2))
            if e.duty < 1:
                s *= ((tt % e.period_s) / e.period_s) < e.duty
            Sg = sfft.fft(s, workers=-1)
            amp = math.sqrt(10 ** (e.snr_db / 10) * e.bw_hz / fs)                # in-band SNR -> power over fs
            for rel_db, d_az, el, dly_ns in ((0.0, 0.0, e.el_deg, 0.0),) + tuple(e.paths):
                a = steering(self.pos, cfg.fc_hz + e.f_off_hz, e.az_at(t_mid) + d_az - cfg.heading_deg, el)
                c = amp * 10 ** (rel_db / 20) * math.sqrt(element_gain(el)) * \
                    np.exp(-2j * np.pi * cfg.fc_hz * dly_ns * 1e-9)
                sig = s if dly_ns == 0 else sfft.ifft(Sg * np.exp(-2j * np.pi * f * dly_ns * 1e-9), workers=-1)
                X += (c * a)[:, None] * sig[None, :]
        X = self.cpl @ X                                                         # coupling acts on the incident field
        X += (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)) * math.sqrt(0.5)
        X *= self.g[:, None]
        t_ns = int(round(self.t * 1e9))
        self.t += n / fs
        return self._adc(X).astype(np.complex64), t_ns

    def cal_tones(self, az_true_deg, el_deg: float = 0.0, snr_db: float = 40.0, n: int = 1 << 16) -> list:
        """Raw covariances of single CW-like tones at known true bearings (a turntable / range calibration run):
        the input of ManifoldCorrection.fit. Captures with this receiver's own coupling, gains and element errors."""
        saved, t = self.em, self.t
        Rs = []
        for az in az_true_deg:
            self.em = [Emitter(float(az), el_deg, 0.25e6, 0.1e6, snr_db)]
            X = self.read(n)[0].astype(np.complex128)
            Rs.append(X @ X.conj().T / n)
        self.em, self.t = saved, t
        return Rs

    def _adc(self, X: np.ndarray) -> np.ndarray:
        """AGC to -12 dBFS RMS, then b-bit I/Q quantisation with clipping."""
        full = np.sqrt(np.mean(np.abs(X) ** 2)) * 10 ** (12 / 20) * math.sqrt(2)
        q = 2 ** (self.bits - 1)
        re = np.clip(np.round(X.real / full * q), -q, q - 1)
        im = np.clip(np.round(X.imag / full * q), -q, q - 1)
        return (re + 1j * im) * (full / q)


class AirTSource:
    """AIR7311 via SoapySDR (Deepwave AirStack driver "SoapyAIRT"). Four RX channels in one CF32 stream, started at
    a common hardware time so all channels are sample-aligned. AGC is OFF and the same manual gain is set on all
    channels: per-channel AGC would change channel gains and phases under the DF. Both AD9371s are tuned to the
    same LO (setting RX1 tunes RX2, RX3 tunes RX4). Deepwave recommends terminating the RF ports during radio
    initialisation (calibration signals may be emitted)."""

    def __init__(self, cfg: DFConfig, gain_db: float | None, channels=(0, 1, 2, 3), args: dict | None = None):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_HAS_TIME, SOAPY_SDR_RX
        self._S, self._RX, self._HAS_TIME = SoapySDR, SOAPY_SDR_RX, SOAPY_SDR_HAS_TIME
        self.ch = list(channels)
        self.sdr = SoapySDR.Device(dict(driver="SoapyAIRT", **(args or {})))
        for c in self.ch:
            self.sdr.setSampleRate(SOAPY_SDR_RX, c, cfg.fs_hz)
            self.sdr.setGainMode(SOAPY_SDR_RX, c, False)
            if gain_db is not None:
                self.sdr.setGain(SOAPY_SDR_RX, c, gain_db)
            self.sdr.setFrequency(SOAPY_SDR_RX, c, cfg.fc_hz)
        fs = self.sdr.getSampleRate(SOAPY_SDR_RX, self.ch[0])
        if abs(fs - cfg.fs_hz) > 1:
            print(f"[music_df] sample rate set to {fs} (asked {cfg.fs_hz}); using {fs}", file=sys.stderr)
            cfg.fs_hz = fs
        if gain_db is None:
            print(f"[music_df] no --gain-db: gain range {self.sdr.getGainRange(SOAPY_SDR_RX, self.ch[0])}; "
                  f"current {self.sdr.getGain(SOAPY_SDR_RX, self.ch[0])} dB", file=sys.stderr)
        self.stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, self.ch)
        t_start = self.sdr.getHardwareTime("now") + int(0.1e9)
        self.sdr.activateStream(self.stream, SOAPY_SDR_HAS_TIME, t_start)       # all channels start together

    def read(self, n: int) -> tuple[np.ndarray, int]:
        buf = np.empty((len(self.ch), n), np.complex64)
        off, t_ns = 0, 0
        while off < n:
            sr = self.sdr.readStream(self.stream, [buf[i, off:] for i in range(len(self.ch))], n - off,
                                     timeoutUs=1_000_000)
            if sr.ret == -4:                                                     # SOAPY_SDR_OVERFLOW: restart the block
                off = 0
                continue
            if sr.ret < 0:
                raise RuntimeError(f"readStream error {sr.ret}")
            if off == 0:
                t_ns = sr.timeNs if sr.flags & self._HAS_TIME else time.time_ns()
            off += sr.ret
        return buf, int(t_ns)

    def close(self):
        self.sdr.deactivateStream(self.stream)
        self.sdr.closeStream(self.stream)


# ----------------------------------------------------------------------------------------------------------------
# Runner, self-test and CLI
# ----------------------------------------------------------------------------------------------------------------

def run(source, df: DirectionFinder, tracker: BearingTracker, n_blocks: int | None, block: int, out=sys.stdout):
    t_proc, k = [], 0
    try:
        while n_blocks is None or k < n_blocks:
            X, t_ns = source.read(block)
            t0 = time.perf_counter()
            dets = df.process(X, t_ns)
            tracks = tracker.step(t_ns * 1e-9 + block / df.cfg.fs_hz, dets)
            t_proc.append(time.perf_counter() - t0)
            out.write(json.dumps({"t": round(t_ns * 1e-9, 6), "detections": dets, "tracks": tracks}) + "\n")
            k += 1
    except KeyboardInterrupt:
        pass
    finally:
        if t_proc:
            dur = block / df.cfg.fs_hz
            print(f"[music_df] {k} blocks, processing p50 {1e3 * np.median(t_proc):.1f} ms per "
                  f"{1e3 * dur:.1f} ms block ({np.median(t_proc) / dur:.2f}x real time)", file=sys.stderr)


def demo_scene(cfg: DFConfig) -> list[Emitter]:
    """Two drones (one with a strong reflection, one bursty) and a calibration beacon."""
    return [Emitter(40.0, 8.0, -7e6, 2e6, 15.0, az_rate_dps=2.0, paths=((-6.0, 75.0, 2.0, 120.0),), name="A"),
            Emitter(220.0, 3.0, 6e6, 1e6, 10.0, az_rate_dps=-1.0, duty=0.5, name="B"),
            Emitter(300.0, 0.0, 12e6, 0.2e6, 25.0, name="beacon")]


def strongest_burst_cov(cfg: DFConfig, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Raw (uncalibrated) covariance of the strongest burst in a block: the calibration measurement for a splitter
    reference or a tone at a known bearing, free of the other signals in the band."""
    df = DirectionFinder(cfg)
    bursts = df._detect(*df._stft(np.asarray(X, np.complex64)), 0.0)
    if not bursts:
        raise RuntimeError("no signal detected: check the reference / tone level and --fc")
    b = max(bursts, key=lambda x: x["n"] * 10 ** (x["snr_cell_db"] / 10))
    return b["R"], b


def selftest() -> int:
    ok = True
    lam = C0 / 2.4835e9
    for side in (0.45 * lam, 0.75 * lam):
        print(f"geometry: side {side / lam:.2f} lambda -> worst manifold correlation "
              f"{ambiguity(square_array(side), 2.4835e9):.4f} (>= 0.995 ambiguous)")
    ok &= ambiguity(square_array(0.45 * lam), 2.4835e9) < 0.995 <= ambiguity(square_array(0.75 * lam), 2.4835e9)

    # manifold correction: fit on 72 noiseless tones of a perturbed array (1 mm errors + coupling), test in between
    rng = np.random.default_rng(1)
    pos = square_array(0.054) - square_array(0.054).mean(0)
    pos_true = pos + rng.normal(0, 1e-3, pos.shape) * [1, 1, 0]
    cpl = np.eye(4) + 0.1 * np.exp(2j * np.pi * rng.random((4, 4)))

    def true(az):
        return steering(pos_true, 2.44e9, az, 0 * az) @ cpl.T

    az_fit, az_test = np.arange(0, 360, 5.0), np.arange(2.5, 360, 5.0)
    corr = ManifoldCorrection.fit([np.outer(v, v.conj()) for v in true(az_fit)], pos, 2.44e9, az_fit)
    a_fit = steering(pos, 2.44e9, az_test, 0 * az_test) * corr(az_test)
    v = true(az_test)
    mis = 1 - np.abs(np.sum(a_fit.conj() * v, 1)) / (np.linalg.norm(a_fit, axis=1) * np.linalg.norm(v, axis=1))
    print(f"manifold correction: worst held-out mismatch 1 - |cos| = {mis.max():.2e}")
    ok &= mis.max() < 1e-4

    # end-to-end on the simulator (coupling -20 dB, 1 mm element errors, 0.5 dB / 5 deg channel errors, 137 deg
    # between the RFICs): no calibration; beacon only; lab tone table, then a retune (new random inter-RFIC phase)
    # that the beacon tracks
    for label, beacon, lab in (("no calibration", None, False), ("beacon only", 2.4535e9, False),
                               ("tone table + beacon, retuned", 2.4535e9, True)):
        cfg = DFConfig(heading_deg=10.0, beacon_f_hz=beacon, beacon_az_deg=300.0, beacon_el_deg=0.0)
        em = demo_scene(cfg)
        sim = Simulator(cfg, em, seed=3, interchip_deg=137.0)
        corr = None
        if lab:
            az_cal = np.arange(0.0, 360.0, 5.0)
            corr = ManifoldCorrection.fit(sim.cal_tones(az_cal), cfg.positions_m, cfg.fc_hz, az_cal - cfg.heading_deg)
            sim.g[2:] *= np.exp(1j * np.radians(sim.rng.uniform(0, 360)))       # retune: new inter-RFIC phase
        df, trk = DirectionFinder(cfg, correction=corr), BearingTracker()
        block, errs, tp, crbs = 1 << 18, {"A": [], "B": []}, [], []
        for _ in range(40):
            X, t_ns = sim.read(block)
            t0 = time.perf_counter()
            dets = df.process(X, t_ns)
            tracks = trk.step(t_ns * 1e-9 + block / cfg.fs_hz, dets)
            tp.append(time.perf_counter() - t0)
            for e in em[:2]:
                mine = [d for d in dets if abs(d["f_hz"] - cfg.fc_hz - e.f_off_hz) < e.bw_hz and not d["beacon"]]
                if mine:
                    d = max(mine, key=lambda d: d["snr_db"])
                    errs[e.name].append(float(wrap180(d["az_deg"] - e.az_at(d["t_s"]))))
                    crbs.append(d["std_deg"])
        med = {k: float(np.median(np.abs(v))) if v else float("inf") for k, v in errs.items()}
        truth = [em[0].az_at(sim.t), em[0].az_at(sim.t) + em[0].paths[0][1], em[1].az_at(sim.t)]
        false = [t for t in tracks if min(abs(float(wrap180(t["az_deg"] - x))) for x in truth) > 3.0]
        prim = [t for t in tracks if not t["multipath_suspect"]]
        print(f"{label}: median |az error| A {med['A']:.3f} deg, B {med['B']:.3f} deg; reported std p50 "
              f"{np.median(crbs):.3f} deg; confirmed tracks {len(tracks)} ({len(prim)} primary, {len(false)} off "
              f"truth by > 3 deg); processing p50 {1e3 * np.median(tp):.1f} ms per "
              f"{1e3 * block / cfg.fs_hz:.1f} ms block")
        if lab:
            ok &= med["A"] < 0.3 and med["B"] < 0.3 and len(prim) == 2 and not false
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _cfg_from_args(a) -> DFConfig:
    return DFConfig(positions_m=square_array(a.side), heading_deg=a.heading, fc_hz=a.fc, fs_hz=a.fs, nfft=a.nfft,
                    iq_conj=a.iq_conj, estimate_el=a.estimate_el, beacon_f_hz=a.beacon_freq,
                    beacon_az_deg=a.beacon_az, beacon_el_deg=a.beacon_el)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("sim", "hw", "calibrate", "fit-manifold", "geometry"):
        p = sub.add_parser(name)
        p.add_argument("--fc", type=float, default=2.4415e9)
        p.add_argument("--fs", type=float, default=31.25e6)
        p.add_argument("--side", type=float, default=0.054, help="square side (m), <= 0.45 lambda at the top freq")
        p.add_argument("--heading", type=float, default=0.0, help="true bearing of the array +y axis (deg)")
        p.add_argument("--nfft", type=int, default=256)
        p.add_argument("--block", type=int, default=1 << 18)
        p.add_argument("--gain-db", type=float, default=None, help="manual RX gain, same on all channels")
        p.add_argument("--cal", default=None, help="channel gains .npz from `calibrate` (splitter)")
        p.add_argument("--manifold", default=None, help="manifold correction .npz from `fit-manifold`")
        p.add_argument("--tone-az", type=float, default=None, help="calibrate: true bearing of the tone source")
        p.add_argument("--tones", default="tones.npz", help="tone captures appended by `calibrate --tone-az`")
        p.add_argument("--harmonics", type=int, default=8)
        p.add_argument("--beacon-freq", type=float, default=None)
        p.add_argument("--beacon-az", type=float, default=0.0)
        p.add_argument("--beacon-el", type=float, default=0.0)
        p.add_argument("--estimate-el", action="store_true", help="also search elevation (low confidence)")
        p.add_argument("--iq-conj", action="store_true")
        p.add_argument("--seconds", type=float, default=2.0)
        p.add_argument("--element-diameter", type=float, default=None, help="measured omni diameter (m)")
        p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    cfg = _cfg_from_args(a)
    pos = square_array(a.side)
    if a.cmd == "geometry":
        lam = C0 / a.fc
        print(f"side {a.side * 1e3:.1f} mm = {a.side / lam:.3f} lambda at {a.fc / 1e6:.1f} MHz; "
              f"worst manifold correlation {ambiguity(pos, a.fc):.4f} (>= 0.995: ambiguous)")
        print(f"usable up to ~{max_unambiguous_freq(pos) / 1e6:.0f} MHz; side for 0.45 lambda here: "
              f"{0.45 * lam * 1e3:.1f} mm")
        crb = crb_az_deg(pos - pos.mean(0), a.fc, np.arange(0, 360, 5.0), 0.0, 10.0, 1000)
        print(f"CRB (10 dB per element, 1000 snapshots, el 0): {crb.mean():.3f} deg mean, {crb.max():.3f} max; "
              f"a 1 mm element error is {360 * 1e-3 / lam:.1f} deg of phase")
        if a.element_diameter:
            gap = a.side - a.element_diameter
            print(f"gap between omnis {gap * 1e3:.1f} mm = {gap / lam:.2f} lambda"
                  + ("  -> physically impossible" if gap <= 0 else
                     "  -> strong coupling: a tone table (fit-manifold) is required" if gap < lam / 4 else ""))
        print(f"element: {ANTENNA['model']} {ANTENNA['gain_dbi']} dBi, -3 dB at el +-{ANTENNA['vbw_deg'] / 2} deg "
              f"(gain at el 45: {10 * math.log10(element_gain(45.0)):.1f} dB)")
        return 0
    if a.cmd == "fit-manifold":
        z = np.load(a.tones)
        corr = ManifoldCorrection.fit(list(z["R"]), pos, float(z["fc"]), z["az_array"], n_harm=a.harmonics)
        corr.save(a.out or "manifold.npz")
        print(f"saved {a.out or 'manifold.npz'} from {len(z['az_array'])} tones, {a.harmonics} harmonics")
        return 0
    gains = None
    if a.cal:
        z = np.load(a.cal)
        gains = z["gains"]
        if abs(float(z["fc"]) - a.fc) > 1:
            print("[music_df] WARNING: calibration LO differs from --fc; inter-RFIC phase is invalid", file=sys.stderr)
        print("[music_df] note: inter-RFIC phase is re-randomised on retune and device re-open; prefer --beacon-*",
              file=sys.stderr)
    corr = ManifoldCorrection.load(a.manifold) if a.manifold else None
    if a.cmd == "sim":
        src = Simulator(cfg, demo_scene(cfg), seed=0)
        if a.beacon_freq is None:
            cfg.beacon_f_hz, cfg.beacon_az_deg = a.fc + 12e6, 300.0
        run(src, DirectionFinder(cfg, gains, corr), BearingTracker(), max(1, int(a.seconds * a.fs / a.block)), a.block)
        return 0
    src = AirTSource(cfg, a.gain_db)
    try:
        if a.cmd == "calibrate":
            R, b = strongest_burst_cov(cfg, src.read(a.block)[0])
            print(f"reference at {b['f_hz'] / 1e6:.3f} MHz, {b['snr_cell_db']:.1f} dB per cell, {b['n']} snapshots")
            if a.tone_az is None:                                                # splitter: equal-phase reference
                g = gains_from_reference(R, np.ones(len(R)))
                out = a.out or "cal.npz"
                np.savez(out, gains=g, fc=a.fc, fs=cfg.fs_hz, t=time.time())
                print(f"saved {out}: |g| dB {np.round(20 * np.log10(np.abs(g)), 2)}, "
                      f"arg deg {np.round(np.degrees(np.angle(g)), 1)}")
            else:                                                                # one tone of the table
                try:
                    z = np.load(a.tones)
                    Rs, az = list(z["R"]) + [R], list(z["az_array"]) + [(a.tone_az - a.heading) % 360]
                except FileNotFoundError:
                    Rs, az = [R], [(a.tone_az - a.heading) % 360]
                np.savez(a.tones, R=np.stack(Rs), az_array=np.array(az), fc=a.fc)
                print(f"{a.tones}: {len(az)} tones; capture all bearings in ONE session (no retune), then fit-manifold")
            return 0
        run(src, DirectionFinder(cfg, gains, corr), BearingTracker(), None, a.block)
    finally:
        src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
