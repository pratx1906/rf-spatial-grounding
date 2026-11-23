"""Coupling-aware DF calibration: the BELIEF-side array manifold (RF convergence follow-up to Task 2c, 2026-10-27).

A real calibrated DF system does not steer with the textbook manifold a(az, el). It measures the array once (a
factory or site calibration: CW tones from surveyed bearings) and steers with a calibrated manifold. Here the
calibrated manifold is  a_cal(az, el) = C_hat a(az, el), renormalised to |a| = 2, where C_hat is the 4 x 4 mutual
coupling matrix fitted to the calibration measurements by least squares.

Truth separation: this module is belief side. It sees only what a calibration procedure would record: the measured
tone vectors Y [4, K] and the surveyed tone bearings. The tone vectors are produced on the world side
(sim.obs_model.observe_v2.coupling_cal_tones) from the node's true coupling plus the procedure's error terms; the
true coupling never reaches this module. The fitted matrix is normalised to a unit diagonal: per-channel gains and
phases belong to the receiver's internal (injection) calibration, whose residual stays modelled separately
(observe_v2.cal_residuals), so the antenna calibration does not double-count it.

Switch: `df_calibration.coupling_cal` in configs/hardware/air7311.yaml (default true for new builds); the
environment variable SCENESCOUT_DF_COUPLING_CAL=0|1 overrides it. Off reproduces the nominal-steering builds
(demo v2, v21 and the v3 sets built before 2026-10-27). Residual model and its parameters: estimated, [HUMAN] to
confirm against real AIR7311 calibration data (docs/RESULTS/df_coupling_calibration.md).
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

C0 = 299792458.0
ENV_SWITCH = "SCENESCOUT_DF_COUPLING_CAL"

# calibration procedure (estimated; [HUMAN] to confirm against the real AIR7311 / Aksha360 calibration):
DEFAULTS = {
    "n_tones": 72,                # tones at 5 deg steps around the horizon (turntable / positioner)
    "tone_el_deg": 0.0,           # tone source at the array's horizon
    "tone_snr_db": 40.0,          # per-element SNR of one tone snapshot
    "tone_snapshots": 1024,       # snapshots averaged per tone
    "bearing_sigma_deg": 0.1,     # positioner / survey error of each tone bearing (1 sigma)
    "multipath_rel_db": -35.0,    # range reflection per tone, relative to the direct tone (random direction, phase)
    "drift_rel_db": -40.0,        # change of the coupling between calibration and use (temperature, mast, ageing)
}


def cal_config(hw: dict) -> dict:
    """The calibration procedure parameters: hw['df_calibration'] over DEFAULTS (the switch itself not included)."""
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (hw.get("df_calibration") or {}).items() if k in DEFAULTS})
    return c


def coupling_cal_enabled(hw: dict) -> bool:
    """Coupling-aware steering on? SCENESCOUT_DF_COUPLING_CAL (0/1) overrides hw['df_calibration']['coupling_cal'];
    a config without the section (older builds) means off."""
    env = os.environ.get(ENV_SWITCH)
    if env not in (None, ""):
        return env.strip().lower() not in ("0", "false", "off", "no")
    return bool((hw.get("df_calibration") or {}).get("coupling_cal", False))


def nominal_steering(az_deg, el_deg, pos: np.ndarray, f_hz: float) -> np.ndarray:
    """[K, 4] textbook plane-wave manifold (same convention as DFEngine.scan_steering and sim.rf.channel.steering)."""
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)
    return np.exp(1j * 2 * math.pi * f_hz / C0 * (u @ np.asarray(pos, float).T))


def estimate_coupling(Y, az_deg, el_deg, pos, f_hz) -> np.ndarray:
    """Least-squares coupling fit from calibration measurements.

    Y [4, K]: measured array vectors of K tones; az_deg, el_deg [K] (or scalar el): the SURVEYED tone directions;
    pos [4, 3]: element positions; f_hz: tone frequency. Model Y = M A + E with A [4, K] the nominal manifold at the
    surveyed directions; M_hat = Y A^H (A A^H)^-1, then normalised to a unit diagonal (row i divided by M_ii), which
    removes the cal-time channel gains/phases (the receiver's injection calibration handles those)."""
    Y = np.asarray(Y, complex)
    az = np.atleast_1d(np.asarray(az_deg, float))
    el = np.broadcast_to(np.asarray(el_deg, float), az.shape)
    A = nominal_steering(az, el, pos, f_hz).T                              # [4, K]
    M = Y @ A.conj().T @ np.linalg.inv(A @ A.conj().T)
    return M / np.diag(M)[:, None]


def calibrated_steering(A: torch.Tensor, cal: torch.Tensor | None) -> torch.Tensor:
    """a_cal = C_hat a for every steering vector of A [..., 4], renormalised to the nominal norm (|a|^2 = 4), so
    Bartlett powers and MUSIC projections keep their scale. cal None -> A unchanged."""
    if cal is None:
        return A
    cal = cal.to(device=A.device, dtype=A.dtype)
    Ac = A @ cal.T
    scale = A.abs().pow(2).sum(-1, keepdim=True).sqrt() / Ac.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return Ac * scale
