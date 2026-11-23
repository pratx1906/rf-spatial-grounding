"""CRB helpers for the 4-element square AIR7311 array (runs/s81/crb; CPU only, offline analysis).

Deterministic (conditional) CRB, single source, M = 4 elements, N snapshots, per-element SNR rho
(signal power / noise power per element, per snapshot), unknown (az, el) (Stoica & Nehorai 1989):
    FIM = 2 N rho Re(D^H P_a_perp D),  D = [da/daz, da/del],  P_a_perp = I - a a^H / M
    CRB_det(az) = [FIM^-1]_00
Stochastic (unconditional) CRB, single source: CRB_sto = CRB_det * (1 + 1 / (M rho)).
Steering convention as sim.df.air7311.scan_steering: a_m = exp(j 2 pi f / c u(az, el) . r_m),
u = (sin az cos el, cos az cos el, sin el), positions sim.df.air7311.square_positions(side).
"""

from __future__ import annotations

import math

import numpy as np

C = 299792458.0
SIDES = {"sub_ghz_868_915": 0.145, "ism_2400": 0.054, "ism_5800": 0.023}   # configs/hardware/air7311.yaml track_b_custom
BAND_TOP_HZ = {"sub_ghz_868_915": 928e6, "ism_2400": 2483.5e6, "ism_5800": 5875e6}


def band_of(f_mhz: float) -> str:   # same rule as runs/s77/d_q32_v3.py
    return "sub_ghz_868_915" if f_mhz < 1500 else ("ism_2400" if f_mhz < 3000 else "ism_5800")


def square_positions(side_m: float) -> np.ndarray:
    h = side_m / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def steer(az_rad, el_rad, pos, f_hz):
    u = np.array([math.sin(az_rad) * math.cos(el_rad), math.cos(az_rad) * math.cos(el_rad), math.sin(el_rad)])
    return np.exp(1j * 2 * math.pi * f_hz / C * (pos @ u))


def crb_az_deg(az_deg, el_deg, side_m, f_hz, rho, n_snap, stochastic=False, el_known=False):
    """sqrt CRB on azimuth (deg). Vectorised over numpy arrays of equal shape (or scalars)."""
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    side, f = np.broadcast_to(side_m, az.shape), np.broadcast_to(f_hz, az.shape)
    rho, n = np.broadcast_to(np.asarray(rho, float), az.shape), np.broadcast_to(np.asarray(n_snap, float), az.shape)
    out = np.empty(az.shape)
    for i in np.ndindex(az.shape):
        pos = square_positions(float(side[i]))
        k = 2 * math.pi * f[i] / C
        a, e = az[i], el[i]
        u = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])
        du_a = np.array([math.cos(a) * math.cos(e), -math.sin(a) * math.cos(e), 0.0])
        du_e = np.array([-math.sin(a) * math.sin(e), -math.cos(a) * math.sin(e), math.cos(e)])
        av = np.exp(1j * k * (pos @ u))
        D = np.stack([1j * k * (pos @ du_a) * av, 1j * k * (pos @ du_e) * av], 1)     # [4, 2]
        P = np.eye(4) - np.outer(av, av.conj()) / 4
        J = 2 * n[i] * rho[i] * np.real(D.conj().T @ P @ D)
        det = J[0, 0] * J[1, 1] - J[0, 1] ** 2
        # planar array: el is unidentifiable at el = 0 (J11 = 0); the centred square array makes J01 = 0 anyway
        # (checked in mc_check.py), so az is decoupled and 1 / J00 is the az bound
        if el_known or J[1, 1] <= 1e-12 * J[0, 0] or det <= 0:
            v = 1.0 / J[0, 0]
        else:
            v = J[1, 1] / det
        if stochastic:
            v *= 1 + 1 / (4 * rho[i])
        out[i] = math.degrees(math.sqrt(v))
    return out if out.shape else float(out)
