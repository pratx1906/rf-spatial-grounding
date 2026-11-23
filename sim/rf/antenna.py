"""Antenna patterns, drone attitude and mutual coupling (RF phase 2, Task 8; world side).

Node element (AIR7311 array, 4 omnis on a horizontal square; real element data pending [HUMAN]):
  quarter-wave monopole over a finite ground plane, approximated by the half-wave dipole elevation pattern
  F(el) = cos(pi/2 sin el) / cos el, times a ground-plane factor that rolls off below the horizon
  (1 + tanh(el / EL_ROLL_DEG)) / 2 (estimated). Omnidirectional in azimuth, so per-path gains change but array
  phases do not: arrivals from below the horizon (sea and ground bounces) are attenuated.
Drone antenna: a half-wave dipole along the airframe's vertical axis, tilted by bank (coordinated turn,
  tan(bank) = v * turn_rate / g) and pitch (tan(pitch) = -a_forward / g, plus the climb angle). Toward the path's
  departure direction u the radiated field is the dipole axis projected on the plane normal to u: amplitude
  |a x u|, split into V (theta) and H (phi) components. With dual-pol caches the path coefficient is c_V h_VV + c_H h_VH (reciprocity).
Mutual coupling: calibration removes the nominal coupling matrix; a residual coupling of RESIDUAL_DB (random
  phase) stays. observe_v2 applies it per node and scenario (y <- C y at the antenna ports) in addition to the
  per-channel gain/phase residuals (S7.2).
"""

from __future__ import annotations

import math

import numpy as np

EL_ROLL_DEG = 8.0
RESIDUAL_DB = -30.0
G = 9.81


def node_element_gain(el_deg) -> np.ndarray:
    """Field gain (linear amplitude, 1 at the horizon) of the node element versus arrival elevation."""
    el = np.radians(np.asarray(el_deg, float))
    c = np.cos(el)
    f = np.where(np.abs(c) > 1e-6, np.cos(np.pi / 2 * np.sin(el)) / np.where(np.abs(c) > 1e-6, c, 1.0), 0.0)
    return np.abs(f) * (1 + np.tanh(np.degrees(el) / EL_ROLL_DEG)) / 2 * 2      # x2: 1 at the horizon


def attitude(pts_xyz: np.ndarray, t_s: np.ndarray, climb: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Bank and pitch (rad) along a trajectory from finite differences.

    bank: coordinated turn, tan(bank) = v_h * turn_rate / g (right turn = right bank).
    pitch: nose up = positive; -atan(a_forward / g) (accelerating tilts the nose down) plus, with climb, the
    flight-path angle atan(v_z / v_h) (S7.2: the airframe follows the climb; an assumption for multirotors, which
    climb nearly level [HUMAN])."""
    v = np.gradient(pts_xyz, t_s, axis=0)
    a = np.gradient(v, t_s, axis=0)
    sp = np.linalg.norm(v[:, :2], axis=1)
    heading = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    turn = np.gradient(heading, t_s)
    bank = np.arctan(sp * turn / G)
    fwd = np.where(sp[:, None] > 0.1, v[:, :2] / np.maximum(sp[:, None], 1e-9), 0)
    pitch = -np.arctan((a[:, :2] * fwd).sum(1) / G)
    if climb:
        pitch = pitch + np.where(sp > 0.5, np.arctan2(v[:, 2], np.maximum(sp, 1e-9)), 0.0)
    return bank, pitch


def dipole_axis(heading_rad: float, bank_rad: float, pitch_rad: float) -> np.ndarray:
    """Airframe vertical axis in ENU after pitch (about the right wing) and bank (about the nose)."""
    f = np.array([math.sin(heading_rad), math.cos(heading_rad), 0.0])            # nose
    r = np.array([math.cos(heading_rad), -math.sin(heading_rad), 0.0])           # right wing
    up = np.array([0.0, 0.0, 1.0])
    up = up * math.cos(pitch_rad) - f * math.sin(pitch_rad)                       # nose up tilts the axis back
    up = up * math.cos(bank_rad) + r * math.sin(bank_rad)                         # right bank tilts it right
    return up / np.linalg.norm(up)


def drone_pol_coeffs(axis: np.ndarray, az_dep_deg, el_dep_deg) -> tuple[np.ndarray, np.ndarray]:
    """(c_V, c_H) of a dipole along `axis` radiating toward departure directions (compass az, elevation)."""
    az, el = np.radians(np.asarray(az_dep_deg, float)), np.radians(np.asarray(el_dep_deg, float))
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)
    theta_hat = np.stack([np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), -np.cos(el)], -1)     # down = +theta
    phi_hat = np.stack([np.cos(az), -np.sin(az), np.zeros_like(az)], -1)
    e = axis[None, :] - (u @ axis)[..., None] * u                                 # dipole field direction
    c_v = -(e * theta_hat).sum(-1)                                                # V = up-pointing component
    c_h = (e * phi_hat).sum(-1)
    return c_v, c_h


def residual_coupling(rng: np.random.Generator, m: int = 4) -> np.ndarray:
    """Identity plus a residual coupling of RESIDUAL_DB with random phases between all element pairs."""
    c = np.eye(m, dtype=complex)
    a = 10 ** (RESIDUAL_DB / 20)
    for i in range(m):
        for j in range(m):
            if i != j:
                c[i, j] = a * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return c
