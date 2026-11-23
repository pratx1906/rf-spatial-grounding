"""Antenna patterns and drone attitude (RF phase 2, Task 8)."""

import math

import numpy as np
import pytest

from sim.rf import antenna as A


def test_node_element_is_unity_at_the_horizon_and_rolls_off_below_it():
    g = A.node_element_gain([0.0, 30.0, -10.0, -30.0, 89.0])
    assert g[0] == pytest.approx(1.0, abs=1e-6)
    assert g[2] < 0.25 and g[3] < g[2]                    # sea/ground bounces arrive from below and are cut
    assert g[4] < 0.05                                   # monopole null at zenith


def test_level_dipole_is_pure_v_and_a_banked_one_leaks_h():
    axis = A.dipole_axis(0.0, 0.0, 0.0)
    cv, ch = A.drone_pol_coeffs(axis, [90.0], [0.0])
    assert cv[0] == pytest.approx(1.0, abs=1e-9) and abs(ch[0]) < 1e-9
    axis = A.dipole_axis(0.0, math.radians(45), 0.0)    # heading north, 45 deg right bank
    cv, ch = A.drone_pol_coeffs(axis, [90.0], [0.0])     # toward the east (along the wing)
    assert math.hypot(cv[0], ch[0]) == pytest.approx(math.cos(math.radians(45)), abs=1e-6)   # dipole factor
    cv2, ch2 = A.drone_pol_coeffs(axis, [0.0], [0.0])    # toward the north (nose): pure roll -> V and H mix
    assert abs(ch2[0]) == pytest.approx(math.sin(math.radians(45)), abs=1e-6)


def test_attitude_of_a_coordinated_turn():
    t = np.linspace(0, 20, 201)
    r, w = 100.0, 0.15                                   # 15 m/s, 0.15 rad/s
    pts = np.stack([r * np.sin(w * t), r * np.cos(w * t), np.full_like(t, 40.0)], 1)
    bank, pitch = A.attitude(pts, t)
    expect = math.atan(r * w * w / A.G)                  # tan(bank) = v * turn rate / g
    assert np.median(np.abs(bank[5:-5])) == pytest.approx(expect, rel=0.02)
    assert np.max(np.abs(pitch[5:-5])) < 0.01


def test_residual_coupling_level():
    c = A.residual_coupling(np.random.default_rng(0))
    off = np.abs(c[~np.eye(4, dtype=bool)])
    assert np.allclose(20 * np.log10(off), A.RESIDUAL_DB) and np.allclose(np.diag(c), 1)
