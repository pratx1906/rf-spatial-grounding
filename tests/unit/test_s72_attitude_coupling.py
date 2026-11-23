"""S7.2 items 4 and 5 (audit A Q9, C Q21): drone attitude in the polarisation; residual mutual coupling applied."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sim.obs_model.observe_v2 import coupling, drone_axis
from sim.rf import antenna as A


def _track(fn, t0=0.0, t1=60.0, dt=2.0):
    return SimpleNamespace(points=[(float(t), *fn(t)) for t in np.arange(t0, t1 + 1e-9, dt)])


def test_level_straight_drone_is_pure_v():
    e = _track(lambda t: (0.0, 15.0 * t, 60.0))                         # north at 15 m/s, level
    axis = drone_axis(e, 30.0)
    assert axis == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    cv, ch = A.drone_pol_coeffs(axis, [0.0, 90.0, 200.0], [0.0, -10.0, -30.0])
    assert np.all(np.abs(ch) < 1e-9)


def test_banked_drone_leaks_h_at_the_expected_level():
    r, w = 100.0, 0.15                                                   # 15 m/s right turn, 0.15 rad/s
    e = _track(lambda t: (r * math.sin(w * t), r * math.cos(w * t), 60.0), dt=2.0)
    t = 30.0
    axis = drone_axis(e, t)
    bank = math.atan(r * w * w / A.G)                                     # coordinated turn: tan(bank) = v w / g
    # heading at t is clockwise from north (w t); toward the nose (horizontal) the field is the full axis:
    # V = cos(bank), H = sin(bank)  ->  H/V = tan(bank) (here 13.0 deg bank: -12.6 dB cross-pol)
    hd = math.degrees(w * t + math.pi / 2)
    cv, ch = A.drone_pol_coeffs(axis, [hd], [0.0])
    assert abs(ch[0]) / abs(cv[0]) == pytest.approx(math.tan(bank), rel=0.05)
    assert 20 * math.log10(abs(ch[0]) / abs(cv[0])) == pytest.approx(20 * math.log10(math.tan(bank)), abs=0.5)


def test_climbing_drone_pitches_nose_up():
    gam = math.radians(10.0)
    e = _track(lambda t: (0.0, 15.0 * math.cos(gam) * t, 40.0 + 15.0 * math.sin(gam) * t))
    axis = drone_axis(e, 30.0)
    # nose up by the climb angle tilts the vertical axis back (toward -north)
    assert axis == pytest.approx([0.0, -math.sin(gam), math.cos(gam)], abs=1e-6)


def test_residual_coupling_is_minus_30_db_and_seeded_per_scenario_and_node():
    c = coupling(3, "S1")
    off = np.abs(c[~np.eye(4, dtype=bool)])
    assert np.allclose(20 * np.log10(off), -30.0) and np.allclose(np.diag(c), 1.0)
    x = np.zeros(4, complex)
    x[0] = 1.0                                                           # one element excited
    y = c @ x
    assert 20 * np.log10(np.abs(y[1:])) == pytest.approx([-30.0] * 3, abs=1e-6)
    assert np.allclose(c, coupling(3, "S1"))                             # fixed for the deployment
    assert not np.allclose(c, coupling(3, "S2")) and not np.allclose(c, coupling(4, "S1"))
