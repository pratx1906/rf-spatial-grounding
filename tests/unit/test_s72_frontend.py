"""S7.2 item 3 (audit C Q18/Q23, B Q16): strong-signal front end in the IQ path (Rapp LNA, preselector, IM3,
desensitisation, blocking)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.obs_model.observe_v2 import in_window, link_gain_db
from sim.rf import frontend as FE
from sim.rf.spectrum import CLUTTER_DB, fspl_db

REPO = Path(__file__).resolve().parents[2]
FE_CFG = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())["front_end"]
FS, N = 125e6, 1 << 14


def _tone(p_dbm, k):
    a = math.sqrt(10 ** ((p_dbm - 30) / 10))
    return a * torch.exp(2j * math.pi * k * torch.arange(N, dtype=torch.float64) / N).to(torch.complex64)


def _bin_dbm(y, k):
    return 20 * math.log10(abs(complex(torch.fft.fft(y.to(torch.complex128))[k % N])) / N) + 30


def test_single_tone_compresses_1_db_at_p1db():
    p1 = FE_CFG["p1db_in_dbm"]
    y = FE.rapp(_tone(p1, 100), p1)
    assert _bin_dbm(y, 100) - p1 == pytest.approx(-1.0, abs=1e-3)
    y = FE.rapp(_tone(p1 - 40, 100), p1)
    assert _bin_dbm(y, 100) - (p1 - 40) == pytest.approx(0.0, abs=1e-3)   # linear at low level


@pytest.mark.parametrize("backoff_db", [20.0, 15.0])
def test_two_tone_im3_matches_theory(backoff_db):
    """Two equal tones of power P each at bins k1, k2: IM3 at 2 k1 - k2 sits at 20 log10(P / (2 Ps)) dBc
    (Rapp p = 1 is the cubic x (1 - |x|^2 / (2 Ps)) at low level); IM3 rises 3 dB per dB of input."""
    p1 = FE_CFG["p1db_in_dbm"]
    p = p1 - backoff_db
    k1, k2 = 1000, 1100
    y = FE.rapp(_tone(p, k1) + _tone(p, k2), p1)
    im3_dbc = _bin_dbm(y, 2 * k1 - k2) - _bin_dbm(y, k1)
    ps = FE.psat_w(p1)
    theory = 20 * math.log10(10 ** ((p - 30) / 10) / (2 * ps))
    assert im3_dbc == pytest.approx(theory, abs=0.5)
    assert _bin_dbm(y, 2 * k2 - k1) - _bin_dbm(y, k2) == pytest.approx(theory, abs=0.5)
    assert _bin_dbm(FE.rapp(_tone(p, k1) + _tone(p, k2), p1), 3 * k1 - 2 * k2) < _bin_dbm(y, 2 * k1 - k2) - 15  # IM5 lower


def test_in_window_desensitisation_against_blocker_power():
    """A weak tone 50 dB under a strong in-window blocker: its gain follows (1 + u/2) / (1 + u)^1.5, u = P_b / Ps."""
    p1 = FE_CFG["p1db_in_dbm"]
    for pb in (p1 - 20, p1 - 10, p1 - 5, p1, p1 + 5, p1 + 10):
        y = FE.rapp(_tone(pb, 3000) + _tone(pb - 50, 500), p1)
        g_meas = _bin_dbm(y, 500) - (pb - 50)
        g_theory = 20 * math.log10(FE.blocker_gain(10 ** ((pb - 30) / 10), p1))
        assert g_meas == pytest.approx(g_theory, abs=0.2), pb
    assert 20 * math.log10(FE.blocker_gain(10 ** ((p1 + 10 - 30) / 10), p1)) < -9       # 9.4 dB at P1dB + 10 dB


def test_out_of_window_blocker_desensitises_and_blocks_through_the_preselector():
    rej, p_lna = FE_CFG["out_of_window_rejection_db"], FE_CFG["p1db_lna_dbm"]
    weak = _tone(-80.0, 500)[None].repeat(4, 1)
    prev = 1.0
    for pb in (-30.0, 0.0, 20.0, 30.0, 35.0, 40.0):                          # received before the preselector
        y, info = FE.front_end(weak, FE_CFG, [("cctv", pb)])
        g = _bin_dbm(y[0], 500) - (-80.0)
        assert g == pytest.approx(20 * math.log10(FE.blocker_gain(10 ** ((pb - rej - 30) / 10), p_lna)), abs=0.05)
        assert g <= prev + 1e-9
        prev = g
        assert info["blocked"] == (pb - rej > p_lna)
        assert info["blocker_dbm"] == pytest.approx(pb - rej)
    assert prev < -3.0
    _, info = FE.front_end(_tone(FE_CFG["p1db_in_dbm"] + 3, 500)[None].repeat(4, 1), FE_CFG, [])
    assert info["blocked"] and info["reasons"][0].startswith("in_window_compression")


def test_transmitters_beyond_60_mhz_are_blockers_not_dropped():
    ibw = 100e6
    assert in_window(2441.75e6 + 40e6, 2441.75e6, "ism_2400", ibw)
    assert not in_window(2441.75e6 + 80e6, 2441.75e6, "ism_2400", ibw)      # -> blocker list, through preselector
    assert not in_window(5775e6, 2441.75e6, "ism_2400", ibw)
    # link gain without a cache cell: free space (+ clutter beyond 30 m), as SpectrumWorld does
    assert link_gain_db(None, (0.0, 20.0, 10.0), (0.0, 0.0, 10.0), 2.44e9) == pytest.approx(-fspl_db(20.0, 2440.0))
    assert link_gain_db(None, (0.0, 500.0, 10.0), (0.0, 0.0, 10.0), 2.44e9) == \
        pytest.approx(-fspl_db(500.0, 2440.0) - CLUTTER_DB)
