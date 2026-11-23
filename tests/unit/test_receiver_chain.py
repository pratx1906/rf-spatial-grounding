"""Receiver chain + burst detection + per-burst DF end to end (RF phase 2, Task 6)."""

import math

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from sim.df.air7311 import DFEngine, square_positions
from sim.obs_model.observe_air7311 import select_peaks
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import receiver as rx
from sim.rf import waveforms as W

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
FS, DWELL = 125e6, 4e-3


def _dwell(az, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    w = W.ofdm_ocusync_like(10.0, 3e-3, rng)
    n = int(FS * DWELL)
    s = ch.resample_shift(w.iq, w.fs, FS, 12e6, n, 0.3e-3)
    pos = square_positions(HW["antenna"]["candidate_sides_m"]["B_2g4"])
    p_n = 10 ** ((-174 + 10 * math.log10(9e6) + 3 - 30) / 10)                     # noise in the signal band
    amp = math.sqrt(p_n * 10 ** (snr_db / 10))
    y = ch.propagate(s * amp, FS, np.array([1.0 + 0j]), np.array([3e-6]), np.array([az]), np.array([0.0]), pos,
                     2.45e9, 2.44e9)
    g = torch.Generator().manual_seed(seed)
    x, info = rx.receive(y, FS, HW, g, interchip_deg=0.0)
    return x, info, pos, w


def test_bursts_are_found_with_their_band_and_timing():
    x, info, _, w = _dwell(60.0, 20.0)
    bs = B.detect(x, FS)
    assert bs and info["clipped_frac"] < 1e-3
    b = max(bs, key=lambda q: q.dur)
    assert b.fc == pytest.approx(12e6, abs=0.5e6) and b.bw == pytest.approx(9e6, rel=0.15)
    assert b.t0 == pytest.approx(0.3e-3 + w.bursts[0][0], abs=20e-6)


@pytest.mark.parametrize("az", [60.0, 200.0])
def test_per_burst_df_recovers_the_bearing(az):
    x, _, pos, _ = _dwell(az, 15.0, seed=1)
    b = max(B.detect(x, FS), key=lambda q: q.dur)
    R = B.burst_cov(x, FS, b)
    eng = DFEngine(HW, "cpu")
    A = eng.scan_steering(pos, 2.45e9 + 0.0)
    bart, music = eng.spectra(R[None].to(torch.complex64), A, 256)
    pk, _ = select_peaks(eng.az, music.numpy(), bart.numpy(), 10.0, 1)
    assert abs((pk[0, 0] - az + 180) % 360 - 180) < 2.0


def test_hop_fusion_is_a_weighted_circular_mean():
    assert B.fuse_bearings([359.0, 1.0], [10.0, 10.0]) == pytest.approx(0.0, abs=1e-6) or \
        B.fuse_bearings([359.0, 1.0], [10.0, 10.0]) == pytest.approx(360.0, abs=1e-6)
    assert B.fuse_bearings([10.0, 50.0], [20.0, 0.0]) < 11.0
