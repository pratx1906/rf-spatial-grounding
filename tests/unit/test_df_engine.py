"""Known-answer tests for the AIR7311 DF engine (CPU)."""

import math

import numpy as np
import pytest
import torch
import yaml

from sim.df.air7311 import DFEngine, square_positions

HW = yaml.safe_load(open("configs/hardware/air7311.yaml"))
F24 = 2.44e9


def _run(paths, side, f=F24, snr_db=30.0, interchip=0.0, seed=0):
    """paths: list of (az, el, rel_amp_db, phase_deg, delay_ns). Returns (music_peaks_az, bartlett_top_az)."""
    eng = DFEngine(HW, device="cpu")
    K = len(paths)
    h = np.array([[10 ** (p[2] / 20) * np.exp(1j * np.radians(p[3])) for p in paths]])
    az = np.array([[p[0] for p in paths]], "f4")
    el = np.array([[p[1] for p in paths]], "f4")
    tau = np.array([[p[4] for p in paths]], "f4")
    valid = np.ones((1, K), bool)
    noise = 10 ** (-snr_db / 10)
    rng = torch.Generator().manual_seed(seed)
    z4 = torch.zeros(4)
    X = eng.snapshots(h, tau, az, el, valid, square_positions(side), f, 20e6, 1.0, noise, rng, interchip, z4, z4)
    R = eng.covariance(X)
    A = eng.scan_steering(square_positions(side), f)
    bart, music = eng.spectra(R, A, X.shape[-1])
    pk, _ = eng.rank_by_power(music, bart, 10.0)
    bk, _ = eng.peaks(bart, 10.0)
    return pk[0], bk[0, 0]


def _err(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("az", [0.0, 37.5, 90.0, 181.0, 271.5, 359.0])
def test_single_direct_path_is_found_everywhere(az):
    pk, bk = _run([(az, 10.0, 0.0, 0.0, 0.0)], 0.061)
    assert _err(pk[0], az) <= 1.0 and _err(bk, az) <= 2.0


def test_strong_reflection_appears_as_a_second_peak():
    pk, _ = _run([(60.0, 5.0, 0.0, 0.0, 0.0), (200.0, 5.0, -3.0, 70.0, 180.0)], 0.061)
    found = [p for p in pk if not math.isnan(p)]
    assert any(_err(p, 60.0) <= 3 for p in found) and any(_err(p, 200.0) <= 3 for p in found)


def test_oversized_array_is_ambiguous_at_5g8():
    # 17 cm square at 5.8 GHz: spacing ~3.3 lambda -> grating lobes; small array is clean
    errs_big, errs_small = [], []
    for az in np.arange(5, 360, 23.0):
        pk_b, _ = _run([(az, 10.0, 0.0, 0.0, 0.0)], 0.17, f=5.8e9, snr_db=20)
        pk_s, _ = _run([(az, 10.0, 0.0, 0.0, 0.0)], 0.023, f=5.8e9, snr_db=20)
        errs_big.append(_err(pk_b[0], az))
        errs_small.append(_err(pk_s[0], az))
    assert max(errs_small) <= 2.0
    assert sum(e > 10 for e in errs_big) >= 1           # at least one grating-lobe confusion


def test_weak_reflection_does_not_win_over_strong_direct_path():
    # the case found on the Kandla data: a -10 dB reflection used to give the tallest MUSIC peak
    for az in (21.8, 60.3, 34.2, 140.0, 300.0):
        pk, _ = _run([(az, 3.0, 0.0, 0.0, 0.0), ((az + 150) % 360, 3.0, -10.0, 40.0, 300.0)], 0.061, snr_db=18)
        assert _err(pk[0], az) <= 3.0


def test_interchip_phase_offset_biases_bearing():
    e0 = [_err(_run([(az, 10.0, 0.0, 0.0, 0.0)], 0.061, interchip=0.0)[0][0], az) for az in (20, 110, 250)]
    e10 = [_err(_run([(az, 10.0, 0.0, 0.0, 0.0)], 0.061, interchip=10.0)[0][0], az) for az in (20, 110, 250)]
    assert np.mean(e10) > np.mean(e0)


def test_music_model_order_keeps_two_noise_dimensions():
    """4 channels: MUSIC needs >= 2 noise eigenvectors to survive calibration errors (measured 2026-10-18: order 3
    put the strongest peak on the direct path only 55% of the time for a LOS + reflection cell, order 2 100%)."""
    import torch

    from sim.df.air7311 import DFEngine

    ev = torch.tensor([[1e-4, 1e-1, 1.0, 10.0]])          # three strong eigenvalues, one noise
    assert int(DFEngine.mdl_order(ev, 256)[0]) <= 2
