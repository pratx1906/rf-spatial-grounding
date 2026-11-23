"""Coupling-aware DF calibration (RF convergence follow-up to Task 2c, 2026-10-27).

WORLD side (sim.obs_model.observe_v2): the node's true mutual coupling C stays in the IQ; a simulated calibration
procedure (tones at surveyed bearings, with estimation noise, cal-site multipath, positioner error and post-cal
drift) produces measurements. BELIEF side (sim.df.calibration): the estimator sees only those measurements and the
nominal manifold, fits C_hat, and steers with a_cal = C_hat a.
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.df import calibration as cal
from sim.df.air7311 import DFEngine, square_positions
from sim.obs_model import observe_v2 as ov
from sim.rf import bursts as B
from sim.rf.channel import steering

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
HW = {**HW, "processing": {**HW["processing"], "model_error_floor_db": -20.0}}
POS = square_positions(0.054)
F0 = 2.44175e9


def _offdiag_db(m):
    off = m[~np.eye(4, dtype=bool)]
    return 10 * math.log10(float(np.mean(np.abs(off) ** 2)))


# ---------------------------------------------------------------------------------------------------------- config
def test_config_default_on_and_env_switch(monkeypatch):
    monkeypatch.delenv(cal.ENV_SWITCH, raising=False)
    assert HW["df_calibration"]["coupling_cal"] is True                 # default ON for new builds
    assert cal.coupling_cal_enabled(HW)
    off = {**HW, "df_calibration": {**HW["df_calibration"], "coupling_cal": False}}
    assert not cal.coupling_cal_enabled(off)
    assert not cal.coupling_cal_enabled({k: v for k, v in HW.items() if k != "df_calibration"})   # old configs
    monkeypatch.setenv(cal.ENV_SWITCH, "0")                             # demo v2 / pre-cal reproduction
    assert not cal.coupling_cal_enabled(HW)
    monkeypatch.setenv(cal.ENV_SWITCH, "1")
    assert cal.coupling_cal_enabled(off)


def test_cal_config_fills_defaults():
    c = cal.cal_config({})
    for k in ("n_tones", "tone_el_deg", "tone_snr_db", "tone_snapshots", "bearing_sigma_deg", "multipath_rel_db",
              "drift_rel_db"):
        assert k in c
    assert cal.cal_config({"df_calibration": {"n_tones": 8}})["n_tones"] == 8


# ---------------------------------------------------------------------------------------------------- estimator
def test_noise_free_calibration_recovers_the_coupling_exactly():
    c_true = ov.coupling(5, "S2")
    ideal = {"n_tones": 24, "tone_el_deg": 0.0, "tone_snr_db": 300.0, "tone_snapshots": 1, "bearing_sigma_deg": 0.0,
             "multipath_rel_db": -300.0, "drift_rel_db": -300.0}
    Y, az, el = ov.coupling_cal_tones(c_true, POS, F0, ideal, np.random.default_rng(0), HW)
    c_hat = cal.estimate_coupling(Y, az, el, POS, F0)
    assert np.allclose(np.diag(c_hat), 1.0)                             # channel gains/phases normalised out
    assert np.allclose(c_hat, c_true, atol=1e-6)


def test_residual_model_level_and_trends():
    """Default procedure: residual coupling well below the -30 dB true coupling; better with a cleaner range, worse
    with fewer tones; never zero (no oracle)."""
    def resid(cfg_over, n=24):
        cfg = {**cal.cal_config(HW), **cfg_over}
        out = []
        for s in range(n):
            c_true = ov.coupling(s, "S1")
            Y, az, el = ov.coupling_cal_tones(c_true, POS, F0, cfg, np.random.default_rng(100 + s), HW)
            out.append(_offdiag_db(cal.estimate_coupling(Y, az, el, POS, F0) - c_true))
        return float(np.median(out))

    base = resid({})
    assert -46.0 <= base <= -34.0, base                                  # ~10 dB below the -30 dB true coupling
    assert resid({"multipath_rel_db": -50.0, "drift_rel_db": -55.0}) < base - 6.0
    assert resid({"n_tones": 6}) > base + 2.0


def test_node_cal_is_an_estimate_not_the_true_matrix_and_is_deterministic():
    c_true = ov.coupling(13, "S3")
    c_hat = ov.node_coupling_cal(HW, 13, "S3", "ism_2400", POS, F0)
    assert not np.allclose(c_hat, c_true, atol=1e-4)                    # residual stays
    assert _offdiag_db(c_hat - c_true) < -33.0
    assert np.allclose(c_hat, ov.node_coupling_cal(HW, 13, "S3", "ism_2400", POS, F0))    # one cal per deployment
    assert not np.allclose(c_hat, ov.node_coupling_cal(HW, 13, "S3", "ism_5800", POS, 5.8e9))   # per band


def test_estimator_sees_only_calibration_measurements(monkeypatch):
    """Truth separation: the belief-side estimator's inputs are the tone vectors and the surveyed bearings; the
    true coupling never reaches it, and the belief module does not import the world-side coupling."""
    params = list(inspect.signature(cal.estimate_coupling).parameters)
    assert params == ["Y", "az_deg", "el_deg", "pos", "f_hz"]
    src = (REPO / "sim" / "df" / "calibration.py").read_text()
    for bad in ("obs_model", "residual_coupling", "antenna", "stable_seed", "truth", "world_cache"):
        assert bad not in src.split('"""', 2)[-1], bad                  # code (not the docstring)
    c_true = ov.coupling(13, "S1")
    seen = []
    real = cal.estimate_coupling

    def spy(*a, **k):
        seen.extend(a)
        seen.extend(k.values())
        return real(*a, **k)

    monkeypatch.setattr(cal, "estimate_coupling", spy)
    ov.node_coupling_cal(HW, 13, "S1", "ism_2400", POS, F0)
    assert seen
    for v in seen:
        if isinstance(v, np.ndarray) and v.shape == (4, 4):
            assert not np.allclose(v, c_true)


# ------------------------------------------------------------------------------------------------ steering / DF
def test_calibrated_steering_identity_is_nominal_and_norm_is_kept():
    eng = DFEngine(HW, "cpu")
    A = eng.steering_at(POS, F0)
    same = cal.calibrated_steering(A, torch.eye(4, dtype=torch.complex64))
    assert torch.allclose(same, A, atol=1e-6)
    Ac = cal.calibrated_steering(A, torch.as_tensor(ov.coupling(1, "S1"), dtype=torch.complex64))
    assert torch.allclose(Ac.abs().pow(2).sum(-1), torch.full(Ac.shape[:-1], 4.0), atol=1e-4)
    assert not torch.allclose(Ac, A, atol=1e-3)


def _az_errs(c_true, c_hat, el=12.0):
    eng = DFEngine(HW, "cpu")
    b = B.Burst(0.0, 1e-3, -5e6, 5e6, 30.0)
    errs = {"nominal": [], "cal": []}
    for az0 in np.arange(1.0, 360.0, 7.0):
        a = c_true @ steering([az0], [el], POS, F0)[0]
        R = torch.as_tensor(np.outer(a, a.conj()) + 1e-4 * np.eye(4), dtype=torch.complex64)
        for k, m in (("nominal", None), ("cal", torch.as_tensor(c_hat, dtype=torch.complex64))):
            pk = ov.burst_df(eng, R, b, POS, F0, HW, 0.0, cal=m)
            errs[k].append(abs((pk[0][0] - az0 + 180) % 360 - 180))
    return {k: np.array(v) for k, v in errs.items()}


def test_calibrated_steering_removes_most_of_the_coupling_bias():
    nom, calb = [], []
    for s in range(6):
        c_true = ov.coupling(s, "S4")
        c_hat = ov.node_coupling_cal(HW, s, "S4", "ism_2400", POS, F0)
        e = _az_errs(c_true, c_hat)
        nom.append(e["nominal"])
        calb.append(e["cal"])
    nom, calb = np.concatenate(nom), np.concatenate(calb)
    assert np.percentile(nom, 90) > 0.6                                  # the bias exists (0.5 deg grid)
    assert np.mean(calb) < 0.5 * np.mean(nom)
    assert np.percentile(calb, 90) <= 0.5 * np.percentile(nom, 90) + 0.25


def test_burst_df_batch_matches_burst_df_with_cal():
    eng = DFEngine(HW, "cpu")
    m = torch.as_tensor(ov.node_coupling_cal(HW, 2, "S2", "ism_2400", POS, F0), dtype=torch.complex64)
    bs, Rs = [], []
    for k, az0 in enumerate((20.0, 140.0, 255.0)):
        a = ov.coupling(2, "S2") @ steering([az0], [8.0], POS, F0 + 1e6 * k)[0]
        Rs.append(torch.as_tensor(np.outer(a, a.conj()) + 1e-3 * np.eye(4), dtype=torch.complex64))
        bs.append(B.Burst(0.0, 1e-3, 1e6 * k - 5e6, 1e6 * k + 5e6, 25.0))
    got = ov.burst_df_batch(eng, torch.stack(Rs), bs, POS, F0, HW, 1.0, cal=m)
    for b, R, g in zip(bs, Rs, got):
        ref = ov.burst_df(eng, R, b, POS, F0, HW, 1.0, cal=m)
        assert len(ref) == len(g)
        for (a1, e1, s1), (a2, e2, s2) in zip(ref, g):
            assert abs((a1 - a2 + 180) % 360 - 180) < 1e-6 and e1 == e2 and abs(s1 - s2) < 1e-3
