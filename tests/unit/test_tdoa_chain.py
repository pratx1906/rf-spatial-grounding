"""TDOA chain (demo-v2 S1.1, audit D Q30): linear delays, truncated bursts ignored, same-burst matching across
nodes, residual-gated 3-D refinement."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.belief import tdoa as TD
from sim.df.air7311 import square_positions
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import receiver as rx
from sim.rf import toa as T
from sim.rf import waveforms as W

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
FS = 125e6
N = int(FS * 8e-3)
POS = square_positions(0.054)
C0 = 299_792_458.0


def test_delay_is_linear_not_circular():
    """A burst ending at the dwell end must not wrap to the dwell start; content on air before the dwell (the
    pre-roll) must flow in."""
    pre = ch.pre_roll(N)
    s = torch.zeros(pre + N, dtype=torch.complex64)
    s[-20000:] = 1.0                                       # burst in the last 160 us of the dwell
    y = ch.propagate(s, FS, np.array([1.0 + 0j]), np.array([6e-6]), np.array([0.0]), np.array([0.0]), POS,
                     2.44e9, 2.44e9, n_pre=pre)
    assert y.shape == (4, N)
    assert float(y[:, :700].abs().max()) < 1e-3            # no wrapped tail in the first 6 us
    s2 = torch.ones(pre + N, dtype=torch.complex64)        # continuous carrier: on air before the dwell too
    y2 = ch.propagate(s2, FS, np.array([1.0 + 0j]), np.array([6e-6]), np.array([0.0]), np.array([0.0]), POS,
                      2.44e9, 2.44e9, n_pre=pre)
    assert float(y2[:, :700].abs().min()) > 0.99           # no gap at the dwell start


def _node_dwell(w, off, tau, seed):
    pre = ch.pre_roll(N)
    iq = w.iq[off - int(round(pre * w.fs / FS)):]
    s = ch.resample_shift(iq, w.fs, FS, 0.0, pre + N, 0.0)
    y = ch.propagate(s, FS, np.array([1.0 + 0j]), np.array([tau]), np.array([30.0]), np.array([0.0]), POS,
                     2.44e9, 2.44e9, n_pre=pre)
    bw = 0.9 * 20e6
    p_n = 10 ** ((-174 + 10 * math.log10(bw) + HW["receiver"]["noise_figure_db"] - 30) / 10)
    x, _ = rx.receive(y * math.sqrt(p_n * 10 ** 2.5), FS, HW, torch.Generator().manual_seed(seed), interchip_deg=0.0)
    return x


def test_one_and_six_us_give_five_us_from_complete_bursts_only():
    """Audit case: true delays 1 us and 6 us, a burst already on air at the dwell start. The truncated burst gives
    no TOA; the matched complete burst gives TDOA = 5 us within 1/B."""
    w = W.ofdm_ocusync_like(20.0, 40e-3, np.random.default_rng(3))
    b0 = w.bursts[1]
    off = int((b0[0] + 0.5e-3) * w.fs)                    # dwell starts 0.5 ms into a burst
    toas = {}
    for sid, tau, sd in (("A", 1e-6, 1), ("B", 6e-6, 2)):
        x = _node_dwell(w, off, tau, sd)
        bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
        assert bs and bs[0].t0 < 5e-6                      # the truncated burst is detected ...
        out = [T.toa_s(x, FS, b, np.random.default_rng(0), 0.0) for b in bs]
        assert math.isnan(out[0])                          # ... but gives no TOA
        toas[sid] = [t for t in out if np.isfinite(t)]
        assert toas[sid]
    pose = {"A": (0.0, 0.0, 10.0), "B": (1500.0, 0.0, 10.0)}
    m = TD.match_toas(toas, pose, min_nodes=2)                # two-node chain check (the tracker needs >= 3)
    assert set(m) == {"A", "B"}
    assert m["B"] - m["A"] == pytest.approx(5e-6, abs=1 / (0.9 * 20e6))


def test_match_picks_the_same_burst_not_the_earliest():
    pose = {"A": (0.0, 0.0, 0.0), "B": (1000.0, 0.0, 0.0), "C": (0.0, 1000.0, 0.0)}
    toas = {"A": [1.000e-3, 3.000e-3], "B": [3.002e-3], "C": [1.001e-3, 3.0015e-3]}
    m = TD.match_toas(toas, pose)
    assert m == {"A": 3.000e-3, "B": 3.002e-3, "C": 3.0015e-3}


def _group(p, sensors, toa_err=None, brg_bias=0.0):
    g = []
    for i, (sid, s) in enumerate(sensors.items()):
        brg = math.degrees(math.atan2(p[0] - s[0], p[1] - s[1])) % 360
        toa = math.dist(p, s) / C0 + (toa_err or {}).get(sid, 0.0)
        g.append({"sensor": sid, "brg": brg + brg_bias, "snr": 20.0, "toa": toa, "bw_khz": 20000.0})
    return g


SENS = {"S1": (0.0, 0.0, 12.0), "S2": (1800.0, 200.0, 15.0), "S3": (600.0, 1700.0, 30.0)}


def test_refine_uses_3d_ranges():
    p = (700.0, 600.0, 150.0)                              # high drone: 2-D ranges would be off by tens of metres
    xy, ok, info = TD.tdoa_fix((760.0, 540.0), _group(p, SENS), SENS, 0.0, z_m=150.0)
    assert ok
    assert math.hypot(xy[0] - p[0], xy[1] - p[1]) < 3.0


def test_inconsistent_tdoa_falls_back_to_aoa():
    p = (700.0, 600.0, 50.0)
    aoa = (720.0, 610.0)
    xy, ok, info = TD.tdoa_fix(aoa, _group(p, SENS, {"S2": 3e-6}, brg_bias=1.0), SENS, 5.0, z_m=50.0)
    assert not ok
    assert tuple(xy) == aoa


def test_refine_skips_nodes_without_toa():
    p = (700.0, 600.0, 50.0)
    g = _group(p, SENS)
    g.append({"sensor": "S4", "brg": 10.0, "snr": 3.0, "toa": float("nan"), "bw_khz": 20000.0})
    sens = {**SENS, "S4": (-500.0, -500.0, 10.0)}
    xy = TD.tdoa_refine((710.0, 590.0), g[:3] + [g[3]], sens, 0.0, z_m=50.0)
    assert np.all(np.isfinite(xy))


def test_match_merges_one_burst_split_in_frequency():
    """An OFDM burst with a DC null is detected as two bursts with leading edges ~0.1 us apart at each node."""
    pose = {"A": (0.0, 0.0, 0.0), "B": (1000.0, 0.0, 0.0), "C": (0.0, 1000.0, 0.0)}
    toas = {"A": [1.0e-3, 1.0001e-3], "B": [1.0020e-3, 1.0021e-3], "C": [1.0010e-3]}
    m = TD.match_toas(toas, pose)
    assert set(m) == {"A", "B", "C"}
    assert m["B"] - m["A"] == pytest.approx(2.0e-6, abs=1e-9)


def test_emissions_keep_every_burst_and_only_complete_toas():
    from sim.obs_model.observe_v2 import Detection
    from sim.obs_model.tracker_v2 import emissions

    mk = lambda t0, toa: Detection(10.0, "S1", "ism_5800", 5800.0, 18000.0, t0, 1e-3, "ocusync_like", 0.9,
                                   [(30.0, 0.0, 20.0)], toa, 20.0, -60.0)
    em = emissions([mk(0.0, float("nan")), mk(2e-3, 10.002001), mk(5e-3, 10.005001)])
    assert len(em) == 1
    e = em[0]
    assert sorted(e["t0s"]) == [0.0, 2e-3, 5e-3]                # every burst links to the track
    assert e["toas"] == [10.002001, 10.005001]                  # the cut burst has no TOA
    assert e["toa"] == 10.002001
