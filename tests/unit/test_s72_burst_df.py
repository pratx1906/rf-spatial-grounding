"""S7.2 items 2 and 6 (audit D Q26/Q27, A Q10): MDL with N = B x T of the burst, DF steering at the burst frequency."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.df.air7311 import DFEngine, square_positions
from sim.obs_model.observe_v2 import burst_df
from sim.rf import bursts as B
from sim.rf.channel import steering

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
POS = square_positions(0.054)


def _bl_noise(g, n, fs, bw):
    z = torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g)) / math.sqrt(2)
    Z = torch.fft.fft(z)
    Z[torch.fft.fftfreq(n, 1 / fs).abs() > bw / 2] = 0
    z = torch.fft.ifft(Z)
    return z / z.abs().pow(2).mean().sqrt()


def _burst(g, fs, bw, dur, srcs, snr_db, f_hz=2.44e9):
    n = int(dur * fs)
    x = torch.complex(torch.randn((4, n), generator=g), torch.randn((4, n), generator=g)) / math.sqrt(2)
    for az, rel in srcs:
        a = torch.as_tensor(steering([az], [0.0], POS, f_hz)[0], dtype=torch.complex64)
        x = x + 10 ** ((snr_db + rel) / 20) * a[:, None] * _bl_noise(g, n, fs, bw)[None]
    return B.burst_cov(x.to(torch.complex64), fs, B.Burst(0.0, dur, -bw / 2, bw / 2, 0.0))


def _orders(fs, bw, dur, srcs, snr, n_snap, trials=12):
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(trials):
        R = _burst(g, fs, bw, dur, srcs, snr)
        ev = torch.linalg.eigvalsh(R[None].to(torch.complex64)).real
        out.append(int(DFEngine.mdl_order(ev, n_snap, rel_floor_db=-20.0)[0]))
    return out


def test_effective_snapshots_are_bandwidth_times_duration_capped():
    assert DFEngine.mdl_snapshots(64e3, 2e-3, 125e6) == 128
    assert DFEngine.mdl_snapshots(500e3, 1e-3, 125e6) == 500
    assert DFEngine.mdl_snapshots(20e6, 1e-3, 125e6) == 4096           # cap: model errors, not noise, limit there
    assert DFEngine.mdl_snapshots(40e3, 150e-6, 125e6) == 16           # floor


def test_model_order_is_right_for_wide_long_and_narrow_short_bursts():
    fs = 50e6
    # wideband long burst (20 MHz x 1 ms, N_eff 20,000): a weak (-18 dB) second source is real and must be found;
    # the old fixed N = 256 misses it
    srcs = [(30.0, 0.0), (150.0, -16.0)]
    assert _orders(fs, 20e6, 1e-3, srcs, 3.0, DFEngine.mdl_snapshots(20e6, 1e-3, fs)) == [2] * 12
    assert _orders(fs, 20e6, 1e-3, srcs, 3.0, 256) == [1] * 12
    # narrowband short burst (64 kHz x 0.3 ms, N_eff 19): one source stays one source
    assert _orders(fs, 64e3, 0.3e-3, [(30.0, 0.0)], 10.0, DFEngine.mdl_snapshots(64e3, 0.3e-3, fs)) == [1] * 12
    # narrowband, two strong sources (-3 dB): still two with only 100 effective snapshots
    assert _orders(fs, 500e3, 0.2e-3, [(30.0, 0.0), (150.0, -3.0)], 15.0,
                   DFEngine.mdl_snapshots(500e3, 0.2e-3, fs)) == [2] * 12


def test_steering_at_the_burst_frequency_removes_the_elevation_bias():
    """A burst 45 MHz above the window centre at el 12 deg: steering built at the window centre scales every
    phase by f0/fc and reads the elevation low (cos el' = cos el * fc / f0); at the burst frequency it is exact."""
    eng = DFEngine(HW, "cpu")
    f0, off, el = 2.44175e9, 45e6, 12.0
    a = steering([77.0], [el], POS, f0 + off)[0]
    R = torch.as_tensor(np.outer(a, a.conj()) + 1e-3 * np.eye(4), dtype=torch.complex64)
    b = B.Burst(0.0, 1e-3, off - 5e6, off + 5e6, 30.0)
    pk = burst_df(eng, R, b, POS, f0, HW, 0.0)
    assert abs((pk[0][0] - 77.0 + 180) % 360 - 180) < 1.0
    assert pk[0][1] == pytest.approx(el, abs=2.0)
    pk_old = burst_df(eng, R, b, POS, f0, HW, 0.0, steer_at_burst=False)
    assert pk_old[0][1] <= el - 4.0                                     # the bias the fix removes
