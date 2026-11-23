"""S7.2 item 1 (audit B Q14): per-path Doppler nu_p = v . u_p f_c / c, applied per path in propagate."""

import math

import numpy as np
import pytest
import torch

from sim.rf import channel as ch


def tone(fs, n, f):
    return np.exp(2j * np.pi * f * np.arange(n) / fs).astype(np.complex64)


def test_two_paths_with_their_own_doppler_match_the_closed_form():
    fs, n = 125e6, 1 << 16
    f1 = 512 * fs / n                                                    # integer FFT bin: exact circular delay
    taus = np.array([1.0e-6, 1.3e-6])
    h = np.array([1.0 + 0j, 0.5j])
    nus = np.array([3000.0, -2000.0])                                   # large enough to rotate inside 0.5 ms
    y = ch.propagate(torch.as_tensor(tone(fs, n, f1)), fs, h, taus, np.zeros(2), np.zeros(2), np.zeros((1, 3)),
                     2.44e9, 2.44e9, doppler_hz=nus).numpy()[0]
    t = np.arange(n) / fs
    expect = sum(h[p] * np.exp(2j * np.pi * f1 * (t - taus[p])) * np.exp(2j * np.pi * nus[p] * t) for p in range(2))
    assert np.allclose(y, expect, atol=2e-3)
    # the old model (one Doppler for all paths) is measurably different: the paths rotate against each other
    y1 = ch.propagate(torch.as_tensor(tone(fs, n, f1)), fs, h, taus, np.zeros(2), np.zeros(2), np.zeros((1, 3)),
                      2.44e9, 2.44e9, doppler_hz=float(nus[0])).numpy()[0]
    assert np.max(np.abs(y1 - expect)) > 0.5


def test_scalar_doppler_is_unchanged():
    fs, n = 125e6, 4096
    f1 = 64 * fs / n
    y = ch.propagate(torch.as_tensor(tone(fs, n, f1)), fs, np.array([1.0 + 0j]), np.array([0.0]), np.zeros(1),
                     np.zeros(1), np.zeros((1, 3)), 2.44e9, 2.44e9, doppler_hz=1000.0).numpy()[0]
    t = np.arange(n) / fs
    assert np.allclose(y, np.exp(2j * np.pi * (f1 + 1000.0) * t), atol=2e-3)


def test_per_path_doppler_uses_each_paths_drone_end_direction():
    v = np.array([0.0, 20.0, 0.0])                                      # 20 m/s north
    az_r = np.array([0.0, 180.0, 90.0, 0.0])                            # toward the node: N, S, E, N and up 60 deg
    el_r = np.array([0.0, 0.0, 0.0, 60.0])
    nu = ch.doppler_hz(v, az_r, el_r, 5.8e9)
    k = 20 * 5.8e9 / ch.C
    assert nu == pytest.approx([k, -k, 0.0, k * math.cos(math.radians(60))], abs=1e-6)
    assert ch.doppler_hz(v, 0.0, 0.0, 5.8e9) == pytest.approx(k)       # scalar call still returns a float
