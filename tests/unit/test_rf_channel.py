"""Wideband channel (RF phase 2, Task 6)."""

import math

import numpy as np
import pytest
import torch

from sim.rf import channel as ch

POS = np.array([[-0.027, 0.027, 0], [0.027, 0.027, 0], [0.027, -0.027, 0], [-0.027, -0.027, 0]])


def tone(fs, n, f):
    return np.exp(2j * np.pi * f * np.arange(n) / fs).astype(np.complex64)


def test_resample_keeps_a_tone_and_its_shift():
    x = tone(20e6, 4000, 1e6)
    y = ch.resample_shift(x, 20e6, 125e6, 5e6, 30000, 1e-5)
    seg = y[2000:20000].numpy()
    f = np.fft.fftfreq(len(seg), 1 / 125e6)[np.argmax(np.abs(np.fft.fft(seg)))]
    assert f == pytest.approx(6e6, abs=125e6 / len(seg))
    assert np.allclose(np.abs(seg), 1.0, atol=0.05)
    assert torch.all(y[:1000] == 0)                                       # placed at t0 = 10 us


def test_single_path_delay_and_steering_match_closed_form():
    fs, n = 125e6, 4096
    f0 = 64 * fs / n                                                     # integer FFT bin: no circular leakage
    s = torch.as_tensor(tone(fs, n, f0))
    tau = 1.234e-6
    y = ch.propagate(s, fs, np.array([1.0 + 0j]), np.array([tau]), np.array([60.0]), np.array([10.0]), POS,
                     2.44e9, 2.44e9).numpy()
    ref = np.exp(2j * np.pi * f0 * (np.arange(n) / fs - tau))[None, :] * ch.steering([60.0], [10.0], POS, 2.44e9).T
    assert np.allclose(y[:, 100:-100], ref[:, 100:-100], atol=2e-3)      # circular edges aside


def test_carrier_offset_corrects_the_cached_band_centre_phase():
    fs, n = 125e6, 2048
    s = torch.as_tensor(tone(fs, n, 0.0))
    tau = 2.0e-6
    h0 = np.exp(-2j * np.pi * 2.44e9 * tau)                             # cached at the band centre
    y = ch.propagate(s, fs, np.array([h0]), np.array([tau]), np.array([0.0]), np.array([0.0]), np.zeros((1, 3)),
                     2.47e9, 2.44e9).numpy()
    expect = np.exp(-2j * np.pi * 2.47e9 * tau)                         # true phase at the carrier
    assert abs(y[0, 1000] - expect) < 1e-3


def test_doppler_sign_and_size():
    v = np.array([0.0, 20.0, 0.0])                                      # flying north at 20 m/s
    assert ch.doppler_hz(v, 0.0, 0.0, 5.8e9) == pytest.approx(20 * 5.8e9 / ch.C, rel=1e-6)   # node due north: closing
    assert ch.doppler_hz(v, 180.0, 0.0, 5.8e9) < 0
    assert abs(ch.doppler_hz(v, 0.0, 0.0, 5.8e9)) < 400


def test_offset_carrier_phase_is_counted_once():
    """Audit Q3/Q10/Q13: the signal is already shifted to fc - f0 (LO at f0); each path must carry
    exp(-j2pi fc tau) once, i.e. baseband y = exp(j2pi (fc-f0) t) exp(-j2pi fc tau)."""
    fs, n = 125e6, 2048
    df = 320 * fs / n                                                   # 19.53 MHz, integer FFT bin
    f0, fband = 2.44e9, 2.44e9
    fc = f0 + df
    tau = 1.0e-6
    t = np.arange(n) / fs
    s = torch.as_tensor(tone(fs, n, df))                                # waveform already at fc - f0
    h0 = np.exp(-2j * np.pi * fband * tau)                              # cache convention: band-centre phase
    y = ch.propagate(s, fs, np.array([h0]), np.array([tau]), np.array([0.0]), np.array([0.0]), np.zeros((1, 3)),
                     fc, fband, f_lo_hz=f0).numpy()
    expect = np.exp(2j * np.pi * df * t) * np.exp(-2j * np.pi * fc * tau)
    assert np.allclose(y[0, 200:-200], expect[200:-200], atol=2e-3)


def test_relative_multipath_phase_matches_passband_for_offset_carrier():
    fs, n = 125e6, 4096
    df = -640 * fs / n                                                  # -19.53 MHz
    f0 = 2.4415e9
    fc = f0 + df
    taus = np.array([1.0e-6, 1.3e-6])
    h = np.array([1.0, 0.5]) * np.exp(-2j * np.pi * 2.44e9 * taus)
    t = np.arange(n) / fs
    y = ch.propagate(torch.as_tensor(tone(fs, n, df)), fs, h, taus, np.zeros(2), np.zeros(2), np.zeros((1, 3)),
                     fc, 2.44e9, f_lo_hz=f0).numpy()
    expect = np.exp(2j * np.pi * df * t) * (1.0 * np.exp(-2j * np.pi * fc * taus[0]) + 0.5 * np.exp(-2j * np.pi * fc * taus[1]))
    assert np.allclose(y[0, 400:-400], expect[400:-400], atol=3e-3)
