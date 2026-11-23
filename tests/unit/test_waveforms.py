"""Waveform structure checks (RF phase 2, Task 6): the properties an RF analyst would measure."""

import numpy as np
import pytest

from sim.rf import waveforms as W


def rng():
    return np.random.default_rng(3)


@pytest.mark.parametrize("bw", [10.0, 20.0])
def test_ocusync_like_occupied_bandwidth_and_cyclic_prefix(bw):
    w = W.ofdm_ocusync_like(bw, 6e-3, rng())
    obw = W.occupied_bw_hz(w.iq, w.fs)
    assert obw == pytest.approx(0.9 * bw * 1e6, rel=0.05)
    nfft, cp = w.meta["nfft"], w.meta["cp"]
    t0, dur = w.bursts[0][0], w.bursts[0][1]
    x = w.iq[int(t0 * w.fs): int((t0 + dur) * w.fs)]
    def corr(lag):
        a, b = x[:-lag], x[lag:]
        return abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    assert corr(nfft) > 3 * max(corr(nfft - 40), corr(nfft + 40))      # CP repeats the symbol tail at lag N_fft
    assert w.meta["scs_hz"] == 15e3


def test_wifi_preamble_and_bandwidth():
    w = W.wifi_ofdm(5e-3, rng())
    obw = W.occupied_bw_hz(w.iq, w.fs)
    assert 15.0e6 < obw < 18.5e6                                        # 52 x 312.5 kHz = 16.25 MHz
    t0 = w.bursts[0][0]
    stf = w.iq[int(t0 * w.fs): int(t0 * w.fs) + 160]
    assert np.allclose(stf[:16], stf[16:32], atol=1e-5)                 # L-STF: 0.8 us period


def test_lora_hops_and_dechirps_to_a_tone():
    chans = [-1.0e6, 0.0, 1.0e6]
    w = W.lora_elrs(500e3, 7, 150.0, chans, 30e-3, rng())
    assert len({b[2] for b in w.bursts}) == 3                            # one channel per packet, all used
    t0, _, f0, _ = w.bursts[0]
    over = int(w.fs / 500e3)
    n = 2 ** 7 * over
    i = int(t0 * w.fs)
    seg = w.iq[i:i + n] * np.exp(-2j * np.pi * f0 * (np.arange(i, i + n) / w.fs))
    d = seg * W._chirp(2 ** 7, False, over=over)
    spec = np.abs(np.fft.fft(d)) ** 2
    assert spec.max() > 50 * spec.mean()                                 # preamble up-chirp dechirps to one bin
    single = W.lora_elrs(500e3, 7, 150.0, [0.0], 20e-3, rng())
    assert W.occupied_bw_hz(single.iq, single.fs) == pytest.approx(500e3, rel=0.1)   # continuous-phase chirp


def test_fm_video_demodulates_to_its_deviation_line_rate_and_audio_subcarrier():
    w = W.fm_fpv(4e-3, rng())
    assert np.std(np.abs(w.iq)) < 1e-4                                  # constant envelope
    f_inst = np.angle(w.iq[1:] * np.conj(w.iq[:-1])) * w.fs / (2 * np.pi)
    assert (f_inst.max() - f_inst.min()) == pytest.approx(2 * w.meta["dev_hz"], rel=0.05)
    spec = np.abs(np.fft.rfft(f_inst - f_inst.mean())) ** 2
    fr = np.fft.rfftfreq(len(f_inst), 1 / w.fs)
    band = lambda f0, df: spec[(fr > f0 - df) & (fr < f0 + df)].max()  # noqa: E731
    floor = np.median(spec[(fr > 1e6) & (fr < 5e6)])
    assert band(6.5e6, 60e3) > 100 * floor                              # audio subcarrier
    assert band(15.625e3, 300) > 100 * np.median(spec[(fr > 20e3) & (fr < 1e6)])   # 64 us lines
    assert W.occupied_bw_hz(w.iq, w.fs) < 2 * (w.meta["dev_hz"] + w.meta["audio_hz"])   # within Carson


def test_gfsk_burst_bandwidth_and_period():
    w = W.gfsk_hop(1e6, 0.5e-3, 2e-3, [0.0], 20e-3, rng())
    assert 0.8e6 < W.occupied_bw_hz(w.iq, w.fs) < 2.0e6
    gaps = np.diff([b[0] for b in w.bursts])
    assert np.allclose(gaps, 2e-3)
    assert np.std(np.abs(w.iq[np.abs(w.iq) > 0])) < 1e-4                # GFSK: constant envelope
