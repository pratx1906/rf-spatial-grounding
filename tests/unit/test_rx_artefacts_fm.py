"""S7.2b fixes found by the S7.3 characterisation (docs/RESULTS/characterisation.md):

1. FM video was `unknown` 80-88 % of the time at 12-20 dB (its in-box occupied bandwidth, 3.9-4.4 MHz, fell in the
   3-5 MHz gap between the gfsk and fm_video rules); SiK was 37-39 % unknown at high SNR.
2. A strong carrier produced a burst at its own IQ image (mirror frequency about the LO) from -70 dBm, and the DC
   offset a second false burst from -60 dBm, in every dwell.

The chain is the one observe_v2 uses: plane wave into the AIR7311 square -> front end -> receiver -> detect."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.df.air7311 import square_positions
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import classify as C
from sim.rf import frontend as FE
from sim.rf import receiver as rx
from sim.rf import waveforms as W

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
FS, F0, DWELL = 125e6, 2.44e9, 8e-3
N = int(FS * DWELL)
POS = square_positions(0.054)


def _chain(sig: torch.Tensor, f_off: float, az: float, seed: int) -> torch.Tensor:
    y = ch.propagate(sig, FS, np.array([1.0 + 0j]), np.array([0.0]), np.array([az]), np.array([0.0]), POS,
                     F0 + f_off, F0, f_lo_hz=F0)
    y, _ = FE.front_end(y, HW["front_end"], [])
    x, _ = rx.receive(y, FS, HW, torch.Generator().manual_seed(seed), interchip_deg=2.0)
    return x


def _waveform_dwell(w, snr_db: float, f_off: float, az: float, seed: int) -> torch.Tensor:
    s = ch.resample_shift(w.iq, w.fs, FS, f_off, N, 0.0, ibw_hz=100e6)
    b99 = W.occupied_bw_hz(w.iq, w.fs, nfft=4096 if w.fs > 10e6 else 1024)
    p_n = 10 ** ((-174 + 10 * math.log10(b99) + HW["receiver"]["noise_figure_db"] - 30) / 10)
    return _chain(s * math.sqrt(p_n * 10 ** (snr_db / 10)), f_off, az, seed)


def _tone(p_dbm: float, f_hz: float, phase: float) -> torch.Tensor:
    t = torch.arange(N, dtype=torch.float64) / FS
    return (torch.polar(torch.ones_like(t), 2 * math.pi * f_hz * t + phase) * math.sqrt(10 ** ((p_dbm - 30) / 10))
            ).to(torch.complex64)


def _labels(x, f_lo, f_hi):
    """Classifier labels of the detections (observe_v2's 150 us minimum) overlapping [f_lo, f_hi]."""
    return [C.classify(C.features(x, FS, b))[0] for b in B.detect(x, FS)
            if b.dur >= 150e-6 and b.f_lo < f_hi and b.f_hi > f_lo]


def test_fm_video_is_classified_at_6_db_and_above():
    labs = []
    for snr in (6.0, 10.0, 14.0, 20.0):
        for k in range(3):
            w = W.fm_fpv(DWELL, np.random.default_rng(100 + k))
            x = _waveform_dwell(w, snr, 10e6, 30.0 + 70 * k, 10 * k + int(snr))
            got = _labels(x, 10e6 - 12e6, 10e6 + 12e6)
            assert got, (snr, k)
            labs += got
    assert labs.count("fm_video") / len(labs) >= 0.9, labs


def test_sik_telemetry_is_gfsk_at_high_snr():
    labs = []
    for snr in (14.0, 18.0):
        for k in range(3):
            w = W.gfsk_hop(64e3, 2e-3, 3e-3, [0.0], DWELL, np.random.default_rng(200 + k))
            x = _waveform_dwell(w, snr, 10e6, 45.0 + 90 * k, 20 * k + int(snr))
            labs += _labels(x, 9.5e6, 10.5e6)
    assert labs and labs.count("gfsk") / len(labs) >= 0.9, labs


def test_image_rejection_matches_the_receiver_model():
    """The detector's image model uses receiver.image_rejection_db(); check it against a noiseless tone."""
    y = _tone(-50.0, 10e6, 0.3)[None].repeat(4, 1)
    hw = {**HW, "receiver": {**HW["receiver"], "noise_figure_db": -200.0}}
    x, _ = rx.receive(y, FS, hw, torch.Generator().manual_seed(0), interchip_deg=0.0)
    X = torch.fft.fft(x[0] * torch.hann_window(N, dtype=torch.float32))
    f = torch.fft.fftfreq(N, 1 / FS)
    k_s, k_i = int(torch.argmin((f - 10e6).abs())), int(torch.argmin((f + 10e6).abs()))
    irr = 10 * math.log10(float(X[k_s - 3:k_s + 4].abs().pow(2).sum() / X[k_i - 3:k_i + 4].abs().pow(2).sum()))
    assert irr == pytest.approx(rx.image_rejection_db(), abs=0.3)
    assert 30.0 < irr < 45.0


@pytest.mark.parametrize("p_dbm", [-60.0, -50.0, -40.0])
def test_strong_carrier_makes_no_image_or_dc_burst(p_dbm):
    for k in range(3):
        x = _chain(_tone(p_dbm, 10e6, 0.7 * k), 10e6, 40.0 + 100 * k, 300 + k)
        bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
        far = [(round(b.fc / 1e6, 2), round(b.snr_db, 1)) for b in bs if not (b.f_lo < 11e6 and b.f_hi > 9e6)]
        assert any(b.f_lo < 11e6 and b.f_hi > 9e6 for b in bs)
        assert not far, (p_dbm, k, far)


def test_weak_signal_at_the_image_frequency_is_still_detected():
    """A real tone exactly at the strong carrier's mirror frequency, 20 dB below it (about 18 dB above the image
    level for IRR ~38 dB, 40 dB per-bin SNR), is not taken for the image."""
    assert rx.image_rejection_db() > 30.0
    for k in range(3):
        s = _tone(-50.0, 10e6, 0.4 * k) + _tone(-70.0, -10e6, 1.1 + k)
        x = _chain(s, 10e6, 60.0 + 90 * k, 400 + k)
        bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
        assert any(b.f_lo < -9e6 and b.f_hi > -11e6 for b in bs), (k, [(b.fc / 1e6, b.snr_db) for b in bs])


def test_weak_wideband_signal_on_the_image_band_is_still_detected():
    """A 10 MHz OFDM signal centred on the image of a strong carrier, 25 dB below the carrier."""
    w = W.ofdm_ocusync_like(10.0, DWELL, np.random.default_rng(9), duty=1.0)
    o = ch.resample_shift(w.iq, w.fs, FS, -10e6, N, 0.0, ibw_hz=100e6)
    o = o / o.abs().pow(2).mean().sqrt() * math.sqrt(10 ** ((-75.0 - 30) / 10))
    x = _chain(_tone(-50.0, 10e6, 0.2) + o, 10e6, 75.0, 500)
    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
    assert any(b.f_lo < -8e6 and b.f_hi > -12e6 and b.bw > 5e6 for b in bs), [(b.fc / 1e6, b.bw / 1e6) for b in bs]
