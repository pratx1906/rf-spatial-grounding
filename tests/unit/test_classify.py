"""Protocol classifier and TOA (RF phase 2, Task 6)."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import classify as C
from sim.rf import receiver as rx
from sim.rf import toa as T
from sim.rf import waveforms as W

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
FS = 125e6


def _rx(w, snr_db, f_off=8e6, t0=0.2e-3, dwell=13e-3, seed=0):
    n = int(FS * dwell)
    s = ch.resample_shift(w.iq, w.fs, FS, f_off, n, t0)
    bw = max(b[3] for b in w.bursts)
    p_n = 10 ** ((-174 + 10 * math.log10(bw) + 3 - 30) / 10)
    y = s[None, :].repeat(4, 1) * math.sqrt(p_n * 10 ** (snr_db / 10))
    x, _ = rx.receive(y, FS, HW, torch.Generator().manual_seed(seed), interchip_deg=0.0)
    return x


GEN = {
    "ocusync_like": lambda r: W.ofdm_ocusync_like(10.0, 4e-3, r),
    "wifi": lambda r: W.wifi_ofdm(4e-3, r, period_us=1500.0),
    "lora": lambda r: W.lora_elrs(500e3, 7, 150.0, [0.0], 12e-3, r),
    "fm_video": lambda r: W.fm_fpv(3e-3, r),
    "gfsk": lambda r: W.gfsk_hop(1e6, 0.5e-3, 1e-3, [0.0], 4e-3, r),
}


@pytest.mark.parametrize("proto", list(GEN))
def test_each_protocol_is_recognised_at_good_snr(proto):
    w = GEN[proto](np.random.default_rng(5))
    x = _rx(w, 20.0)
    bs = [b for b in B.detect(x, FS) if b.dur > 100e-6]
    assert bs
    b = max(bs, key=lambda q: q.dur * q.bw)
    got, conf = C.classify(C.features(x, FS, b))
    assert got == proto and 0 < conf <= 1


def test_toa_tracks_the_leading_edge_within_bandwidth_resolution():
    w = W.ofdm_ocusync_like(20.0, 2e-3, np.random.default_rng(2))
    x = _rx(w, 25.0, t0=0.5e-3)
    b = max(B.detect(x, FS), key=lambda q: q.dur)
    t = T.toa_s(x, FS, b, np.random.default_rng(0), gnss_ns=0.0)
    assert t == pytest.approx(0.5e-3 + w.bursts[0][0], abs=0.2e-6)
