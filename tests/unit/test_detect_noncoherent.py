"""Detection on the non-coherent channel sum (demo-v2 S1.2, audit C Q24): sum_m |STFT(x_m)|^2, so the detection
statistic of a single plane wave does not depend on its azimuth (the unsteered x.sum(0) had a ~4.4 dB ripple)."""

import math
from pathlib import Path

import numpy as np
import torch
import yaml

from sim.df.air7311 import square_positions
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import receiver as rx
from sim.rf import waveforms as W

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
FS = 125e6


def _snr_at(az, el, s, side, f):
    y = ch.propagate(s, FS, np.array([1.0 + 0j]), np.array([0.0]), np.array([az]), np.array([el]),
                     square_positions(side), f, f)
    x, _ = rx.receive(y, FS, HW, torch.Generator().manual_seed(7), interchip_deg=0.0)
    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
    return max(b.snr_db for b in bs) if bs else -99.0


def test_single_path_detection_ripple_over_azimuth_below_1_db():
    n = int(FS * 2e-3)
    w = W.ofdm_ocusync_like(20.0, 1.5e-3, np.random.default_rng(1), duty=1.0)
    s = ch.resample_shift(w.iq, w.fs, FS, 10e6, n, 0.2e-3)
    p_n = 10 ** ((-174 + 10 * math.log10(18e6) + HW["receiver"]["noise_figure_db"] - 30) / 10)
    s = s * math.sqrt(p_n * 10 ** 1.2)                      # 12 dB per element in the occupied band
    for side, f in ((0.054, 2.4835e9), (0.145, 0.928e9)):
        snr = [_snr_at(az, 20.0, s, side, f) for az in range(0, 360, 15)]
        assert min(snr) > 0
        assert max(snr) - min(snr) < 1.0, (side, snr)
