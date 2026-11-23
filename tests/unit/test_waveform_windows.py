"""Waveform window fixes (demo-v2 S1.3, audit B Q12/Q16): ELRS hop sequences are random (start and order), content
beyond the +-50 MHz IBW is dropped instead of aliasing, RC hops stay inside 2400-2483.5 MHz."""

import numpy as np
import torch

from sim.obs_model import observe_v2 as O
from sim.rf import channel as ch
from sim.rf import waveforms as W


def test_elrs_hops_do_not_restart_at_channel_zero():
    chans = O.ELRS_24
    first, steps = set(), []
    for k in range(20):
        w = W.lora_elrs(812.5e3, 6, 250.0, chans, 40e-3, np.random.default_rng(k), data_symbols=8)
        idx = [chans.index(b[2]) for b in w.bursts]
        first.add(idx[0])
        steps += list(np.diff(idx))
    assert len(first) >= 10                                 # random start channel
    assert np.mean(np.array(steps) == 1) < 0.2              # not a sequential sweep


def test_content_beyond_the_ibw_is_dropped_not_aliased():
    fs_in, n_in = 30.72e6, 30720
    tone = np.exp(2j * np.pi * 5e6 * np.arange(n_in) / fs_in).astype(np.complex64)
    n = 125000
    y = ch.resample_shift(tone, fs_in, 125e6, 68e6, n, 0.0, ibw_hz=100e6)    # tone at +73 MHz: outside the IBW
    assert float(y.abs().pow(2).mean()) < 1e-6
    y2 = ch.resample_shift(tone, fs_in, 125e6, 40e6, n, 0.0, ibw_hz=100e6)   # +45 MHz: inside, kept
    assert float(y2[1000:-1000].abs().pow(2).mean()) > 0.9


def test_rc_hops_stay_inside_the_ism_band():
    for fc in (2401.0e6, 2441.0e6, 2482.5e6):
        for k in range(5):
            w = O.waveform("rc_fhss", np.random.default_rng(k), 20e-3, 1500.0, fc_hz=fc)
            for t0, dur, f_off, bw in w.bursts:
                assert 2400e6 <= fc + f_off - bw / 2 and fc + f_off + bw / 2 <= 2483.5e6, (fc, f_off)
    for fc in (2402.0e6, 2480.0e6):
        w = O.waveform("elrs24", np.random.default_rng(0), 40e-3, 812.5, fc_hz=fc)
        for t0, dur, f_off, bw in w.bursts:
            assert 2400e6 <= fc + f_off - bw / 2 and fc + f_off + bw / 2 <= 2483.5e6, (fc, f_off)


def test_hw_errors_change_with_the_scenario_seed():
    """S1.4: heading bias and ovality ripple phase are drawn per scenario and node, not per node only."""
    hw = {"heading_datum": {"bias_deg_1sigma": 0.5}}
    b1, r1 = O.hw_errors(hw, 1, "S1")
    b2, r2 = O.hw_errors(hw, 2, "S1")
    b1b, r1b = O.hw_errors(hw, 1, "S1")
    b3, _ = O.hw_errors(hw, 1, "S2")
    assert b1 == b1b and np.allclose(r1, r1b)
    assert b1 != b2 and not np.allclose(r1, r2)
    assert b1 != b3
    assert torch.is_tensor(torch.as_tensor(r1)) and len(r1) == 4
