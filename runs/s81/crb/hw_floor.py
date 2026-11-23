"""Hardware / calibration error floor of the per-burst DF without multipath (CPU only).

Same per-burst chain as the v3 build (sim.obs_model.observe_v2) for ONE free-space plane wave (no multipath, no other
emitters, no front-end blockers): ideal array steering at the burst carrier -> residual mutual coupling C (per seed,
node) -> sim.rf.receiver.receive (thermal noise, per-channel gain/phase residuals from cal_residuals, 2 deg inter-chip
offset, LO phase noise, IQ imbalance, DC, 14-bit ADC) -> bursts.detect -> burst_cov -> burst_df with the node's
calibrated manifold C_hat (node_coupling_cal), heading bias 0. Conditions: 'ideal' (no coupling, no receiver
impairments, cal None: thermal noise + grid only) and 'hw' (all of the above). True in-band per-element SNR 25 dB,
2 MHz x 1 ms burst. Seeds 15, 16, 51-60 x nodes S1-S5 x 3 bands.
Output: runs/s81/crb/hw_floor.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from crb_lib import SIDES, crb_az_deg, square_positions, steer  # noqa: E402
from sim.df.air7311 import DFEngine  # noqa: E402
from sim.obs_model.observe_air7311 import HW_FILE  # noqa: E402
from sim.obs_model.observe_v2 import burst_df, cal_residuals, coupling, node_coupling_cal  # noqa: E402
from sim.rf import bursts as B  # noqa: E402
from sim.rf import receiver as rx  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
hw = yaml.safe_load(HW_FILE.read_text())
hw = {**hw, "processing": {**hw["processing"], "model_error_floor_db": -20.0}}
eng = DFEngine(hw, "cpu")
FS, n = 125e6, 1 << 19
WIN = {"sub_ghz_868_915": 895e6, "ism_2400": 2441.75e6, "ism_5800": 5775e6}
SEEDS = [15, 16] + list(range(51, 61))
TR = int(os.environ.get("CRB_HW_TRIALS", "4"))
SNR_DB, BW, DUR, T0, OFF = 25.0, 2e6, 1e-3, 1.5e-3, 5e6
p_n = 10 ** ((-174 + 10 * math.log10(FS) + hw["receiver"]["noise_figure_db"] - 30) / 10)
rng = np.random.default_rng(4242)
i0, i1 = int(T0 * FS), int((T0 + DUR) * FS)
fr = np.fft.fftfreq(i1 - i0, 1 / FS)
keep = np.abs(fr - OFF) <= BW / 2


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


res = {"ideal": {b: [] for b in WIN}, "hw": {b: [] for b in WIN}}
for seed in SEEDS:
    for sid in ("S1", "S2", "S3", "S4", "S5"):
        C = coupling(seed, sid)
        ge, pe = cal_residuals(hw, seed, sid)
        for band, f0 in WIN.items():
            pos = square_positions(SIDES[band])
            chat = torch.as_tensor(node_coupling_cal(hw, seed, sid, band, pos, f0), dtype=torch.complex64)
            for _ in range(TR):
                az, el = rng.uniform(0, 360), rng.uniform(2, 20)
                S = np.zeros(i1 - i0, complex)
                S[keep] = rng.normal(size=keep.sum()) + 1j * rng.normal(size=keep.sum())
                s = np.fft.ifft(S)
                s *= math.sqrt(10 ** (SNR_DB / 10) * p_n * BW / FS / np.mean(np.abs(s) ** 2))
                a = steer(math.radians(az), math.radians(el), pos, f0 + OFF)
                y = np.zeros((4, n), complex)
                y[:, i0:i1] = a[:, None] * s[None]
                for cond in ("ideal", "hw"):
                    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
                    if cond == "ideal":
                        x = torch.as_tensor(y + (np.asarray(torch.randn(4, n, generator=g)) + 1j * np.asarray(
                            torch.randn(4, n, generator=g))) * math.sqrt(p_n / 2), dtype=torch.complex64)
                        cal = None
                    else:
                        yt = torch.as_tensor(C @ y, dtype=torch.complex64)
                        x, _ = rx.receive(yt, FS, hw, g, interchip_deg=2.0, gain_err_db=ge, phase_err_deg=pe)
                        cal = chat
                    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6 and b.f_lo < OFF + BW and b.f_hi > OFF - BW
                          and b.t0 < T0 + DUR and b.t0 + b.dur > T0]
                    if not bs:
                        continue
                    b = max(bs, key=lambda q: q.dur * q.bw)
                    pk = burst_df(eng, B.burst_cov(x, FS, b), b, pos, f0, hw, 0.0, cal=cal)
                    if pk:
                        res[cond][band].append((float(wrap(pk[0][0] - az)), el, b.snr_db))
        print(seed, sid, {c: {b: len(v) for b, v in d.items()} for c, d in res.items()}, flush=True)

out = {}
for cond, d in res.items():
    for band, v in list(d.items()) + [("all", [r for vv in d.values() for r in vv])]:
        e = np.array([r[0] for r in v])
        out[f"{cond}/{band}/n"] = len(e)
        out[f"{cond}/{band}/err_med_deg"] = float(np.median(np.abs(e)))
        out[f"{cond}/{band}/err_rms_deg"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"{cond}/{band}/err_rsigma_deg"] = float(1.4826 * np.median(np.abs(e - np.median(e))))
        out[f"{cond}/{band}/err_p90_deg"] = float(np.percentile(np.abs(e), 90))
        out[f"{cond}/{band}/reported_snr_med_db"] = float(np.median([r[2] for r in v]))
out["crb_sto_deg_2g4_el10"] = float(crb_az_deg(0, 10, SIDES["ism_2400"], 2446.75e6, 10 ** (SNR_DB / 10), BW * DUR,
                                               stochastic=True))
(HERE / "hw_floor.json").write_text(json.dumps(out, indent=1, sort_keys=True))
for k, v in sorted(out.items()):
    print(k, round(v, 4) if isinstance(v, float) else v)
