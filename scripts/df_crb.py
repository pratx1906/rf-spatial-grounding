"""AIR7311 DF engine vs the Cramer-Rao bound (validation case C; WSL, GPU).

    python scripts/df_crb.py [--n 3000]  -> runs/rfval/df_crb.json

One line-of-sight path per cell, random azimuth, elevation 0, at 2.44 GHz with the matched 0.45-lambda square
array. SNR is the engine's own definition (received power over kTB + NF in the signal bandwidth, per element);
N = subbands x snapshots_per_subband = 256 snapshots. Three conditions: the engine as configured (residual gain
0.3 dB, phase 1 deg, inter-chip 2 deg), impairments off, and the bound. Floors expected without noise: the 0.5 deg
scan grid (uniform quantisation, 0.144 deg RMS) and, with impairments, the calibration residual.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import sim.obs_model.observe_air7311 as obs  # noqa: E402
from sim.df.air7311 import square_positions  # noqa: E402
from sim.rf import refs  # noqa: E402


def run(hw: dict, snr_db: float, n: int, seed: int, interchip: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n)
    band = "ism_2400"
    nw = obs.noise_w(hw, band)
    p_w = 10 ** ((20.0 - hw["receiver"]["cable_loss_db"] - 30) / 10)
    g = 10 ** (snr_db / 10) * nw / p_w                                       # path power gain giving this SNR
    K = 2
    h = np.zeros((n, K), complex)
    h[:, 0] = math.sqrt(g)
    tau = np.zeros((n, K))
    tau[:, 0] = 1000.0
    azs = np.stack([az, np.zeros(n)], 1)
    el = np.zeros((n, K))
    valid = np.zeros((n, K), bool)
    valid[:, 0] = True
    old = obs.INTERCHIP_DEG
    obs.INTERCHIP_DEG = interchip
    try:
        pk = obs.air7311_peaks(hw, band, h, tau, azs, el, valid, np.full(n, 20.0), seed=seed, max_peaks=1)
    finally:
        obs.INTERCHIP_DEG = old
    err = np.array([abs((p[0]["az_deg"] - a + 180) % 360 - 180) if p else np.nan for p, a in zip(pk, az)])
    return err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    hw = yaml.safe_load(obs.HW_FILE.read_text())
    ideal = copy.deepcopy(hw)
    ideal["impairments"]["channel_gain_sigma_db"] = 0.0
    ideal["impairments"]["channel_phase_sigma_deg"] = 0.0
    lam = refs.C0 / 2.44e9
    pos = square_positions(hw["antenna"]["candidate_sides_m"]["B_2g4"])[:, :2]
    n_snap = hw["processing"]["subbands"] * hw["processing"]["snapshots_per_subband"]
    grid = hw["processing"]["az_step_deg"] / math.sqrt(12)
    rows = []
    floored = copy.deepcopy(hw)
    floored["processing"]["model_error_floor_db"] = -20.0
    for snr in (7, 8, 10, 15, 20, 25, 30, 40):
        crb = refs.crb_bearing_deg(pos, lam, 0.0, snr, n_snap)
        e_cfg = run(hw, snr, a.n, 1, obs.INTERCHIP_DEG)
        e_ideal = run(ideal, snr, a.n, 2, 0.0)
        e_floor = [run(floored, snr, a.n, sd, obs.INTERCHIP_DEG) for sd in (1, 2, 3)]
        e_cfg3 = [e_cfg] + [run(hw, snr, a.n, sd, obs.INTERCHIP_DEG) for sd in (2, 3)]
        rms = lambda e: float(np.sqrt(np.nanmean(e ** 2)))  # noqa: E731
        rows.append({"snr_db": snr, "crb_deg": round(crb, 4), "crb_plus_grid_deg": round(math.hypot(crb, grid), 4),
                     "rmse_ideal_deg": round(rms(e_ideal), 4), "rmse_configured_deg": round(rms(e_cfg), 4),
                     "rmse_configured_3draws_deg": [round(rms(e), 3) for e in e_cfg3],
                     "flip_share_configured": round(float(np.mean([np.nanmean(e > 90) for e in e_cfg3])), 4),
                     "rmse_floor20_3draws_deg": [round(rms(e), 3) for e in e_floor],
                     "flip_share_floor20": round(float(np.mean([np.nanmean(e > 90) for e in e_floor])), 4),
                     "p95_configured_deg": round(float(np.nanpercentile(e_cfg, 95)), 3),
                     "detected_share": round(float(np.isfinite(e_cfg).mean()), 3)})
        print(rows[-1], flush=True)
    noise_limited = [r for r in rows if r["snr_db"] <= 10]
    ok = all(r["rmse_ideal_deg"] <= 2 * r["crb_plus_grid_deg"] for r in noise_limited)
    out = {"array": "square 4, 0.45 lambda at 2.44 GHz", "n_snapshots": n_snap, "grid_floor_deg": round(grid, 4),
           "rows": rows, "pass": ok,
           "accept": "impairments off, SNR <= 10 dB: RMSE <= 2 x sqrt(CRB^2 + grid^2); higher SNR and the configured "
                     "engine are reported (calibration residuals set a floor above the bound by design)"}
    (REPO / "runs" / "rfval").mkdir(parents=True, exist_ok=True)
    (REPO / "runs" / "rfval" / "df_crb.json").write_text(json.dumps(out, indent=1))
    print("pass", ok)
    return 0


if __name__ == "__main__" and "--two-path" not in sys.argv:
    sys.exit(main())


def two_path(hw: dict, snr_db: float, rel_db: float, sep_deg: float, n: int, seed: int) -> dict:
    """Direct path + one coherent reflection rel_db weaker, sep_deg apart, 30 ns later: share of cells where the
    top peak is the direct path (<5 deg) and where a second peak lies within 5 deg of the reflection."""
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, n)
    band = "ism_2400"
    nw = obs.noise_w(hw, band)
    p_w = 10 ** ((20.0 - hw["receiver"]["cable_loss_db"] - 30) / 10)
    g = 10 ** (snr_db / 10) * nw / p_w
    h = np.zeros((n, 2), complex)
    h[:, 0] = math.sqrt(g)
    h[:, 1] = math.sqrt(g * 10 ** (rel_db / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    tau = np.stack([np.full(n, 1000.0), np.full(n, 1030.0)], 1)
    az2 = (az + sep_deg) % 360
    pk = obs.air7311_peaks(hw, band, h, tau, np.stack([az, az2], 1), np.zeros((n, 2)), np.ones((n, 2), bool),
                           np.full(n, 20.0), seed=seed, max_peaks=3)
    d = lambda x, y: abs((x - y + 180) % 360 - 180)  # noqa: E731
    top = np.mean([bool(p) and d(p[0]["az_deg"], a) < 5 for p, a in zip(pk, az)])
    sec = np.mean([any(d(q["az_deg"], b) < 5 for q in p[1:]) for p, b in zip(pk, az2)])
    return {"top_is_direct": round(float(top), 3), "reflection_found": round(float(sec), 3)}


if __name__ == "__main__" and "--two-path" in sys.argv:
    hw = yaml.safe_load(obs.HW_FILE.read_text())
    fl = copy.deepcopy(hw)
    fl["processing"]["model_error_floor_db"] = -20.0
    res = {}
    for snr in (15, 30):
        for rel in (-3, -6, -10):
            res[f"snr{snr}_rel{rel}"] = {"plain": two_path(hw, snr, rel, 40, 1500, 5), "floor20": two_path(fl, snr, rel, 40, 1500, 5)}
            print(f"snr{snr}_rel{rel}", res[f"snr{snr}_rel{rel}"], flush=True)
    (REPO / "runs" / "rfval" / "df_two_path.json").write_text(json.dumps(res, indent=1))
