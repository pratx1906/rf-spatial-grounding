"""AIR7311 DF study at one node: Sionna RT paths -> receiver model -> MUSIC/Bartlett -> error maps and summary.

    python scripts/df_eval.py --sensor S4 [--quick]

Sweeps band x propagation setup x array size (calibrated), plus an inter-chip phase-offset sweep for the
band-matched array with full propagation. Writes runs/air7311_<sensor>/<combo>.npz maps and summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import add_allow_flag, data_roots, df_run_dir, guard_write  # noqa: E402
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402
from sim.df.air7311 import DFEngine, square_positions  # noqa: E402

SITE = REPO / "data" / "site_kandla"
MATCHED = {"ism_5800": "A_5g8", "ism_2400": "B_2g4", "sub_ghz_868_915": "C_sub1g"}
MATCH_DEG = 5.0


def ang_err(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def run_combo(eng: DFEngine, hw: dict, z: dict, sensor: dict, band: str, array: str, interchip: float,
              seed: int, batch: int, radius: float) -> tuple[dict, dict]:
    pos = square_positions(hw["antenna"]["candidate_sides_m"][array])
    f = float(z["f_hz"])
    bw = hw["emitter"]["signal_bw_mhz"][band] * 1e6
    nf = hw["receiver"]["noise_figure_db"]
    noise_w = 10 ** ((-174 + 10 * math.log10(bw) + nf - 30) / 10)
    p_scale = 10 ** ((hw["emitter"]["eirp_dbm"] - hw["receiver"]["cable_loss_db"] - 30) / 10)
    A = eng.scan_steering(pos, f)
    rng = torch.Generator(device=eng.dev).manual_seed(seed)
    imp = hw["impairments"]
    g_err = torch.randn(4, generator=rng, device=eng.dev) * imp["channel_gain_sigma_db"]
    p_err = torch.randn(4, generator=rng, device=eng.dev) * imp["channel_phase_sigma_deg"]
    n_snap = hw["processing"]["subbands"] * hw["processing"]["snapshots_per_subband"]
    xs, ys, hs = z["xs"], z["ys"], z["heights"]
    X, Y = np.meshgrid(xs, ys)
    rngm = np.hypot(X - sensor["x_m"], Y - sensor["y_m"])
    shape = (len(hs), len(ys), len(xs))
    out = {k: np.full(shape, np.nan, "f4") for k in ("err_music", "err_bart", "err_simple", "snr_db")}
    out["cls"] = np.full(shape, 255, "u1")          # 0 direct wins, 1 ghost wins, 2 spurious/ambiguous
    out["n_peaks"] = np.zeros(shape, "u1")
    lrng = np.random.default_rng(seed)
    for ih, hgt in enumerate(hs):
        h = z["h_re"][ih] + 1j * z["h_im"][ih]                                        # [Ny,Nx,K]
        valid = (z["ptype"][ih] != 255)
        tot = (np.abs(h) ** 2 * valid).sum(-1)
        snr = 10 * np.log10(np.maximum(p_scale * tot / noise_w, 1e-30))
        # exclude the cells next to the mast (bearing undefined straight above the node)
        sel = np.flatnonzero(((rngm <= radius) & (rngm >= 30.0) & (tot > 0)).ravel())
        true_az = (np.degrees(np.arctan2(X - sensor["x_m"], Y - sensor["y_m"]))) % 360
        hz = h.reshape(-1, h.shape[-1])
        for i in range(0, len(sel), batch):
            idx = sel[i:i + batch]
            iy, ix = np.unravel_index(idx, X.shape)
            Xs = eng.snapshots(hz[idx], z["tau_ns"][ih][iy, ix], z["az_deg"][ih][iy, ix], z["el_deg"][ih][iy, ix],
                               valid[iy, ix], pos, f, bw, p_scale, noise_w, rng, interchip, g_err, p_err)
            R = eng.covariance(Xs)
            bart, music = eng.spectra(R, A, n_snap)
            pk_az, _ = eng.rank_by_power(music, bart, hw["processing"]["peak_floor_db"])
            bk_az, _ = eng.peaks(bart, hw["processing"]["peak_floor_db"])
            ta = true_az[iy, ix]
            e_m = ang_err(pk_az[:, 0], ta)
            out["err_music"][ih, iy, ix] = e_m
            out["err_bart"][ih, iy, ix] = ang_err(bk_az[:, 0], ta)
            out["n_peaks"][ih, iy, ix] = np.sum(~np.isnan(pk_az), 1)
            out["snr_db"][ih, iy, ix] = snr[iy, ix]
            # which physical path does the winning peak correspond to?
            paz, ptp = z["az_deg"][ih][iy, ix], z["ptype"][ih][iy, ix]
            pw = np.abs(hz[idx]) ** 2
            strong = pw >= pw.max(1, keepdims=True) * 10 ** (-20 / 10)
            ghost_hit = np.any((ptp != 0) & (ptp != 255) & strong & (ang_err(paz, pk_az[:, :1]) <= MATCH_DEG), 1)
            direct = e_m <= MATCH_DEG
            out["cls"][ih, iy, ix] = np.where(direct, 0, np.where(ghost_hit, 1, 2))
            # previous simple model: strongest path direction + SNR-dependent Gaussian noise
            top = np.argmax(np.where(valid[iy, ix], pw, -1), 1)
            s_az = paz[np.arange(len(idx)), top]
            s_snr = snr[iy, ix]
            sig = 0.8 + 12.0 / (1.0 + 10 ** (s_snr / 20.0))
            out["err_simple"][ih, iy, ix] = ang_err(s_az + lrng.normal(0, sig), ta)
    det_snr = hw["processing"]["detect_snr_db"]
    summ = {}
    for ih, hgt in enumerate(hs):
        m = (out["snr_db"][ih] >= det_snr) & ~np.isnan(out["err_music"][ih])
        n_all = int(np.sum(~np.isnan(out["err_music"][ih])))
        if not m.any():
            summ[f"{int(hgt)}m"] = {"cells": n_all, "detected_pct": 0.0}
            continue
        em, eb, es, c = out["err_music"][ih][m], out["err_bart"][ih][m], out["err_simple"][ih][m], out["cls"][ih][m]
        summ[f"{int(hgt)}m"] = {
            "cells": n_all, "detected_pct": round(100 * m.sum() / max(n_all, 1), 1),
            "music_median_deg": round(float(np.median(em)), 2), "music_p90_deg": round(float(np.percentile(em, 90)), 2),
            "music_gt10_pct": round(100 * float(np.mean(em > 10)), 2),
            "bartlett_median_deg": round(float(np.median(eb)), 2), "bartlett_gt10_pct": round(100 * float(np.mean(eb > 10)), 2),
            "simple_model_gt10_pct": round(100 * float(np.mean(es > 10)), 2),
            "direct_wins_pct": round(100 * float(np.mean(c == 0)), 2), "ghost_wins_pct": round(100 * float(np.mean(c == 1)), 2),
            "spurious_pct": round(100 * float(np.mean(c == 2)), 2),
            "mean_peaks": round(float(np.mean(out["n_peaks"][ih][m])), 2),
        }
    return out, summ


@exit_if_busy
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="S4")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--quick", action="store_true", help="one combo, for a smoke test")
    ap.add_argument("--stack", choices=["v1", "v2"], default="v1",
                    help="v2: rf_cache_v2 caches, DF model-error floor -20 dB, band-matched arrays only (RF phase 2)")
    add_allow_flag(ap)
    a = ap.parse_args()
    hw = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
    if a.stack == "v2":
        hw["processing"]["model_error_floor_db"] = -20.0
    sensor = next(s for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
                  if s["id"] == a.sensor)["pose"]
    if a.stack == "v1":
        cache = SITE / "rf_cache" / f"air7311_{a.sensor}"
        out_dir = REPO / "runs" / f"air7311_{a.sensor}"
    else:                               # data version (SCENESCOUT_DATA_VERSION): rf_cache_v2|_v3 -> runs/..._v2|_v3
        cache = data_roots()["rf_cache"] / f"air7311_{a.sensor}"
        out_dir = guard_write(df_run_dir(a.sensor), a.allow_demo_roots)
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = DFEngine(hw)
    combos = []
    for band in ("ism_2400", "ism_5800", "sub_ghz_868_915"):
        for setup in ("los", "spec", "spec_diff"):
            for array in hw["antenna"]["candidate_sides_m"]:
                combos.append((band, setup, array, 0.0))
        for ic in hw["impairments"]["interchip_phase_deg_cases"][1:]:
            combos.append((band, "spec_diff", MATCHED[band], ic))
    if a.quick:
        combos = [("ism_2400", "spec_diff", "B_2g4", 0.0)]
    elif a.stack == "v2":
        combos = [(b, "spec_diff", MATCHED[b], 2.0) for b in ("ism_2400", "ism_5800", "sub_ghz_868_915")]
    summary = {"sensor": a.sensor, "pose": sensor, "hardware": "configs/hardware/air7311.yaml", "stack": a.stack,
               "processing": hw["processing"], "combos": {}}
    with gpu_lock("df_sim"):
        loaded = {}
        for band, setup, array, ic in combos:
            key = (setup, band)
            if key not in loaded:
                loaded = {key: dict(np.load(cache / setup / f"{band}.npz"))}
            t0 = time.time()
            maps, summ = run_combo(eng, hw, loaded[key], sensor, band, array, ic, seed=7, batch=a.batch,
                                   radius=a.radius_m)
            name = f"{band}__{setup}__{array}__ic{ic:g}"
            np.savez_compressed(out_dir / f"{name}.npz", **maps)
            summary["combos"][name] = {"band": band, "setup": setup, "array": array, "interchip_deg": ic,
                                       "seconds": round(time.time() - t0, 1), "by_height": summ}
            print(f"[df] {name}: {round(time.time() - t0, 1)} s  40m: {summ.get('40m')}", flush=True)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
