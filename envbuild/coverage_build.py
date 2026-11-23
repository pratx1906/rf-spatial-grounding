"""Belief-side coverage and siting products from the AIR7311 DF study and the 2.5D site (offline).

    python -m envbuild.coverage_build        (after scripts/df_eval.py --sensor S1..S5)

1. `rf_cache/coverage/<band>.npz`: one 20 m grid over the union of the node caches (heights 5/15/40/80 m) holding,
   per node, the predicted SNR (dB) and the AIR7311 DF bearing error (deg, MUSIC with the band-matched array,
   2 deg inter-chip residual, `spec_diff` propagation) where the node detects (SNR >= 6 dB), else NaN.
2. `rf_cache/siting/candidates.npz`: mount candidates (roof masts on large buildings, jetty-head masts) with a 2.5D
   line-of-sight mask over a 40 m grid at 15 m and 40 m drone heights within 1.5 km (a hypothetical node has no
   Sionna cache; LOS is the documented proxy used by `suggest_node`).
Both are NOMINAL model products (what the agent may know), like `df_err.zarr` in the SPEC 4 contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from shapely import STRtree, wkt

from sim.geometry import los_clear

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
RUNS = REPO / "runs"
STACK = {"rf": "rf_cache", "run_suffix": ""}      # --stack v2: rf_cache_v2 caches and runs/air7311_<S>_v2 maps
MATCHED = {"ism_5800": "A_5g8", "ism_2400": "B_2g4", "sub_ghz_868_915": "C_sub1g"}
DETECT_SNR_DB = 6.0
STEP = 20.0
SIT_STEP = 40.0
SIT_HEIGHTS = (15.0, 40.0)
RADIUS = 1500.0


def sensors() -> dict:
    return {s["id"]: s for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"] if s["type"] == "fixed_df"}


def coverage() -> None:
    sens = sensors()
    out = SITE / STACK["rf"] / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    xs_all, ys_all = [], []
    grids = {}
    for sid in sens:
        z = np.load(SITE / STACK["rf"] / f"air7311_{sid}" / "spec_diff" / "ism_2400.npz")
        grids[sid] = (z["xs"], z["ys"], z["heights"])
        xs_all += [z["xs"][0], z["xs"][-1]]
        ys_all += [z["ys"][0], z["ys"][-1]]
    xs = np.arange(min(xs_all), max(xs_all) + STEP / 2, STEP)
    ys = np.arange(min(ys_all), max(ys_all) + STEP / 2, STEP)
    heights = grids[next(iter(grids))][2]
    for band, arr in MATCHED.items():
        ids = sorted(sens)
        snr = np.full((len(ids), len(heights), len(ys), len(xs)), np.nan, np.float32)
        err = np.full_like(snr, np.nan)
        for k, sid in enumerate(ids):
            f = RUNS / f"air7311_{sid}{STACK['run_suffix']}" / f"{band}__spec_diff__{arr}__ic2.npz"
            if not f.exists():
                f = RUNS / f"air7311_{sid}{STACK['run_suffix']}" / f"{band}__spec_diff__{arr}__ic0.npz"
            m = np.load(f)
            gx, gy, _ = grids[sid]
            ix0 = int(round((gx[0] - xs[0]) / STEP))
            iy0 = int(round((gy[0] - ys[0]) / STEP))
            s_ = m["snr_db"]
            e_ = np.where(s_ >= DETECT_SNR_DB, m["err_music"], np.nan)
            snr[k, :, iy0:iy0 + len(gy), ix0:ix0 + len(gx)] = s_
            err[k, :, iy0:iy0 + len(gy), ix0:ix0 + len(gx)] = e_
        np.savez_compressed(out / f"{band}.npz", xs=xs, ys=ys, heights=heights, sensors=np.array(ids),
                            snr_db=snr, df_err_deg=err)
        print(band, snr.shape)
    (out / "manifest.json").write_text(json.dumps({
        "source": "scripts/df_eval.py maps (AIR7311 engine, order-2 MUSIC, spec_diff Sionna paths)",
        "grid_m": STEP, "detect_snr_db": DETECT_SNR_DB, "sensors": sorted(sens)}, indent=1))


def siting() -> None:
    sens = sensors()
    rows = pq.read_table(SITE / "objects.parquet").to_pylist()
    objs = [(wkt.loads(r["wkt"]), float(r["height_m"] or 0)) for r in rows
            if r["kind"] not in ("tank_farm", "road") and (r["height_m"] or 0) > 1]
    tree = STRtree([g for g, _ in objs])
    cands = []
    for r in rows:
        if str(r["mount_candidate"]) != "True":
            continue
        g = wkt.loads(r["wkt"])
        c = g.representative_point()
        z = 25.0 if r["kind"] == "jetty" else float(r["height_m"]) + 4.0
        if min(math.hypot(c.x - s["pose"]["x_m"], c.y - s["pose"]["y_m"]) for s in sens.values()) < 100:
            continue                                                    # already a DF node there
        cands.append({"id": r["id"], "name": r["name"], "kind": r["kind"], "x": c.x, "y": c.y, "z": z})
    cov = np.load(SITE / STACK["rf"] / "coverage" / "ism_2400.npz")
    xs = np.arange(cov["xs"][0], cov["xs"][-1] + 1, SIT_STEP)
    ys = np.arange(cov["ys"][0], cov["ys"][-1] + 1, SIT_STEP)
    los = np.zeros((len(cands), len(SIT_HEIGHTS), len(ys), len(xs)), np.uint8)
    X, Y = np.meshgrid(xs, ys)
    for k, c in enumerate(cands):
        near = np.hypot(X - c["x"], Y - c["y"]) <= RADIUS
        for ih, h in enumerate(SIT_HEIGHTS):
            for iy, ix in zip(*np.nonzero(near), strict=True):
                los[k, ih, iy, ix] = los_clear((c["x"], c["y"], c["z"]), (X[iy, ix], Y[iy, ix], h), objs, tree)
        print(c["id"], int(los[k].sum()))
    out = SITE / STACK["rf"] / "siting"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "candidates.npz", xs=xs, ys=ys, heights=np.array(SIT_HEIGHTS), los=los,
                        ids=np.array([c["id"] for c in cands]), xyz=np.array([[c["x"], c["y"], c["z"]] for c in cands]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["v1", "v2"], default="v1")
    from envbuild.variant_roots import add_allow_flag, data_roots, df_run_dir, guard_write

    add_allow_flag(ap)
    a = ap.parse_args()
    if a.stack == "v2":                 # data version (SCENESCOUT_DATA_VERSION): rf_cache_v2|_v3, runs/air7311_<S>_v2|_v3
        STACK.update(rf=data_roots()["rf_cache"].name, run_suffix=df_run_dir("S1").name.removeprefix("air7311_S1"))
        for sub in ("coverage", "siting"):
            guard_write(SITE / STACK["rf"] / sub, a.allow_demo_roots)
    coverage()
    siting()
