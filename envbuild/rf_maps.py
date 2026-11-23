"""Belief-side RF maps from the trust ensemble (nominal + belief variants B1-B3, D-031).

    python -m envbuild.rf_maps          (WSL, CPU)  -> data/site_kandla/rf_cache/maps/<band>.npz

Per sensor, on its own AIR7311 node grid, across the ensemble members' strongest arrival:
  ghost_risk     fraction of members whose strongest azimuth is > 5 deg off the geometric sensor->cell bearing
                 (members with no path are excluded; with 4 members the values are steps of 0.25)
  nlos_frac      fraction of members whose strongest path is not line of sight (reflections on the geometric
                 bearing, e.g. sea or ground bounce, are NLOS but not ghosts)
  az_spread_deg  circular spread of the strongest azimuth across members
  trust          exp(-spread/10) * (1 - 0.6 * ghost_risk)
Then resampled (nearest cell, NaN outside the node's 1.5 km footprint) onto the coverage union grid
(rf_cache/coverage/<band>.npz), with site layers: site_ghost_risk = min over sensors, site_trust = max over
sensors (the best node decides). Only nominal and belief-ensemble caches are read: the world side is never used.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
RF = SITE / "rf_cache"                     # --stack v2: rf_cache_v2 (nominal + ensemble B1-B7)
BANDS = ("sub_ghz_868_915", "ism_2400", "ism_5800")
GHOST_DEG = 5.0
SPREAD_SCALE_DEG = 10.0
FOOTPRINT_M = 1500.0                     # AIR7311 precompute radius; the node grid is the square around it


def _strongest(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pw = (m["h_re"].astype("f8") ** 2 + m["h_im"].astype("f8") ** 2) * (m["ptype"] != 255)
    i = pw.argmax(-1)[..., None]
    valid = np.take_along_axis(pw, i, -1)[..., 0] > 0
    az = np.take_along_axis(m["az_deg"], i, -1)[..., 0].astype("f8")
    pt = np.take_along_axis(m["ptype"], i, -1)[..., 0]
    return az, pt, valid


def node_metrics(members: list[dict], geo_bearing: np.ndarray) -> dict[str, np.ndarray]:
    """members: node caches (same grid) with az_deg/h_re/h_im/ptype [H,Y,X,K]; geo_bearing [H,Y,X] (deg)."""
    az, pt, ok = (np.stack(x) for x in zip(*(_strongest(m) for m in members)))      # [M,H,Y,X]
    n = ok.sum(0).astype("f8")
    off = np.abs((az - geo_bearing[None] + 180.0) % 360.0 - 180.0) > GHOST_DEG
    with np.errstate(invalid="ignore", divide="ignore"):
        ghost = np.where(n > 0, (off & ok).sum(0) / n, np.nan)
        nlos = np.where(n > 0, ((pt != 0) & ok).sum(0) / n, np.nan)
        rad = np.radians(np.where(ok, az, 0.0))
        c, s = (np.cos(rad) * ok).sum(0), (np.sin(rad) * ok).sum(0)
        R = np.clip(np.hypot(c, s) / np.maximum(n, 1), 1e-12, 1.0)
        spread = np.where(n > 0, np.degrees(np.sqrt(-2.0 * np.log(R))), np.nan)
    trust = np.exp(-spread / SPREAD_SCALE_DEG) * (1.0 - 0.6 * ghost)
    return {"ghost_risk": ghost, "nlos_frac": nlos, "az_spread_deg": spread, "trust": trust}


def _load(f: Path) -> dict:
    z = np.load(f)
    return {k: z[k] for k in ("xs", "ys", "heights", "az_deg", "h_re", "h_im", "ptype")}


def _members(sensor: str, band: str) -> list[dict]:
    files = [RF / f"air7311_{sensor}" / "spec_diff" / f"{band}.npz"]
    files += sorted((RF / "ensemble").glob(f"*/air7311_{sensor}/spec_diff/{band}.npz"))
    return [_load(f) for f in files if f.exists()]


def build_band(band: str) -> dict:
    cov = np.load(RF / "coverage" / f"{band}.npz")
    xs, ys, hs = cov["xs"], cov["ys"], cov["heights"]
    sensors = [s for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"] if s["type"] == "fixed_df"]
    keys = ("ghost_risk", "nlos_frac", "az_spread_deg", "trust")
    out = {k: np.full((len(sensors), len(hs), len(ys), len(xs)), np.nan, "f4") for k in keys}
    n_members = []
    for si, s in enumerate(sensors):
        mem = _members(s["id"], band)
        n_members.append(len(mem))
        if not mem:
            continue
        g = mem[0]
        gx, gy = np.meshgrid(g["xs"], g["ys"])
        geo = (np.degrees(np.arctan2(gx - s["pose"]["x_m"], gy - s["pose"]["y_m"])) % 360.0)
        met = node_metrics(mem, np.broadcast_to(geo, (len(g["heights"]),) + geo.shape))
        step = g["xs"][1] - g["xs"][0]
        ix = np.rint((xs - g["xs"][0]) / step).astype(int)
        iy = np.rint((ys - g["ys"][0]) / step).astype(int)
        inx = (ix >= 0) & (ix < len(g["xs"]))
        iny = (iy >= 0) & (iy < len(g["ys"]))
        ih = np.array([int(np.argmin(np.abs(g["heights"] - h))) for h in hs])
        for k in keys:
            src = met[k][ih][:, np.clip(iy, 0, len(g["ys"]) - 1)][:, :, np.clip(ix, 0, len(g["xs"]) - 1)]
            mask = iny[:, None] & inx[None, :]
            mask &= np.hypot(xs[None, :] - s["pose"]["x_m"], ys[:, None] - s["pose"]["y_m"]) <= FOOTPRINT_M
            out[k][si] = np.where(mask[None], src, np.nan)
    with warnings.catch_warnings():                           # cells outside every footprint stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        site_ghost = np.nanmin(out["ghost_risk"], axis=0)
        site_trust = np.nanmax(out["trust"], axis=0)
    return {"xs": xs, "ys": ys, "heights": hs, "sensors": np.array([s["id"] for s in sensors]), **out,
            "site_ghost_risk": site_ghost.astype("f4"), "site_trust": site_trust.astype("f4"),
            "members": np.array(n_members)}


def main() -> None:
    import argparse

    global RF
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["v1", "v2"], default="v1")
    from envbuild.variant_roots import add_allow_flag, data_roots, guard_write

    add_allow_flag(ap)
    a = ap.parse_args()
    if a.stack == "v2":
        RF = data_roots()["rf_cache"]                 # data version (SCENESCOUT_DATA_VERSION): rf_cache_v2 | _v3
        guard_write(RF / "maps", a.allow_demo_roots)
    (RF / "maps").mkdir(parents=True, exist_ok=True)
    summary = {}
    for band in BANDS:
        m = build_band(band)
        np.savez_compressed(RF / "maps" / f"{band}.npz", **m)
        g = m["site_ghost_risk"]
        summary[band] = {"members_per_sensor": m["members"].tolist(),
                         "site_ghost_risk_mean": round(float(np.nanmean(g)), 3),
                         "site_trust_mean": round(float(np.nanmean(m["site_trust"])), 3),
                         "cells_covered": int(np.isfinite(g).sum())}
    (RF / "maps" / "manifest.json").write_text(json.dumps({"ensemble": f"nominal + {RF.name}/ensemble/*", "D": "D-031",
                                                           "ghost_deg": GHOST_DEG, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
