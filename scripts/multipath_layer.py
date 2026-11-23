"""Per-sensor multipath layer for the operator console (web product, CPU, offline).

    python scripts/multipath_layer.py            (WSL, .venv-wsl)  -> ui/kandla/data/multipath/{index.json,<band>_h<H>.json}

Reads the demo-stack AIR7311 node caches (rf_cache_v2/air7311_<S>/spec_diff/<band>.npz: top-24 Sionna paths per
20 m cell, transmitter at the node by reciprocity) and the belief maps (rf_cache_v2/maps/<band>.npz, D-031). Nothing
is written into the read-only demo roots; only the nominal and belief-side caches are read (never world_cache).

Per cell, only where the drone is detectable (strongest path >= DETECT_DB) and inside the node's 1.5 km footprint:
  n_echo      number of other paths within 10 dB of the strongest (echoes strong enough to pull a bearing)
  not_direct  1 if the strongest path is not line of sight (specular or diffracted)
  spread_deg  power-weighted circular spread of the path azimuths (paths within 20 dB of the strongest)
  score       0.4 * min(n_echo / 4, 1) + 0.3 * not_direct + 0.3 * min(spread_deg / 30, 1)      (0 clean .. 1 heavy)
Hot spots: cells in the top 5 % of the 3x3-smoothed score per sensor (and >= MIN_HOT), 8-connected clusters of
>= MIN_CELLS cells, ranked by summed score; up to TOP_N per sensor with centroid, area, outline rings (holes filled)
and a plain-language label.
Simulation only: a Sionna RT model of the Blender port geometry, synthetic sensor sites.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
RF = SITE / "rf_cache_v2"                       # the demo stack (data/scenarios/kandla_s0_X/meta.yaml rf_cache_dir)
OUT = REPO / "ui" / "kandla" / "data" / "multipath"
BANDS = ("sub_ghz_868_915", "ism_2400", "ism_5800")
BAND_LABEL = {"sub_ghz_868_915": "868/915 MHz", "ism_2400": "2.4 GHz", "ism_5800": "5.8 GHz"}
HEIGHTS = (5.0, 40.0)
SENSORS = ("S1", "S2", "S3", "S4", "S5")
DETECT_DB = -115.0
ECHO_DB = 10.0
SPREAD_DB = 20.0
FOOTPRINT_M = 1500.0
HOT_FRAC = 0.05
MIN_HOT = 0.35
MIN_CELLS = 3
TOP_N = 5
NODATA = 255


# ------------------------------------------------------------------ scoring
def cell_metrics(h_re, h_im, az_deg, ptype) -> dict[str, np.ndarray]:
    """Path arrays [..., K] -> per-cell metrics [...]. Cells with no path get NaN metrics and p0_db = -inf."""
    pw = (h_re.astype("f8") ** 2 + h_im.astype("f8") ** 2) * (ptype != 255)
    p0 = pw.max(-1)
    i0 = pw.argmax(-1)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p0_db = 10.0 * np.log10(p0)
        rel = 10.0 * np.log10(pw / p0[..., None])                       # dB below the strongest (-inf for no path)
    valid = p0 > 0
    n_echo = (rel >= -ECHO_DB).sum(-1) - 1                              # the strongest itself is not an echo
    not_direct = np.take_along_axis(ptype, i0, -1)[..., 0] != 0
    w = np.where(rel >= -SPREAD_DB, pw, 0.0)
    rad = np.radians(np.nan_to_num(az_deg.astype("f8")))
    c, s = (w * np.cos(rad)).sum(-1), (w * np.sin(rad)).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.clip(np.hypot(c, s) / w.sum(-1), 1e-12, 1.0)
    spread = np.degrees(np.sqrt(-2.0 * np.log(R)))
    score = 0.4 * np.minimum(n_echo / 4.0, 1.0) + 0.3 * not_direct + 0.3 * np.minimum(spread / 30.0, 1.0)
    nan = lambda x: np.where(valid, x, np.nan)  # noqa: E731
    return {"p0_db": np.where(valid, p0_db, -np.inf), "n_echo": nan(n_echo.astype("f8")),
            "not_direct": nan(not_direct.astype("f8")), "spread_deg": nan(spread), "score": nan(score)}


def label_cells(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels (0 = background), no scipy needed."""
    lab = np.zeros(mask.shape, "i4")
    n = 0
    ny, nx = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        stack.append((yy, xx))
    return lab, n


def smooth3(score: np.ndarray) -> np.ndarray:
    """NaN-aware 3x3 mean over the finite cells (NaN cells stay NaN): breaks score plateaus and single-cell speckle."""
    ok = np.isfinite(score)
    v = np.pad(np.where(ok, score, 0.0), 1)
    n = np.pad(ok.astype("f8"), 1)
    ny, nx = score.shape
    s = sum(v[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    c = sum(n[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, s / c, np.nan)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Cells not reachable from the border through False cells (4-connected) become True."""
    ny, nx = mask.shape
    outside = np.zeros(mask.shape, bool)
    stack = [(y, x) for y in range(ny) for x in (0, nx - 1)] + [(y, x) for x in range(nx) for y in (0, ny - 1)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < ny and 0 <= x < nx and not mask[y, x] and not outside[y, x]:
            outside[y, x] = True
            stack += [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    return ~outside


def outline(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Closed rings of cell-corner indices (ix, iy) around the True cells (corner (ix, iy) = lower-left of cell)."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for iy, ix in zip(*np.nonzero(mask)):
        corners = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]      # counter-clockwise
        nbr = [(iy - 1, ix), (iy, ix + 1), (iy + 1, ix), (iy, ix - 1)]           # the cell across each edge
        for k, (ny_, nx_) in enumerate(nbr):
            inside = 0 <= ny_ < mask.shape[0] and 0 <= nx_ < mask.shape[1] and mask[ny_, nx_]
            if not inside:
                edges.setdefault(corners[k], []).append(corners[(k + 1) % 4])
    rings = []
    while edges:
        start = next(iter(edges))
        ring, cur = [start], start
        while True:
            nxt = edges[cur].pop()
            if not edges[cur]:
                del edges[cur]
            if nxt == start:
                break
            ring.append(nxt)
            cur = nxt
        keep = [p for i, p in enumerate(ring)                                    # drop collinear corners
                if (ring[i - 1][0] - p[0]) * (ring[(i + 1) % len(ring)][1] - p[1])
                != (ring[i - 1][1] - p[1]) * (ring[(i + 1) % len(ring)][0] - p[0])]
        rings.append(keep)
    return rings


def hot_clusters(score: np.ndarray, frac: float = HOT_FRAC, min_hot: float = MIN_HOT, min_cells: int = MIN_CELLS,
                 top_n: int = TOP_N) -> list[dict]:
    """Top-`frac` score cells (finite only), 8-connected, >= min_cells; ranked by summed score.
    Returns [{cells: (iy, ix) arrays, n, mean, sum, cy, cx (score-weighted centroid, cell index), rings}]."""
    ok = np.isfinite(score)
    if not ok.any():
        return []
    thr = max(float(np.quantile(score[ok], 1.0 - frac)), min_hot)
    hot = ok & (score >= thr)
    lab, n = label_cells(hot)
    out = []
    for k in range(1, n + 1):
        iy, ix = np.nonzero(lab == k)
        if len(iy) < min_cells:
            continue
        w = score[iy, ix]
        out.append({"iy": iy, "ix": ix, "n": len(iy), "mean": float(w.mean()), "sum": float(w.sum()),
                    "cy": float((iy * w).sum() / w.sum()), "cx": float((ix * w).sum() / w.sum()),
                    "rings": outline(fill_holes(lab == k)), "thr": thr})
    out.sort(key=lambda c: -c["sum"])
    return out[:top_n]


# ------------------------------------------------------------------ plain labels
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass(dx: float, dy: float) -> str:
    return COMPASS[int(((math.degrees(math.atan2(dx, dy)) % 360.0) + 22.5) // 45) % 8]


def landmarks() -> list[dict]:
    """Named site objects to anchor labels: tank farms, jetties, substations, silos (no synthetic ships)."""
    import pandas as pd
    from shapely import wkt

    o = pd.read_parquet(SITE / "objects.parquet")
    out = []
    for r in o.itertuples():
        if r.kind in ("tank_farm", "jetty", "substation", "silo", "tank", "building"):
            out.append({"id": r.id, "name": r.name, "kind": r.kind, "geom": wkt.loads(r.wkt),
                        "major": r.kind in ("tank_farm", "jetty", "substation", "silo")})
    return out


def object_names() -> dict[str, str]:
    import pandas as pd

    o = pd.read_parquet(SITE / "objects.parquet")
    return dict(zip(o["id"], o["name"]))


def place_label(x: float, y: float, sensor: str, sx: float, sy: float, marks: list[dict]) -> str:
    from shapely.geometry import Point

    pt = Point(x, y)
    d = math.hypot(x - sx, y - sy) / 1000.0
    where = f"{d:.1f} km {compass(x - sx, y - sy)} of {sensor}"
    for major, near_m in ((True, 200.0), (False, 100.0)):       # a tank farm or jetty first, else one tank or building
        best = min((m for m in marks if m["major"] == major), key=lambda m: m["geom"].distance(pt), default=None)
        if best is None:
            continue
        dist = best["geom"].distance(pt)
        if dist == 0.0:
            return f"{best['name']}, {where}"
        if dist <= near_m:
            return f"Near {best['name']}, {where}"
    return where[0].upper() + where[1:]


def severity(mean_score: float) -> str:
    return "heavy" if mean_score >= 0.75 else "moderate" if mean_score >= 0.5 else "light"


def reflector_name(raw: str, names: dict[str, str]) -> str | None:
    """Blender RF object name 'tank_t044__steel' -> site name; imagery extras get a generic word."""
    base = raw.split("__")[0]
    if base in names:
        return names[base]
    if base.startswith("KW_tower"):
        return "a tower"
    if base.startswith("KW_mlbuildings") or base.startswith("KW_"):
        return "buildings (from imagery)"
    return None


# ------------------------------------------------------------------ build
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _maps_at(band: str) -> dict | None:
    f = RF / "maps" / f"{band}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in ("xs", "ys", "heights", "sensors", "ghost_risk", "trust")}


def build_sensor(sensor: str, band: str, height: float, pose: dict, marks, names, maps) -> dict | None:
    f = RF / f"air7311_{sensor}" / "spec_diff" / f"{band}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    hs = z["heights"]
    ih = int(np.argmin(np.abs(hs - height)))
    m = cell_metrics(z["h_re"][ih], z["h_im"][ih], z["az_deg"][ih], z["ptype"][ih])
    xs, ys = z["xs"], z["ys"]
    gx, gy = np.meshgrid(xs, ys)
    inside = np.hypot(gx - pose["x_m"], gy - pose["y_m"]) <= FOOTPRINT_M
    det = inside & (m["p0_db"] >= DETECT_DB)
    score = np.where(det, m["score"], np.nan)
    hot_score = smooth3(score)                       # hot spots rank the smoothed score; the drape shows the raw cells
    grid =np.where(det, np.clip(np.rint(np.nan_to_num(score) * 250), 0, 250), NODATA).astype("u1")
    dx = float(xs[1] - xs[0])
    pw = z["h_re"][ih].astype("f8") ** 2 + z["h_im"][ih].astype("f8") ** 2
    i0 = pw.argmax(-1)
    via0 = np.take_along_axis(z["via"][ih], i0[..., None], -1)[..., 0]
    rf_names = z["names"]
    si = None if maps is None else int(np.nonzero(maps["sensors"] == sensor)[0][0])
    mh = None if maps is None else int(np.argmin(np.abs(maps["heights"] - height)))
    clusters = []
    for rank, c in enumerate(hot_clusters(hot_score), 1):
        cx = float(xs[0] + c["cx"] * dx)
        cy = float(ys[0] + c["cy"] * dx)
        iy, ix = c["iy"], c["ix"]
        vias = [int(v) for v in via0[iy, ix] if v >= 0]
        refl = None
        if vias:
            raw = str(rf_names[Counter(vias).most_common(1)[0][0]])
            refl = reflector_name(raw, names)
        ctx = {}
        if maps is not None:
            mx = np.clip(np.rint((xs[ix] - maps["xs"][0]) / (maps["xs"][1] - maps["xs"][0])).astype(int), 0, len(maps["xs"]) - 1)
            my = np.clip(np.rint((ys[iy] - maps["ys"][0]) / (maps["ys"][1] - maps["ys"][0])).astype(int), 0, len(maps["ys"]) - 1)
            g = maps["ghost_risk"][si, mh, my, mx]
            t = maps["trust"][si, mh, my, mx]
            if np.isfinite(g).any():
                ctx = {"ghost_risk_mean": round(float(np.nanmean(g)), 2), "trust_mean": round(float(np.nanmean(t)), 2)}
        ring_xy = [[[round(float(xs[0] + (px - 0.5) * dx), 1), round(float(ys[0] + (py - 0.5) * dx), 1)] for px, py in r]
                   for r in c["rings"]]
        clusters.append({
            "rank": rank, "centroid": [round(cx, 1), round(cy, 1)], "cells": c["n"],
            "area_km2": round(c["n"] * dx * dx / 1e6, 3), "mean_score": round(c["mean"], 2),
            "severity": severity(c["mean"]), "dist_km": round(math.hypot(cx - pose["x_m"], cy - pose["y_m"]) / 1000.0, 2),
            "dir": compass(cx - pose["x_m"], cy - pose["y_m"]),
            "label": place_label(cx, cy, sensor, pose["x_m"], pose["y_m"], marks),
            "echoes_mean": round(float(np.nanmean(m["n_echo"][iy, ix])), 1),
            "not_direct_frac": round(float(np.nanmean(m["not_direct"][iy, ix])), 2),
            "spread_deg_mean": round(float(np.nanmean(m["spread_deg"][iy, ix])), 1),
            "main_reflector": refl, **ctx, "rings": ring_xy})
    sc = hot_score[np.isfinite(hot_score)]
    return {"sensor": sensor, "pose": [pose["x_m"], pose["y_m"], pose["z_m"]], "height_m": float(hs[ih]),
            "nx": len(xs), "ny": len(ys), "x0": float(xs[0]), "y0": float(ys[0]), "dx": dx, "dy": float(ys[1] - ys[0]),
            "detectable_cells": int(det.sum()),
            "hot_threshold": round(max(float(np.quantile(sc, 1 - HOT_FRAC)), MIN_HOT) if sc.size else MIN_HOT, 2),
            "score_u8": _b64(grid), "clusters": clusters}


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    poses = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
             if s["type"] == "fixed_df"}
    marks, names = landmarks(), object_names()
    files = []
    summary = {}
    for band in BANDS:
        maps = _maps_at(band)
        for h in HEIGHTS:
            per = [r for s in SENSORS if (r := build_sensor(s, band, h, poses[s], marks, names, maps))]
            name = f"{band}_h{int(h)}.json"
            (out / name).write_text(json.dumps({"band": band, "band_label": BAND_LABEL[band], "height_m": h,
                                                "sensors": per}, separators=(",", ":")))
            files.append({"band": band, "height_m": h, "file": name})
            summary[f"{band}@{int(h)}m"] = {r["sensor"]: (r["clusters"][0]["label"] if r["clusters"] else None) for r in per}
    (out / "index.json").write_text(json.dumps({
        "product": "multipath", "stack": RF.name, "source": f"data/site_kandla/{RF.name}/air7311_<S>/spec_diff/<band>.npz",
        "simulation": True,
        "method": {"detect_db": DETECT_DB, "echo_db": ECHO_DB, "spread_db": SPREAD_DB, "footprint_m": FOOTPRINT_M,
                   "score": "0.4*min(echoes/4,1) + 0.3*strongest_not_direct + 0.3*min(spread_deg/30,1)",
                   "score_u8": "score*250 (255 = not detectable or outside the node footprint); rows south to north",
                   "hot_spots": f"top {int(HOT_FRAC * 100)} % cells of the 3x3-smoothed score (>= {MIN_HOT}), "
                                f"8-connected, >= {MIN_CELLS} cells, holes filled, top {TOP_N} by summed score"},
        "bands": [{"id": b, "label": BAND_LABEL[b]} for b in BANDS], "heights_m": list(HEIGHTS),
        "sensors": list(SENSORS), "files": files}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
