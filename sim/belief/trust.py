"""BELIEF side trust layer (main spec 4A.3), from the nominal model only.

Two propagation variants of the same Sionna scene exist near each DF node (AIR7311 caches, 1.5 km): `spec`
(specular only) and `spec_diff` (specular + edge diffraction). Where they disagree about the strongest arrival, the
nominal prediction is sensitive to modelling choices and should be trusted less. Multipath richness (how much of
the energy is outside the strongest path) lowers trust too. Outside the node caches only the site-wide cache exists:
trust is capped at OUTSIDE_CAP because no variant comparison is possible.

When the belief ensemble maps exist (rf_cache/maps/<band>.npz, envbuild.rf_maps, D-031), trust comes from them
instead: the spread of the strongest arrival across the nominal model and three perturbed belief variants (tide,
berths, materials) and the fraction of members whose strongest arrival is off the geometric bearing (ghost risk).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sim.geometry import angle_diff_deg

OUTSIDE_CAP = 0.6
SPREAD_SCALE_DEG = 10.0
_CACHE: dict[tuple, dict | None] = {}


def _node(rf_dir: Path, sensor: str, setup: str, band: str) -> dict | None:
    key = (str(rf_dir), sensor, setup, band)
    if key not in _CACHE:
        f = Path(rf_dir) / f"air7311_{sensor}" / setup / f"{band}.npz"
        if not f.exists():
            _CACHE[key] = None
        else:
            z = np.load(f)
            _CACHE[key] = {k: z[k] for k in ("xs", "ys", "heights", "az_deg", "h_re", "h_im", "ptype")}
    return _CACHE[key]


def _strongest(c: dict, x: float, y: float, z: float) -> tuple[float, float] | None:
    xs, ys, hs = c["xs"], c["ys"], c["heights"]
    if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
        return None
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    ih = int(np.argmin(np.abs(hs - z)))
    ok = c["ptype"][ih, iy, ix] != 255
    if not ok.any():
        return None
    pw = (c["h_re"][ih, iy, ix] ** 2 + c["h_im"][ih, iy, ix] ** 2) * ok
    k = int(np.argmax(pw))
    return float(c["az_deg"][ih, iy, ix, k]), float(pw[k] / max(pw.sum(), 1e-30))


def _inside(c: dict, x: float, y: float) -> bool:
    xs, ys = c["xs"], c["ys"]
    return bool(xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10)


def _maps(rf_dir: Path, band: str) -> dict | None:
    key = (str(rf_dir), "maps", band)
    if key not in _CACHE:
        f = Path(rf_dir) / "maps" / f"{band}.npz"
        if f.exists():
            z = np.load(f)
            _CACHE[key] = {k: z[k] for k in ("xs", "ys", "heights", "sensors", "trust", "ghost_risk", "az_spread_deg",
                                             "nlos_frac")}
        else:
            _CACHE[key] = None
    return _CACHE[key]


def _from_maps(m: dict, sensor: str, x: float, y: float, z: float) -> tuple[float, float, float] | None:
    names = [str(s) for s in m["sensors"]]
    if sensor not in names:
        return None
    xs, ys, hs = m["xs"], m["ys"], m["heights"]
    if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
        return None
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    ih = int(np.argmin(np.abs(hs - z)))
    si = names.index(sensor)
    t = float(m["trust"][si, ih, iy, ix])
    if not math.isfinite(t):
        return None
    return t, float(m["ghost_risk"][si, ih, iy, ix]), float(m["az_spread_deg"][si, ih, iy, ix])


def per_sensor_at(rf_dir: Path, band: str, x: float, y: float, z: float) -> list[dict]:
    """Every node's ensemble-map values at a point (nearest map cell and height), best trust first. Empty when no
    map is built or the point is outside every node footprint."""
    m = _maps(rf_dir, band)
    if m is None:
        return []
    xs, ys, hs = m["xs"], m["ys"], m["heights"]
    if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
        return []
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    ih = int(np.argmin(np.abs(hs - z)))
    rows = []
    for si, s in enumerate(str(v) for v in m["sensors"]):
        t = float(m["trust"][si, ih, iy, ix])
        if not math.isfinite(t):
            continue
        rows.append({"sensor": s, "ghost_risk": round(float(m["ghost_risk"][si, ih, iy, ix]), 3),
                     "nlos_frac": round(float(m["nlos_frac"][si, ih, iy, ix]), 3),
                     "spread_deg": round(float(m["az_spread_deg"][si, ih, iy, ix]), 1), "trust": round(t, 2),
                     "map_height_m": float(hs[ih])})
    return sorted(rows, key=lambda r: -r["trust"])


def trust_at(rf_dir: Path, sensor: str, band: str, x: float, y: float, z: float) -> dict:
    """{trust, spread_deg, dominance, flags, backends}. trust in [0, 1]."""
    m = _maps(rf_dir, band)
    if m is not None:
        hit = _from_maps(m, sensor, x, y, z)
        if hit is not None:
            t, ghost, spread = hit
            b = _node(rf_dir, sensor, "spec_diff", band)
            sb = _strongest(b, x, y, z) if b else None
            dom = sb[1] if sb else None
            flags = []
            if spread > 5:
                flags.append("ensemble_spread")
            if ghost >= 0.5:
                flags.append("ghost_prone")
            if dom is not None and dom < 0.5:
                flags.append("rich_multipath")
            return {"trust": round(t, 2), "spread_deg": round(spread, 1),
                    "dominance": round(dom, 2) if dom is not None else None, "flags": flags,
                    "backends": ["sionna_rt:nominal", "ensemble:belief_variants"]}
    a = _node(rf_dir, sensor, "spec", band)
    b = _node(rf_dir, sensor, "spec_diff", band)
    sa = _strongest(a, x, y, z) if a else None
    sb = _strongest(b, x, y, z) if b else None
    if a is None or b is None or (sa is None and sb is None and not _inside(b, x, y)):
        return {"trust": OUTSIDE_CAP, "spread_deg": None, "dominance": None,
                "flags": ["outside_node_cache: no variant comparison, nominal model only"], "backends": ["sionna_rt"]}
    if sa is None or sb is None:
        return {"trust": 0.3, "spread_deg": None, "dominance": None,
                "flags": ["no_paths: the node model predicts no path to this cell in at least one variant"],
                "backends": ["sionna_rt:spec", "sionna_rt:spec_diff"]}
    spread = abs(angle_diff_deg(sa[0], sb[0]))
    dom = sb[1]
    trust = math.exp(-spread / SPREAD_SCALE_DEG) * (0.5 + 0.5 * min(1.0, dom / 0.7))
    flags = []
    if spread > 5:
        flags.append("variants_disagree")
    if dom < 0.5:
        flags.append("rich_multipath")
    return {"trust": round(trust, 2), "spread_deg": round(spread, 1), "dominance": round(dom, 2), "flags": flags,
            "backends": ["sionna_rt:spec", "sionna_rt:spec_diff"]}
