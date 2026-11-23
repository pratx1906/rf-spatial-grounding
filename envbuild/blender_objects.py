"""Site object catalogue from the Blender world (RF phase 2; STRICT: Blender is the only production geometry).

v2 scenarios describe the port to the agent's tools and the console with the Blender world's objects:
  * objects present in both (tanks, OSM buildings, jetties, substations, silos, tank farms; the Blender build names
    them with the twin's ids) keep their OSM footprint (the Blender meshes are built on it) and take the MODELLED
    height (top of the Blender mesh above the quay level);
  * ships are the Sentinel-2-observed vessels of the Blender world (`ship_obs_*`, estimated; hull outline = convex
    hull of the mesh, height = top above mean sea level); the twin's synthetic tankers are dropped;
  * mobile harbour cranes and yard machinery stay out of the catalogue (ILLUSTRATIVE, RF geometry only).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint

REPO = Path(__file__).resolve().parents[1]
BL = REPO / "data" / "site_kandla" / "rf_scene_blender"
MSL_M = -3.5


def _ply_xyz(p: Path) -> np.ndarray:
    raw = p.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    nv = int(next(x for x in head.decode().splitlines() if x.startswith("element vertex")).split()[-1])
    return np.frombuffer(body[: nv * 12], "<f4").reshape(nv, 3)


def catalogue(objs: list) -> tuple[list, dict]:
    m = json.loads((BL / "objects.json").read_text())
    tops: dict[str, float] = {}
    shapes_of: dict[str, list[str]] = {}
    for r in m["shapes"]:
        if r["group"] != "structure":
            continue
        tops[r["object"]] = max(tops.get(r["object"], -1e9), r["bbox"][5])
        shapes_of.setdefault(r["object"], []).append(r["shape"])
    out, stats = [], {"height_from_blender": 0, "kept_twin_only": 0, "ships_observed": 0, "ships_dropped": 0}
    for o in objs:
        if o.kind == "ship":
            stats["ships_dropped"] += 1
            continue
        if o.id in tops:
            out.append(replace(o, height_m=round(float(tops[o.id]), 2)))
            stats["height_from_blender"] += 1
        else:
            out.append(o)
            stats["kept_twin_only"] += 1
    Obj = type(objs[0])
    for name in sorted(n for n in shapes_of if n.startswith("ship_obs_")):
        xyz = np.concatenate([_ply_xyz(BL / "meshes" / f"{s}.ply") for s in shapes_of[name]])
        hull = MultiPoint([tuple(p) for p in xyz[:, :2][:: max(1, len(xyz) // 2000)]]).convex_hull
        k = name.rsplit("_", 1)[-1]
        out.append(Obj(id=name, name=f"Observed ship {k}", kind="ship", geom=hull,
                       height_m=round(float(xyz[:, 2].max() - MSL_M), 1), material="steel",
                       aliases=[f"ship {k}", f"vessel {k}"], tags=["sentinel2_observed", "estimated"]))
        stats["ships_observed"] += 1
    return out, stats
