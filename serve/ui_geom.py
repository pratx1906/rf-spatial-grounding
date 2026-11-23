"""Map geometry for UI events (never sent to the model): episode-scoped results the console must draw.

weak_zones -> polygons; effect_footprint -> footprint outline, graded level bands (threshold +20/+10/+0 dB) and the
effector; collateral_check -> victim positions (assets, emitters, zones, own DF sensors);
predict_aoa / fit_candidates -> positions of the objects named as reflectors. ENU metres; the browser converts.
"""

from __future__ import annotations

import numpy as np
import shapely


def _ring(g, tol: float = 10.0) -> list[list[list[float]]]:
    g = g.simplify(tol)
    polys = list(g.geoms) if hasattr(g, "geoms") else [g]
    return [[[round(x, 1), round(y, 1)] for x, y in p.exterior.coords] for p in polys if p.geom_type == "Polygon"]


def _pos(site, ref: str) -> list[float] | None:
    kind, _, rid = ref.partition(":")
    if kind == "obj" and rid in site.objects:
        c = site.objects[rid].geom.representative_point()
        return [round(c.x, 1), round(c.y, 1)]
    if kind == "zone" and rid in site.zones:
        g = site.zones[rid].geom
        c = g.representative_point()
        return [round(c.x, 1), round(c.y, 1)]
    if kind == "sensor":
        v = site.sensors.get(rid) or site.sensors.get(rid.upper())
        return [round(v["pose"]["x_m"], 1), round(v["pose"]["y_m"], 1)] if v else None
    if kind == "fe":
        e = next((e for e in site.friendly if e["id"] == rid), None)
        return [e["x_m"], e["y_m"]] if e else None
    return None


def _mask_rings(m, xs, ys, tol: float = 25.0) -> list[list[list[float]]]:
    """Grid cells above a level as display rings: cells merged, small gaps closed, specks dropped, edges simplified at
    25 m so outlines read as zones rather than pixel traces."""
    iy, ix = np.nonzero(m)
    if not len(ix):
        return []
    step = float(xs[1] - xs[0])
    boxes = shapely.box(xs[ix] - step / 2, ys[iy] - step / 2, xs[ix] + step / 2, ys[iy] + step / 2)
    g = shapely.union_all(boxes).buffer(step, 4).buffer(-step, 4)     # display: close gaps between cells
    parts = [q for q in (g.geoms if hasattr(g, "geoms") else [g]) if q.area >= 6 * step * step]
    return _ring(shapely.union_all(parts), max(tol, step / 2)) if parts else []


def ui_extras(env, name: str, res: dict) -> dict | None:
    if not res.get("ok"):
        return None
    ep = env.executor.ctx.episode
    site = env.site
    d = res.get("data") or {}
    if name == "weak_zones":
        ras = ep.get("ras", {})
        return {"zones": [{"ref": z["ref"], "rings": _ring(ras[z["ref"]]["geom"]), "worst": z["worst"]}
                          for z in d.get("zones", []) if z["ref"] in ras]}
    if name == "effect_footprint":
        fp = ep.get("fp", {}).get(d.get("ref"))
        if fp is None:
            return None
        xs, ys = fp["xs"], fp["ys"]
        if not fp["mask"].any():
            return {"rings": [], "bands": []}
        thr = float(fp["threshold_dbm"])
        bands = []
        for lv in (thr + 20.0, thr + 10.0, thr):          # inner (strongest) first; the outer band is the footprint
            m = fp["mask"] if lv == thr else (fp["rx"] >= lv)
            bands.append({"level_dbm": round(lv, 1), "rings": _mask_rings(m, xs, ys)})
        out = {"rings": bands[-1]["rings"], "bands": bands}
        eff = fp["effector"]
        out["effector"] = _pos(site, eff) if isinstance(eff, str) and eff.startswith("obj") else (
            [site.sensors[eff.split(":")[1]]["pose"]["x_m"], site.sensors[eff.split(":")[1]]["pose"]["y_m"]]
            if isinstance(eff, str) else [eff["x_m"], eff["y_m"]])
        return out
    if name == "collateral_check":
        return {"victims": [{"ref": v["ref"], "pos": _pos(site, v["ref"]), "kind": v["kind"],
                             "severity": v["severity"], "level_dbm": v["predicted_level_dbm"],
                             "name": v.get("victim"), "note": v.get("note")}
                            for v in d.get("violations", [])]}
    if name in ("predict_aoa", "fit_candidates"):
        vias = {r for r in res.get("refs", []) if r.startswith("obj:")}
        return {"vias": {r: _pos(site, r) for r in vias}}
    return None
