"""Family E (SPEC 6.2 E): effect_footprint and collateral_check. ASSESSMENT ONLY: no tool performs mitigation.

effect_footprint predicts where a hypothetical effector (e.g. a jammer co-located with a DF node) would put more
than `threshold_dbm` into a receiver, from the nominal Sionna model by reciprocity (paths node <-> cell, 40 m cells,
nearest cached height). A free-standing point effector uses free-space loss (no site geometry) with lower trust.
collateral_check lists what the footprint touches: protected site objects, friendly emitters and civil comms on
overlapping bands (from the site's register `friendly_emitters.yaml`), and aviation zones.
"""

from __future__ import annotations

import hashlib
import math
from typing import Annotated, Literal

from pathlib import Path

import numpy as np
import shapely
from pydantic import Field

from sim.geometry import fspl_db
from tools.df import predictor, xyz_of
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _entity
from tools.types import Band, Strict, pt, r1
from tools.types import Point as PointArg

Kinds = Literal["protected_asset", "friendly_emitter", "civil_comms", "aviation", "own_sensor"]
CIVIL = {"lte_site", "vhf_base"}


class EffectFootprintArgs(Strict):
    effector: str | PointArg
    band: Band
    eirp_dbm: Annotated[float, Field(ge=0, le=60)]
    threshold_dbm: Annotated[float, Field(ge=-130, le=0)] = -90.0
    height_m: Annotated[float, Field(ge=0, le=400)] = 5.0


def _rx_grid(ctx: ToolContext, a: EffectFootprintArgs):
    pred = predictor(ctx)
    if isinstance(a.effector, str):
        kind, ent, _ = _entity(ctx, a.effector)
        if kind != "sensor":
            raise ToolError("INVALID_ARGS", "effector must be a sensor: ref (co-located effector) or a point",
                            "Pass sensor:S1 or {x_m, y_m, z_m}.")
        c = pred.cache(a.band, ent["id"])
        if c is None:
            raise ToolError("OUT_OF_COVERAGE", f"no propagation model for {a.band} at sensor:{ent['id']}",
                            "Use sub_ghz_868_915, ism_2400 or ism_5800.")
        ih = int(np.argmin(np.abs(c["heights"] - a.height_m)))
        g = np.asarray(c["gain_db"][ih])
        ok = np.asarray(c["ptype"][ih]) != 255
        lin = np.where(ok, 10 ** (g / 10), 0.0).sum(-1)
        rx = a.eirp_dbm + 10 * np.log10(np.maximum(lin, 1e-30))
        return rx, c["xs"], c["ys"], float(c["heights"][ih]), f"sensor:{ent['id']}", "sionna_rt", 0.8, \
            ent["pose"]["x_m"], ent["pose"]["y_m"]
    x0, y0, z0, _ = xyz_of(ctx, a.effector)
    c = next((pred.cache(a.band, s) for s, v in ctx.site.sensors.items()
              if v["type"] == "fixed_df" and pred.cache(a.band, s) is not None), None)
    if c is None:
        raise ToolError("OUT_OF_COVERAGE", f"no propagation grid for {a.band}",
                        "Use sub_ghz_868_915, ism_2400 or ism_5800.")
    xs, ys = c["xs"], c["ys"]
    X, Y = np.meshgrid(xs, ys)
    d = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (a.height_m - z0) ** 2)
    f = float(np.mean(ctx.site.bands[a.band]))
    rx = a.eirp_dbm - np.vectorize(lambda v: fspl_db(v, f))(d)
    return rx, xs, ys, a.height_m, None, "freespace", 0.4, x0, y0


@register("effect_footprint", "E", "Predicted footprint (fp: ref) where an effector exceeds a received-power threshold.",
          EffectFootprintArgs)
def effect_footprint(ctx: ToolContext, a: EffectFootprintArgs) -> ToolOutput:
    rx, xs, ys, h, eref, src, trust, ex, ey = _rx_grid(ctx, a)
    mask = rx >= a.threshold_dbm
    step = float(xs[1] - xs[0])
    key = repr((str(a.effector), a.band, a.eirp_dbm, a.threshold_dbm, h))
    ref = "fp:" + hashlib.sha1(key.encode()).hexdigest()[:8]
    iy, ix = np.nonzero(mask)
    area = float(mask.sum()) * step * step
    rng = float(np.max(np.hypot(xs[ix] - ex, ys[iy] - ey))) if len(ix) else 0.0
    ctx.episode.setdefault("fp", {})[ref] = {"rx": rx, "xs": xs, "ys": ys, "mask": mask, "band": a.band,
                                              "threshold_dbm": a.threshold_dbm, "eirp_dbm": a.eirp_dbm,
                                              "effector": eref or pt(ex, ey), "height_m": h, "model_source": src}
    bbox = [r1(xs[ix].min() - step / 2), r1(ys[iy].min() - step / 2), r1(xs[ix].max() + step / 2),
            r1(ys[iy].max() + step / 2)] if len(ix) else None
    return ToolOutput({"ref": ref, "band": a.band, "eirp_dbm": a.eirp_dbm, "threshold_dbm": a.threshold_dbm,
                       "height_m": h, "area_m2": round(area), "max_range_m": r1(rng), "bbox": bbox,
                       "model_source": src, "trust": trust}, [ref] + ([eref] if eref else []),
                      {"footprint": f"40 m cells at {h:g} m receiver height where predicted received power >= "
                                    "threshold (isotropic antennas)"},
                      ["free-space model: ignores buildings and tanks"] if src == "freespace" else [])


class CollateralCheckArgs(Strict):
    fp_ref: str
    kinds: list[Kinds] | None = None


def _level(fp: dict, x: float, y: float) -> float:
    xs, ys = fp["xs"], fp["ys"]
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    return float(fp["rx"][iy, ix])


def _geom_level(fp: dict, geom) -> float:
    xs, ys = fp["xs"], fp["ys"]
    minx, miny, maxx, maxy = geom.bounds
    jx = np.flatnonzero((xs >= minx - 20) & (xs <= maxx + 20))
    jy = np.flatnonzero((ys >= miny - 20) & (ys <= maxy + 20))
    best = _level(fp, geom.representative_point().x, geom.representative_point().y)
    if len(jx) and len(jy):
        X, Y = np.meshgrid(xs[jx], ys[jy])
        inside = shapely.contains_xy(geom.buffer(20), X, Y)
        if inside.any():
            best = max(best, float(fp["rx"][np.ix_(jy, jx)][inside].max()))
    return best


def _overlap(site, a: str, b: str) -> bool:
    ra, rb = site.bands[a], site.bands[b]
    return ra[0] < rb[1] and rb[0] < ra[1]


def _sev(margin: float) -> str:
    return "high" if margin >= 20 else "medium" if margin >= 10 else "low"


DEFAULT_KINDS = {"protected_asset", "friendly_emitter", "civil_comms", "aviation"}


def _own_sensors(ctx: ToolContext, fp: dict, kinds: set) -> list[dict]:
    """Fratricide (RF phase 2): our own DF nodes desensitised by the effector. Uses the traced node-to-node links
    (rf_cache/node_links.json, v2 belief caches only, so v1 answers are unchanged). Desense = 10 log10(1 + I/N),
    N = kTB F in the band's typical signal bandwidth; the transmitting node itself is blinded (limiter / T-R)."""
    import json
    import math

    f = Path(ctx.site.rf_dir) / "node_links.json"
    if not f.exists() or ("own_sensor" not in kinds and kinds != DEFAULT_KINDS):
        return []
    links = json.loads(f.read_text())["links"].get(fp["band"], {})
    bw = {"sub_ghz_868_915": 0.5e6, "ism_2400": 20e6, "ism_5800": 20e6}[fp["band"]]
    n_dbm = -174 + 10 * math.log10(bw) + 3.0
    src = fp["effector"] if isinstance(fp["effector"], str) else None
    out = []
    for sid, v in ctx.site.sensors.items():
        if v["type"] != "fixed_df":
            continue
        if src == f"sensor:{sid}":
            out.append({"ref": f"sensor:{sid}", "kind": "own_sensor", "severity": "high", "predicted_level_dbm": 99.0,
                        "predicted_desense_db": None, "note": "transmitting node: receiver blanked (limiter / T-R switch)"})
            continue
        if src and f"{src[7:]}->{sid}" in links:
            lv = fp["eirp_dbm"] - 2.0 + links[f"{src[7:]}->{sid}"]["gain_db"]
        else:
            lv = _level(fp, v["pose"]["x_m"], v["pose"]["y_m"])
        des = 10 * math.log10(1 + 10 ** ((lv - n_dbm) / 10))
        if des >= 10:
            out.append({"ref": f"sensor:{sid}", "kind": "own_sensor", "severity": _sev(des - 10),
                        "predicted_level_dbm": r1(lv), "predicted_desense_db": r1(des)})
    return out


@register("collateral_check", "E", "What a footprint would hit: assets, friendly emitters, comms, aviation, own DF sensors.", CollateralCheckArgs)
def collateral_check(ctx: ToolContext, a: CollateralCheckArgs) -> ToolOutput:
    fp = ctx.episode.get("fp", {}).get(a.fp_ref.strip())
    if fp is None:
        raise ToolError("UNKNOWN_REF", f"{a.fp_ref} was not produced in this episode",
                        "Call effect_footprint first and pass its fp: ref.")
    kinds = set(a.kinds or DEFAULT_KINDS)
    thr = fp["threshold_dbm"]
    site = ctx.site
    out = []
    if "protected_asset" in kinds:
        for o in site.objects.values():
            if not o.protected or o.kind not in ("tank_farm", "jetty", "substation"):
                continue
            lv = _geom_level(fp, o.geom)
            if lv >= thr:
                out.append({"ref": o.ref, "kind": "protected_asset", "severity": _sev(lv - thr),
                            "predicted_level_dbm": r1(lv)})
    for e in site.friendly:
        k = "civil_comms" if e["kind"] in CIVIL else "friendly_emitter"
        if k not in kinds or not _overlap(site, e["band"], fp["band"]):
            continue
        lv = _level(fp, e["x_m"], e["y_m"])
        if lv >= thr:
            out.append({"ref": f"fe:{e['id']}", "kind": k, "severity": _sev(lv - thr),
                        "predicted_level_dbm": r1(lv), "victim": e["name"]})
    if "aviation" in kinds:
        for z in site.zones.values():
            if z.kind != "aviation":
                continue
            lv = _geom_level(fp, z.geom)
            if lv >= thr:
                out.append({"ref": z.ref, "kind": "aviation", "severity": _sev(lv - thr),
                            "predicted_level_dbm": r1(lv)})
    out += _own_sensors(ctx, fp, kinds)
    out.sort(key=lambda v: -v["predicted_level_dbm"])
    # full list (no 20-row cap) for the verifier; a second check with other kinds adds to it, never hides earlier finds
    seen = ctx.episode.setdefault("collateral", {}).get(a.fp_ref.strip(), [])
    ctx.episode["collateral"][a.fp_ref.strip()] = list(dict.fromkeys(seen + [v["ref"] for v in out]))
    return ToolOutput({"fp_ref": a.fp_ref, "violations": out[:20], "n_violations": len(out),
                       "checked": sorted(kinds), "requires_human_confirm": True},
                      [a.fp_ref] + [v["ref"] for v in out[:20]],
                      {"violation": "predicted level at the asset >= footprint threshold; friendly and civil "
                                    "emitters only on overlapping bands; severity by margin (>=20 dB high, >=10 medium)",
                       "requires_human_confirm": "always: the agent recommends, the operator decides"})
