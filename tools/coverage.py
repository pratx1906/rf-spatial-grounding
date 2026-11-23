"""Family C (SPEC 6.2 C): propagation, coverage, weak zones, node siting, model trust. Nominal model only.

Rasters come from `rf_cache/coverage/<band>.npz` (AIR7311 DF study per node, envbuild/coverage_build.py). Per cell:
- snr_db: best node SNR;
- detect_prob: P(at least one node detects), soft threshold at 6 dB;
- df_err_p90_deg (per cell): the 2-node DF error = second-smallest bearing error among detecting nodes (a fix needs
  two bearings), 90 deg where fewer than two nodes detect. `agg` then summarises the region (p90 by convention).
Weak zones and rasters get deterministic `ras:` refs; their geometry is kept in the episode for the verifier and UI.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Annotated, Literal

import numpy as np
import shapely
from pydantic import Field

from sim.belief.trust import per_sensor_at, trust_at
from tools.df import DEFAULT_Z_M, predictor, xyz_of
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _entity, region_geom
from tools.types import Band, Region, Strict, pt, r1
from tools.types import Point as PointArg

Metric = Literal["df_err_p90_deg", "snr_db", "detect_prob"]
DETECT_SNR_DB = 6.0
ERR_CAP_DEG = 90.0
NEW_NODE_LOS_ERR = (1.0, 1.0)        # deg: base + per km, AIR7311 LOS median error scale at matched array
_COV: dict[tuple[str, str], dict] = {}
_SIT: dict[str, dict] = {}


def cov(ctx: ToolContext, band: str) -> dict:
    key = (str(ctx.site.rf_dir), band)
    if key not in _COV:
        f = ctx.site.rf_dir / "coverage" / f"{band}.npz"
        if not f.exists():
            raise ToolError("OUT_OF_COVERAGE", f"no coverage model for {band}",
                            "Coverage exists for sub_ghz_868_915, ism_2400 and ism_5800.")
        z = np.load(f)
        _COV[key] = {k: z[k] for k in z.files}
        _COV[key]["sensors"] = [str(s) for s in _COV[key]["sensors"]]
    return _COV[key]


def _sensor_idx(ctx, c: dict, refs: list[str] | None) -> list[int]:
    if not refs:
        return list(range(len(c["sensors"])))
    out = []
    for r in refs:
        kind, ent, _ = _entity(ctx, r)
        if kind != "sensor" or ent["id"] not in c["sensors"]:
            raise ToolError("INVALID_ARGS", f"{r} is not a fixed DF node", f"Nodes: {', '.join(c['sensors'])}.")
        out.append(c["sensors"].index(ent["id"]))
    return out


def _h(c: dict, height_m: float) -> int:
    return int(np.argmin(np.abs(c["heights"] - height_m)))


def metric_grid(c: dict, metric: str, idx: list[int], ih: int, extra_err: np.ndarray | None = None) -> np.ndarray:
    snr = c["snr_db"][idx, ih]
    if metric == "snr_db":
        return np.nanmax(np.where(np.isnan(snr), -np.inf, snr), 0)
    if metric == "detect_prob":
        p = 1.0 / (1.0 + np.exp(-(np.nan_to_num(snr, nan=-60.0) - DETECT_SNR_DB) / 1.5))
        return 1.0 - np.prod(1.0 - p, 0)
    err = c["df_err_deg"][idx, ih]
    if extra_err is not None:
        err = np.concatenate([err, extra_err[None]], 0)
    e = np.sort(np.where(np.isnan(err), ERR_CAP_DEG, err), 0)
    return np.minimum(e[1] if e.shape[0] > 1 else e[0], ERR_CAP_DEG)


def _mask(c: dict, geom) -> np.ndarray:
    X, Y = np.meshgrid(c["xs"], c["ys"])
    return shapely.contains_xy(geom, X, Y)


def ras_ref(*parts) -> str:
    return "ras:" + "_".join(str(p) for p in parts[:1]) + "_" + hashlib.sha1(repr(parts).encode()).hexdigest()[:8]


def _store(ctx: ToolContext, ref: str, rec: dict) -> None:
    ctx.episode.setdefault("ras", {})[ref] = rec


AGG = {"mean": np.mean, "p10": lambda v: np.percentile(v, 10), "p90": lambda v: np.percentile(v, 90),
       "min": np.min, "max": np.max}


# --- prop_query ------------------------------------------------------------------------------------------------
class PropQueryArgs(Strict):
    sensor_ref: str
    point: PointArg
    band: Band


@register("prop_query", "C", "Path loss, number of paths, LOS flag and delay spread between a DF node and a point.",
          PropQueryArgs)
def prop_query(ctx: ToolContext, a: PropQueryArgs) -> ToolOutput:
    kind, ent, _ = _entity(ctx, a.sensor_ref)
    if kind != "sensor":
        raise ToolError("INVALID_ARGS", f"{a.sensor_ref} is not a sensor ref", "Pass sensor: refs, e.g. sensor:S1.")
    z = a.point.z_m if a.point.z_m is not None else DEFAULT_Z_M
    ps = predictor(ctx).paths(a.point.x_m, a.point.y_m, z, ent["id"], a.band)
    if ps is None:
        raise ToolError("OUT_OF_COVERAGE", "point outside the propagation model (+-3.2 km) or band not modelled",
                        "Query points inside the port area for sub_ghz_868_915, ism_2400 or ism_5800.")
    tr = trust_at(ctx.site.rf_dir, ent["id"], a.band, a.point.x_m, a.point.y_m, z)
    if not ps:
        return ToolOutput({"sensor": f"sensor:{ent['id']}", "point": pt(a.point.x_m, a.point.y_m, z), "n_paths": 0,
                           "los": False, "path_loss_db": None, "model_source": "sionna_rt", "trust": tr["trust"]},
                          [f"sensor:{ent['id']}"], warnings=["no propagation path (fully blocked)"])
    lin = np.array([10 ** (p["gain_db"] / 10) for p in ps])
    tau = np.array([p["tau_ns"] for p in ps])
    mt = float((lin * tau).sum() / lin.sum())
    ds = math.sqrt(max(float((lin * (tau - mt) ** 2).sum() / lin.sum()), 0.0))
    data = {"sensor": f"sensor:{ent['id']}", "point": pt(a.point.x_m, a.point.y_m, z),
            "path_loss_db": r1(-10 * math.log10(lin.sum())), "n_paths": len(ps),
            "los": any(p["kind"] == "direct" for p in ps), "strongest_kind": ps[0]["kind"],
            "delay_spread_ns": r1(ds), "model_source": "sionna_rt", "trust": tr["trust"]}
    return ToolOutput(data, [f"sensor:{ent['id']}"], {"path_loss_db": "-10 log10(sum of path gains), isotropic",
                                                      "los": "a direct path exists in the nominal model"})


# --- coverage_query --------------------------------------------------------------------------------------------
class CoverageQueryArgs(Strict):
    sensor_refs: Annotated[list[str], Field(max_length=8)] | None = None
    band: Band
    metric: Metric
    region: Region
    agg: Literal["mean", "p10", "p90", "min", "max"] = "mean"
    height_m: Annotated[float, Field(ge=0, le=400)] = 40.0


@register("coverage_query", "C", "Summarise DF coverage (snr, detect_prob, 2-node DF error) over a region.",
          CoverageQueryArgs)
def coverage_query(ctx: ToolContext, a: CoverageQueryArgs) -> ToolOutput:
    c = cov(ctx, a.band)
    idx = _sensor_idx(ctx, c, a.sensor_refs)
    geom = region_geom(ctx, a.region)
    m = _mask(c, geom)
    if not m.any():
        raise ToolError("OUT_OF_COVERAGE", "region is outside the coverage model (1.5 km around the DF nodes)",
                        "Use a region inside the port core, e.g. zone:hero_zone.")
    ih = _h(c, a.height_m)
    g = metric_grid(c, a.metric, idx, ih)
    v = g[m]
    v = v[np.isfinite(v)] if a.metric == "snr_db" else v
    two = metric_grid(c, "df_err_p90_deg", idx, ih)[m] < ERR_CAP_DEG
    ref = ras_ref(a.band, a.metric, float(c["heights"][ih]), tuple(idx))
    _store(ctx, ref, {"band": a.band, "metric": a.metric, "height_m": float(c["heights"][ih]),
                      "sensors": [c["sensors"][i] for i in idx]})
    data = {"metric": a.metric, "agg": a.agg, "value": r1(float(AGG[a.agg](v))) if len(v) else None,
            "height_m": float(c["heights"][ih]), "n_cells": int(m.sum()), "area_m2": round(float(m.sum()) * 400.0),
            "two_node_fix_fraction": round(float(two.mean()), 2), "sensors": [f"sensor:{c['sensors'][i]}" for i in idx],
            "ras_ref": ref, "model_source": "air7311_df_study"}
    if a.metric == "detect_prob" and data["value"] is not None:
        data["value"] = round(float(AGG[a.agg](v)), 3)
    return ToolOutput(data, [ref] + data["sensors"], {
        "df_err_p90_deg": "per 20 m cell: 2nd-best node bearing error (deg; 90 if < 2 nodes detect); agg over region",
        "detect_prob": "P(>= 1 node detects), soft 6 dB threshold", "snr_db": "best node SNR"})


# --- weak_zones ------------------------------------------------------------------------------------------------
class WeakZonesArgs(Strict):
    metric: Metric
    band: Band
    threshold: float
    region: Region
    limit: Annotated[int, Field(ge=1, le=10)] = 5
    height_m: Annotated[float, Field(ge=0, le=400)] = 40.0
    sensor_refs: Annotated[list[str], Field(max_length=8)] | None = None


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, bool)
    out = []
    H, W = mask.shape
    for i0, j0 in zip(*np.nonzero(mask), strict=True):
        if seen[i0, j0]:
            continue
        comp, q = [], deque([(i0, j0)])
        seen[i0, j0] = True
        while q:
            i, j = q.popleft()
            comp.append((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    q.append((a, b))
        out.append(comp)
    return out


@register("weak_zones", "C", "Connected areas where coverage is worse than a threshold, largest first (ras: refs).",
          WeakZonesArgs)
def weak_zones(ctx: ToolContext, a: WeakZonesArgs) -> ToolOutput:
    c = cov(ctx, a.band)
    idx = _sensor_idx(ctx, c, a.sensor_refs)
    geom = region_geom(ctx, a.region)
    inside = _mask(c, geom)
    if not inside.any():
        raise ToolError("OUT_OF_COVERAGE", "region is outside the coverage model (1.5 km around the DF nodes)",
                        "Use a region inside the port core, e.g. zone:hero_zone.")
    ih = _h(c, a.height_m)
    g = metric_grid(c, a.metric, idx, ih)
    bad = (g > a.threshold) if a.metric == "df_err_p90_deg" else (g < a.threshold)
    bad &= inside
    comps = sorted(components(bad), key=len, reverse=True)
    step = float(c["xs"][1] - c["xs"][0])
    zones = []
    for k, comp in enumerate(comps):
        if len(comp) < 3:
            break
        ii = np.array(comp)
        xs, ys = c["xs"][ii[:, 1]], c["ys"][ii[:, 0]]
        poly = shapely.union_all(shapely.box(xs - step / 2, ys - step / 2, xs + step / 2, ys + step / 2))
        vals = g[ii[:, 0], ii[:, 1]]
        worst = float(vals.max() if a.metric == "df_err_p90_deg" else vals.min())
        ref = ras_ref(f"wz{k}", a.band, a.metric, a.threshold, float(c["heights"][ih]), tuple(idx), geom.wkt[:200])
        simp = poly.simplify(step / 2)
        _store(ctx, ref, {"geom": simp, "band": a.band, "metric": a.metric})
        cen = simp.centroid
        zones.append({"ref": ref, "area_m2": round(float(poly.area)), "centroid": pt(cen.x, cen.y),
                      "worst": r1(worst), "bbox": [r1(v) for v in simp.bounds]})
        if len(zones) == a.limit:
            break
    total = float(bad.sum()) * step * step
    return ToolOutput({"metric": a.metric, "threshold": a.threshold, "height_m": float(c["heights"][ih]),
                       "zones": zones, "n_zones": len([q for q in comps if len(q) >= 3]),
                       "total_area_m2": round(total), "region_area_m2": round(float(inside.sum()) * step * step)},
                      [z["ref"] for z in zones], {"weak": f"{'above' if a.metric == 'df_err_p90_deg' else 'below'} "
                                                          f"threshold, 20 m cells, components of >= 3 cells"},
                      threshold_warnings(a.metric, a.threshold))


# SFT v1.4 guardrail (PLAN rf-convergence Task 7): v1.3 invented a 90 deg threshold when the question gave none, so
# almost nothing came back weak. Answered as asked, with a hint that redirects to the documented default.
DF_THRESHOLD_PLAUSIBLE_MAX_DEG = 45.0


def threshold_warnings(metric: str, threshold: float) -> list[str]:
    if metric == "df_err_p90_deg" and threshold > DF_THRESHOLD_PLAUSIBLE_MAX_DEG:
        return [f"implausible threshold {threshold:g} deg for df_err_p90_deg (90 deg only marks cells that fewer "
                f"than 2 nodes detect): if the operator gave no threshold, use the default 10 deg and re-call"]
    if metric == "detect_prob" and threshold > 1.0:
        return [f"detect_prob is 0..1, got {threshold:g}: pass a probability (e.g. 0.5 for 50 %; default 0.5)"]
    return []


# --- suggest_node ----------------------------------------------------------------------------------------------
class NodeConstraints(Strict):
    mount_kinds: list[Literal["building", "jetty"]] | None = None
    min_sep_m: Annotated[float, Field(ge=0, le=3000)] = 300.0


class SuggestNodeArgs(Strict):
    region: Region
    band: Band = "ism_2400"
    objective: Literal["df_err", "detect"] = "df_err"
    constraints: NodeConstraints = NodeConstraints()
    k: Annotated[int, Field(ge=1, le=5)] = 3
    height_m: Literal[15, 40] = 40


def sit(ctx: ToolContext) -> dict:
    key = str(ctx.site.rf_dir)
    if key not in _SIT:
        f = ctx.site.rf_dir / "siting" / "candidates.npz"
        if not f.exists():
            raise ToolError("OUT_OF_COVERAGE", "no siting model built", "Siting is unavailable for this site.")
        z = np.load(f)
        _SIT[key] = {k: z[k] for k in z.files}
    return _SIT[key]


@register("suggest_node", "C", "Rank mount sites for one more DF node by predicted coverage gain in a region.",
          SuggestNodeArgs)
def suggest_node(ctx: ToolContext, a: SuggestNodeArgs) -> ToolOutput:
    c = cov(ctx, a.band)
    s = sit(ctx)
    geom = region_geom(ctx, a.region)
    hs = list(s["heights"])
    ih_s = hs.index(float(a.height_m))
    ih_c = _h(c, a.height_m)
    X, Y = np.meshgrid(s["xs"], s["ys"])
    reg = shapely.contains_xy(geom, X, Y)
    if not reg.any():
        raise ToolError("OUT_OF_COVERAGE", "region outside the siting grid", "Use a region inside the port core.")
    # existing per-node errors resampled to the siting grid
    jx = np.clip(np.round((s["xs"] - c["xs"][0]) / (c["xs"][1] - c["xs"][0])).astype(int), 0, len(c["xs"]) - 1)
    jy = np.clip(np.round((s["ys"] - c["ys"][0]) / (c["ys"][1] - c["ys"][0])).astype(int), 0, len(c["ys"]) - 1)
    err_nodes = np.where(np.isnan(c["df_err_deg"][:, ih_c][:, jy][:, :, jx]), ERR_CAP_DEG,
                         c["df_err_deg"][:, ih_c][:, jy][:, :, jx])
    poses = [(v["pose"]["x_m"], v["pose"]["y_m"]) for v in ctx.site.sensors.values() if v["type"] == "fixed_df"]
    kinds = {str(i): ctx.site.objects[str(i)].kind for i in s["ids"] if str(i) in ctx.site.objects}

    def score(errs: np.ndarray) -> tuple[float, float]:
        e = np.sort(errs, 0)
        e2 = e[1] if len(e) > 1 else e[0]
        return float(np.mean(np.minimum(e2, ERR_CAP_DEG)[reg])), float(np.mean((e2 < 10.0)[reg]))

    base_err, base_fix = score(err_nodes)
    chosen, out = [], []
    cur = err_nodes
    for _ in range(a.k):
        best = None
        for j, oid in enumerate(s["ids"]):
            oid = str(oid)
            if oid not in kinds or oid in [q["mount_ref"][4:] for q in out]:
                continue
            if a.constraints.mount_kinds and kinds[oid] not in a.constraints.mount_kinds:
                continue
            x, y, _z = s["xyz"][j]
            seps = [math.hypot(x - px, y - py) for px, py in poses + chosen]
            if min(seps) < a.constraints.min_sep_m:
                continue
            rng = np.hypot(X - x, Y - y) / 1000.0
            new = np.where(s["los"][j, ih_s] > 0, NEW_NODE_LOS_ERR[0] + NEW_NODE_LOS_ERR[1] * rng, ERR_CAP_DEG)
            e, f = score(np.concatenate([cur, new[None]], 0))
            gain = (base_err - e) if a.objective == "df_err" else (f - base_fix)
            if best is None or gain > best[0]:
                best = (gain, j, oid, new, e, f, min(seps))
        if best is None:
            break
        gain, j, oid, new, e, f, sep = best
        cur = np.concatenate([cur, new[None]], 0)
        chosen.append((float(s["xyz"][j][0]), float(s["xyz"][j][1])))
        out.append({"ref": f"nodecand:{oid}", "mount_ref": f"obj:{oid}", "position": pt(*s["xyz"][j]),
                    "predicted_gain": r1(max(gain, 0.0)) if a.objective == "df_err" else round(max(gain, 0.0), 3),
                    "region_mean_df_err_after_deg": r1(e), "two_node_fix_fraction_after": round(f, 3),
                    "min_sep_to_existing_m": r1(sep)})
        ctx.episode.setdefault("nodecand", {})[f"nodecand:{oid}"] = out[-1]
    return ToolOutput({"objective": a.objective, "band": a.band, "height_m": a.height_m,
                       "baseline": {"region_mean_df_err_deg": r1(base_err), "two_node_fix_fraction": round(base_fix, 3)},
                       "candidates": out, "model_source": "air7311_df_study + 2.5D LOS proxy for new nodes"},
                      [q["ref"] for q in out] + [q["mount_ref"] for q in out],
                      {"predicted_gain": "cumulative with the candidates above it. df_err: drop in region mean 2-node DF error (deg); detect: rise in fraction "
                                         "of cells with a 2-node fix under 10 deg",
                       "new_node_model": "LOS cells get 1 deg + 1 deg/km error; NLOS cells no bearing (proxy)"})


# --- model_trust -----------------------------------------------------------------------------------------------
class ModelTrustArgs(Strict):
    target: PointArg | Region | None = None
    track_ref: str | None = Field(None, description="trk:<id>@<t>: the track estimate at that time, instead of target")
    sensor_ref: str | None = None
    band: Band
    per_sensor: bool = Field(False, description="also list every node's ghost risk, NLOS share, spread and trust")


@register("model_trust", "C", "Trust (0..1) of the RF model at a point, region or track; per node: ghost risk.",
          ModelTrustArgs)
def model_trust(ctx: ToolContext, a: ModelTrustArgs) -> ToolOutput:
    sens = [s for s, v in ctx.site.sensors.items() if v["type"] == "fixed_df"]
    if a.sensor_ref:
        kind, ent, _ = _entity(ctx, a.sensor_ref)
        if kind != "sensor":
            raise ToolError("INVALID_ARGS", f"{a.sensor_ref} is not a sensor ref", "Pass sensor: refs.")
        sens = [ent["id"]]
    if (a.target is None) == (a.track_ref is None):
        raise ToolError("INVALID_ARGS", "give exactly one of target or track_ref",
                        "Use track_ref='trk:<id>@<t>' for a drone, or target for a point/region.")
    subject_refs: list[str] = []
    if a.track_ref is not None:
        x, y, z, canon = xyz_of(ctx, a.track_ref)
        pts = [(x, y, z)]
        subject_refs = [canon] if canon else []
    elif isinstance(a.target, PointArg):
        pts = [(a.target.x_m, a.target.y_m, a.target.z_m if a.target.z_m is not None else DEFAULT_Z_M)]
    else:
        g = region_geom(ctx, a.target)
        minx, miny, maxx, maxy = g.bounds
        xs, ys = np.linspace(minx, maxx, 6), np.linspace(miny, maxy, 6)
        pts = [(x, y, DEFAULT_Z_M) for x in xs for y in ys if g.contains(shapely.Point(x, y))] or \
              [(g.centroid.x, g.centroid.y, DEFAULT_Z_M)]
    rows = [trust_at(ctx.site.rf_dir, s, a.band, *p) | {"sensor": s} for s in sens for p in pts]
    tv = [r["trust"] for r in rows]
    sp = [r["spread_deg"] for r in rows if r["spread_deg"] is not None]
    flags = sorted({f.split(":")[0] for r in rows for f in r["flags"]})
    extra = {}
    if a.per_sensor:
        ps = [r for r in per_sensor_at(ctx.site.rf_dir, a.band, *pts[0]) if r["sensor"] in sens]
        extra = {"per_sensor": [{**r, "sensor": f"sensor:{r['sensor']}"} for r in ps],
                 "best_sensor": f"sensor:{ps[0]['sensor']}" if ps else None}
    return ToolOutput({"trust": round(float(np.mean(tv)), 2), "trust_min": round(float(np.min(tv)), 2), **extra,
                       "components": {"spread_deg": r1(float(np.mean(sp))) if sp else None,
                                      "dominance": r1(float(np.mean([r["dominance"] for r in rows
                                                                     if r["dominance"] is not None]))) if sp else None,
                                      "n_samples": len(rows)},
                       "flags": flags, "backends": rows[0]["backends"], "sensors": [f"sensor:{s}" for s in sens]},
                      [f"sensor:{s}" for s in sens] + subject_refs,
                      {"trust": "exp(-spread/10 deg) x (1 - 0.6 ghost risk); spread = disagreement of the strongest "
                                "arrival across the model ensemble (tide, ships, materials), ghost risk = share of "
                                "members whose strongest arrival is off the direct bearing (0.6 cap outside the "
                                "1.5 km node caches)"})
