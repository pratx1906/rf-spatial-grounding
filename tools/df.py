"""Family B completion (SPEC 6.2 B): predict_aoa, geometry_check, fit_candidates. Belief side only.

predict_aoa reads the NOMINAL Sionna model (what the site geometry predicts), never the observation labels. Its
peak classes are named direct / reflection / diffraction / scatter with the object hit as `via_ref`.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
import shapely
from pydantic import Field
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry import Point as SPoint

from sim.belief.aoa import AoAPredictor
from sim.belief.rf_cache import ellipse, triangulate
from sim.belief.trust import trust_at
from sim.geometry import angle_diff_deg, bearing_deg, facades, los_clear, specular_point
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.site import _entity, normalize_ref
from tools.types import Band, Strict, parse_ref, pt, r1
from tools.types import Point as PointArg

DEFAULT_Z_M = 60.0
HYP_SIGMA_DEG = 3.0
RT_GRID_M = 25.0
RT_HALF = 4
RT_CAP_DEG = 25.0
DIRECT_MISS_DEG = 8.0
RT_COARSE_M = 150.0
_PRED: dict[str, AoAPredictor] = {}
_BLD: dict[str, tuple] = {}


def predictor(ctx: ToolContext) -> AoAPredictor:
    key = str(ctx.site.rf_dir)
    if key not in _PRED:
        _PRED[key] = AoAPredictor(ctx.site)
    return _PRED[key]


def blockers(ctx: ToolContext):
    """(objects with height, STRtree) for 2.5D line-of-sight: tanks, buildings, silos, substations, ships, jetties."""
    key = str(ctx.site.dir / "objects.parquet")
    if key not in _BLD:
        objs = [o for o in ctx.site.objects.values() if o.height_m and o.height_m > 1 and o.kind not in
                ("tank_farm", "road") and o.geom.geom_type in ("Polygon", "MultiPolygon")]
        _BLD[key] = (objs, STRtree([o.geom for o in objs]))
    return _BLD[key]


def _sensor(ctx: ToolContext, ref: str) -> tuple[str, dict]:
    kind, ent, _ = _entity(ctx, ref)
    if kind != "sensor":
        raise ToolError("INVALID_ARGS", f"{ref} is not a sensor ref", "Pass sensor: refs, e.g. sensor:S1.")
    return ent["id"], ent


def xyz_of(ctx: ToolContext, subject) -> tuple[float, float, float, str | None]:
    """(x, y, z, canonical ref) for a ref (sensor: mast top, trk: estimate, obj: roof) or a Point."""
    if isinstance(subject, PointArg):
        return subject.x_m, subject.y_m, subject.z_m if subject.z_m is not None else DEFAULT_Z_M, None
    ref = normalize_ref(ctx.site, subject)
    kind, ent, t = _entity(ctx, ref)
    if kind == "sensor":
        p = ent["pose"]
        return p["x_m"], p["y_m"], p["z_m"], f"sensor:{ent['id']}"
    if kind == "trk":
        pts = ctx.site.points[ent]
        i = ctx.site.track_index_at(ent, t)
        return float(pts["x"][i]), float(pts["y"][i]), float(pts["z"][i]), f"trk:{ent}@{pts['t'][i]:g}"
    if kind == "obj":
        c = ent.geom.representative_point()
        return c.x, c.y, float(ent.height_m or 0.0) + 1.0, ent.ref
    raise ToolError("INVALID_ARGS", "zones have no single position", "Pass an obj:, sensor:, trk: ref or a point.")


# --- predict_aoa -----------------------------------------------------------------------------------------------
class PredictAoaArgs(Strict):
    point: PointArg
    sensor_ref: str
    band: Band


@register("predict_aoa", "B", "Nominal predicted AoA peaks at a sensor for an emitter at a point (site geometry).",
          PredictAoaArgs)
def predict_aoa(ctx: ToolContext, a: PredictAoaArgs) -> ToolOutput:
    sid, ent = _sensor(ctx, a.sensor_ref)
    z = a.point.z_m if a.point.z_m is not None else DEFAULT_Z_M
    if a.band not in ent["bands"]:
        raise ToolError("INVALID_ARGS", f"sensor:{sid} does not cover {a.band}", f"Bands of sensor:{sid}: "
                        f"{', '.join(ent['bands'])}.")
    peaks = predictor(ctx).peaks(a.point.x_m, a.point.y_m, z, sid, a.band)
    if peaks is None:
        raise ToolError("OUT_OF_COVERAGE", "point is outside the propagation model's extent (+-3.2 km)",
                        "Predict only for points inside the port area.")
    tr = trust_at(ctx.site.rf_dir, sid, a.band, a.point.x_m, a.point.y_m, z)
    out = [{"bearing_deg": p["bearing_deg"], "rel_power_db": p["rel_power_db"], "kind": p["kind"],
            **({"via_ref": f"obj:{p['via']}"} if p["via"] and p["via"] in ctx.site.objects else {})}
           for p in peaks]
    s = ent["pose"]
    warnings = [] if peaks else ["no propagation path reaches this sensor from the point"]
    refs = [f"sensor:{sid}"] + sorted({p["via_ref"] for p in out if "via_ref" in p})
    return ToolOutput({"sensor": f"sensor:{sid}", "point": pt(a.point.x_m, a.point.y_m, z),
                       "geometric_bearing_deg": r1(bearing_deg(s["x_m"], s["y_m"], a.point.x_m, a.point.y_m)),
                       "peaks": out, "model_source": "sionna_rt", "trust": tr["trust"]}, refs,
                      {"peaks": "resolvable arrivals (8 deg), strongest first, within 15 dB of the strongest",
                       "kind": "direct | reflection (via_ref = object hit) | diffraction | scatter",
                       "trust": "0..1; lower where propagation variants disagree or multipath is rich"}, warnings)


# --- geometry_check --------------------------------------------------------------------------------------------
class GeometryCheckArgs(Strict):
    from_: Annotated[str | PointArg, Field(alias="from")]
    to: str | PointArg
    mode: Literal["los", "reachable", "single_bounce_via"] = "los"
    via_ref: str | None = None

    model_config = {"extra": "forbid", "populate_by_name": True}


def _los_detail(ctx, p0, p1, exclude: set[str]) -> tuple[bool, list[str], float]:
    objs, tree = blockers(ctx)
    seg = LineString([p0[:2], p1[:2]])
    length = seg.length or 1.0
    block, margin = [], math.inf
    for i in tree.query(seg, predicate="intersects"):
        o = objs[int(i)]
        if o.ref in exclude or o.geom.contains(SPoint(p0[:2])) or o.geom.contains(SPoint(p1[:2])):
            continue
        for cx, cy in shapely.get_coordinates(seg.intersection(o.geom)):
            f = math.hypot(cx - p0[0], cy - p0[1]) / length
            m = p0[2] + f * (p1[2] - p0[2]) - o.height_m
            margin = min(margin, m)
            if m < 0 and o.ref not in block:
                block.append(o.ref)
    return not block, block, (round(margin, 1) if margin != math.inf else None)


@register("geometry_check", "B", "Line of sight / single-bounce reachability between two points over the 3D site.",
          GeometryCheckArgs)
def geometry_check(ctx: ToolContext, a: GeometryCheckArgs) -> ToolOutput:
    x0, y0, z0, r0 = xyz_of(ctx, a.from_)
    x1, y1, z1, r1_ = xyz_of(ctx, a.to)
    p0, p1 = (x0, y0, z0), (x1, y1, z1)
    refs = [r for r in (r0, r1_) if r]
    defin = {"los": "3D segment above every footprint (estimated heights) it crosses",
             "single_bounce_via": "specular point on a facade of via_ref, both legs clear",
             "reachable": "los, or a single bounce via any object within 300 m of the segment"}
    ok, blk, margin = _los_detail(ctx, p0, p1, set())
    data = {"mode": a.mode, "from": pt(*p0), "to": pt(*p1), "range_m": r1(math.dist(p0, p1))}
    if a.mode == "los":
        data.update(result=ok, blockers=blk[:8], margin_m=margin)
        return ToolOutput(data, refs + blk[:8], {"los": defin["los"]})
    vias = []
    if a.mode == "single_bounce_via":
        if not a.via_ref:
            raise ToolError("INVALID_ARGS", "single_bounce_via needs via_ref", "Pass via_ref=obj:<id>.")
        kind, ent, _ = _entity(ctx, a.via_ref)
        if kind != "obj":
            raise ToolError("INVALID_ARGS", "via_ref must be an obj: ref", "Use an object from site_resolve.")
        vias = [ent]
    else:
        objs, tree = blockers(ctx)
        corr = LineString([p0[:2], p1[:2]]).buffer(300.0)
        cand = [objs[int(i)] for i in tree.query(corr, predicate="intersects")]
        vias = sorted(cand, key=lambda o: o.geom.distance(LineString([p0[:2], p1[:2]])))[:25]
    bounces = []
    for o in vias:
        g = o.geom if o.geom.geom_type == "Polygon" else max(o.geom.geoms, key=lambda q: q.area)
        for ax, ay, bx, by in facades(g.simplify(1.0)):
            sp = specular_point(x0, y0, x1, y1, ax, ay, bx, by)
            if sp is None:
                continue
            d0, d1 = math.hypot(sp[0] - x0, sp[1] - y0), math.hypot(sp[0] - x1, sp[1] - y1)
            zb = z0 + (z1 - z0) * d0 / max(d0 + d1, 1e-6)
            if zb > o.height_m:
                continue                                  # the bounce would pass over the roof
            q = (sp[0], sp[1], zb)
            ok0, _, _ = _los_detail(ctx, p0, q, {o.ref})
            ok1, _, _ = _los_detail(ctx, q, p1, {o.ref})
            if ok0 and ok1:
                bounces.append({"via_ref": o.ref, "bounce": pt(*q), "path_len_m": r1(d0 + d1 + 0.0),
                                "bearing_at_from_deg": r1(bearing_deg(x0, y0, sp[0], sp[1]))})
                break
    if a.mode == "single_bounce_via":
        data.update(result=bool(bounces), bounce=bounces[0] if bounces else None, los=ok)
        return ToolOutput(data, refs + [vias[0].ref], {"single_bounce_via": defin["single_bounce_via"]})
    data.update(result=ok or bool(bounces), los=ok, blockers=blk[:8], bounces=bounces[:5])
    return ToolOutput(data, refs + blk[:8] + [b["via_ref"] for b in bounces[:5]], {"reachable": defin["reachable"]})


# --- fit_candidates --------------------------------------------------------------------------------------------
class FitCandidatesArgs(Strict):
    hyp_refs: Annotated[list[str], Field(min_length=2, max_length=15)]
    candidate_points: Annotated[list[PointArg], Field(max_length=20)] | None = None
    method: Literal["triangulate", "rt_match"] = "triangulate"


def _hyps(ctx: ToolContext, refs: list[str]) -> list[dict]:
    out = []
    con = ctx.site.cursor()
    for r in refs:
        try:
            kind, hid, _ = parse_ref(normalize_ref(ctx.site, r) if ":" in r else f"hyp:{r}")
        except ValueError:
            raise ToolError("INVALID_ARGS", f"malformed hyp ref {r!r}", "Use hyp: refs from df_hypotheses.") from None
        if kind != "hyp":
            raise ToolError("INVALID_ARGS", f"{r} is not a hyp: ref", "Use hyp: refs from df_hypotheses.")
        row = con.execute("SELECT hyp_id, track_id, sensor_id, t_s, bearing_deg, snr_db, power_db FROM bearings "
                          "WHERE hyp_id = ?", [hid]).fetchone()
        if row is None:
            raise ToolError("UNKNOWN_REF", f"hyp:{hid} does not exist", "Call df_hypotheses and use its hyp: refs.")
        out.append(dict(zip(("hyp", "track", "sensor", "t", "bearing", "snr", "power"), row, strict=True)))
    tracks = {h["track"] for h in out}
    times = {h["t"] for h in out}
    if len(tracks) > 1 or len(times) > 1:
        raise ToolError("INVALID_ARGS", "hypotheses must belong to one track at one time",
                        "Pass the hyp: refs of a single df_hypotheses result.")
    return out


def _sigma(snr: float) -> float:
    return max(1.0, 0.8 + 12.0 / (1.0 + 10 ** (snr / 20.0)))


def _band(ctx, track: str, t: float) -> str:
    p = ctx.site.points[track]
    i = ctx.site.track_index_at(track, t)
    return p["band"][i]


def _consistent_fixes(hs: list[dict], sens: dict, top: int = 3) -> list[tuple[float, float]]:
    """RANSAC-style seeds: one bearing per sensor, every combination (<= 3 hyps per sensor), triangulated; the
    combinations with >= 3 sensors and the smallest angular residual are the most self-consistent geometries."""
    import itertools

    per: dict[str, list[dict]] = {}
    for h in hs:
        per.setdefault(h["sensor"], []).append(h)
    if len(per) < 3:
        return []
    groups = [sorted(v, key=lambda h: -h["power"])[:3] for v in per.values()]
    scored = []
    for combo in itertools.islice(itertools.product(*groups), 400):
        obs = [(sens[h["sensor"]]["pose"]["x_m"], sens[h["sensor"]]["pose"]["y_m"], h["bearing"], _sigma(h["snr"]))
               for h in combo]
        f = triangulate(obs)
        if f is None or not np.all(np.abs(f[0]) < 3200):
            continue
        res = [abs(angle_diff_deg(h["bearing"], bearing_deg(o[0], o[1], f[0][0], f[0][1]))) for h, o in
               zip(combo, obs, strict=True)]
        scored.append((float(np.sqrt(np.mean(np.square(res)))), tuple(f[0])))
    scored.sort(key=lambda s: s[0])
    return [p for _, p in scored[:top]]


@register("fit_candidates", "B", "Locate an emitter from hyp bearings: triangulate, or rt_match against predicted AoAs.",
          FitCandidatesArgs)
def fit_candidates(ctx: ToolContext, a: FitCandidatesArgs) -> ToolOutput:
    if a.method in ctx.options.get("fit_methods_disabled", ()):            # ablation knockout (EnvConfig)
        other = "triangulate" if a.method == "rt_match" else "rt_match"
        raise ToolError("NOT_PERMITTED", f"method {a.method} is disabled in this configuration",
                        f"Use method {other} with one hyp per sensor.")
    hs = _hyps(ctx, a.hyp_refs)
    sens = ctx.site.sensors
    t = hs[0]["t"]
    refs = [f"hyp:{h['hyp']}@{t:g}" for h in hs]
    if a.method == "triangulate":
        per = {}
        for h in hs:
            if h["sensor"] in per:
                raise ToolError("INVALID_ARGS", f"two hyps from sensor:{h['sensor']}",
                                "triangulate takes one bearing per sensor; use rt_match to test several.")
            per[h["sensor"]] = h
        obs = [(sens[s]["pose"]["x_m"], sens[s]["pose"]["y_m"], h["bearing"], _sigma(h["snr"])) for s, h in per.items()]
        fix = triangulate(obs)
        if fix is None:
            raise ToolError("INVALID_ARGS", "bearings are parallel or too few",
                            "Use hyps from at least two sensors with different viewing angles.")
        xy, cov, res = fix
        a_, b_, th = ellipse(cov)
        resid = [{"ref": f"hyp:{h['hyp']}@{t:g}", "residual_deg": r1(angle_diff_deg(
            h["bearing"], bearing_deg(sens[s]["pose"]["x_m"], sens[s]["pose"]["y_m"], xy[0], xy[1])))}
            for s, h in per.items()]
        cand = {"ref": "cand:0", "position": pt(xy[0], xy[1]),
                "uncertainty": {"kind": "ellipse", "a_m": r1(a_), "b_m": r1(b_), "theta_deg": r1(th)},
                "residual_m_rms": r1(res), "hyps": resid}
        return ToolOutput({"method": "triangulate", "t_s": r1(t), "candidates": [cand]}, refs + ["cand:0"],
                          {"triangulate": "weighted least squares of bearing lines (sigma from SNR)",
                           "uncertainty": "~2-sigma ellipse"})
    # rt_match: test candidate positions against the nominal predicted AoAs of every sensor involved
    band = _band(ctx, hs[0]["track"], t)
    p = ctx.site.points[hs[0]["track"]]
    z = float(p["z"][ctx.site.track_index_at(hs[0]["track"], t)]) or DEFAULT_Z_M
    if a.candidate_points:
        cands = [(c.x_m, c.y_m) for c in a.candidate_points]
    else:
        best = {}
        for h in hs:
            if h["sensor"] not in best or h["power"] > best[h["sensor"]]["power"]:
                best[h["sensor"]] = h
        seeds = []
        obs = [(sens[s]["pose"]["x_m"], sens[s]["pose"]["y_m"], h["bearing"], _sigma(h["snr"]))
               for s, h in best.items()]
        f0 = triangulate(obs) if len(obs) >= 2 else None
        if f0 is not None:
            seeds.append(tuple(f0[0]))
        if len(hs) <= 9:                                       # pairwise intersections across sensors
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    if hs[i]["sensor"] == hs[j]["sensor"]:
                        continue
                    f = triangulate([(sens[h["sensor"]]["pose"]["x_m"], sens[h["sensor"]]["pose"]["y_m"],
                                      h["bearing"], 2.0) for h in (hs[i], hs[j])])
                    if f is not None and np.all(np.abs(f[0]) < 3200):
                        seeds.append(tuple(f[0]))
        seeds = seeds[:4] + _consistent_fixes(hs, sens)
        if not seeds:
            raise ToolError("INVALID_ARGS", "cannot seed candidates", "Pass candidate_points or hyps from 2 sensors.")
        cands = []
        for sx, sy in seeds[:6]:
            for i in range(-RT_HALF, RT_HALF + 1, 2 if len(seeds) > 1 else 1):
                for j in range(-RT_HALF, RT_HALF + 1, 2 if len(seeds) > 1 else 1):
                    cands.append((sx + i * RT_GRID_M, sy + j * RT_GRID_M))
    pred = predictor(ctx)
    # a node only hears emitters inside its coverage radius (sensors.yaml): candidates outside are impossible
    heard = {h["sensor"] for h in hs}
    radius = {s: float((sens[s].get("array") or {}).get("coverage_radius_m", 1e9)) + 50.0 for s in heard}
    def ok(cx, cy):
        return all(math.hypot(cx - sens[s]["pose"]["x_m"], cy - sens[s]["pose"]["y_m"]) <= radius[s] for s in heard)

    cands = [(cx, cy) for cx, cy in cands if ok(cx, cy)]
    coarse = False
    if len(cands) < 20 and not a.candidate_points:
        # ghost-biased seeds can all fall outside the joint coverage: search it on a coarse grid, then refine
        area = None
        for s_ in heard:
            disc = SPoint(sens[s_]["pose"]["x_m"], sens[s_]["pose"]["y_m"]).buffer(radius[s_] - 50.0, 32)
            area = disc if area is None else area.intersection(disc)
        if area is not None and not area.is_empty:
            minx, miny, maxx, maxy = area.bounds
            for gx in np.arange(minx, maxx, RT_COARSE_M):
                for gy in np.arange(miny, maxy, RT_COARSE_M):
                    if area.contains(SPoint(gx, gy)):
                        cands.append((float(gx), float(gy)))
            coarse = True
    if not cands:
        raise ToolError("OUT_OF_COVERAGE", "no candidate lies inside the coverage of every node that heard it",
                        "Check the hyps belong to one emitter; pass candidate_points to test specific places.")
    scored = _score_cands(ctx, hs, cands, band, z, heard)
    if coarse:
        ref = []
        for _, bx, by, _ in sorted(scored, key=lambda q: q[0])[:2]:
            ref += [(bx + i * RT_GRID_M, by + j * RT_GRID_M) for i in range(-3, 4) for j in range(-3, 4)
                    if ok(bx + i * RT_GRID_M, by + j * RT_GRID_M)]
        scored += _score_cands(ctx, hs, ref, band, z, heard)
    scored.sort(key=lambda q: q[0])
    out = []
    for k, (rms, cx, cy, per_h) in enumerate(scored[:5]):
        out.append(_cand_row(ctx, k, rms, cx, cy, per_h, hs, sens, t, z))
    vias = sorted({h["via_ref"] for c in out[:1] for h in c["hyps"] if "via_ref" in h})
    return ToolOutput({"method": "rt_match", "t_s": r1(t), "band": band, "n_tested": len(scored), "candidates": out,
                       "model_source": "sionna_rt"}, refs + [c["ref"] for c in out] + vias,
                      {"rt_match": "candidates inside every observing node's coverage, scored by angular distance of "
                                   "each hyp to the nearest predicted peak (+ penalty when a node has no direct path); "
                                   "explained_as = predicted class of that peak (direct = line of sight)"})


def _score_cands(ctx, hs, cands, band, z, heard):
    pred = predictor(ctx)
    scored = []
    for cx, cy in cands:
        cost, per_h, n = 0.0, [], 0
        for h in hs:
            pk = pred.peaks(cx, cy, z, h["sensor"], band)
            if not pk:
                cost += RT_CAP_DEG ** 2
                per_h.append((h, None, RT_CAP_DEG))
                continue
            near = min(pk, key=lambda q: abs(angle_diff_deg(h["bearing"], q["bearing_deg"])))
            d = min(abs(angle_diff_deg(h["bearing"], near["bearing_deg"])), RT_CAP_DEG)
            w = 1.0 if near["rel_power_db"] > -10 else 0.6
            cost += (d / w) ** 2
            per_h.append((h, near, d))
            n += 1
        direct_nodes = {h["sensor"] for h, near, d in per_h if near and near["kind"] == "direct" and d < 8}
        cost += (DIRECT_MISS_DEG ** 2) * len(heard - direct_nodes)      # a node with no direct path is suspicious
        scored.append((math.sqrt(cost / len(hs)), cx, cy, per_h))
    return scored


def _cand_row(ctx, k, rms, cx, cy, per_h, hs, sens, t, z) -> dict:
    direct = [h for h, near, d in per_h if near and near["kind"] == "direct" and d < 8]
    unc = {"kind": "ellipse", "a_m": 60.0, "b_m": 60.0, "theta_deg": 0.0}
    if len({h["sensor"] for h in direct}) >= 2:
        f = triangulate([(sens[h["sensor"]]["pose"]["x_m"], sens[h["sensor"]]["pose"]["y_m"], h["bearing"],
                          _sigma(h["snr"])) for h in {h["sensor"]: h for h in direct}.values()])
        if f is not None and math.hypot(f[0][0] - cx, f[0][1] - cy) < 4 * RT_GRID_M:
            a_, b_, th = ellipse(f[1])
            unc = {"kind": "ellipse", "a_m": r1(max(a_, RT_GRID_M)), "b_m": r1(max(b_, RT_GRID_M)), "theta_deg": r1(th)}
            cx, cy = float(f[0][0]), float(f[0][1])      # refine: triangulate the bearings explained as direct
    hy = []
    for h, near, d in per_h:
        e = {"ref": f"hyp:{h['hyp']}@{t:g}", "residual_deg": r1(d),
             "explained_as": "unexplained" if near is None or d >= 10 else near["kind"]}
        if near is not None and d < 10 and near.get("via") and near["via"] in ctx.site.objects:
            e["via_ref"] = f"obj:{near['via']}"
        hy.append(e)
    return {"ref": f"cand:{k}", "position": pt(cx, cy, z), "uncertainty": unc, "residual_deg_rms": r1(rms), "hyps": hy}
