"""Family A: site and spatial tools (SPEC 6.2 A)."""

from __future__ import annotations

import difflib
import re
from typing import Annotated, Literal

import yaml
from pydantic import Field
from shapely.geometry import LineString, Point, Polygon

from harness.config import REPO
from sim.geometry import bearing_deg, ellipse_polygon
from tools.errors import ToolError
from tools.registry import ToolContext, ToolOutput, register
from tools.types import ObjKind, Region, Strict, ZoneKind, parse_ref, pt, r1
from tools.types import Point as PointArg

SPATIAL = yaml.safe_load((REPO / "configs" / "tools" / "spatial.yaml").read_text())
_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "first": "1", "second": "2"}
_STOP = {"the", "no", "number", "of"}   # never single letters: "Tank Farm A" vs "Tank Farm B" must differ
TOKEN_MATCH = 0.7                        # per-token similarity needed to count as a (typo-tolerant) match


def _norm_tokens(s: str) -> list[str]:
    toks = [_NUM.get(t, t) for t in re.findall(r"[a-z0-9]+", s.lower())]
    return [t for t in toks if t not in _STOP] or toks


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit() or len(a) < 3 or len(b) < 3:
        return 0.0                        # numbers and designators must match exactly
    return difflib.SequenceMatcher(None, a, b).ratio()


def name_score(query: str, name: str) -> float:
    """Coverage-weighted match: every query word must be accounted for (so '5G tower' does not match
    'Port Control Tower'), name words not in the query cost a little (so 'Tank Farm A' beats 'North Tank Farm A')."""
    q, n = _norm_tokens(query), _norm_tokens(name)
    if not q or not n:
        return 0.0
    if q == n:
        return 1.0
    best_q = [max(_tok_sim(t, u) for u in n) for t in q]
    coverage = sum(s for s in best_q if s >= TOKEN_MATCH) / len(q)
    matched_n = sum(1 for u in n if max(_tok_sim(t, u) for t in q) >= TOKEN_MATCH)
    precision = matched_n / len(n)
    return round(0.8 * coverage + 0.2 * precision, 3)


def normalize_ref(site, ref: str) -> str:
    """Accept obvious bare forms the model produces ('s4', '2146', 'tank_farm_a') and return the canonical ref.
    Outputs always use canonical refs, so the agent learns the right form from results."""
    ref = ref.strip()
    head = ref.split("@", 1)[0]
    if ":" in head:
        kind, _, rest = ref.partition(":")
        alias = {"track": "trk", "sens": "sensor", "object": "obj", "hypothesis": "hyp", "event": "evt"}
        return f"{alias.get(kind.lower(), kind.lower())}:{rest}"
    base, _, t = ref.partition("@")
    suffix = f"@{t}" if t else ""
    if base.upper() in site.sensors:
        return f"sensor:{base.upper()}{suffix}"
    if base in site.tracks:
        return f"trk:{base}{suffix}"
    if base in site.objects:
        return f"obj:{base}{suffix}"
    if base in site.zones:
        return f"zone:{base}{suffix}"
    return ref


def _entity(ctx: ToolContext, ref: str):
    ref = normalize_ref(ctx.site, ref)
    try:
        kind, eid, t = parse_ref(ref)
    except ValueError as e:
        raise ToolError("INVALID_ARGS", str(e), "Use refs exactly as returned by tools, e.g. obj:tank_farm_a.") from None
    site = ctx.site
    if kind in ("obj", "zone"):
        ent = site.entity(kind, eid)
        if ent is None:
            raise _unknown(site, kind, ref)
        return kind, ent, t
    if kind == "sensor":
        eid = eid.upper()
        if eid not in site.sensors:
            raise _unknown(site, kind, ref)
        return kind, site.sensors[eid], t
    if kind == "trk":
        if eid not in site.tracks:
            raise _unknown(site, kind, ref)
        return kind, eid, t
    raise ToolError("INVALID_ARGS", f"ref kind {kind!r} not supported here",
                    "Pass an obj:, zone:, sensor: or trk: ref.")


def _unknown(site, kind: str, ref: str) -> ToolError:
    close = difflib.get_close_matches(ref, site.known_refs(kind), n=3, cutoff=0.5)
    return ToolError("UNKNOWN_REF", f"{ref} does not exist", "Use a ref returned by a tool in this episode."
                     + (f" Did you mean: {', '.join(close)}?" if close else " Call site_resolve or track_search."))


def geometry_of(ctx: ToolContext, ref: str):
    """(geometry, kind, entity, time) for obj/zone/sensor/track refs. Tracks use the estimate at @t (or latest)."""
    kind, ent, t = _entity(ctx, ref)
    if kind in ("obj", "zone"):
        return ent.geom, kind, ent, t
    if kind == "sensor":
        p = ent["pose"]
        return Point(p["x_m"], p["y_m"]), kind, ent, t
    tid = ent
    pts = ctx.site.points[tid]
    if t is not None and not (pts["t"][0] - 1e-6 <= t <= pts["t"][-1] + 1e-6):
        raise ToolError("OUT_OF_COVERAGE", f"trk:{tid} has no estimate at t={t}",
                        f"Valid times for trk:{tid}: {pts['t'][0]:.0f}..{pts['t'][-1]:.0f} s.")
    i = ctx.site.track_index_at(tid, t)
    return Point(pts["x"][i], pts["y"][i]), kind, tid, float(pts["t"][i])


def region_geom(ctx: ToolContext, region) -> Polygon:
    if region.type == "circle":
        return Point(region.center.x_m, region.center.y_m).buffer(region.radius_m, 32)
    if region.type == "polygon":
        poly = Polygon([(p.x_m, p.y_m) for p in region.points])
        if not poly.is_valid:
            raise ToolError("INVALID_ARGS", "polygon is self-intersecting", "List polygon points in boundary order.")
        return poly
    geom, kind, _, _ = geometry_of(ctx, region.ref)
    if kind not in ("obj", "zone"):
        raise ToolError("INVALID_ARGS", "region ref must be obj: or zone:", "Use a circle around a point instead.")
    return geom


# --- site_resolve ----------------------------------------------------------------------------------------------
class SiteResolveArgs(Strict):
    query: Annotated[str, Field(min_length=1, max_length=80)]
    kinds: list[ObjKind | ZoneKind] | None = None
    near: PointArg | None = None
    limit: Annotated[int, Field(ge=1, le=10)] = 5


@register("site_resolve", "A", "Resolve a place name to site object/zone refs; returns ranked candidates.",
          SiteResolveArgs)
def site_resolve(ctx: ToolContext, a: SiteResolveArgs) -> ToolOutput:
    site = ctx.site
    cands = []
    for ent in list(site.objects.values()) + list(site.zones.values()):
        if a.kinds and ent.kind not in a.kinds:
            continue
        score = max(name_score(a.query, n) for n in (ent.name, *ent.aliases))
        if a.near is not None:  # small tie-break towards the stated location, never more than 0.05
            d = ent.geom.distance(Point(a.near.x_m, a.near.y_m))
            score -= 0.05 * min(1.0, d / 2000.0)
        cands.append((round(score, 3), ent))
    cands.sort(key=lambda c: (-c[0], c[1].ref))
    cfg = SPATIAL["site_resolve"]

    def row(s, e):
        c = e.geom.centroid
        minx, miny, maxx, maxy = e.geom.bounds
        return {"ref": e.ref, "name": e.name, "kind": e.kind, "score": s, "centroid": pt(c.x, c.y),
                "bbox_m": [r1(minx), r1(miny), r1(maxx), r1(maxy)]}

    top = [row(s, e) for s, e in cands[: a.limit]]
    if not cands or cands[0][0] < cfg["min_score"]:
        if a.kinds:  # a match may exist under another kind: say so instead of a bare NOT_FOUND
            alt =[(max(name_score(a.query, n) for n in (e.name, *e.aliases)), e)
                   for e in list(site.objects.values()) + list(site.zones.values())]
            alt = [x for x in sorted(alt, key=lambda x: -x[0]) if x[0] >= cfg["min_score"]][:3]
            if alt:
                raise ToolError("NOT_FOUND", f"no {'/'.join(a.kinds)} matches {a.query!r}",
                                "Matches exist with other kinds; call again without kinds.",
                                candidates=[{"ref": e.ref, "name": e.name, "kind": e.kind, "score": round(s, 3)}
                                            for s, e in alt])
        raise ToolError("NOT_FOUND", f"no site entity matches {a.query!r}",
                        "This place is not on the site: submit intent abstain with reason entity_not_found.",
                        candidates=[{"ref": c["ref"], "name": c["name"], "score": c["score"]} for c in top[:3]])
    close = [c for c in top if cands[0][0] - c["score"] <= cfg["ambiguity_margin"]]
    if len(close) > 1:
        raise ToolError("AMBIGUOUS_ENTITY", f"{len(close)} candidates within margin {cfg['ambiguity_margin']}",
                        "Call ask_user or choose one and state the assumption in submit.assumptions.",
                        candidates=[{"ref": c["ref"], "name": c["name"], "score": c["score"]} for c in close])
    return ToolOutput({"status": "ok", "candidates": top}, refs=[c["ref"] for c in top])


# --- site_describe ---------------------------------------------------------------------------------------------
class SiteDescribeArgs(Strict):
    ref: str


@register("site_describe", "A", "Describe an obj:, zone: or sensor: ref (geometry, height, material, contents).",
          SiteDescribeArgs)
def site_describe(ctx: ToolContext, a: SiteDescribeArgs) -> ToolOutput:
    kind, ent, _ = _entity(ctx, a.ref)
    site = ctx.site
    if kind == "sensor":
        pose = ent["pose"]
        return ToolOutput({"ref": a.ref, "type": ent["type"], "pos": pt(pose["x_m"], pose["y_m"], pose["z_m"]),
                           "bands": ent["bands"], "mount": ent["mount"]}, refs=[a.ref])
    if kind == "trk":
        raise ToolError("INVALID_ARGS", "use track_detail for tracks", "Call track_detail(track_ref=...).")
    g = ent.geom
    c = g.centroid
    minx, miny, maxx, maxy = g.bounds
    data = {"ref": ent.ref, "name": ent.name, "kind": ent.kind, "aliases": list(ent.aliases),
            "centroid": pt(c.x, c.y), "bbox_m": [r1(minx), r1(miny), r1(maxx), r1(maxy)], "area_m2": round(g.area)}
    refs = [ent.ref]
    if kind == "obj":
        data.update(height_m=ent.height_m, material=ent.material, protected=ent.protected,
                    parents=[f"zone:{z}" for z in site.zones_containing(c.x, c.y)])
        refs += data["parents"]
    else:
        inside = sorted(o.ref for o in site.objects_intersecting(g) if g.contains(o.geom.centroid))
        data.update(children=inside[:20], n_children=len(inside))
        refs += inside[:20]
    return ToolOutput(data, refs=refs)


# --- region_contents -------------------------------------------------------------------------------------------
class RegionContentsArgs(Strict):
    region: Region
    kinds: list[str] | None = None        # validated in the tool so a wrong kind gets a redirecting hint
    limit: Annotated[int, Field(ge=1, le=20)] = 10
    cursor: str | None = None


OBJ_KINDS = set(ObjKind.__args__)


@register("region_contents", "A", "List fixed site objects (tanks, jetties, buildings) in a region. Not tracks.",
          RegionContentsArgs)
def region_contents(ctx: ToolContext, a: RegionContentsArgs) -> ToolOutput:
    bad = [k for k in (a.kinds or []) if k not in OBJ_KINDS]
    if bad:
        raise ToolError("INVALID_ARGS", f"{bad} are not site object kinds",
                        "region_contents lists fixed objects only. To find drones/emitters in an area use "
                        "track_search(filters={'region': ..., 'time_range_s': [t, t]}).",
                        valid_kinds=sorted(OBJ_KINDS))
    g = region_geom(ctx, a.region)
    c = g.centroid
    hits = [o for o in ctx.site.objects_intersecting(g) if not a.kinds or o.kind in a.kinds]
    hits.sort(key=lambda o: (o.geom.centroid.distance(c), o.ref))
    off = _cursor(a.cursor)
    page = hits[off: off + a.limit]
    items = [{"ref": o.ref, "name": o.name, "kind": o.kind, "protected": o.protected,
              "dist_m": r1(o.geom.centroid.distance(c))} for o in page]
    nxt = f"c{off + a.limit}" if off + a.limit < len(hits) else None
    return ToolOutput({"total": len(hits), "items": items, "next_cursor": nxt}, refs=[i["ref"] for i in items],
                      definition={"inside": "footprint intersects region", "dist": "centroid to region centre"})


def _cursor(cur: str | None) -> int:
    if cur is None:
        return 0
    if not re.fullmatch(r"c\d+", cur):
        raise ToolError("INVALID_ARGS", f"bad cursor {cur!r}", "Pass next_cursor exactly as returned.")
    return int(cur[1:])


# --- spatial_relation ------------------------------------------------------------------------------------------
Relation = Literal["distance", "bearing", "inside", "within_m", "north_of", "south_of", "east_of", "west_of",
                   "between"]


class SpatialRelationArgs(Strict):
    a_ref: str
    b_ref: str
    relation: Relation
    param_m: Annotated[float, Field(gt=0, le=20000)] | None = None
    c_ref: str | None = None


@register("spatial_relation", "A", "Compute a relation between refs; result states the exact definition used.",
          SpatialRelationArgs)
def spatial_relation(ctx: ToolContext, a: SpatialRelationArgs) -> ToolOutput:
    ga, ka, ea, ta = geometry_of(ctx, a.a_ref)
    gb, kb, eb, tb = geometry_of(ctx, a.b_ref)
    ca, cb = ga.centroid, gb.centroid
    refs = [a.a_ref, a.b_ref]
    rel = a.relation
    if rel == "distance":
        v = ga.distance(gb)
        return ToolOutput({"value_m": r1(v)}, refs, {"distance": "minimum 2D distance between footprints, m"})
    if rel == "bearing":
        return ToolOutput({"value_deg": r1(bearing_deg(ca.x, ca.y, cb.x, cb.y))}, refs,
                          {"bearing": "from a centroid to b centroid, deg clockwise from north"})
    if rel == "inside":
        if kb not in ("obj", "zone"):
            raise ToolError("INVALID_ARGS", "b_ref must be an area (obj: or zone:)", "Use within_m for points.")
        return ToolOutput({"value": bool(gb.contains(ca))}, refs, {"inside": "a centroid inside b footprint"})
    if rel == "within_m":
        kind_b = getattr(eb, "kind", "default") if kb in ("obj", "zone") else "default"
        radius = a.param_m or SPATIAL["near_radius_m"].get(kind_b, SPATIAL["near_radius_m"]["default"])
        d = ga.distance(gb)
        return ToolOutput({"value": bool(d <= radius), "distance_m": r1(d)}, refs,
                          {"within_m": radius, "distance": "minimum 2D distance between footprints, m",
                           "radius_source": "param_m" if a.param_m else f"default for kind {kind_b}"})
    if rel in ("north_of", "south_of", "east_of", "west_of"):
        minx, miny, maxx, maxy = gb.bounds
        margin = {"north_of": ca.y - maxy, "south_of": miny - ca.y, "east_of": ca.x - maxx,
                  "west_of": minx - ca.x}[rel]
        edge = {"north_of": "maxy", "south_of": "miny", "east_of": "maxx", "west_of": "minx"}[rel]
        return ToolOutput({"value": bool(margin > 0), "margin_m": r1(margin)}, refs,
                          {rel: f"a centroid beyond b bbox {edge}"})
    # between
    if not a.c_ref:
        raise ToolError("INVALID_ARGS", "between needs c_ref", "Pass c_ref: is a between b and c?")
    gc, _, _, _ = geometry_of(ctx, a.c_ref)
    seg = LineString([cb, gc.centroid])
    corridor = a.param_m or SPATIAL["between_corridor_m"]
    proj = seg.project(ca, normalized=True)
    d = seg.distance(ca)
    ok = 0.0 < proj < 1.0 and d <= corridor
    return ToolOutput({"value": bool(ok), "offset_m": r1(d), "fraction": round(proj, 2)}, refs + [a.c_ref],
                      {"between": f"a within {corridor} m of segment b-c, strictly between the ends"})


# --- inside_zone -----------------------------------------------------------------------------------------------
class InsideZoneArgs(Strict):
    subject: str | PointArg
    zone_ref: str
    at_time_s: Annotated[float, Field(ge=0)] | None = None   # for track subjects; default = latest estimate


@register("inside_zone", "A", "Is a track/object/point inside a zone at a time; uncertainty overlap for tracks.",
          InsideZoneArgs)
def inside_zone(ctx: ToolContext, a: InsideZoneArgs) -> ToolOutput:
    zg, kz, zent, _ = geometry_of(ctx, a.zone_ref)
    if kz != "zone":
        raise ToolError("INVALID_ARGS", "zone_ref must be a zone: ref", "Call site_resolve with kinds=[...] zones.")
    refs = [a.zone_ref]
    overlap = None
    t_used = None
    warnings: list[str] = []
    if isinstance(a.subject, str):
        subject = a.subject
        if a.at_time_s is not None and "@" not in subject:
            subject = f"{subject}@{a.at_time_s:g}"
        g, kind, ent, t_used = geometry_of(ctx, subject)
        if kind == "trk" and "@" not in subject:
            warnings.append(f"no time given: used the latest estimate (t={t_used:g} s); pass at_time_s")
        a = a.model_copy(update={"subject": subject})
        refs.insert(0, a.subject)
        p = g.centroid
        if kind == "trk":
            pts = ctx.site.points[ent]
            i = ctx.site.track_index_at(ent, t_used)
            ell = ellipse_polygon(p.x, p.y, pts["a"][i], pts["b"][i], pts["theta"][i])
            overlap = round(ell.intersection(zg).area / ell.area, 2)
    else:
        p = Point(a.subject.x_m, a.subject.y_m)
    inside = bool(zg.contains(p))
    dist = zg.exterior.distance(p)
    data = {"inside": inside, "distance_to_boundary_m": r1(dist if inside else -dist)}
    if overlap is not None:
        data["uncertainty_overlap_fraction"] = overlap
    if t_used is not None:
        data["t_s"] = r1(t_used)
    return ToolOutput(data, refs, {"inside": "estimate centre inside zone polygon",
                                   "distance_to_boundary_m": "positive inside, negative outside"}, warnings)
