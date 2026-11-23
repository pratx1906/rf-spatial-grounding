"""Build the Kandla Creek site (object database, zones, sensors) from OpenStreetMap (ENV spec E1-E4, E8).

    python -m envbuild.site_kandla            -> data/site_kandla/{objects.parquet, zones.geojson, sensors.yaml,
                                                site_build.json, hero_candidates.geojson}

Rules applied (ENV spec 0, 7, 8):
- Geometry and positions come from OSM (ODbL). Heights are not in OSM: every height is `estimated` with a rule.
- Names come from OSM only. Unnamed objects get generated labels marked `name_source = synthetic`.
- Descriptive aliases ("north tank farm") are derived from geometry, never hand-typed.
- Restricted zones, the port perimeter, sensors and moored ships are synthetic/proposed and labelled as such.
"""

from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import unary_union

from envbuild.geo import ENU
from sim.layout import Obj, Sensor, Zone

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "site_kandla"
HERO_SIZE_M = 2000.0


def _load_osm() -> tuple[list[dict], dict]:
    f = sorted(glob.glob(str(REPO / "data" / "raw" / "osm" / "*" / "kandla_osm.json")))[-1]
    manifest = json.loads((Path(f).parent / "manifest.json").read_text())
    return json.load(open(f))["elements"], manifest


def _prov(src_date: str, status: str = "unverified", source: str = "osm", confidence: float = 0.8, **kw) -> dict:
    return {"source": source, "source_date": src_date, "confidence": confidence, "verified_by": None,
            "status": status, **kw}


def _poly(enu: ENU, geom: list[dict]) -> Polygon | None:
    if len(geom) < 4:
        return None
    e, n, _ = enu.to_enu([g["lat"] for g in geom], [g["lon"] for g in geom])
    p = Polygon(np.column_stack([e, n]))
    if not p.is_valid:
        p = p.buffer(0)
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    return p if p.area > 1 else None


def _line(enu: ENU, geom: list[dict]) -> LineString | None:
    if len(geom) < 2:
        return None
    e, n, _ = enu.to_enu([g["lat"] for g in geom], [g["lon"] for g in geom])
    return LineString(np.column_stack([e, n]))


def sea_polygon(els: list[dict], enu: ENU, half_m: float):
    """Water area inside a square frame, built from OSM coastline ways (land on the LEFT of the way direction,
    water on the RIGHT, per OSM convention), plus small natural=water polygons. Kandla Creek is mapped this way."""
    from shapely.ops import linemerge, polygonize

    coast = [_line(enu, e["geometry"]) for e in els
             if e.get("tags", {}).get("natural") == "coastline" and e["type"] == "way" and "geometry" in e]
    coast = [c for c in coast if c is not None]
    water = []
    frame = box(-half_m, -half_m, half_m, half_m)
    if coast:
        merged = linemerge(unary_union(coast))
        lines = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        # A coastline that ends inside the frame cannot split it; shrink the frame to just inside such ends.
        for ln in lines:
            for x, y in (ln.coords[0], ln.coords[-1]):
                if max(abs(x), abs(y)) < half_m:
                    half_m = max(abs(x), abs(y)) - 10.0
        frame = box(-half_m, -half_m, half_m, half_m)
        faces = list(polygonize(unary_union([merged.intersection(frame), frame.boundary])))
        for f in faces:
            p = f.representative_point()
            ln = min(lines, key=lambda g: g.distance(p))
            s = ln.project(p)
            a, b = ln.interpolate(max(s - 2.0, 0.0)), ln.interpolate(min(s + 2.0, ln.length))
            cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            if cross < 0:                                       # right of the coastline: water
                water.append(f)
    ponds = [_poly(enu, e["geometry"]) for e in els
             if e.get("tags", {}).get("natural") == "water" and e["type"] == "way" and "geometry" in e]
    water += [p.intersection(frame) for p in ponds if p is not None]
    return unary_union(water) if water else Polygon()


def _clusters(points: list[tuple[float, float]], eps: float) -> list[list[int]]:
    """Single-linkage clustering (union-find) of points closer than eps."""
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if math.dist(points[i], points[j]) <= eps:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(points)):
        groups[find(i)].append(i)
    return list(groups.values())


def _hero_window(tank_pts: list, pier_pts: list) -> tuple[float, float]:
    """Centre of the 2 km window with the most tanks and jetties (jetties weighted x10). Provisional [HUMAN]."""
    pts = np.array(tank_pts + pier_pts)
    w = np.array([1.0] * len(tank_pts) + [10.0] * len(pier_pts))
    best, best_c = -1, (0, 0)
    xs = np.arange(pts[:, 0].min(), pts[:, 0].max() + 1, 100)
    ys = np.arange(pts[:, 1].min(), pts[:, 1].max() + 1, 100)
    h = HERO_SIZE_M / 2
    for cx in xs:
        for cy in ys:
            inside = (np.abs(pts[:, 0] - cx) <= h) & (np.abs(pts[:, 1] - cy) <= h)
            s = float(w[inside].sum())
            if s > best:
                best, best_c = s, (float(cx), float(cy))
    return best_c


def build(seed: int = 0) -> dict:
    els, manifest = _load_osm()
    src_date = manifest["retrieved_at"][:10]
    # Provisional origin: centre of the OSM tank+pier extent; re-centred on the hero zone below.
    tank_geoms = [e for e in els if e.get("tags", {}).get("man_made") == "storage_tank" and e["type"] == "way"]
    pier_geoms = [e for e in els if e.get("tags", {}).get("man_made") == "pier" and e["type"] == "way"]
    lat_all = [g["lat"] for e in tank_geoms + pier_geoms for g in e["geometry"]]
    lon_all = [g["lon"] for e in tank_geoms + pier_geoms for g in e["geometry"]]
    enu0 = ENU(float(np.mean(lat_all)), float(np.mean(lon_all)))
    tp = [_poly(enu0, e["geometry"]).centroid.coords[0] for e in tank_geoms if _poly(enu0, e["geometry"])]
    pp = [(_poly(enu0, e["geometry"]) or _line(enu0, e["geometry"]).buffer(8)).centroid.coords[0] for e in pier_geoms]
    hx, hy = _hero_window(tp, pp)
    lat0, lon0, _ = enu0.to_geodetic(hx, hy, 0.0)
    enu = ENU(float(lat0), float(lon0))          # origin at the hero-zone centre (ENV spec 2.1)

    objs: list[Obj] = []
    extra: dict[str, dict] = {}

    def add(o: Obj, **kw):
        objs.append(o)
        extra[o.id] = kw

    # --- storage tanks (real footprints) -------------------------------------------------------------------
    tanks = []
    for e in tank_geoms:
        p = _poly(enu, e["geometry"])
        if p is None:
            continue
        d = 2 * math.sqrt(p.area / math.pi)
        tanks.append((e["id"], p, d))
    tanks.sort(key=lambda t: (-round(t[1].centroid.y, -1), t[1].centroid.x))   # north to south, west to east
    for k, (oid, p, d) in enumerate(tanks, start=1):
        h = float(np.clip(0.7 * d, 8, 20))
        add(Obj(f"tank_t{k:03d}", f"Storage tank T{k:03d}", "tank", p, round(h, 1), "steel", False,
                [f"T{k:03d}", f"tank {k}"], ["storage_tank"]),
            osm_id=f"way/{oid}", name_source="synthetic", height_source="estimated: 0.7 x diameter, 8-20 m",
            height_confidence=0.3, diameter_m=round(d, 1), provenance=_prov(src_date))

    # --- tank farms: clusters of real tanks, geometry-derived aliases -------------------------------------
    cents = [t[1].centroid.coords[0] for t in tanks]
    groups = [g for g in _clusters(cents, 60.0) if len(g) >= 3]
    farms = []
    for g in groups:
        hull = unary_union([tanks[i][1] for i in g]).convex_hull.buffer(15)
        farms.append((g, hull))
    farms.sort(key=lambda f: -f[1].centroid.y)
    ys = [f[1].centroid.y for f in farms]
    xs = [f[1].centroid.x for f in farms]
    sizes = [len(f[0]) for f in farms]
    for k, (g, hull) in enumerate(farms, start=1):
        aliases = [f"TF-{k}", f"tank farm {k}", f"tank farm TF{k}"]
        i = k - 1
        if ys[i] == max(ys):
            aliases.append("north tank farm")
        if ys[i] == min(ys):
            aliases.append("south tank farm")
        if xs[i] == max(xs):
            aliases.append("east tank farm")
        if xs[i] == min(xs):
            aliases.append("west tank farm")
        if sizes[i] == max(sizes):
            aliases.append("largest tank farm")
        fid = f"tank_farm_tf{k}"
        add(Obj(fid, f"Tank farm TF-{k}", "tank_farm", hull, round(max(extra_h(tanks[j]) for j in g), 1),
                "steel", True, aliases, ["tank_farm"]),
            name_source="synthetic", height_source="max of member tank estimates", height_confidence=0.3,
            n_tanks=len(g), members=[f"tank_t{j + 1:03d}" for j in g], protected_status="proposed [HUMAN confirm]",
            provenance=_prov(src_date, source="derived: clustering of OSM tanks (60 m)", confidence=0.7))
        for j in g:
            extra[f"tank_t{j + 1:03d}"]["parent_id"] = f"obj:{fid}"

    # --- jetties (real pier polygons) ----------------------------------------------------------------------
    piers = []
    for e in pier_geoms:
        p = _poly(enu, e["geometry"]) or (_line(enu, e["geometry"]).buffer(8) if _line(enu, e["geometry"]) else None)
        if p is not None:
            piers.append((e["id"], p))
    piers.sort(key=lambda t: -t[1].centroid.y)
    for k, (oid, p) in enumerate(piers, start=1):
        aliases = [f"J{k}", f"jetty {k}", f"jetty J{k}"]
        if k == 1:
            aliases.append("northernmost jetty")
        if k == len(piers):
            aliases.append("southernmost jetty")
        add(Obj(f"jetty_j{k}", f"Jetty J{k}", "jetty", p, 7.0, "concrete", True, aliases, ["waterfront"]),
            osm_id=f"way/{oid}", name_source="synthetic", height_source="estimated deck height 7 m",
            height_confidence=0.3, protected_status="proposed [HUMAN confirm]", provenance=_prov(src_date))

    # --- buildings, silos, water tower, substations ---------------------------------------------------------
    nb = 0
    for e in els:
        t = e.get("tags", {})
        if e["type"] != "way" or "geometry" not in e:
            continue
        mm = t.get("man_made")
        if mm in ("storage_tank", "pier"):
            continue
        if t.get("power") == "substation":
            p = _poly(enu, e["geometry"])
            if p is None:
                continue
            name = t.get("name")
            sid = f"substation_{(name or str(e['id'])).lower().replace(' ', '_')}"
            add(Obj(sid, f"{name} substation" if name else f"Substation {e['id']}", "substation", p, 6.0, "concrete",
                    True, [name, f"{name} power substation"] if name else [], ["power"]),
                osm_id=f"way/{e['id']}", name_source="osm" if name else "synthetic",
                height_source="estimated 6 m", height_confidence=0.3, protected_status="proposed [HUMAN confirm]",
                provenance=_prov(src_date))
        elif mm in ("bunker_silo", "water_tower") or "building" in t:
            p = _poly(enu, e["geometry"])
            if p is None or p.area < 30:
                continue
            nb += 1
            kind = {"bunker_silo": "silo", "water_tower": "tower"}.get(mm, "building")
            h = {"silo": 6.0, "tower": 25.0}.get(kind, 12.0 if t.get("building") == "industrial" else 8.0)
            name = t.get("name")
            add(Obj(f"{kind}_b{nb:03d}", name or f"{kind.title()} B{nb:03d}", kind, p, h,
                    "corrugated_metal" if t.get("building") == "industrial" else "concrete", False,
                    [name] if name else [f"B{nb:03d}"], [t.get("building", mm or "")]),
                osm_id=f"way/{e['id']}", name_source="osm" if name else "synthetic",
                height_source=f"estimated class default {h} m", height_confidence=0.2, provenance=_prov(src_date))

    # --- named roads as landmarks (Urbex-style anchors) -----------------------------------------------------
    by_name = defaultdict(list)
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "way" and t.get("highway") and t.get("name") and "geometry" in e:
            ln = _line(enu, e["geometry"])
            if ln is not None:
                by_name[t["name"]].append(ln)
    for k, (name, lines) in enumerate(sorted(by_name.items()), start=1):
        geom = unary_union([ln.buffer(8) for ln in lines])
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        add(Obj(f"road_r{k}", name, "road", geom, 0.0, "ground_dry", False, ["port road"] if "Port Road" in name
                else [], ["road"]),
            name_source="osm", height_source="ground", height_confidence=1.0, provenance=_prov(src_date))

    # --- moored ships (dynamic layer, synthetic) ------------------------------------------------------------
    for k, (oid, p) in enumerate(piers[:4], start=1):
        c = p.centroid
        mrr = p.minimum_rotated_rectangle
        pts = list(mrr.exterior.coords)
        edges = [(pts[i], pts[i + 1]) for i in range(4)]
        a, b = max(edges, key=lambda ab: math.dist(*ab))
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        L, W = 180.0, 30.0
        ox, oy = c.x + 45 * math.cos(ang - math.pi / 2), c.y + 45 * math.sin(ang - math.pi / 2)
        hull = Polygon([(ox + dx * math.cos(ang) - dy * math.sin(ang), oy + dx * math.sin(ang) + dy * math.cos(ang))
                        for dx, dy in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])
        add(Obj(f"ship_s{k}", f"Moored tanker S{k}", "ship", hull, 22.0, "steel", False,
                [f"S{k}", f"tanker at jetty J{k}"], ["dynamic", "ship"]),
            name_source="synthetic", height_source="synthetic tanker (hull + superstructure)", height_confidence=0.0,
            provenance=_prov(src_date, source="synthetic: dynamic layer (ENV spec 9)", confidence=0.0,
                             status="synthetic"))

    # --- zones -------------------------------------------------------------------------------------------------
    zones: list[Zone] = []
    zprov: dict[str, dict] = {}
    h = HERO_SIZE_M / 2
    zones.append(Zone("hero_zone", "Hero zone (provisional)", "named_region", box(-h, -h, h, h),
                      ["hero zone", "demo area"]))
    zprov["hero_zone"] = {"source": "derived: densest 2 km window of OSM tanks and jetties", "status": "proposed [HUMAN]"}
    for e in els:
        t = e.get("tags", {})
        if t.get("landuse") == "industrial" and t.get("name") and "geometry" in e:
            p = _poly(enu, e["geometry"])
            if p is not None:
                zid = "kict" if "Container" in t["name"] else t["name"].lower().replace(" ", "_")[:30]
                zones.append(Zone(zid, t["name"], "named_region", p,
                                  ["KICT", "container terminal", "the container terminal"] if zid == "kict" else []))
                zprov[zid] = {"source": "osm", "status": "unverified"}
    wu = sea_polygon(els, enu, h + 2500)
    if not wu.is_empty:
        creek = max(wu.geoms, key=lambda g: g.area) if hasattr(wu, "geoms") else wu   # the creek channel itself
        zones.append(Zone("kandla_creek", "Kandla Creek", "named_region", creek,
                          ["the creek", "creek", "Kandla creek water"]))
        zprov["kandla_creek"] = {"source": "osm natural=coastline (water on the right of the way)",
                                 "status": "unverified"}
    for e in els:
        t = e.get("tags", {})
        if t.get("natural") == "wetland" and t.get("name") and e["type"] == "relation":
            outer = [m for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            polys = [_poly(enu, m["geometry"]) for m in outer]
            polys = [p for p in polys if p is not None]
            if polys:
                zones.append(Zone("sathsaida_bet", t["name"], "named_region", unary_union(polys).convex_hull,
                                  ["Sathsaida", "the wetland"]))
                zprov["sathsaida_bet"] = {"source": "osm", "status": "unverified"}
    for o in [o for o in objs if o.kind == "tank_farm"]:
        zid = f"{o.id}_restricted"
        zones.append(Zone(zid, f"{o.name} restricted zone", "restricted", o.geom.buffer(100).convex_hull,
                          [f"{a} restricted zone" for a in o.aliases[:2]]))
        zprov[zid] = {"source": "synthetic: 100 m buffer around derived tank farm", "status": "proposed [HUMAN]"}
    # jetties whose centroid lies in or near the hero zone (one OSM pier is a long wedge reaching far south)
    near = box(-h - 300, -h - 300, h + 300, h + 300)
    jets = [o.geom for o in objs if o.kind == "jetty" and near.contains(o.geom.centroid)]
    jet = unary_union(jets).convex_hull.buffer(150)
    zones.append(Zone("jetty_restricted", "Jetty restricted zone", "restricted", jet,
                      ["jetty zone", "oil jetty zone"]))
    zprov["jetty_restricted"] = {"source": f"synthetic: 150 m around the {len(jets)} OSM piers in the hero zone",
                                 "status": "proposed [HUMAN]"}
    footprint = unary_union([o.geom for o in objs if o.kind in ("tank", "jetty", "building", "silo", "substation")])
    zones.append(Zone("perimeter", "Port perimeter (provisional)", "perimeter", footprint.convex_hull.buffer(150),
                      ["port boundary", "the perimeter", "fence line"]))
    zprov["perimeter"] = {"source": "synthetic: hull of OSM port structures + 150 m", "status": "proposed [HUMAN]"}
    # Kandla Airport (IXY) aerodrome traffic zone: 5 NM radius ASSUMED around the published aerodrome reference
    # point (approx. 23.1127 N, 70.1003 E). Clipped to 8 km around the site for size. Status [HUMAN] to confirm.
    ae, an, _ = enu.to_enu(23.1127, 70.1003, 0.0)
    atz = Point(float(ae), float(an)).buffer(9260.0, 64).intersection(box(-8000, -8000, 8000, 8000))
    zones.append(Zone("ixy_atz", "Kandla Airport traffic zone (proposed)", "aviation", atz,
                      ["aviation zone", "airport zone", "ATZ", "Kandla airport"]))
    zprov["ixy_atz"] = {"source": "assumed 5 NM aerodrome traffic zone around Kandla Airport (IXY) reference point",
                        "status": "proposed [HUMAN]"}

    # --- sensors: synthetic DF nodes on mount candidates in and around the hero zone -----------------------
    hero = box(-h - 400, -h - 400, h + 400, h + 400)
    cands = []
    for o in objs:
        if not hero.contains(o.geom.centroid):
            continue
        if o.kind in ("building", "tower") and o.geom.area > 300:
            cands.append((o, o.height_m + 4.0, "roof mast"))
        if o.kind == "jetty":
            cands.append((o, 25.0, "jetty-head mast"))   # above a moored tanker (22 m) so ships do not blind it
    chosen = []
    if cands:
        chosen.append(max(cands, key=lambda c: c[1]))
        while len(chosen) < 5 and len(chosen) < len(cands):
            nxt = max(cands, key=lambda c: min(c[0].geom.centroid.distance(s[0].geom.centroid) for s in chosen))
            chosen.append(nxt)
    df_bands = ["sub_ghz_868_915", "ism_2400", "ism_5800"]
    sensors = []
    for k, (o, z, mount) in enumerate(chosen, start=1):
        c = o.geom.representative_point()
        sensors.append(Sensor(f"S{k}", "fixed_df", round(c.x, 1), round(c.y, 1), round(z, 1), df_bands,
                              mount=f"obj:{o.id} ({mount}, synthetic placement)"))
    sensors.append(Sensor("D1", "drone_df", 0.0, 0.0, 80.0, ["ism_2400", "ism_5800"], mount="drone (synthetic)"))
    for o in objs:
        extra[o.id].setdefault("mount_candidate", any(c[0].id == o.id for c in cands))

    return {"enu": enu, "objs": objs, "extra": extra, "zones": zones, "zprov": zprov, "sensors": sensors,
            "manifest": manifest, "hero_half_m": h}


def friendly_emitters(b: dict) -> list[dict]:
    """Known friendly/civil emitters the operator's system has on file (belief-visible). SYNTHETIC placements on real
    OSM mounts (ENV spec 14.2); bands and powers are typical values, [HUMAN] to replace with the port's register."""
    objs = {o.id: o for o in b["objs"]}
    out: list[dict] = []

    def add(eid, name, kind, band, freq, eirp, oid, z):
        c = objs[oid].geom.representative_point()
        out.append({"id": eid, "name": name, "kind": kind, "band": band, "freq_mhz": freq, "eirp_dbm": eirp,
                    "x_m": round(c.x, 1), "y_m": round(c.y, 1), "z_m": z, "mount_ref": f"obj:{oid}",
                    "placement": "synthetic"})

    jets = sorted([o.id for o in objs.values() if o.kind == "jetty"])
    for k, j in enumerate(jets[:3]):
        add(f"fe_cctv_{k + 1}", f"CCTV link {objs[j].name}", "cctv_link", "ism_5800", 5745.0 + 40 * k, 23.0, j, 12.0)
    farms = sorted([o.id for o in objs.values() if o.kind == "tank_farm"])
    for k, f in enumerate(farms[:4]):
        add(f"fe_lora_{k + 1}", f"Tank gauging telemetry {objs[f].name}", "tank_telemetry", "sub_ghz_868_915",
            866.5, 14.0, f, 15.0)
    bld = sorted([o for o in objs.values() if o.kind == "building" and o.geom.area > 300],
                 key=lambda o: math.hypot(o.geom.centroid.x, o.geom.centroid.y))
    if bld:
        add("fe_wifi_1", f"Port Wi-Fi {bld[0].name}", "port_wifi", "ism_2400", 2437.0, 20.0, bld[0].id, 10.0)
        far = max(bld[:40], key=lambda o: o.geom.centroid.x + o.geom.centroid.y)
        add("fe_lte_1", f"LTE site {far.name}", "lte_site", "cellular_lte", 2350.0, 46.0, far.id, 30.0)
    if len(jets) > 3:
        add("fe_vhf_1", f"Port VHF base {objs[jets[3]].name}", "vhf_base", "vhf_marine", 156.8, 40.0, jets[3], 20.0)
    return out


def extra_h(t) -> float:
    return float(np.clip(0.7 * t[2], 8, 20))


def write_build_files(b: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    enu = b["enu"]
    hero = b["zones"][0].geom
    lat, lon, _ = enu.to_geodetic(*zip(*list(hero.exterior.coords)))
    (OUT / "hero_candidates.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"id": "hero_1", "status": "proposed [HUMAN]",
                                          "rule": "densest 2 km window of OSM tanks (x1) and jetties (x10)"},
        "geometry": {"type": "Polygon", "coordinates": [[[float(lo), float(la)] for la, lo in zip(lat, lon)]]}}]}))
    counts = defaultdict(int)
    for o in b["objs"]:
        counts[o.kind] += 1
    summary = {"origin": enu.meta(), "counts": dict(counts), "zones": [z.id for z in b["zones"]],
               "sensors": [s.id for s in b["sensors"]], "osm_manifest": b["manifest"]["sha256"]}
    (OUT / "site_build.json").write_text(json.dumps(summary, indent=2))
    return OUT


def add_newarea_zones(b: dict) -> None:
    """Unseen-area test zones (cast "newarea" only; paper 2026-10-31). Two synthetic restricted zones on real Blender-
    world structures in the pocket south of the tank farms where no training scenario put a drone (occupancy map of
    seeds 1-12 and 17-28, 250 m cells). Added to the scenario's zones only, after the site-level build files."""
    from shapely.ops import unary_union

    from envbuild.blender_objects import catalogue      # STRICT: anchor on the Blender-world objects (agent-visible ids)

    by_id = {o.id: o for o in catalogue(list(b["objs"]))[0]}
    ware = [o for o in by_id.values() if o.kind == "building"
            and -800 < o.geom.centroid.x < -100 and -2250 < o.geom.centroid.y < -1750]
    if len(ware) < 6 or "jetty_j6" not in by_id:
        raise SystemExit(f"newarea zones: anchors missing (warehouses {len(ware)}, jetty_j6 {'jetty_j6' in by_id})")
    b["zones"].append(Zone("south_warehouse_restricted", "South warehouse block restricted zone", "restricted",
                           unary_union([o.geom for o in ware]).convex_hull.buffer(100),
                           ["south warehouse zone", "warehouse block restricted zone"]))
    b["zprov"]["south_warehouse_restricted"] = {
        "source": f"synthetic: 100 m around the {len(ware)} buildings of the south warehouse block (unseen-area test)",
        "status": "proposed [HUMAN]"}
    b["zones"].append(Zone("jetty_j6_restricted", "Jetty J6 restricted zone", "restricted",
                           by_id["jetty_j6"].geom.convex_hull.buffer(150), ["J6 zone", "south jetty zone"]))
    b["zprov"]["jetty_j6_restricted"] = {"source": "synthetic: 150 m around jetty J6 (unseen-area test)",
                                         "status": "proposed [HUMAN]"}


def hw_variant_desc(tag: str, override: dict, hw: dict, seed: int, sids, realised) -> dict:
    """truth/world_variant.yaml of a hardware-mismatch world (M1-M4): the tag, the override, the merged world-side
    values of the overridden keys and the realised per-node errors (`realised` = observe_v2.realised_hw_errors)."""
    return {"id": tag, "kind": "hw_mismatch", "side": "world", "seed": seed, "override": override,
            "heading_bias_deg_1sigma": hw.get("heading_datum", {}).get("bias_deg_1sigma", 0.0),
            "use_phase_sigma_deg": hw.get("impairments", {}).get("use_phase_sigma_deg", 0.0),
            "df_calibration_drift_rel_db": (hw.get("df_calibration") or {}).get("drift_rel_db"),
            "nodes": realised(hw, seed, sids)}


def geometry_label(obs_version: str, rf_scene: str) -> str:
    """meta.yaml fidelity.geometry (STRICT rule 2026-10-20: the Blender world is the production geometry; the OSM
    extrusion is only the legacy v1 stack)."""
    if obs_version == "v1":
        return "OpenStreetMap footprints (ODbL); heights estimated; no survey data"
    return (f"Blender world (data/site_kandla/blender/kandla_world.blend; built from OpenStreetMap (ODbL), Sentinel-2 "
            f"and Overture inputs); RF scene {rf_scene}; heights estimated; no survey data")


def write_site(seed: int = 0, out: Path = OUT, rf_backend: str = "air7311", observe_fn=None,
               device: str = "cuda", world_variant: str | None = None, obs_version: str = "v1",
               cast: str = "default", allow_demo_roots: bool = False, hw_override: dict | None = None,
               variant_tag: str | None = None) -> dict:
    """Build the site, run the scenario through the observation model, write the SPEC 4 products.
    world_variant (D-031): observations come from the world-side variant RF (world_cache/rf/<id>, plus its weather);
    the belief side keeps the nominal cache and sees only an opaque id; the full description goes to truth/.
    obs_version v2 reads the RF roots of the data version (SCENESCOUT_DATA_VERSION, envbuild.variant_roots): world
    and nominal caches, node links, variant manifests and meta.rf_cache_dir; the spectrum environment is shared.
    cast: sim/world/kandla_scenario.make_emitters cast ("default" unchanged; "varied" for test seeds, S7.3).
    `out` is checked against the demo-root guard (allow_demo_roots: --allow-demo-roots).
    hw_override (mismatch worlds M1-M4, plan 2026-10-28 T6; obs v2 only, never with world_variant): deep-merged into
    the world-side hardware config (observe_v2.load_hw); the belief side keeps the nominal caches and meta. The tag,
    the override and the realised per-node errors go to truth/world_variant.yaml only (audit B1, MINOR 2)."""
    import copy

    from sim.events import extract_events
    from sim.obs_model.grouping import group_tracks
    from sim.obs_model.observe import observe
    from sim.stub_site import write_products
    from envbuild.variant_roots import data_roots, guard_write, rel
    from sim.world.kandla_scenario import make_emitters

    guard_write(out, allow_demo_roots)
    if hw_override is not None and (obs_version != "v2" or world_variant or not variant_tag):
        raise SystemExit("hw_override needs obs_version v2, a variant_tag and no world_variant")
    roots = data_roots()
    b = build(seed)
    write_build_files(b)
    if cast in ("newarea", "na_u", "na_f"):       # both paired arms carry the same zone list
        add_newarea_zones(b)
    zmap = {z.id: z for z in b["zones"]}
    emitters = make_emitters(seed, zmap, b["objs"], cast=cast)
    from sim.belief.rf_cache import RFCache

    rf_root = OUT / "rf_cache"                       # one shared RF cache for every scenario seed
    weather, variant_desc = None, None
    if world_variant and obs_version == "v1":
        from envbuild.rf_variants import load_variants

        var = load_variants()[world_variant]
        if var.side != "world":
            raise SystemExit(f"{world_variant} is not a world variant")
        rf_root = OUT / "world_cache" / "rf" / world_variant
        weather = var.weather if (var.weather.get("rain_mm_h") or var.weather.get("rh_pct", 60) != 60) else None
        variant_desc = json.loads((OUT / "world_cache" / "rf_scene" / world_variant / "manifest.json").read_text())
    cache = RFCache(rf_root / "sionna_rt")
    if obs_version == "v2":
        # RF phase 2: protocols per band + controllers; detection-driven IQ observations (sim.obs_model.observe_v2)
        from shapely import STRtree
        from shapely.geometry import Point

        from sim.obs_model.observe_v2 import observe_v2, products_v2
        from sim.world.scenario_v2 import apply_v2

        solid = [o.geom for o in b["objs"] if o.kind in ("building", "tank", "jetty", "ship", "silo", "substation")]
        tree = STRtree(solid)
        sea = next((z.geom for z in b["zones"] if z.id == "kandla_creek"), None)

        def land_ok(x, y):
            pt = Point(x, y)
            return not (sea is not None and sea.contains(pt)) and not any(solid[i].contains(pt) for i in tree.query(pt))

        emitters = apply_v2(emitters, seed, land_ok)
        from envbuild.blender_objects import catalogue          # STRICT: Blender world objects for tools/console
        b["objs"], cat_stats = catalogue(b["objs"])
        b["extra"] = {k: v for k, v in b["extra"].items() if any(o.id == k for o in b["objs"])}
        w_root = (roots["world_cache"] / "rf" / world_variant) if world_variant else roots["rf_cache"]
        env = roots["spectrum_env"]                       # shared world-side truth (every data version)
        from sim.obs_model.observe_v2 import load_hw, realised_hw_errors

        hw = load_hw(hw_override)                         # world side only (mismatch worlds M*)
        sens = [s for s in b["sensors"]]
        from sim.world.scenario_v2 import jammer_events

        jams = jammer_events(seed, [s.id for s in sens if s.type == "fixed_df"], hw.get("tx", {}).get("max_eirp_dbm", 47.0))
        lf = roots["rf_cache"] / "node_links.json"
        links = json.loads(lf.read_text())["links"] if lf.exists() else None
        dets = observe_v2(seed, emitters, sens, w_root, env if env.exists() else None, device=device, hw=hw,
                          jammers=jams, links=links)
        bgp = {e["id"]: (e["x_m"], e["y_m"]) for e in yaml.safe_load(env.read_text())["emitters"]} if env.exists() else {}
        obs = products_v2(seed, emitters, sens, dets, w_root, hw["receiver"]["gnss_timing_ns_1sigma"], bgp, jams)
        variant_desc = (json.loads((roots["world_cache"] / "rf_scene" / world_variant / "manifest.json").read_text())
                        if world_variant else None)
        if hw_override is not None:
            variant_desc = hw_variant_desc(variant_tag, hw_override, hw, seed,
                                           [s.id for s in sens if s.type == "fixed_df"], realised_hw_errors)
    elif observe_fn is None and rf_backend == "air7311":
        from sim.obs_model.observe_air7311 import observe_air7311

        if not (rf_root / "air7311_S1").exists():
            raise SystemExit("no AIR7311 caches: run envbuild.air7311_precompute (Windows) for S1-S5 first")
        obs = observe_air7311(seed, emitters, b, rf_root, device=device, weather=weather)
        hw = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
        for s in b["sensors"]:
            if s.type == "fixed_df":
                s.array = {"device": "AIR7311", "kind": "square_4_omni", "elements": 4,
                           "side_m": {"ism_5800": hw["antenna"]["candidate_sides_m"]["A_5g8"],
                                      "ism_2400": hw["antenna"]["candidate_sides_m"]["B_2g4"],
                                      "sub_ghz_868_915": hw["antenna"]["candidate_sides_m"]["C_sub1g"]},
                           "coverage_radius_m": 1500.0, "note": "simulation profile; values [HUMAN] to confirm"}
    elif observe_fn is None and rf_backend == "sionna_rt":
        if not cache.available():
            raise SystemExit("no Sionna cache: run envbuild.export_rf then envbuild.rf_precompute (Windows) first")
        from sim.obs_model.observe_rt import observe_rt

        obs = observe_rt(seed, emitters, b, cache)
    elif observe_fn is None:
        # geometric stand-in: reflect off simplified tank-farm hulls, jetties, ships, buildings (not single tanks)
        rf_objs = []
        for o in b["objs"]:
            o2 = copy.copy(o)
            if o.kind == "tank_farm":
                o2.geom = o.geom.simplify(15)
            rf_objs.append(o2)
        obs = observe(seed, emitters, rf_objs, b["sensors"])
    else:
        obs = observe_fn(seed, emitters, b)
    obs["groups"] = group_tracks(obs["points"])
    events = extract_events(obs["points"], b["zones"])
    enu = b["enu"]
    hh = b["hero_half_m"]
    meta = {"site": "site_kandla",
            "frame": {"type": "ENU", "units": "m", "bearing": "deg clockwise from true north", "origin": enu.meta()},
            "extent_m": [-hh - 2500, -hh - 2500, hh + 2500, hh + 2500],
            "hero_zone_m": [-hh, -hh, hh, hh], "rf_backend": rf_backend,
            "rf_cache_dir": "data/site_kandla/rf_cache" if obs_version == "v1" else rel(roots["rf_cache"]),
            "fidelity": {"geometry": geometry_label(obs_version, rel(roots["rf_scene"])),
                         "names": "OSM names where present; other labels synthetic",
                         "sensors": "synthetic placements", "emitters": "synthetic scenario",
                         "rf": {"air7311": "Sionna RT paths + Deepwave AIR7311 4-channel DF model (simulation)"}
                         .get(rf_backend, rf_backend)},
            "attribution": "(c) OpenStreetMap contributors"}
    if world_variant:
        meta["world_variant_id"] = world_variant          # opaque bookkeeping id only (world state lives in truth/)
    if cast != "default":
        meta["cast"] = cast
    if obs_version == "v2":
        meta["obs_version"] = "v2"
        from sim.obs_model.observe_v2 import observe_v2 as _ov2

        meta["df_coupling_cal"] = bool(getattr(_ov2, "coupling_cal", False))   # calibrated DF manifold (sim.df.calibration)
        meta["fidelity"]["rf"] = ("Sionna RT paths (textbook UTD diffraction) + simulated IQ through an AIR7311 "
                                  "receive chain; detection-driven tracking (simulation)")
    full = write_products(out, b["objs"], b["zones"], b["sensors"], emitters, obs, events, seed, meta,
                          obj_extra=b["extra"], zone_extra=b["zprov"])
    if variant_desc is not None:
        (Path(out) / "truth" / "world_variant.yaml").write_text(yaml.safe_dump(variant_desc, sort_keys=False))
    (Path(out) / "friendly_emitters.yaml").write_text(yaml.safe_dump({"emitters": friendly_emitters(b)},
                                                                      sort_keys=False))
    return full


if __name__ == "__main__":
    import sys

    if "--build-only" in sys.argv:
        b = build()
        write_build_files(b)
    else:
        backend = "geometric_stub" if "--geometric" in sys.argv else "sionna_rt" if "--sionna-noise" in sys.argv             else "air7311"
        cast = sys.argv[sys.argv.index("--cast") + 1] if "--cast" in sys.argv else "default"
        m = write_site(rf_backend=backend, cast=cast, allow_demo_roots="--allow-demo-roots" in sys.argv)
        print(json.dumps(m["counts"]), m["version_hash"], m["rf_backend"])
        sys.exit(0)
    print(json.dumps({"origin": b["enu"].meta(), "objects": len(b["objs"]),
                      "kinds": {k: sum(o.kind == k for o in b["objs"]) for k in sorted({o.kind for o in b["objs"]})},
                      "zones": [z.id for z in b["zones"]], "sensors": [(s.id, s.mount) for s in b["sensors"]]},
                     indent=1))
