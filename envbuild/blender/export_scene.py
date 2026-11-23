"""Export the Kandla twin to a Blender-ready bundle (runs in the WSL venv; Blender's Python has no shapely/pyarrow).

    python -m envbuild.blender.export_scene   -> data/site_kandla/blender/{scene_input.json, terrain.npz}

Everything is in the site ENU frame (metres, x east, y north, z up, z = 0 is the land surface used by the RF scene and
the simulation), so the visual world, the Sionna scene and the sensor poses line up exactly.

Fidelity classes carried on every item (ENV spec 0; world brief "verified / estimated / illustrative"):
- `osm`          : footprint or centreline from OpenStreetMap (ODbL), not survey-verified;
- `estimated`    : a value OSM does not hold (heights, road widths, shore slope, water level);
- `illustrative` : procedural detail added for realism (stairs, loading arms, piles, vegetation) and all synthetic
                   objects (moored ships). Never a claim about what is on site.
Nothing is `verified` until a human confirms it against survey data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import shapely
import yaml
from shapely import wkt as swkt
from shapely.geometry import LineString, Point, box

from envbuild.geo import ENU
from envbuild.site_kandla import _load_osm, sea_polygon

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
OUT = SITE / "blender"
TILE = (-4800.0, 4800.0, -6400.0, 3200.0)   # x0, x1, y0, y1 (m): oil jetties, tank farms, township, old cargo port
HALF_M = 4800.0          # half width (legacy field)
GRID_M = 5.0             # terrain heightfield / mask resolution

# Estimated values (OSM has none of these). Kept in one place and written into the bundle.
EST = {
    "land_z_m": 0.0,                 # ENU z = 0 is the port land surface (same as RF scene and sim)
    "water_z_m": -3.5,               # mean water level below quay level (estimated; Kandla has ~6 m tides)
    "bed_z_m": -11.0,                # creek bed near the banks (estimated)
    "shore_land_m": 25.0,            # land falls to the water over this distance (estimated bank slope)
    "shore_water_m": 60.0,           # bed reaches bed_z this far offshore (estimated)
    "road_width_m": {"trunk": 14.0, "primary": 12.0, "secondary": 10.0, "tertiary": 8.0, "tertiary_link": 6.0,
                     "unclassified": 6.0, "service": 4.5},
    "power_tower_h_m": 32.0,
    "jetty_deck_z_m": 4.5,           # deck top above quay level (~8 m above mean water; clears high water with ~6 m tides)
}


def _coords(geom) -> list[list[float]]:
    return [[round(x, 2), round(y, 2)] for x, y in list(geom.coords)]


def _rings(geom) -> dict:
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return {"outer": _coords(geom.exterior)[:-1], "holes": [_coords(r)[:-1] for r in geom.interiors]}


def _objects(frame) -> list[dict]:
    rows = pq.read_table(SITE / "objects.parquet").to_pylist()
    out = []
    for r in rows:
        g = swkt.loads(r["wkt"])
        if not frame.intersects(g):
            continue
        prov = json.loads(r["provenance"]) if r.get("provenance") else {}
        src = str(prov.get("source") or "")
        fid = "illustrative" if src.startswith("synthetic") else "osm"
        item = {"id": r["id"], "name": r["name"], "kind": r["kind"], "height_m": r["height_m"],
                "material": r["material"], "osm_id": r.get("osm_id"), "fidelity": fid,
                "height_fidelity": "estimated", "tags": list(r.get("tags") or [])}
        if g.geom_type in ("Polygon", "MultiPolygon"):
            item.update(_rings(g))
            c = g.centroid
            item["centroid"] = [round(c.x, 2), round(c.y, 2)]
            item["area_m2"] = round(g.area, 1)
            if r["kind"] == "tank":
                item["diameter_m"] = round(2 * float(np.sqrt(g.area / np.pi)), 2)
            if r["kind"] in ("jetty", "ship", "building"):
                mrr = g.minimum_rotated_rectangle
                p = list(mrr.exterior.coords)
                e = [(p[i], p[i + 1]) for i in range(4)]
                a, b = max(e, key=lambda ab: np.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]))
                item["axis_deg"] = round(float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))), 2)
                s = sorted(np.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]) for q in e)
                item["mrr_wl_m"] = [round(float(s[0]), 2), round(float(s[-1]), 2)]
        out.append(item)
    return out


def _osm_lines(els, enu: ENU, frame) -> dict:
    lines = {"road": [], "rail": [], "power_line": [], "pipeline": [], "fence": []}
    points = {"power_tower": []}
    wet, industrial = [], []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "node" and t.get("power") == "tower":
            x, y, _ = enu.to_enu(e["lat"], e["lon"])
            if frame.contains(Point(float(x), float(y))):
                points["power_tower"].append({"osm_id": f"node/{e['id']}", "xy": [round(float(x), 2), round(float(y), 2)],
                                              "fidelity": "osm", "height_fidelity": "estimated"})
            continue
        if e["type"] != "way" or "geometry" not in e:
            continue
        g = e["geometry"]
        x, y, _ = enu.to_enu([p["lat"] for p in g], [p["lon"] for p in g])
        ln = LineString(np.column_stack([x, y]))
        if not frame.intersects(ln):
            continue
        rec = {"osm_id": f"way/{e['id']}", "fidelity": "osm"}
        if t.get("highway") in EST["road_width_m"]:
            parts = ln.intersection(frame)
            for p in getattr(parts, "geoms", [parts]):
                if p.geom_type == "LineString" and p.length > 2:
                    lines["road"].append({**rec, "class": t["highway"], "name": t.get("name"),
                                          "width_m": EST["road_width_m"][t["highway"]], "width_fidelity": "estimated",
                                          "xy": _coords(p)})
        elif t.get("railway") in ("rail", "construction"):
            lines["rail"].append({**rec, "status": t["railway"], "xy": _coords(ln)})
        elif t.get("power") == "line":
            lines["power_line"].append({**rec, "xy": _coords(ln)})
        elif t.get("man_made") == "pipeline":
            lines["pipeline"].append({**rec, "xy": _coords(ln)})
        elif t.get("barrier") == "fence":
            lines["fence"].append({**rec, "xy": _coords(ln)})
        elif t.get("natural") == "wetland" and len(g) >= 4:
            wet.append(ln.convex_hull if not ln.is_ring else shapely.Polygon(ln.coords))
        elif t.get("landuse") in ("industrial", "port", "railway") and len(g) >= 4 and ln.is_ring:
            industrial.append(shapely.Polygon(ln.coords).buffer(0))
    for e in els:  # wetland relations (Sathsaida Bet)
        t = e.get("tags", {})
        if e["type"] == "relation" and t.get("natural") == "wetland":
            for m in e.get("members", []):
                if m.get("role") == "outer" and m.get("geometry") and len(m["geometry"]) >= 4:
                    gx, gy, _ = enu.to_enu([p["lat"] for p in m["geometry"]], [p["lon"] for p in m["geometry"]])
                    pg = shapely.Polygon(np.column_stack([gx, gy])).buffer(0)
                    if pg.area > 1:
                        wet.append(pg)
    return {"lines": lines, "points": points, "wetland": wet, "industrial": industrial}


OVERTURE = REPO / "data" / "raw" / "overture" / "2026-09-23.1" / "kandla_buildings.parquet"


def _est_height(area: float, elong: float) -> tuple[float, str]:
    """Height rule for footprints without heights (estimated; tuned by eye against the imagery)."""
    if area < 80:
        return 3.5, "hut/container/small structure"
    if area < 300:
        return 6.0, "small building"
    if area < 1000:
        return 8.5, "building"
    if elong > 2.5 or area > 3000:
        return 13.0, "warehouse/shed"
    return 11.0, "industrial building"


def _overture(enu: ENU, frame, osm_objs: list[dict]) -> list[dict]:
    """Overture Maps buildings (Google Open Buildings CC BY 4.0 / Microsoft ML Buildings ODbL / OSM ODbL) that are
    not already in the OSM twin. Footprints are ML-derived; heights are estimated by size and shape."""
    if not OVERTURE.exists():
        return []
    have = [shapely.Polygon(o["outer"]) for o in osm_objs if o.get("outer") and len(o["outer"]) >= 3]
    tree = shapely.STRtree(have)
    out = []
    for r in pq.read_table(OVERTURE).to_pylist():
        if r.get("class") == "storage_tank":
            continue
        g = shapely.from_wkb(r["wkb"])
        g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)
        xy = np.asarray(g.exterior.coords)
        e, n, _ = enu.to_enu(xy[:, 1], xy[:, 0])
        p = shapely.Polygon(np.column_stack([e, n])).buffer(0)
        if p.is_empty or p.area < 25 or not frame.contains(p.centroid):
            continue
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda q: q.area)
        hits = tree.query(p)
        if any(have[i].intersection(p).area > 0.3 * min(p.area, have[i].area) for i in hits):
            continue
        mrr = p.minimum_rotated_rectangle
        q = list(mrr.exterior.coords)
        edges = sorted(((float(np.hypot(q[i + 1][0] - q[i][0], q[i + 1][1] - q[i][1])), i) for i in range(4)))
        w, ln = edges[0][0], edges[-1][0]
        i = edges[-1][1]
        axis = float(np.degrees(np.arctan2(q[i + 1][1] - q[i][1], q[i + 1][0] - q[i][0])))
        h, kind_note = _est_height(p.area, ln / max(w, 0.1))
        src = (r.get("datasets") or ["?"])[0]
        out.append({"id": f"ovt_{r['id'][:12]}", "name": r.get("name") or "", "kind": "building", "height_m": h,
                    "material": "corrugated_metal" if h >= 11 else "concrete", "osm_id": None,
                    "fidelity": "ml_footprint", "height_fidelity": "estimated", "tags": [kind_note],
                    "footprint_source": src, "outer": _coords(p.exterior)[:-1], "holes": [],
                    "centroid": [round(p.centroid.x, 2), round(p.centroid.y, 2)], "area_m2": round(p.area, 1),
                    "axis_deg": round(axis, 2), "mrr_wl_m": [round(w, 2), round(ln, 2)]})
    return out


def _raster(polys, xs, ys) -> np.ndarray:
    """Boolean mask of grid points inside any polygon."""
    m = np.zeros((len(ys), len(xs)), bool)
    if not polys:
        return m
    u = shapely.union_all([p for p in polys if not p.is_empty])
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel(), Y.ravel())
    shapely.prepare(u)
    return shapely.contains(u, pts).reshape(m.shape)


def _terrain(water, wet, industrial, objs) -> dict:
    xs = np.arange(TILE[0], TILE[1] + 0.1, GRID_M)
    ys = np.arange(TILE[2], TILE[3] + 0.1, GRID_M)
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel(), Y.ravel())
    is_water = _raster([water], xs, ys)
    # signed distance to the shoreline: + on land, - on water (computed on a 10 m grid, then upsampled)
    shore = water.boundary
    shapely.prepare(shore)
    xc = np.arange(TILE[0], TILE[1] + 0.1, 2 * GRID_M)
    yc = np.arange(TILE[2], TILE[3] + 0.1, 2 * GRID_M)
    Xc, Yc = np.meshgrid(xc, yc)
    dc = shapely.distance(shore, shapely.points(Xc.ravel(), Yc.ravel())).reshape(Xc.shape)
    d = np.repeat(np.repeat(dc, 2, 0), 2, 1)[: len(ys), : len(xs)]
    sd = np.where(is_water, -d, d)
    t_land = np.clip(sd / EST["shore_land_m"], 0, 1)
    t_water = np.clip(-sd / EST["shore_water_m"], 0, 1)
    smooth = lambda t: t * t * (3 - 2 * t)  # noqa: E731
    z = np.where(sd >= 0,
                 EST["water_z_m"] + (EST["land_z_m"] - EST["water_z_m"]) * smooth(t_land),
                 EST["water_z_m"] + (EST["bed_z_m"] - EST["water_z_m"]) * smooth(t_water))
    # built-up ground (tank farms, buildings, jetty roots) is flat quay level
    built = [shapely.buffer(shapely.Polygon(o["outer"]), 12.0) for o in objs
             if o["kind"] in ("tank", "tank_farm", "building", "silo", "substation") and len(o["outer"]) >= 3]
    m_built = _raster(built + industrial, xs, ys) & ~is_water
    z = np.where(m_built, np.maximum(z, EST["land_z_m"]), z)
    del pts
    return {"x": xs.astype(np.float32), "y": ys.astype(np.float32), "z": z.astype(np.float32),
            "sd": sd.astype(np.float32), "wet": _raster(wet, xs, ys), "industrial": _raster(industrial, xs, ys),
            "built": m_built}


def main() -> None:
    meta = yaml.safe_load((SITE / "meta.yaml").read_text())
    o = meta["frame"]["origin"]
    enu = ENU(float(o["lat"]), float(o["lon"]))
    frame = box(TILE[0], TILE[2], TILE[1], TILE[3])
    els, manifest = _load_osm()
    # wider OSM coastline (the original fetch stops 5.3 km south; the creek runs on to the gulf)
    import glob
    extra = sorted(glob.glob(str(REPO / "data" / "raw" / "osm" / "extra_*" / "coastline.json")))
    coast_els = list(els)
    if extra:
        have = {e["id"] for e in els}
        coast_els += [e for e in json.loads(Path(extra[-1]).read_text())["elements"] if e["id"] not in have]
    water = sea_polygon(coast_els, enu, max(abs(v) for v in TILE) + 200.0).intersection(frame)
    objs = _objects(frame)
    for ob in objs:    # the cargo-wharf polygon reaches into the creek: keep land plus a 25 m quay apron
        if ob["kind"] == "jetty" and ob.get("area_m2", 0) > 100000:
            p = shapely.Polygon(ob["outer"]).buffer(0).difference(water.buffer(-25.0))
            p = max(p.geoms, key=lambda g: g.area) if p.geom_type == "MultiPolygon" else p
            ob["outer"] = _coords(p.exterior)[:-1]
            ob["area_m2"] = round(p.area, 1)
            ob["centroid"] = [round(p.centroid.x, 2), round(p.centroid.y, 2)]
            ob["wharf"] = True
    ovt = _overture(enu, frame, objs)
    objs += ovt
    osm = _osm_lines(els, enu, frame)
    terr = _terrain(water, osm["wetland"], osm["industrial"], objs)
    sensors = yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
    zones = json.loads((SITE / "zones.geojson").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "terrain.npz", **terr)
    bundle = {
        "schema": "scenescout-blender-1",
        "origin": {"lat": o["lat"], "lon": o["lon"], "frame": "ENU metres, z=0 land surface"},
        "half_m": HALF_M, "tile": list(TILE), "grid_m": GRID_M, "estimated": EST,
        "osm": {"attribution": manifest["attribution"], "license": manifest["license"],
                "retrieved_at": manifest["retrieved_at"], "sha256": manifest["sha256"]},
        "overture": {"release": "2026-09-23.1", "n": len(ovt),
                     "attribution": "Overture Maps Foundation; Google Open Buildings (CC BY 4.0); Microsoft ML "
                                    "Buildings (ODbL); OpenStreetMap contributors (ODbL)"},
        "objects": objs, "lines": osm["lines"], "points": osm["points"],
        "sensors": [{"id": s["id"], "type": s["type"], "xyz": [s["pose"]["x_m"], s["pose"]["y_m"], s["pose"]["z_m"]],
                     "fidelity": "illustrative", "note": "synthetic placement"} for s in sensors],
        "zones": [{"id": f["properties"]["id"], "kind": f["properties"].get("kind")} for f in zones["features"]],
    }
    (OUT / "scene_input.json").write_text(json.dumps(bundle))
    counts = {k: len(v) for k, v in osm["lines"].items()}
    kinds: dict[str, int] = {}
    for ob in objs:
        kinds[ob["kind"]] = kinds.get(ob["kind"], 0) + 1
    print(json.dumps({"objects": kinds, "lines": counts, "towers": len(osm["points"]["power_tower"]),
                      "terrain": list(terr["z"].shape), "water_frac": round(float((terr["sd"] < 0).mean()), 3),
                      "out": str(OUT)}))


if __name__ == "__main__":
    main()
