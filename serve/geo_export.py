"""Belief-side site and track data converted to WGS-84 lon/lat for the 3D operator UI (ENV SPEC 13).

Reads only the belief products (meta.yaml, objects.parquet, zones.geojson, sensors.yaml, observed.duckdb) through
SiteView plus the provenance columns of objects.parquet. Nothing here touches the ground-truth side (SPEC 12.1).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from envbuild.geo import ENU
from sim.belief.site_view import SiteView

# Sites without a georeferenced frame (the procedural stub) are drawn at a nominal origin so the 3D UI still works.
NOMINAL_ORIGIN = {"lat": 23.0327, "lon": 70.2125}


def origin(site: SiteView) -> tuple[float, float, bool]:
    o = (site.meta.get("frame") or {}).get("origin") or {}
    if o.get("lat") is not None and o.get("lon") is not None:
        return float(o["lat"]), float(o["lon"]), True
    return NOMINAL_ORIGIN["lat"], NOMINAL_ORIGIN["lon"], False


def _enu(site: SiteView) -> ENU:
    lat0, lon0, _ = origin(site)
    return ENU(lat0, lon0)


def _r(v: float, nd: int) -> float:
    return round(float(v), nd)


def ring_lonlat(enu: ENU, xy) -> list[list[float]]:
    xy = np.asarray(xy, float)
    lat, lon, _ = enu.to_geodetic(xy[:, 0], xy[:, 1], 0.0)
    return [[_r(a, 7), _r(b, 7)] for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat), strict=True)]


def point_lonlat(enu: ENU, x: float, y: float) -> tuple[float, float]:
    lat, lon, _ = enu.to_geodetic(x, y, 0.0)
    return _r(lon, 7), _r(lat, 7)


def _exterior(geom):
    """Exterior ring of a polygon (largest part of a multipolygon); None for other geometries."""
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return None
    return list(geom.exterior.coords)


def _json(v):
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return v if isinstance(v, dict) else {}


def _clean(v):
    """parquet NaN / numpy scalars -> JSON-safe python values."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _provenance_rows(site_dir: Path) -> dict[str, dict]:
    """Per-object provenance columns (absent on older sites such as the stub)."""
    table = pq.read_table(site_dir / "objects.parquet")
    cols = set(table.column_names)
    want = [c for c in ("id", "osm_id", "name_source", "height_source", "height_confidence", "parent_id",
                        "mount_candidate", "protected_status", "provenance", "material", "aliases") if c in cols]
    rows = table.select(want).to_pylist()
    return {r["id"]: r for r in rows}


def provenance_summary(row: dict) -> dict:
    prov = _json(row.get("provenance"))
    src = _clean(prov.get("source")) or ("osm" if _clean(row.get("osm_id")) else None)
    status = _clean(prov.get("status"))
    hsrc = _clean(row.get("height_source"))
    if src and src.startswith("synthetic"):
        cls = "synthetic"
    elif src and src.startswith("derived"):
        cls = "derived"
    elif src == "osm":
        cls = "osm_estimated" if hsrc and str(hsrc).startswith("estimated") else "osm"
    else:
        cls = "unknown"
    return {"source": src, "status": status, "name_source": _clean(row.get("name_source")),
            "height_source": hsrc, "height_confidence": _clean(row.get("height_confidence")),
            "confidence": _clean(prov.get("confidence")), "source_date": _clean(prov.get("source_date")),
            "osm_id": _clean(row.get("osm_id")), "protected_status": _clean(row.get("protected_status")),
            "class": cls}


@lru_cache(maxsize=4)
def site_geo(site_dir: str, version: str, site: SiteView) -> dict:  # noqa: ARG001 - version keys the cache
    enu = _enu(site)
    lat0, lon0, georef = origin(site)
    prov_rows = _provenance_rows(Path(site_dir))
    zone_props = {}
    zpath = Path(site_dir) / "zones.geojson"
    for f in json.loads(zpath.read_text())["features"]:
        zone_props[f["properties"]["id"]] = f["properties"]

    objects = []
    for o in site.objects.values():
        ring = _exterior(o.geom)
        if ring is None:
            continue
        row = prov_rows.get(o.id, {})
        c = o.geom.centroid
        clon, clat = point_lonlat(enu, c.x, c.y)
        objects.append({
            "ref": o.ref, "name": o.name, "kind": o.kind, "aliases": list(o.aliases),
            "height_m": _clean(o.height_m), "material": _clean(o.material), "protected": bool(o.protected),
            "parent": _clean(row.get("parent_id")), "mount_candidate": str(_clean(row.get("mount_candidate"))) == "True",
            "provenance": provenance_summary(row), "center": [clon, clat],
            "polygon": ring_lonlat(enu, ring),
        })

    zones = []
    for z in site.zones.values():
        ring = _exterior(z.geom)
        if ring is None:
            continue
        props = zone_props.get(z.id, {})
        prov = props.get("provenance") or {}
        c = z.geom.centroid
        clon, clat = point_lonlat(enu, c.x, c.y)
        zones.append({"ref": z.ref, "name": z.name, "kind": z.kind, "aliases": list(z.aliases),
                      "provenance": {"source": prov.get("source"), "status": prov.get("status")},
                      "center": [clon, clat], "polygon": ring_lonlat(enu, ring)})

    sensors = []
    for s in site.sensors.values():
        p = s["pose"]
        lon, lat = point_lonlat(enu, p["x_m"], p["y_m"])
        sensors.append({"ref": f"sensor:{s['id']}", "id": s["id"], "type": s.get("type"), "lon": lon, "lat": lat,
                        "z_m": p.get("z_m", 0.0), "x_m": p["x_m"], "y_m": p["y_m"], "mount": s.get("mount"),
                        "placement": s.get("placement"), "bands": s.get("bands", [])})

    m = site.meta
    hz = m.get("hero_zone_m")
    hero = ring_lonlat(enu, [(hz[0], hz[1]), (hz[2], hz[1]), (hz[2], hz[3]), (hz[0], hz[3]), (hz[0], hz[1])]) \
        if hz else None
    ex = m.get("extent_m")
    extent = ring_lonlat(enu, [(ex[0], ex[1]), (ex[2], ex[1]), (ex[2], ex[3]), (ex[0], ex[3]), (ex[0], ex[1])]) \
        if ex else None
    fid = m.get("fidelity") or {}
    return {
        "objects": objects, "zones": zones, "sensors": sensors,
        "meta": {"site": m.get("site"), "data_version": site.version,
                 "origin": {"lat": lat0, "lon": lon0, "georeferenced": georef},
                 "fidelity": fid, "rf_backend": m.get("rf_backend") or fid.get("rf") or "unknown",
                 "attribution": m.get("attribution") or "", "hero_zone": hero, "hero_zone_m": hz,
                 "extent": extent, "extent_m": ex, "t0_iso": (m.get("time") or {}).get("t0_iso"),
                 "duration_s": (m.get("time") or {}).get("duration_s"), "counts": m.get("counts", {})},
    }


@lru_cache(maxsize=8)
def tracks_geo(site_dir: str, version: str, site: SiteView, step_s: float) -> list[dict]:  # noqa: ARG001
    enu = _enu(site)
    out = []
    for tid, tr in site.tracks.items():
        p = site.points[tid]
        t = p["t"]
        if len(t) == 0:
            continue
        keep = [i for i in range(len(t)) if abs(t[i] / step_s - round(t[i] / step_s)) < 1e-6 or i == len(t) - 1]
        lat, lon, _ = enu.to_geodetic(p["x"][keep], p["y"][keep], 0.0)
        lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
        pts = [[_r(t[i], 1), _r(lon[k], 7), _r(lat[k], 7), _r(p["z"][i], 1), _r(p["a"][i], 1), _r(p["b"][i], 1),
                _r(p["theta"][i], 1)] for k, i in enumerate(keep)]
        probs = tr.get("klass_probs") or {}
        out.append({"ref": f"trk:{tid}", "band": tr.get("band"), "freq_mhz": _clean(tr.get("freq_mhz")),
                    "klass": max(probs, key=probs.get) if probs else "unknown", "klass_probs": probs,
                    "confidence": _clean(tr.get("confidence")), "group": _clean(tr.get("group_id")) or None,
                    "first_seen": _clean(tr.get("first_seen")), "last_seen": _clean(tr.get("last_seen")),
                    "pts": pts})
    return out


def bearings_at(site: SiteView, t: float | None, track: str | None, window_s: float = 10.0) -> dict:
    """Observed DF bearings nearest to t (optionally for one track), grouped by sensor with geodetic sensor poses."""
    enu = _enu(site)
    con = site.cursor()
    tid = track.split(":", 1)[1].split("@", 1)[0] if track else None
    where, args = [], []
    if tid is not None:
        where.append("track_id = ?")
        args.append(tid)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    if t is None:
        row = con.execute(f"SELECT max(t_s) FROM bearings{clause}", args).fetchone()
    else:
        row = con.execute(f"SELECT t_s FROM bearings{clause} ORDER BY abs(t_s - ?), t_s LIMIT 1",
                          [*args, t]).fetchone()
    if row is None or row[0] is None or (t is not None and abs(row[0] - t) > window_s):
        return {"t_s": None, "track": track, "sensors": []}
    ts = float(row[0])
    rows = con.execute(f"SELECT hyp_id, track_id, sensor_id, bearing_deg, snr_db, power_db FROM bearings"
                       f"{clause}{' AND' if clause else ' WHERE'} t_s = ? ORDER BY sensor_id, track_id, bearing_deg",
                       [*args, ts]).fetchall()
    per: dict[str, dict] = {}
    for hid, trk, sid, brg, snr, pwr in rows:
        s = site.sensors.get(sid)
        if s is None:
            continue
        pose = s["pose"]
        if sid not in per:
            lon, lat = point_lonlat(enu, pose["x_m"], pose["y_m"])
            per[sid] = {"ref": f"sensor:{sid}", "lon": lon, "lat": lat, "z_m": pose.get("z_m", 0.0),
                        "x_m": pose["x_m"], "y_m": pose["y_m"], "bearings": []}
        per[sid]["bearings"].append({"hyp": f"hyp:{hid}@{ts:g}", "track": f"trk:{trk}", "bearing_deg": _r(brg, 1),
                                     "snr_db": _r(snr, 1) if snr is not None else None,
                                     "power_db": _r(pwr, 1) if pwr is not None else None})
    return {"t_s": ts, "track": track, "sensors": list(per.values())}
