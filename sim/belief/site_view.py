"""Read-only view of a site's BELIEF-side products. The only data access tools have.

Loads exactly: meta.yaml, objects.parquet, zones.geojson, sensors.yaml, observed.duckdb (read_only).
There is deliberately no code path to truth/ or world_cache/ (SPEC 12.1); the runtime guard enforces it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import yaml
from shapely import STRtree, wkt
from shapely.geometry import Point, shape
from shapely.prepared import prep

BELIEF_FILES = ("meta.yaml", "objects.parquet", "zones.geojson", "sensors.yaml", "observed.duckdb")
REPO = Path(__file__).resolve().parents[2]


@dataclass
class SiteObj:
    ref: str
    id: str
    name: str
    kind: str
    geom: object
    height_m: float | None = None
    material: str | None = None
    protected: bool = False
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    family: str = "obj"          # obj | zone


class SiteView:
    def __init__(self, site_dir: str | Path):
        self.dir = Path(site_dir)
        missing = [f for f in BELIEF_FILES if not (self.dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"site {self.dir} is missing belief products: {missing}")
        self.meta = yaml.safe_load((self.dir / "meta.yaml").read_text())
        self.version = self.meta["version_hash"]
        self.bands: dict[str, list[float]] = self.meta["bands"]

        t = pq.read_table(self.dir / "objects.parquet").to_pylist()
        self.objects = {r["id"]: SiteObj(f"obj:{r['id']}", r["id"], r["name"], r["kind"], wkt.loads(r["wkt"]),
                                         r["height_m"], r["material"], r["protected"], tuple(r["aliases"]),
                                         tuple(r["tags"])) for r in t}
        feats = json.loads((self.dir / "zones.geojson").read_text())["features"]
        self.zones = {f["properties"]["id"]: SiteObj(f"zone:{f['properties']['id']}", f["properties"]["id"],
                                                     f["properties"]["name"], f["properties"]["kind"],
                                                     shape(f["geometry"]), aliases=tuple(f["properties"]["aliases"]),
                                                     family="zone") for f in feats}
        self.sensors = {s["id"]: s for s in yaml.safe_load((self.dir / "sensors.yaml").read_text())["sensors"]}
        # RF cache (belief side, nominal model): scenario seeds share the site's cache (meta.rf_cache_dir)
        rel = self.meta.get("rf_cache_dir")
        self.rf_dir = (REPO / rel) if rel else (self.dir / "rf_cache")
        fe = self.dir / "friendly_emitters.yaml"
        self.friendly = yaml.safe_load(fe.read_text())["emitters"] if fe.exists() else []

        self._obj_list = list(self.objects.values())
        self.obj_tree = STRtree([o.geom for o in self._obj_list])
        self.zone_prep = {zid: prep(z.geom) for zid, z in self.zones.items()}

        self._local = threading.local()
        self._db_path = str(self.dir / "observed.duckdb")
        con = self.cursor()
        rows = con.execute("SELECT * FROM tracks ORDER BY track_id").fetchall()
        cols = [d[0] for d in con.description]
        self.tracks = {r[0]: dict(zip(cols, r, strict=True)) for r in rows}
        for tr in self.tracks.values():
            tr["klass_probs"] = json.loads(tr["klass_probs"])
        self.groups = {tid: tr.get("group_id") for tid, tr in self.tracks.items()}
        ev = con.execute("SELECT * FROM events ORDER BY t_s, evt_id").fetchall()
        ecols = [d[0] for d in con.description]
        self.events = [dict(zip(ecols, r, strict=True)) for r in ev]
        # Track estimates are small and hot: keep them as arrays for vectorised queries (SPEC 10.3).
        tp_cols = {r[1] for r in con.execute("PRAGMA table_info('track_points')").fetchall()}
        nlos = ", nlos_suspect" if "nlos_suspect" in tp_cols else ", FALSE"     # column added in S7.1 (v2 tracker)
        pts = con.execute("SELECT track_id, t_s, x_m, y_m, z_m, a_m, b_m, theta_deg, band, n_sensors" + nlos +
                          " FROM track_points ORDER BY track_id, t_s").fetchall()
        self.points: dict[str, dict[str, np.ndarray]] = {}
        for tid in self.tracks:
            rows_t = [p for p in pts if p[0] == tid]
            self.points[tid] = {
                "t": np.array([p[1] for p in rows_t]), "x": np.array([p[2] for p in rows_t]),
                "y": np.array([p[3] for p in rows_t]), "z": np.array([p[4] for p in rows_t]),
                "a": np.array([p[5] for p in rows_t]), "b": np.array([p[6] for p in rows_t]),
                "theta": np.array([p[7] for p in rows_t]), "band": [p[8] for p in rows_t],
                "n": np.array([p[9] for p in rows_t]),
                "nlos": np.array([bool(p[10]) for p in rows_t], dtype=bool),
            }

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """One read-only connection per thread (the executor runs parallel tool calls on worker threads)."""
        con = getattr(self._local, "con", None)
        if con is None:
            con = duckdb.connect(self._db_path, read_only=True)
            self._local.con = con
        return con

    # --- lookups -------------------------------------------------------------------------------------------
    def entity(self, kind: str, eid: str) -> SiteObj | None:
        return (self.objects if kind == "obj" else self.zones if kind == "zone" else {}).get(eid)

    def known_refs(self, kind: str) -> list[str]:
        if kind == "obj":
            return [o.ref for o in self._obj_list]
        if kind == "zone":
            return [z.ref for z in self.zones.values()]
        if kind == "sensor":
            return [f"sensor:{s}" for s in self.sensors]
        if kind == "trk":
            return [f"trk:{t}" for t in self.tracks]
        return []

    def track_index_at(self, tid: str, t: float | None) -> int:
        """Index of the latest estimate at or before t (or the latest overall)."""
        ts = self.points[tid]["t"]
        if t is None:
            return len(ts) - 1
        i = int(np.searchsorted(ts, t, side="right")) - 1
        return max(i, 0)

    def objects_intersecting(self, geom) -> list[SiteObj]:
        return [self._obj_list[int(i)] for i in self.obj_tree.query(geom, predicate="intersects")]

    def zones_containing(self, x: float, y: float) -> list[str]:
        p = Point(x, y)
        return [zid for zid, pz in self.zone_prep.items() if pz.contains(p)]
