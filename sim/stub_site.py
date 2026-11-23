"""Generate a site to the SPEC 4 data contract. The stub site and the Kandla site share `write_products`.

    python -m sim.stub_site --out data/stub_site --seed 0

Writes (belief / agent-visible):  meta.yaml, objects.parquet, zones.geojson, sensors.yaml, observed.duckdb
Writes (truth, never visible):    truth/truth.duckdb  (emitters, true trajectories, bearing path labels, track labels)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from shapely.geometry import mapping

from harness.config import REPO
from sim.events import extract_events
from sim.layout import EXTENT, SHORE_Y, build_layout
from sim.obs_model.observe import BEARING_DT_S, observe
from sim.world.scenario import BANDS, DT_S, DURATION_S, make_emitters

SCHEMA_VERSION = "stub-1"
EXTRA_COLUMNS = ("osm_id", "name_source", "height_source", "height_confidence", "parent_id", "mount_candidate",
                 "protected_status", "provenance")


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def write_products(out: Path, objs, zones, sensors, emitters, obs: dict, events: list, seed: int, meta: dict,
                   obj_extra: dict | None = None, zone_extra: dict | None = None,
                   duration_s: float = DURATION_S) -> dict:
    """Write the belief-side products and the truth side for any site (SPEC 4 contract)."""
    out = Path(out)
    if out.exists():
        for name in ("objects.parquet", "zones.geojson", "sensors.yaml", "observed.duckdb", "meta.yaml"):
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(out / "truth", ignore_errors=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    obj_extra = obj_extra or {}
    zone_extra = zone_extra or {}

    # --- belief side ---------------------------------------------------------------------------------------
    cols = {
        "id": [o.id for o in objs], "name": [o.name for o in objs], "aliases": [list(o.aliases) for o in objs],
        "kind": [o.kind for o in objs], "wkt": [o.geom.wkt for o in objs],
        "height_m": [float(o.height_m) for o in objs], "material": [o.material for o in objs],
        "protected": [bool(o.protected) for o in objs], "tags": [list(o.tags) for o in objs],
    }
    if obj_extra:
        for c in EXTRA_COLUMNS:
            vals = [obj_extra.get(o.id, {}).get(c) for o in objs]
            cols[c] = [json.dumps(v) if isinstance(v, (dict, list)) else (None if v is None else str(v)) for v in vals]
    pq.write_table(pa.table(cols), out / "objects.parquet")
    (out / "zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(z.geom),
         "properties": {"id": z.id, "name": z.name, "kind": z.kind, "aliases": z.aliases,
                        "provenance": zone_extra.get(z.id, {"source": "synthetic", "status": "synthetic"})}}
        for z in zones]}, indent=1))
    (out / "sensors.yaml").write_text(yaml.safe_dump({"sensors": [
        {"id": s.id, "type": s.type, "pose": {"x_m": s.x, "y_m": s.y, "z_m": s.z}, "bands": s.bands,
         "array": s.array, "mount": s.mount, "placement": "synthetic"} for s in sensors]}, sort_keys=False))

    db = out / "observed.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE tracks (track_id VARCHAR PRIMARY KEY, band VARCHAR, freq_mhz DOUBLE,
        klass_probs VARCHAR, protocol_guess VARCHAR, bandwidth_khz DOUBLE, power_dbm DOUBLE,
        first_seen DOUBLE, last_seen DOUBLE, n_sensors INTEGER, confidence DOUBLE, group_id VARCHAR)""")
    con.executemany("INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(t[0], t[1], t[2], json.dumps(t[3]), *t[4:11], obs.get("groups", {}).get(t[0]))
                     for t in obs["tracks"]])
    con.execute("""CREATE TABLE track_points (track_id VARCHAR, t_s DOUBLE, x_m DOUBLE, y_m DOUBLE, z_m DOUBLE,
        a_m DOUBLE, b_m DOUBLE, theta_deg DOUBLE, band VARCHAR, n_sensors INTEGER,
        nlos_suspect BOOLEAN DEFAULT FALSE)""")          # per-fix NLOS / inconsistency flag (v2 tracker, S7.1)
    tp_cols = "track_id, t_s, x_m, y_m, z_m, a_m, b_m, theta_deg, band, n_sensors"
    p10 = [r for r in obs["points"] if len(r) == 10]
    p11 = [r for r in obs["points"] if len(r) == 11]
    if p10:
        con.executemany(f"INSERT INTO track_points ({tp_cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", p10)
    if p11:
        con.executemany(f"INSERT INTO track_points ({tp_cols}, nlos_suspect) VALUES (?,?,?,?,?,?,?,?,?,?,?)", p11)
    con.execute("""CREATE TABLE bearings (hyp_id VARCHAR PRIMARY KEY, track_id VARCHAR, sensor_id VARCHAR,
        t_s DOUBLE, bearing_deg DOUBLE, snr_db DOUBLE, power_db DOUBLE)""")
    con.executemany("INSERT INTO bearings VALUES (?,?,?,?,?,?,?)", obs["bearings"])
    con.execute("""CREATE TABLE events (evt_id VARCHAR PRIMARY KEY, type VARCHAR, track_id VARCHAR,
        zone_id VARCHAR, t_s DOUBLE, x_m DOUBLE, y_m DOUBLE, detail VARCHAR)""")
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", events)
    if obs.get("bursts") is not None:          # v2 observation sets only (RF phase 2): receiver-level detections
        con.execute("""CREATE TABLE bursts (burst_id VARCHAR PRIMARY KEY, sensor_id VARCHAR, t_s DOUBLE,
            f_mhz DOUBLE, bw_khz DOUBLE, t0_us DOUBLE, dur_us DOUBLE, protocol_est VARCHAR, protocol_conf DOUBLE,
            bearing_deg DOUBLE, snr_db DOUBLE, toa_s DOUBLE, track_id VARCHAR)""")
        if obs["bursts"]:
            con.executemany("INSERT INTO bursts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", obs["bursts"])
        con.execute("""CREATE TABLE dwells (t_s DOUBLE, sensor_id VARCHAR, band VARCHAR, window_mhz DOUBLE,
            noise_dbm_per_mhz DOUBLE, clipped_frac DOUBLE, n_bursts INTEGER)""")
        con.execute("""CREATE TABLE effector_log (event_id VARCHAR, sensor_id VARCHAR, band VARCHAR, f_mhz DOUBLE,
            bw_mhz DOUBLE, eirp_dbm DOUBLE, t0_s DOUBLE, t1_s DOUBLE)""")     # own mitigation TX (the system knows)
        if obs.get("effector_log"):
            con.executemany("INSERT INTO effector_log VALUES (?,?,?,?,?,?,?,?)", obs["effector_log"])
        if obs.get("dwells"):
            con.executemany("INSERT INTO dwells VALUES (?,?,?,?,?,?,?)", obs["dwells"])
    for stmt in ("CREATE INDEX tp_idx ON track_points (track_id, t_s)",
                 "CREATE INDEX br_idx ON bearings (track_id, t_s)",
                 "CREATE INDEX ev_idx ON events (type, t_s)"):
        con.execute(stmt)
    con.close()

    # --- truth side (NEVER agent-visible) ------------------------------------------------------------------
    tcon = duckdb.connect(str(out / "truth" / "truth.duckdb"))
    tcon.execute("CREATE TABLE scenario (seed INTEGER, duration_s DOUBLE, n_emitters INTEGER)")
    tcon.execute("INSERT INTO scenario VALUES (?,?,?)", (seed, duration_s, len(emitters)))
    tcon.execute("CREATE TABLE emitters (emitter_id VARCHAR, kind VARCHAR, bands VARCHAR, eirp_dbm DOUBLE, "
                 "swarm_id VARCHAR)")
    tcon.executemany("INSERT INTO emitters VALUES (?,?,?,?,?)",
                     [(e.id, e.kind, json.dumps(e.band_schedule), e.eirp_dbm, getattr(e, "swarm_id", None))
                      for e in emitters])
    tcon.execute("CREATE TABLE emitter_points (emitter_id VARCHAR, t_s DOUBLE, x_m DOUBLE, y_m DOUBLE, z_m DOUBLE)")
    tcon.executemany("INSERT INTO emitter_points VALUES (?,?,?,?,?)",
                     [(e.id, *p) for e in emitters for p in e.points])
    if obs.get("bursts") is not None:          # v2 sets: protocol and controller pairing (truth, world side)
        tcon.execute("CREATE TABLE emitters_v2 (emitter_id VARCHAR, protocol VARCHAR, pair_id VARCHAR)")
        tcon.executemany("INSERT INTO emitters_v2 VALUES (?,?,?)",
                         [(e.id, e.protocol, getattr(e, "pair_id", None)) for e in emitters])
    tcon.execute("CREATE TABLE track_labels (track_id VARCHAR, emitter_id VARCHAR, emitter_kind VARCHAR)")
    tcon.executemany("INSERT INTO track_labels VALUES (?,?,?)", obs["track_labels"])
    tcon.execute("CREATE TABLE hyp_labels (hyp_id VARCHAR, path_type VARCHAR, via_obj VARCHAR)")
    tcon.executemany("INSERT INTO hyp_labels VALUES (?,?,?)", obs["bearing_labels"])
    tcon.close()

    belief_files = [out / f for f in ("objects.parquet", "zones.geojson", "sensors.yaml")]
    full = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "time": {"t0_iso": "2026-01-01T00:00:00Z", "duration_s": duration_s, "track_dt_s": DT_S,
                 "bearing_dt_s": BEARING_DT_S},
        "bands": {k: list(v) for k, v in BANDS.items()},
        **meta,
        "version_hash": _hash_files(belief_files),   # data version recorded in every trace (SPEC 4)
        "counts": {"objects": len(objs), "zones": len(zones), "sensors": len(sensors),
                   "tracks": len(obs["tracks"]), "track_points": len(obs["points"]),
                   "bearings": len(obs["bearings"]), "events": len(events)},
    }
    (out / "meta.yaml").write_text(yaml.safe_dump(full, sort_keys=False))
    return full


def generate(out: Path, seed: int = 0) -> dict:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    objs, zones, sensors = build_layout(seed)
    emitters = make_emitters(seed, {z.id: z for z in zones})
    obs = observe(seed, emitters, objs, sensors)
    events = extract_events(obs["points"], zones)
    meta = {"site": "stub_site",
            "frame": {"type": "ENU", "units": "m", "bearing": "deg clockwise from true north",
                      "origin": {"lat": None, "lon": None, "alt_m": 0.0, "note": "fictitious stub site"}},
            "extent_m": list(EXTENT), "shore_y_m": SHORE_Y, "rf_backend": "geometric_stub"}
    return write_products(out, objs, zones, sensors, emitters, obs, events, seed, meta)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "data" / "stub_site")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    meta = generate(a.out, a.seed)
    print(json.dumps(meta["counts"]), meta["version_hash"])


if __name__ == "__main__":
    main()
