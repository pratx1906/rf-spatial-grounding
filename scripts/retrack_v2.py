"""Offline re-tracking of v2 scenario sets (S7.1): run only products_v2 (tracker + belief tables) again, without
rebuilding the IQ observations. WORLD/offline side: it reads truth only to attach the products_v2 world-side labels.

  capture      run the real v2 build for one seed (CPU, SCENESCOUT_V2_MAX_T honoured) into runs/s71/, and pickle
               the receiver detections so trackers can be compared on identical inputs;
  from-bursts  rebuild Detection records from a set's stored `bursts` table (bearing = strongest peak, no elevation,
               power from the dwell noise floor + SNR) and re-run products_v2;
  from-pickle  re-run products_v2 on captured detections.

Outputs (observed.duckdb with tracks / track_points / bearings / bursts / dwells, plus truth/track_labels) go to the
--out directory; data/scenarios/* is never written.

  python scripts/retrack_v2.py from-bursts --set data/scenarios/kandla_s13_X --out runs/s71/new/kandla_s13_X
  python scripts/retrack_v2.py capture --seed 40 --max-t 200 --out runs/s71/build_s40
  python scripts/retrack_v2.py from-pickle --dets runs/s71/build_s40/dets.pkl --set runs/s71/build_s40 --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SITE = REPO / "data" / "site_kandla"


def _sensors(set_dir: Path):
    out = []
    for s in yaml.safe_load((set_dir / "sensors.yaml").read_text())["sensors"]:
        p = s["pose"]
        out.append(SimpleNamespace(id=s["id"], type=s["type"], x=p["x_m"], y=p["y_m"], z=p["z_m"]))
    return out


def _emitters(set_dir: Path):
    t = duckdb.connect(str(set_dir / "truth" / "truth.duckdb"), read_only=True)
    rows = t.execute("select emitter_id, kind, bands from emitters").fetchall()
    pts: dict = {}
    for e, ts, x, y, z in t.execute("select emitter_id, t_s, x_m, y_m, z_m from emitter_points order by t_s").fetchall():
        pts.setdefault(e, []).append((ts, x, y, z))
    t.close()
    # band_schedule: the world-side labels match on band as well as position (observe_v2.match_emitter)
    return [SimpleNamespace(id=e, kind=k, points=pts[e], band_schedule=[tuple(b) for b in json.loads(bands)])
            for e, k, bands in rows if pts.get(e)]


def _world_root(set_dir: Path) -> Path:
    v = (yaml.safe_load((set_dir / "meta.yaml").read_text()) or {}).get("world_variant_id")
    return (SITE / "world_cache_v2" / "rf" / v) if v else (SITE / "rf_cache_v2")


def _bg():
    env = SITE / "world_cache_v2" / "spectrum_env.yaml"
    return {e["id"]: (e["x_m"], e["y_m"]) for e in yaml.safe_load(env.read_text())["emitters"]} if env.exists() else {}


def _gnss() -> float:
    hw = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
    return hw["receiver"]["gnss_timing_ns_1sigma"]


def dets_from_bursts(con) -> tuple[list, list]:
    """Detection records from a stored bursts table (+ the dwells table)."""
    from sim.obs_model.observe_v2 import Detection
    from sim.rf.spectrum import band_of

    dwells = con.execute("select * from dwells").fetchall()
    floor = {(r[0], r[1], r[2]): r[4] for r in dwells}
    dets = []
    for (_bid, s, t, f, bw, t0, dur, proto, conf, az, snr, toa, _tid) in con.execute(
            "select * from bursts order by burst_id").fetchall():
        band = band_of(f) or "ism_5800"
        nf = floor.get((t, s, band), -100.0)
        p = nf + 10 * math.log10(max(bw / 1000.0, 1e-3)) + snr
        peaks = [] if az is None else [(float(az), math.nan, float(snr))]
        dets.append(Detection(t, s, band, f, bw, t0 * 1e-6, dur * 1e-6, proto, conf, peaks,
                              math.nan if toa is None else toa, snr, p))
    return dets, dwells


def run_products(seed: int, set_dir: Path, dets: list, dwells: list, out: Path, jammers=None) -> dict:
    from sim.obs_model import observe_v2 as ov

    ov.observe_v2.dwells = dwells
    obs = ov.products_v2(seed, _emitters(set_dir), _sensors(set_dir), dets, _world_root(set_dir), _gnss(), _bg(),
                         jammers)
    write_observed(out, obs)
    return obs


def write_observed(out: Path, obs: dict) -> None:
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for f in (out / "observed.duckdb", out / "truth" / "truth.duckdb"):
        if f.exists():
            f.unlink()
    con = duckdb.connect(str(out / "observed.duckdb"))
    con.execute("""CREATE TABLE tracks (track_id VARCHAR PRIMARY KEY, band VARCHAR, freq_mhz DOUBLE,
        klass_probs VARCHAR, protocol_guess VARCHAR, bandwidth_khz DOUBLE, power_dbm DOUBLE,
        first_seen DOUBLE, last_seen DOUBLE, n_sensors INTEGER, confidence DOUBLE, group_id VARCHAR)""")
    con.executemany("INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(t[0], t[1], t[2], json.dumps(t[3]), *t[4:11], None) for t in obs["tracks"]])
    con.execute("""CREATE TABLE track_points (track_id VARCHAR, t_s DOUBLE, x_m DOUBLE, y_m DOUBLE, z_m DOUBLE,
        a_m DOUBLE, b_m DOUBLE, theta_deg DOUBLE, band VARCHAR, n_sensors INTEGER, nlos_suspect BOOLEAN DEFAULT FALSE)""")
    cols = "track_id, t_s, x_m, y_m, z_m, a_m, b_m, theta_deg, band, n_sensors"
    for r in obs["points"]:
        if len(r) == 11:
            con.execute(f"INSERT INTO track_points ({cols}, nlos_suspect) VALUES (?,?,?,?,?,?,?,?,?,?,?)", r)
        else:
            con.execute(f"INSERT INTO track_points ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?)", r)
    con.execute("""CREATE TABLE bearings (hyp_id VARCHAR, track_id VARCHAR, sensor_id VARCHAR,
        t_s DOUBLE, bearing_deg DOUBLE, snr_db DOUBLE, power_db DOUBLE)""")
    if obs["bearings"]:
        con.executemany("INSERT INTO bearings VALUES (?,?,?,?,?,?,?)", obs["bearings"])
    con.execute("""CREATE TABLE bursts (burst_id VARCHAR, sensor_id VARCHAR, t_s DOUBLE,
        f_mhz DOUBLE, bw_khz DOUBLE, t0_us DOUBLE, dur_us DOUBLE, protocol_est VARCHAR, protocol_conf DOUBLE,
        bearing_deg DOUBLE, snr_db DOUBLE, toa_s DOUBLE, track_id VARCHAR)""")
    if obs["bursts"]:
        con.executemany("INSERT INTO bursts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", obs["bursts"])
    con.execute("""CREATE TABLE dwells (t_s DOUBLE, sensor_id VARCHAR, band VARCHAR, window_mhz DOUBLE,
        noise_dbm_per_mhz DOUBLE, clipped_frac DOUBLE, n_bursts INTEGER)""")
    if obs.get("dwells"):
        con.executemany("INSERT INTO dwells VALUES (?,?,?,?,?,?,?)", obs["dwells"])
    con.close()
    t = duckdb.connect(str(out / "truth" / "truth.duckdb"))
    t.execute("CREATE TABLE track_labels (track_id VARCHAR, emitter_id VARCHAR, emitter_kind VARCHAR)")
    if obs["track_labels"]:
        t.executemany("INSERT INTO track_labels VALUES (?,?,?)", obs["track_labels"])
    t.close()
    (out / "fix_modes.json").write_text(json.dumps(obs.get("fix_modes", {})))


def cmd_from_bursts(a) -> None:
    s = Path(a.set)
    con = duckdb.connect(str(s / "observed.duckdb"), read_only=True)
    dets, dwells = dets_from_bursts(con)
    jams = [{"id": r[0], "sensor": r[1], "band": r[2], "f_mhz": r[3], "bw_mhz": r[4], "eirp_dbm": r[5], "t0_s": r[6],
             "t1_s": r[7]} for r in con.execute("select * from effector_log").fetchall()]
    con.close()
    seed = int(yaml.safe_load((s / "meta.yaml").read_text())["seed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    obs = run_products(seed, s, dets, dwells, out, jams)
    print(f"{s.name}: {len(dets)} dets -> {len(obs['tracks'])} tracks, {len(obs['points'])} points -> {out}")


def cmd_capture(a) -> None:
    """Real v2 build of one seed on CPU into --out; pickles detections + dwells + jammers for re-tracking."""
    os.environ["SCENESCOUT_V2_MAX_T"] = str(a.max_t)
    import envbuild.site_kandla as sk
    from sim.obs_model import observe_v2 as ov

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sk.write_build_files = lambda b: None            # never touch the shared site build files
    real = ov.products_v2

    def grab(seed, emitters, sensors, dets, world_root, gnss_ns, bg=None, jammers=None):
        with open(out / "dets.pkl", "wb") as f:
            pickle.dump({"seed": seed, "dets": dets, "dwells": list(getattr(ov.observe_v2, "dwells", [])),
                         "jammers": jammers}, f)
        return real(seed, emitters, sensors, dets, world_root, gnss_ns, bg, jammers)

    ov.products_v2 = grab
    m = sk.write_site(a.seed, out, rf_backend="air7311", device="cpu", obs_version="v2")
    print(json.dumps(m["counts"]))


def cmd_from_pickle(a) -> None:
    with open(a.dets, "rb") as f:
        d = pickle.load(f)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    obs = run_products(d["seed"], Path(a.set), d["dets"], d["dwells"], out, d.get("jammers"))
    print(f"{len(d['dets'])} dets -> {len(obs['tracks'])} tracks, {len(obs['points'])} points -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-bursts")
    p.add_argument("--set", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-t", type=float, default=200.0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("from-pickle")
    p.add_argument("--dets", required=True)
    p.add_argument("--set", required=True, help="dir with sensors.yaml, meta.yaml, truth/ (e.g. the capture dir)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"from-bursts": cmd_from_bursts, "capture": cmd_capture, "from-pickle": cmd_from_pickle}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
