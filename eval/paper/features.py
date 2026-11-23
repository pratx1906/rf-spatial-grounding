"""P9 (a): per-task difficulty features for any frozen set, computed OFFLINE from scenario and world products.

    SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr \
        python -m eval.paper.features --set eval/frozen/v6/tasks.jsonl --out runs/paper/results/P9/features_v6.parquet

Reads truth/ (emitter positions, bearing path labels) through tasks.truth_access and the RF ghost/trust maps of the
scenario's rf cache (meta.yaml rf_cache_dir, `maps/<band>.npz`). Analysis only: never imported by tools/harness/serve.

Features (one row per task; NaN where a task has no track, e.g. siting, collateral, most unanswerable):
- identity: task_id, area, template (family), seed, world_variant, site
- band: the task's band param, else the primary track's observed band
- n_track_refs, has_track, track_kind (truth label of the primary track: drone / clutter / ...)
- emitter_z_m: true emitter height (median over the window), nearest_sensor_m: horizontal distance to the nearest
  fixed DF sensor
- n_sensors: fixed sensors with a bearing on the track in the window; n_bearings
- hyps_per_look: bearings per (sensor, epoch); multipath severity from the observations
- ghost_frac_label: share of the track's bearings in the window whose path label is not LOS (truth)
- ghost_frac_5deg: share of bearings more than 5 deg from the true sensor->emitter azimuth (geometric)
- gdop_m_per_deg: bearing-only GDOP of the contributing sensors at the true position (metres of position error per
  degree of bearing noise; capped at GDOP_CAP when fewer than two sensors or collinear)
- map_ghost_risk, map_nlos_frac, map_az_spread_deg: RF map products (D-031 ensemble) at the emitter cell, linear in
  height between map levels, band, averaged over contributing sensors (all five if none); map_site_ghost_risk: site-level ghost risk
- n_emitters_active (truth) and n_tracks_active (observed) in the window
- los_look_frac, strongest_los_frac, los_stratum (LOS / NLOS): LOS/NLOS stratification (plan Revision 1, item 10)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GDOP_CAP = 1000.0          # m/deg; ~ "unobservable" (one sensor or collinear)
GHOST_DEG = 5.0
TRK = re.compile(r"\btrk:(\w+)")
HALF_WINDOW_S = 5.0        # bearings come every 10 s; +-5 s picks the epoch of a point-in-time question


# ----------------------------------------------------------------------------------------------- pure functions

def angdiff(a_deg, b_deg):
    """Smallest absolute difference of two azimuths (deg), vectorised."""
    d = (np.asarray(a_deg, float) - np.asarray(b_deg, float) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def azimuth_deg(sx: float, sy: float, tx: float, ty: float) -> float:
    """Azimuth clockwise from north (ENU: x east, y north) from sensor (sx, sy) to target (tx, ty)."""
    return math.degrees(math.atan2(tx - sx, ty - sy)) % 360.0


def gdop_bearing(sensors_xy, target_xy, cap: float = GDOP_CAP) -> float:
    """Bearing-only GDOP in metres per degree: sqrt(trace(J^-1)) with J = sum_i u_i u_i^T / r_i^2 (u_i the unit
    vector perpendicular to the line of sight), times pi/180. Equal 1-deg noise on every sensor. `cap` for fewer than
    two sensors or a (near) singular geometry."""
    s = np.asarray(sensors_xy, float).reshape(-1, 2)
    t = np.asarray(target_xy, float).reshape(2)
    if len(s) < 2:
        return cap
    d = t - s
    r = np.hypot(d[:, 0], d[:, 1])
    if np.any(r < 1e-6):
        r = np.maximum(r, 1e-6)
    u = np.stack([-d[:, 1], d[:, 0]], axis=1) / r[:, None]
    J = (u[:, :, None] * u[:, None, :] / (r ** 2)[:, None, None]).sum(0)
    if np.linalg.cond(J) > 1e10:
        return cap
    g = math.sqrt(max(float(np.trace(np.linalg.inv(J))), 0.0)) * math.pi / 180.0
    return min(g, cap)


def nearest_sensor_distance(sensors_xy, xy) -> float:
    s = np.asarray(sensors_xy, float).reshape(-1, 2)
    if not len(s):
        return float("nan")
    return float(np.min(np.hypot(s[:, 0] - xy[0], s[:, 1] - xy[1])))


def ghost_fraction(bearings_deg, true_deg, thr: float = GHOST_DEG) -> float:
    """Share of bearings further than `thr` from the true azimuth (NaN for no bearings)."""
    b = np.asarray(bearings_deg, float)
    if not b.size:
        return float("nan")
    return float(np.mean(angdiff(b, true_deg) > thr))


def grid_index(xs, ys, heights, x: float, y: float, z: float) -> tuple[int, int, int] | None:
    """Nearest cell (iy, ix) and nearest height level hi; None outside the grid (half a cell of slack)."""
    xs, ys, hs = (np.asarray(v, float) for v in (xs, ys, heights))
    dx = (xs[-1] - xs[0]) / max(len(xs) - 1, 1)
    dy = (ys[-1] - ys[0]) / max(len(ys) - 1, 1)
    if not (xs[0] - dx / 2 <= x <= xs[-1] + dx / 2 and ys[0] - dy / 2 <= y <= ys[-1] + dy / 2):
        return None
    return int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(hs - z)))


def los_strata(looks) -> dict:
    """LOS/NLOS stratification (Revision 1 item 10) from a bearings frame with sensor_id, t_s, power_db, label:
    los_look_frac = share of (sensor, epoch) looks with a LOS-labelled bearing; strongest_los_frac = share of looks
    whose strongest bearing is LOS; los_stratum = LOS (>= 0.5 of looks have a LOS bearing), NLOS, or None."""
    lab = looks[looks.label.notna()]
    if not len(lab):
        return {"los_look_frac": float("nan"), "strongest_los_frac": float("nan"), "los_stratum": None}
    g = lab.groupby(["sensor_id", "t_s"])
    has = g.label.apply(lambda s: bool((s == "LOS").any()))
    top = lab.loc[g.power_db.idxmax()].label.eq("LOS")
    f = float(has.mean())
    return {"los_look_frac": f, "strongest_los_frac": float(top.mean()), "los_stratum": "LOS" if f >= 0.5 else "NLOS"}


def task_track_ids(task: dict) -> list[str]:
    """Observed track ids (without 'trk:') a task is about: params.track, else answer-key track refs, else refs in
    the question; in that order, de-duplicated."""
    out: list[str] = []
    p, k = task.get("params") or {}, task.get("answer_key") or {}
    if isinstance(p.get("track"), str):
        out.append(p["track"])
    out += [r for r in (k.get("track_refs") or []) if isinstance(r, str)]
    out += ["trk:" + m for m in TRK.findall(task.get("question", ""))]
    ids = [r.split("@", 1)[0].removeprefix("trk:") for r in out]
    return list(dict.fromkeys(ids))


def task_window(task: dict, life: tuple[float, float] | None) -> tuple[float, float] | None:
    """Time window of a task: a point time (+- HALF_WINDOW_S), else [t0, t1], else the track's life."""
    p, k = task.get("params") or {}, task.get("answer_key") or {}
    t = p.get("t", k.get("t_s"))
    if isinstance(t, (int, float)):
        return float(t) - HALF_WINDOW_S, float(t) + HALF_WINDOW_S
    if isinstance(p.get("t0"), (int, float)) and isinstance(p.get("t1"), (int, float)):
        return float(p["t0"]), float(p["t1"])
    return life


# ----------------------------------------------------------------------------------------------- world access

class SiteData:
    """Observed + truth + map products of one scenario dir (offline)."""

    def __init__(self, site: str | Path):
        import duckdb
        import yaml

        from tasks.truth_access import TruthStore

        self.dir = (REPO / site) if not Path(site).is_absolute() else Path(site)
        obs = duckdb.connect(str(self.dir / "observed.duckdb"), read_only=True)
        self.tracks = {r[0]: {"band": r[1], "first": r[2], "last": r[3]} for r in obs.execute(
            "SELECT track_id, band, first_seen, last_seen FROM tracks").fetchall()}
        self.bearings = obs.execute("SELECT hyp_id, track_id, sensor_id, t_s, bearing_deg, power_db FROM bearings").fetchdf()
        obs.close()
        sens = yaml.safe_load((self.dir / "sensors.yaml").read_text())["sensors"]
        self.sensors = {s["id"]: (s["pose"]["x_m"], s["pose"]["y_m"]) for s in sens if s.get("type") == "fixed_df"}
        ts = TruthStore(self.dir)
        self.track_of = ts.track_of()
        self.labels = ts.hyp_labels()
        self.kind = ts.emitters()
        pts = ts.con.execute("SELECT emitter_id, t_s, x_m, y_m, z_m FROM emitter_points ORDER BY 1, 2").fetchdf()
        ts.close()
        self.points = {e: g[["t_s", "x_m", "y_m", "z_m"]].to_numpy() for e, g in pts.groupby("emitter_id")}
        meta = yaml.safe_load((self.dir / "meta.yaml").read_text())
        self.rf_cache = REPO / meta.get("rf_cache_dir", "data/site_kandla/rf_cache_v3")

    def emitter_xyz(self, eid: str, w: tuple[float, float]) -> np.ndarray | None:
        p = self.points.get(eid)
        if p is None or not len(p):
            return None
        m = (p[:, 0] >= w[0]) & (p[:, 0] <= w[1])
        if not m.any():
            i = int(np.argmin(np.abs(p[:, 0] - 0.5 * (w[0] + w[1]))))
            if abs(p[i, 0] - 0.5 * (w[0] + w[1])) > 30:
                return None
            return p[i, 1:]
        return np.median(p[m, 1:], axis=0)

    def n_emitters_active(self, w) -> int:
        return sum(bool(((p[:, 0] >= w[0]) & (p[:, 0] <= w[1])).any()) for p in self.points.values())

    def n_tracks_active(self, w) -> int:
        return sum(1 for t in self.tracks.values() if t["first"] <= w[1] and t["last"] >= w[0])


@lru_cache(maxsize=16)
def _maps(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=64)
def site_data(site: str) -> SiteData:
    return SiteData(site)


def task_features(task: dict, sd: SiteData) -> dict:
    p = task.get("params") or {}
    f = {"task_id": task["task_id"], "area": task["area"], "template": task["template"],
         "seed": task.get("scenario_seed"), "world_variant": task.get("world_variant") or "N", "site": task.get("site"),
         "band": p.get("band")}
    tids = [t for t in task_track_ids(task) if t in sd.tracks]
    f["n_track_refs"] = len(tids)
    f["has_track"] = int(bool(tids))
    nan = float("nan")
    for k in ("emitter_z_m", "nearest_sensor_m", "n_sensors", "n_bearings", "hyps_per_look", "ghost_frac_label",
              "ghost_frac_5deg", "gdop_m_per_deg", "map_ghost_risk", "map_nlos_frac", "map_az_spread_deg",
              "map_site_ghost_risk", "n_emitters_active", "n_tracks_active", "los_look_frac",
              "strongest_los_frac"):
        f[k] = nan
    f["track_kind"] = None
    f["los_stratum"] = None
    if not tids:
        return f
    tid = tids[0]
    tr = sd.tracks[tid]
    f["band"] = f["band"] or tr["band"]
    w = task_window(task, (tr["first"], tr["last"]))
    f["n_emitters_active"] = sd.n_emitters_active(w)
    f["n_tracks_active"] = sd.n_tracks_active(w)
    eid, kind = sd.track_of.get(tid, (None, None))
    f["track_kind"] = kind
    xyz = sd.emitter_xyz(eid, w) if eid else None
    b = sd.bearings
    bw = b[(b.track_id == tid) & (b.t_s >= w[0]) & (b.t_s <= w[1])]
    if task.get("params", {}).get("sensor"):
        pass                                      # a per-sensor question still depends on the whole track's DF
    contrib = sorted(set(bw.sensor_id) & set(sd.sensors))
    f["n_sensors"] = len(contrib)
    f["n_bearings"] = int(len(bw))
    if len(bw):
        f["hyps_per_look"] = float(bw.groupby(["sensor_id", "t_s"]).size().mean())
        lab = [sd.labels.get(h) for h in bw.hyp_id]
        known = [x for x in lab if x is not None]
        f["ghost_frac_label"] = float(np.mean([x != "LOS" for x in known])) if known else nan
        f.update(los_strata(bw.assign(label=lab)))
    if xyz is None:
        return f
    x, y, z = (float(v) for v in xyz)
    f["emitter_z_m"] = z
    f["nearest_sensor_m"] = nearest_sensor_distance(list(sd.sensors.values()), (x, y))
    if len(bw):
        true = np.array([azimuth_deg(*sd.sensors[s], x, y) if s in sd.sensors else np.nan for s in bw.sensor_id])
        ok = ~np.isnan(true)
        f["ghost_frac_5deg"] = ghost_fraction(bw.bearing_deg.to_numpy()[ok], true[ok]) if ok.any() else nan
    f["gdop_m_per_deg"] = gdop_bearing([sd.sensors[s] for s in contrib], (x, y)) if contrib else GDOP_CAP
    band = f["band"]
    if band:
        m = _maps(str(sd.rf_cache / "maps" / f"{band}.npz"))
        if m is not None:
            gi = grid_index(m["xs"], m["ys"], m["heights"], x, y, z)
            if gi is not None:
                iy, ix, hi = gi
                names = [str(s) for s in m["sensors"]]
                use = [names.index(s) for s in (contrib or names) if s in names]
                hs = np.asarray(m["heights"], float)

                def at_z(col):                            # sensor mean per height level, linear in height
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        v = np.nanmean(col, axis=0) if col.ndim == 2 else col
                    ok = ~np.isnan(v)
                    return float(np.interp(z, hs[ok], v[ok])) if ok.any() else float("nan")

                f["map_ghost_risk"] = at_z(m["ghost_risk"][use][:, :, iy, ix])
                f["map_nlos_frac"] = at_z(m["nlos_frac"][use][:, :, iy, ix])
                f["map_az_spread_deg"] = at_z(m["az_spread_deg"][use][:, :, iy, ix])
                f["map_site_ghost_risk"] = at_z(m["site_ghost_risk"][:, iy, ix])
    return f


def build(set_path: str | Path) -> list[dict]:
    rows = [json.loads(x) for x in Path(REPO / set_path).read_text().splitlines() if x.strip()]
    out = []
    for t in rows:
        site = t.get("site")
        if not site or not (REPO / site).exists():
            out.append({"task_id": t["task_id"], "area": t["area"], "template": t["template"], "has_track": 0})
            continue
        out.append(task_features(t, site_data(site)))
    return out


def write(rows: list[dict], out: str | Path) -> Path:
    import pandas as pd

    out = Path(REPO / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        out.write_text("".join(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                           for k, v in r.items()}) + "\n" for r in rows))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="frozen tasks.jsonl")
    ap.add_argument("--out", required=True, help=".parquet or .jsonl")
    a = ap.parse_args(argv)
    rows = build(a.set)
    p = write(rows, a.out)
    if p.suffix == ".parquet":                       # a jsonl twin for tools without pyarrow
        write(rows, p.with_suffix(".jsonl"))
    n_tr = sum(r.get("has_track", 0) for r in rows)
    print(json.dumps({"set": a.set, "tasks": len(rows), "with_track": n_tr, "out": str(p)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
