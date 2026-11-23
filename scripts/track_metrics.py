"""Tracking metrics for v2 observation sets (S7.1; OFFLINE, reads truth: never run inside an env step).

Per set: observed.duckdb (track_points) against truth/truth.duckdb (emitters, emitter_points) and the background
emitters (world_cache_v2/spectrum_env.yaml).

Track labelling (independent of the stored track_labels): a track belongs to the scenario emitter with the smallest
median xy distance over the track's points (truth sample at the same 2 s time), among emitters seen at >= max(2, n/2)
of them, if that median is < 250 m; otherwise `clutter` if a background emitter is within 300 m (median), else
`spurious`. Classes: single (drone, no swarm), swarm, friendly, controller, decoy.

Metrics
  false_track_rate  spurious / all tracks;  nontarget_rate  (spurious + clutter) / all tracks (audit E's "42 %")
  tracks/emitter    mean over all emitters of the class (fragmentation; 0 for undetected)
  coverage          median over emitters of the fraction of active 2 s samples with a point of an own track
  id_switches       per 2 s step, the GOSPA assignment truth -> track (d < c); a switch when an emitter's track
                    changes from its previous one; total per class and per emitter
  containment       fraction of fix-time points (t % 10 == 0) of labelled tracks whose truth lies inside the 2-sigma
                    ellipse (nominal 1 - e^-2 = 0.865)
  fix error         xy error at the track's own fix times (bearings rows; median, p90); |z| error median
  GOSPA             c = 200 m, p = 2, alpha = 2 per 2 s step over all scenario emitters (truth) and the points of
                    tracks not labelled clutter (real background transmitters are not false tracks); mean total,
                    mean localisation term, mean missed and false counts

  python scripts/track_metrics.py --sets runs/s71/new --seeds 13-16 [--truth-root data/scenarios] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.belief.assign import linear_sum_assignment  # noqa: E402

C_M, P_EXP = 200.0, 2
CLASSES = ("single", "swarm", "friendly", "controller", "decoy")
NOMINAL = 1 - math.exp(-2)


def gospa(X: np.ndarray, Y: np.ndarray, c: float = C_M, p: int = P_EXP) -> dict:
    """GOSPA (alpha = 2) between truth X [n, 2] and estimates Y [m, 2]. Returns total, loc (sum d^p of assigned
    pairs), missed, false and the assignment pairs (i, j) with d < c."""
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return {"total": (c ** p / 2 * (n + m)) ** (1 / p), "loc": 0.0, "missed": n, "false": m, "pairs": []}
    D = np.hypot(X[:, None, 0] - Y[None, :, 0], X[:, None, 1] - Y[None, :, 1])
    cost = np.minimum(D, c) ** p
    r, k = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(r, k) if D[i, j] < c]
    loc = float(sum(D[i, j] ** p for i, j in pairs))
    missed, false = n - len(pairs), m - len(pairs)
    return {"total": (loc + c ** p / 2 * (missed + false)) ** (1 / p), "loc": loc, "missed": missed, "false": false,
            "pairs": pairs}


def _bg() -> dict:
    env = REPO / "data" / "site_kandla" / "world_cache_v2" / "spectrum_env.yaml"
    return {e["id"]: (e["x_m"], e["y_m"]) for e in yaml.safe_load(env.read_text())["emitters"]} if env.exists() else {}


def _cls(kind: str, swarm) -> str:
    if kind == "drone":
        return "swarm" if swarm else "single"
    return {"friendly_drone": "friendly"}.get(kind, kind)


def evaluate(obs_dir: Path | None, truth_dir: Path, bg: dict | None = None, points: list | None = None,
             fix_times: set | None = None) -> dict:
    """Metrics of one set; `points` (track_points rows) and `fix_times` ({(track_id, t)} of the track's own fixes)
    replace reading obs_dir/observed.duckdb (track_points; bearings rows mark the fix times)."""
    bg = _bg() if bg is None else bg
    t = duckdb.connect(str(truth_dir / "truth" / "truth.duckdb"), read_only=True)
    em = {r[0]: _cls(r[1], r[2]) for r in t.execute("select emitter_id, kind, swarm_id from emitters").fetchall()}
    P: dict = defaultdict(dict)
    for e, ts, x, y, z in t.execute("select emitter_id, t_s, x_m, y_m, z_m from emitter_points").fetchall():
        P[e][int(round(ts))] = (x, y, z)
    t.close()
    tp = defaultdict(list)
    if points is None:
        o = duckdb.connect(str(obs_dir / "observed.duckdb"), read_only=True)
        points = o.execute("select track_id, t_s, x_m, y_m, z_m, a_m, b_m, theta_deg from track_points").fetchall()
        fix_times = {(r[0], int(round(r[1]))) for r in o.execute("select distinct track_id, t_s from bearings").fetchall()}
        o.close()
    fix_times = {(a, int(round(b))) for a, b in (fix_times or set())}
    for r in sorted(points, key=lambda q: q[1]):
        tp[r[0]].append(tuple(r[1:8]))
    # --- label tracks
    lab = {}
    for tid, rows in tp.items():
        best, bd = None, math.inf
        for e, pe in P.items():
            ds = [math.hypot(pe[k][0] - x, pe[k][1] - y) for (ts, x, y, *_r) in rows
                  if (k := int(round(ts))) in pe]
            if len(ds) >= max(2, len(rows) // 2) and np.median(ds) < bd:
                best, bd = e, float(np.median(ds))
        if best is not None and bd < 250:
            lab[tid] = best
        else:
            dm = min((np.median([math.hypot(bx - x, by - y) for (_ts, x, y, *_r) in rows]) for bx, by in bg.values()),
                     default=1e9)
            lab[tid] = "clutter" if dm < 300 else "spurious"
    n_tr = len(tp)
    n_sp = sum(v == "spurious" for v in lab.values())
    n_cl = sum(v == "clutter" for v in lab.values())
    # --- per emitter: tracks, coverage
    own = defaultdict(list)
    for tid, e in lab.items():
        own[e].append(tid)
    per = {}
    for e, c in em.items():
        act = set(P[e])
        if not act:
            continue
        cov = set()
        for tid in own.get(e, []):
            cov |= {int(round(r[0])) for r in tp[tid]}
        per[e] = {"cls": c, "n_tracks": len(own.get(e, [])), "coverage": len(cov & act) / len(act), "switches": 0}
    # --- containment and errors at fix times
    inside, err, zerr = defaultdict(list), defaultdict(list), defaultdict(list)
    for tid, e in lab.items():
        if e not in em:
            continue
        for ts, x, y, z, a, b, th in tp[tid]:
            k = int(round(ts))
            if k % 10 or k not in P[e]:
                continue
            px, py, pz = P[e][k]
            dx, dy = px - x, py - y
            u = math.radians(th)
            al, ac = dx * math.sin(u) + dy * math.cos(u), dx * math.cos(u) - dy * math.sin(u)
            c = em[e]
            inside[c].append((al / max(a, 1e-6)) ** 2 + (ac / max(b, 1e-6)) ** 2 <= 1)
            if (tid, k) in fix_times:                     # error at the track's own fixes (filtered estimate)
                err[c].append(math.hypot(dx, dy))
                if z is not None and np.isfinite(z):
                    zerr[c].append(abs(z - pz))
    # --- GOSPA and ID switches per 2 s step
    steps = sorted({k for pe in P.values() for k in pe} | {int(round(r[0])) for rows in tp.values() for r in rows})
    tpk = defaultdict(list)
    for tid, rows in tp.items():
        if lab[tid] == "clutter":
            continue
        for r in rows:
            tpk[int(round(r[0]))].append((tid, r[1], r[2]))
    g_tot, g_loc, g_mis, g_fal = [], [], [], []
    last = {}
    for k in steps:
        es = [e for e in per if k in P[e]]
        ys = tpk.get(k, [])
        X = np.array([P[e][k][:2] for e in es]).reshape(-1, 2)
        Y = np.array([(q[1], q[2]) for q in ys]).reshape(-1, 2)
        g = gospa(X, Y)
        g_tot.append(g["total"])
        g_loc.append(g["loc"])
        g_mis.append(g["missed"])
        g_fal.append(g["false"])
        for i, j in g["pairs"]:
            e, tid = es[i], ys[j][0]
            if e in last and last[e] != tid:
                per[e]["switches"] += 1
            last[e] = tid
    out = {"tracks": n_tr, "spurious": n_sp, "clutter": n_cl, "gospa": float(np.mean(g_tot)),
           "gospa_missed": float(np.mean(g_mis)), "gospa_false": float(np.mean(g_fal)), "per_class": {}}
    for c in CLASSES:
        v = [p for p in per.values() if p["cls"] == c]
        if not v:
            continue
        n = np.array([p["n_tracks"] for p in v])
        out["per_class"][c] = {"N": len(v), "tracks_per_emitter": float(n.mean()), "undetected": float(np.mean(n == 0)),
                               "fragmented": float(np.mean(n > 1)),
                               "coverage_med": float(np.median([p["coverage"] for p in v])),
                               "id_switches": int(sum(p["switches"] for p in v)),
                               "_in": inside[c], "_err": err[c], "_z": zerr[c]}
    return out


def combine(res: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in res) for k in ("tracks", "spurious", "clutter")}
    out = {**tot, "false_track_rate": tot["spurious"] / max(tot["tracks"], 1),
           "nontarget_rate": (tot["spurious"] + tot["clutter"]) / max(tot["tracks"], 1),
           "gospa": float(np.mean([r["gospa"] for r in res])),
           "gospa_missed": float(np.mean([r["gospa_missed"] for r in res])),
           "gospa_false": float(np.mean([r["gospa_false"] for r in res])), "per_class": {}}
    ins_all, err_all = [], []
    for c in CLASSES:
        v = [r["per_class"][c] for r in res if c in r["per_class"]]
        if not v:
            continue
        N = sum(x["N"] for x in v)
        ins = [b for x in v for b in x["_in"]]
        er = [b for x in v for b in x["_err"]]
        zz = [b for x in v for b in x["_z"]]
        ins_all += ins
        err_all += er
        out["per_class"][c] = {
            "N": N, "tracks_per_emitter": sum(x["tracks_per_emitter"] * x["N"] for x in v) / N,
            "undetected": sum(x["undetected"] * x["N"] for x in v) / N,
            "fragmented": sum(x["fragmented"] * x["N"] for x in v) / N,
            "coverage_med": float(np.median([x["coverage_med"] for x in v])),
            "id_switches": sum(x["id_switches"] for x in v),
            "containment": float(np.mean(ins)) if ins else None, "n_fix": len(ins),
            "err_med": float(np.median(er)) if er else None, "err_p90": float(np.percentile(er, 90)) if er else None,
            "z_abs_med": float(np.median(zz)) if zz else None}
    out["containment"] = float(np.mean(ins_all)) if ins_all else None
    out["err_med"] = float(np.median(err_all)) if err_all else None
    out["err_p90"] = float(np.percentile(err_all, 90)) if err_all else None
    return out


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def table(name: str, r: dict) -> str:
    lines = [f"## {name}",
             f"tracks {r['tracks']}  false (spurious) {r['false_track_rate']:.3f}  non-target (spurious+clutter) "
             f"{r['nontarget_rate']:.3f}  GOSPA {r['gospa']:.1f} m (missed {r['gospa_missed']:.1f}, false "
             f"{r['gospa_false']:.1f} per step)  containment {r['containment'] or 0:.3f}  fix err med "
             f"{r['err_med'] or 0:.1f} m p90 {r['err_p90'] or 0:.1f} m",
             "| class | N | tracks/emitter | undetected | fragmented | coverage med | ID switches | containment (n) | "
             "err med / p90 m | abs z err med m |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in r["per_class"].items():
        f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
        lines.append(f"| {c} | {v['N']} | {v['tracks_per_emitter']:.2f} | {v['undetected']:.2f} | {v['fragmented']:.2f}"
                     f" | {v['coverage_med']:.2f} | {v['id_switches']} | {f(v['containment'], 3)} ({v['n_fix']}) | "
                     f"{f(v['err_med'], 1)} / {f(v['err_p90'], 1)} | {f(v['z_abs_med'], 1)} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True, help="root holding kandla_s<N>_X dirs with observed.duckdb, or one set dir")
    ap.add_argument("--seeds", default="13-16")
    ap.add_argument("--suffix", default="_X")
    ap.add_argument("--truth-root", default=None, help="where truth/ lives (default: the set dir itself)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.sets)
    bg = _bg()
    res = []
    if (root / "observed.duckdb").exists():
        res.append(evaluate(root, Path(a.truth_root) if a.truth_root else root, bg))
    else:
        for s in parse_seeds(a.seeds):
            d = root / f"kandla_s{s}{a.suffix}"
            res.append(evaluate(d, (Path(a.truth_root) / d.name) if a.truth_root else d, bg))
    r = combine(res)
    print(table(a.name or str(root), r))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
