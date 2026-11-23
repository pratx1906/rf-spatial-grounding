"""E7 (exploratory, offline, CPU): would choosing each node's DF hypothesis by cross-node consistency beat 'strongest
peak' on multipath? 2026-11-07. Nominal test worlds (seeds 15, 16, 51-60). Observations select; truth only scores.

Per tracked epoch (track, t) with >= 3 nodes: every node has 1-3 bearing hypotheses (MUSIC peaks: direct path and
reflections). Rules:
  strongest   : max power per node (current production choice)
  consistent  : the combination (one hyp per node) minimising  sum_s (perp. angular residual to the joint LS fix / 1.5 deg)^2
                + LAMBDA * sum_s (power_max_s - power_s) / 3 dB    (power drop as a soft prior on the direct path)
  oracle      : per node, the hyp closest to the truth bearing (upper bound)
Score: |bearing - truth bearing| per node-epoch (truth = emitter position of the track's label at the nearest
emitter_points sample within 2 s; tracks labelled none/clutter/spurious skipped), and the resulting fix error.
    python runs/s82/e7_hyp_select.py -> runs/s82/e7_hyp_select.json
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "data/scenarios_v3"
SEEDS = [15, 16] + list(range(51, 61))
SIG = 1.5
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def wrap(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def brg(s, x, y):
    return math.degrees(math.atan2(x - s[0], y - s[1])) % 360.0


def lsfix(sens, combo):
    A, rhs = [], []
    for s, b in combo:
        th = math.radians(b)
        n = np.array([math.cos(th), -math.sin(th)])
        A.append(n)
        rhs.append(n @ np.array(sens[s]))
    A, rhs = np.array(A), np.array(rhs)
    M = A.T @ A
    if np.linalg.cond(M) > 1e8:
        return None
    return np.linalg.solve(M, A.T @ rhs)


def main():
    out = {"lambda": {}, "per_seed": {}}
    acc = {k: {"err": [], "fix": []} for k in ["strongest", "oracle"] + [f"consistent_{l}" for l in LAMBDAS]}
    nep = 0
    for seed in SEEDS:
        d = SC / f"kandla_s{seed}_X"
        sens = {s["id"]: (s["pose"]["x_m"], s["pose"]["y_m"]) for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]
                if s.get("type") == "fixed_df"}
        o = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
        rows = o.execute("SELECT track_id, t_s, sensor_id, bearing_deg, power_db FROM bearings WHERE track_id IS NOT NULL").fetchall()
        t = duckdb.connect(str(d / "truth/truth.duckdb"), read_only=True)        # offline scoring only
        lab = {tid: (e, k) for tid, e, k in t.execute("SELECT track_id, emitter_id, emitter_kind FROM track_labels").fetchall()}
        pts = {}
        for e, ts, x, y in t.execute("SELECT emitter_id, t_s, x_m, y_m FROM emitter_points").fetchall():
            pts.setdefault(e, []).append((ts, x, y))
        pts = {e: np.array(sorted(v)) for e, v in pts.items()}
        ep = {}
        for tid, ts, s, b, pw in rows:
            if s in sens:
                ep.setdefault((tid, ts), {}).setdefault(s, []).append((float(b), float(pw)))
        for (tid, ts), hy in ep.items():
            if len(hy) < 3 or tid not in lab or lab[tid][1] in (None, "none", "clutter", "spurious") or lab[tid][0] not in pts:
                continue
            P = pts[lab[tid][0]]
            i = int(np.argmin(np.abs(P[:, 0] - ts)))
            if abs(P[i, 0] - ts) > 2.0:
                continue
            tx, ty = P[i, 1], P[i, 2]
            truth = {s: brg(sens[s], tx, ty) for s in hy}
            nodes = sorted(hy)
            nep += 1

            def score(name, combo):
                xy = lsfix(sens, combo)
                for s, b in combo:
                    acc[name]["err"].append(abs(float(wrap(b - truth[s]))))
                if xy is not None:
                    acc[name]["fix"].append(math.hypot(xy[0] - tx, xy[1] - ty))

            score("strongest", [(s, max(hy[s], key=lambda h: h[1])[0]) for s in nodes])
            score("oracle", [(s, min(hy[s], key=lambda h: abs(float(wrap(h[0] - truth[s]))))[0]) for s in nodes])
            combos = list(itertools.product(*[[(s, h) for h in hy[s]] for s in nodes]))
            pmax = {s: max(h[1] for h in hy[s]) for s in nodes}
            for lam in LAMBDAS:
                best = None
                for c in combos:
                    cb = [(s, h[0]) for s, h in c]
                    xy = lsfix(sens, cb)
                    if xy is None:
                        continue
                    r = sum((float(wrap(b - brg(sens[s], xy[0], xy[1]))) / SIG) ** 2 for s, b in cb)
                    r += lam * sum((pmax[s] - h[1]) / 3.0 for s, h in c)
                    if best is None or r < best[0]:
                        best = (r, cb)
                if best is not None:
                    score(f"consistent_{lam}", best[1])
        print(seed, nep, flush=True)
    for k, v in acc.items():
        e, f = np.array(v["err"]), np.array(v["fix"])
        out["lambda"][k] = {"n_node_epochs": len(e), "bearing_err_med_deg": round(float(np.median(e)), 3),
                            "bearing_err_p90_deg": round(float(np.percentile(e, 90)), 3),
                            "frac_gt5deg": round(float(np.mean(e > 5)), 4),
                            "fix_err_med_m": round(float(np.median(f)), 1), "fix_err_p90_m": round(float(np.percentile(f, 90)), 1)}
    out["n_epochs"] = nep
    (REPO / "runs/s82/e7_hyp_select.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["lambda"], indent=1))


if __name__ == "__main__":
    main()
