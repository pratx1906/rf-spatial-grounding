"""E6: joint truth-free heading self-calibration (bundle adjustment over all tracked epochs). 2026-11-07.

The T-FIX estimator (tools/consistency.py: leave-one-out + greedy sparse) assumes most nodes are calibrated and
recovers N(0, 10 deg) per-node biases poorly (median |err| 5 deg on M2; docs/paper/TWIN_FIX.md). Here every node bias
and every epoch position are estimated together:

    minimise  sum_{epoch e, node s}  rho( wrap(obs_es - bearing(p_s -> x_e) - b_s) / sigma )  +  sum_s (b_s / PRIOR)^2

rho = soft-L1 (robust to multipath ghosts taken as strongest); obs = the strongest-power hypothesis per node and epoch
(as tools/consistency); epochs need >= 3 nodes. A common rotation of all nodes is only weakly observable with 3+
non-collinear nodes (exactly unobservable with 2), so a weak prior (PRIOR deg) keeps it bounded. Observations only
(observed.duckdb + sensors.yaml); the realised biases (truth/world_variant.yaml) are read afterwards, offline, only to
score the recovery.
    python runs/s82/joint_calib.py --tags M0 M5 M1 M6 M2 M7 M8 --seeds 51-60 --out runs/s82/joint_calib.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "data/scenarios_v3"
SIGMA = 1.5          # deg, per-bearing scale
F_SCALE = 2.0        # soft-L1 knee in sigma units (~3 deg)
PRIOR = 20.0         # deg, weak prior on each bias
SITE_HALF = 3500.0
MIN_NODES = 2
ACC_M = 60.0        # m, sd of the second difference of a track's position between consecutive epochs


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def bearing(sx, sy, x, y):
    return np.degrees(np.arctan2(x - sx, y - sy)) % 360.0


def load(d: Path):
    sens = {s["id"]: (s["pose"]["x_m"], s["pose"]["y_m"]) for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]
            if s.get("type") == "fixed_df"}
    c = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    rows = c.execute("SELECT track_id, t_s, sensor_id, arg_max(bearing_deg, power_db * 1000 + snr_db) FROM bearings "
                     "WHERE track_id IS NOT NULL GROUP BY track_id, t_s, sensor_id").fetchall()
    ep = {}
    for tid, t, s, b in rows:
        if s in sens:
            ep.setdefault((tid, t), {})[s] = float(b)
    keys = sorted(k for k, e in ep.items() if len(e) >= MIN_NODES)
    return sens, [ep[k] for k in keys], keys


def tri(sens, e, bias):
    A, rhs = [], []
    for s, b in e.items():
        th = math.radians(b - bias.get(s, 0.0))
        n = np.array([math.cos(th), -math.sin(th)])
        A.append(n)
        rhs.append(n @ np.array(sens[s]))
    A, rhs = np.array(A), np.array(rhs)
    M = A.T @ A
    if np.linalg.cond(M) > 1e8:
        return None
    return np.linalg.solve(M, A.T @ rhs)


def fit(sens, epochs, keys):
    ids = sorted(sens)
    ix = {s: i for i, s in enumerate(ids)}
    x0 = [tri(sens, e, {}) for e in epochs]
    keep = [i for i, x in enumerate(x0) if x is not None and np.all(np.abs(x) < SITE_HALF)]
    epochs = [epochs[i] for i in keep]
    keys = [keys[i] for i in keep]
    x0 = np.array([x0[i] for i in keep])
    # motion prior: consecutive epochs of one track (same spacing) -> second difference ~ N(0, ACC_M)
    trip = []
    for k in range(1, len(keys) - 1):
        (t0, a), (t1, b), (t2, c) = keys[k - 1], keys[k], keys[k + 1]
        if t0 == t1 == t2 and abs((b - a) - (c - b)) < 1e-6 and (b - a) <= 30:
            trip.append((k - 1, k, k + 1))
    trip = np.array(trip, dtype=int).reshape(-1, 3)
    E, S = len(epochs), len(ids)
    obs = [(k, ix[s], b) for k, e in enumerate(epochs) for s, b in e.items()]
    ek = np.array([o[0] for o in obs])
    si = np.array([o[1] for o in obs])
    ob = np.array([o[2] for o in obs])
    sx = np.array([sens[ids[i]][0] for i in si])
    sy = np.array([sens[ids[i]][1] for i in si])

    def res(p):
        b, xy = p[:S], p[S:].reshape(E, 2)
        pred = bearing(sx, sy, xy[ek, 0], xy[ek, 1]) + b[si]
        acc = (xy[trip[:, 0]] - 2 * xy[trip[:, 1]] + xy[trip[:, 2]]).ravel() / ACC_M
        return np.concatenate([wrap(ob - pred) / SIGMA, b / PRIOR, acc])

    J = lil_matrix((len(obs) + S + 2 * len(trip), S + 2 * E), dtype=int)
    for r, (k, i, _) in enumerate(obs):
        J[r, i] = 1
        J[r, S + 2 * k] = 1
        J[r, S + 2 * k + 1] = 1
    for i in range(S):
        J[len(obs) + i, i] = 1
    for j, tr in enumerate(trip):
        for d in range(2):
            for k in tr:
                J[len(obs) + S + 2 * j + d, S + 2 * k + d] = 1
    p0 = np.concatenate([np.zeros(S), x0.ravel()])
    sol = least_squares(res, p0, jac_sparsity=J, loss="soft_l1", f_scale=F_SCALE, x_scale="jac", max_nfev=200)
    b = sol.x[:S]
    r = wrap(ob - (bearing(sx, sy, sol.x[S:].reshape(E, 2)[ek, 0], sol.x[S:].reshape(E, 2)[ek, 1]) + b[si]))
    # per-node uncertainty from the Gauss-Newton normal matrix (robust-weighted), bias block
    try:
        Jd = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac
        cov = np.linalg.pinv(Jd.T @ Jd)[:S, :S] * SIGMA ** 2
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:  # noqa: BLE001
        sd = np.full(S, np.nan)
    inl = np.abs(r) < 3 * SIGMA * F_SCALE
    return {s: {"bias_deg": round(float(b[i]), 3), "sd_deg": round(float(sd[i]), 3),
                "n": int((si == i).sum()), "inlier_frac": round(float(inl[si == i].mean()), 3) if (si == i).any() else None}
            for i, s in enumerate(ids)}, E, bool(sol.success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["M0", "M5", "M1", "M6", "M2", "M7", "M8"])
    ap.add_argument("--seeds", default="51-60")
    ap.add_argument("--out", default="runs/s82/joint_calib.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rep, summ = {}, {}
    for tag in a.tags:
        errs, raw = [], []
        for s in range(lo, hi + 1):
            d = SC / (f"kandla_s{s}_X" if tag == "N" else f"kandla_s{s}_X_{tag}")
            if not (d / "observed.duckdb").exists():
                continue
            sens, epochs, keys = load(d)
            if len(epochs) < 5:
                rep[f"{tag}/s{s}"] = {"ok": False, "n_epochs": len(epochs)}
                continue
            est, E, ok = fit(sens, epochs, keys)
            wv = d / "truth/world_variant.yaml"            # offline scoring only, after the estimate
            tb = {k: v["heading_bias_deg"] for k, v in yaml.safe_load(wv.read_text())["nodes"].items()} if wv.exists() else {}
            for k, v in est.items():
                if k in tb:
                    v["true_deg"] = round(tb[k], 3)
                    v["err_deg"] = round(v["bias_deg"] - tb[k], 3)
                    errs.append(abs(v["err_deg"]))
                    raw.append(abs(tb[k]))
            rep[f"{tag}/s{s}"] = {"ok": ok, "n_epochs": E, "nodes": est}
        if errs:
            summ[tag] = {"n_nodes": len(errs), "median_abs_err_deg": round(float(np.median(errs)), 3),
                         "p90_abs_err_deg": round(float(np.percentile(errs, 90)), 3),
                         "median_abs_true_bias_deg": round(float(np.median(raw)), 3),
                         "frac_nodes_improved": round(float(np.mean(np.array(errs) < np.array(raw))), 3)}
            print(tag, summ[tag], flush=True)
    out = REPO / a.out
    out.write_text(json.dumps({"summary": summ, "per_scenario": rep,
                               "params": {"SIGMA": SIGMA, "F_SCALE": F_SCALE, "PRIOR": PRIOR, "MIN_NODES": MIN_NODES, "ACC_M": ACC_M}}, indent=1))


if __name__ == "__main__":
    main()
