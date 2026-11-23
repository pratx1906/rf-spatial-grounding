"""E6: anchored truth-free heading self-calibration. 2026-11-07. Observations + twin only; truth read afterwards to score.

Stage 1, known emitters (standard DF practice: calibrate on reference emitters at surveyed positions). The twin lists
friendly emitters with positions (friendly_emitters.yaml, belief side). For node s, every burst whose carrier is
within FREQ_TOL of a friendly emitter's frequency gives candidate residuals r = bearing - bearing(s -> emitter) for
every emitter on that frequency (several emitters can share a channel). Consensus: b_s = the value supported by the
most bursts (a candidate within +-TOL), refined as the median of the supporting residuals. Accept if support >= MIN_SUP
bursts and support fraction >= MIN_FRAC (else NLOS / reflections dominate, e.g. a rooftop node that only hears the
reference emitters off a wall).
Stage 2, drone tracks. Nodes not anchored in stage 1 are fitted by robust least squares over the tracker's multi-node
epochs, with the anchored nodes held at their stage-1 values (a biased free node can no longer tilt the others' fixes,
the failure mode of the T-FIX leave-one-out estimator).
    python runs/s82/anchored_calib.py --tags M0 M5 M1 M6 M2 M7 M8 --seeds 51-60 --out runs/s82/anchored_calib.json
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

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "data/scenarios_v3"
FREQ_TOL = 0.3      # MHz
TOL = 2.0           # deg, consensus window
SEARCH = 90.0       # deg, bias search range
MIN_SUP = 20
MIN_FRAC = 0.6
MIN_EPOCHS = 20     # stage 2: fewer usable epochs -> unresolved (no correction, node flagged)
MIN_INLIER = 0.6
MIN_DIST = 50.0     # m
SIGMA = 1.5
F_SCALE = 2.0
PRIOR_FREE = 30.0


def wrap(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def brg(sx, sy, x, y):
    return np.degrees(np.arctan2(np.asarray(x) - sx, np.asarray(y) - sy)) % 360.0


def stage1(d: Path, sens: dict) -> dict:
    fe = yaml.safe_load((d / "friendly_emitters.yaml").read_text()).get("emitters", [])
    c = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    out = {}
    grid = np.arange(-SEARCH, SEARCH + 1e-9, 0.1)
    for s, (sx, sy) in sens.items():
        B = c.execute("SELECT f_mhz, bearing_deg FROM bursts WHERE sensor_id = ? AND bearing_deg IS NOT NULL", [s]).fetchall()
        cands = []                         # per burst: array of candidate residuals
        for f, b in B:
            r = [wrap(b - brg(sx, sy, e["x_m"], e["y_m"])) for e in fe
                 if abs(f - e["freq_mhz"]) < FREQ_TOL and math.hypot(e["x_m"] - sx, e["y_m"] - sy) > MIN_DIST]
            if r:
                cands.append(np.array(r, dtype=float))
        if not cands:
            out[s] = {"anchored": False, "n_ref_bursts": 0}
            continue
        sup = np.zeros_like(grid)
        for r in cands:
            sup += (np.abs(wrap(grid[:, None] - r[None, :])) <= TOL).any(axis=1)
        g = grid[int(np.argmax(sup))]
        match = [r[np.argmin(np.abs(wrap(r - g)))] for r in cands]
        match = np.array([m for m in match if abs(wrap(m - g)) <= TOL])
        b = float(np.median(match))
        frac = len(match) / len(cands)
        out[s] = {"anchored": bool(len(match) >= MIN_SUP and frac >= MIN_FRAC), "bias_deg": round(b, 3),
                  "support": int(len(match)), "n_ref_bursts": len(cands), "support_frac": round(frac, 3),
                  "mad_deg": round(float(np.median(np.abs(match - b))), 3)}
    return out


def epochs(d: Path, sens: dict):
    c = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    rows = c.execute("SELECT track_id, t_s, sensor_id, arg_max(bearing_deg, power_db * 1000 + snr_db) FROM bearings "
                     "WHERE track_id IS NOT NULL GROUP BY track_id, t_s, sensor_id").fetchall()
    ep = {}
    for tid, t, s, b in rows:
        if s in sens:
            ep.setdefault((tid, t), {})[s] = float(b)
    return [e for e in ep.values() if len(e) >= 2]


def stage2(sens: dict, eps: list, fixed: dict) -> dict:
    free = sorted(s for s in sens if s not in fixed)
    use = [e for e in eps if any(s in e for s in free) and sum(s in fixed for s in e) >= 2]
    if not free or not use:
        return {s: {"bias_deg": None, "n_epochs": 0} for s in free}
    fi = {s: i for i, s in enumerate(free)}
    # init each epoch position from the anchored nodes only (de-biased)
    x0 = []
    for e in use:
        A, rhs = [], []
        for s, b in e.items():
            if s in fixed:
                th = math.radians(b - fixed[s])
                n = np.array([math.cos(th), -math.sin(th)])
                A.append(n)
                rhs.append(n @ np.array(sens[s]))
        A, rhs = np.array(A), np.array(rhs)
        x0.append(np.linalg.lstsq(A, rhs, rcond=None)[0])
    x0 = np.array(x0)
    E, F = len(use), len(free)
    obs = [(k, s, b) for k, e in enumerate(use) for s, b in e.items()]
    ek = np.array([o[0] for o in obs])
    sx = np.array([sens[o[1]][0] for o in obs])
    sy = np.array([sens[o[1]][1] for o in obs])
    ob = np.array([o[2] for o in obs])
    isfree = np.array([o[1] in fi for o in obs])
    fidx = np.array([fi.get(o[1], 0) for o in obs])
    fixb = np.array([fixed.get(o[1], 0.0) for o in obs])

    def res(p):
        bf, xy = p[:F], p[F:].reshape(E, 2)
        bias = np.where(isfree, bf[fidx], fixb)
        return np.concatenate([wrap(ob - brg(sx, sy, xy[ek, 0], xy[ek, 1]) - bias) / SIGMA, bf / PRIOR_FREE])

    best = None
    for b0 in (-40, -20, 0, 20, 40):          # multi-start over the free biases (the residual is periodic)
        sol = least_squares(res, np.concatenate([np.full(F, float(b0)), x0.ravel()]), loss="soft_l1",
                            f_scale=F_SCALE, x_scale="jac", max_nfev=300)
        if best is None or sol.cost < best.cost:
            best = sol
    r = res(best.x)[:len(obs)] * SIGMA
    return {s: {"bias_deg": round(float(wrap(best.x[i])), 3), "n_epochs": int(len({k for k, ff in zip(ek, isfree & (fidx == i)) if ff})),
                "inlier_frac": round(float(np.mean(np.abs(r[isfree & (fidx == i)]) < 6)), 3)} for s, i in fi.items()}


def calibrate(d: Path) -> dict:
    sens = {s["id"]: (s["pose"]["x_m"], s["pose"]["y_m"]) for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]
            if s.get("type") == "fixed_df"}
    s1 = stage1(d, sens)
    fixed = {s: v["bias_deg"] for s, v in s1.items() if v.get("anchored")}
    s2 = stage2(sens, epochs(d, sens), fixed) if len(fixed) >= 2 else {}
    est = {}
    for s in sens:
        if s in fixed:
            est[s] = {"bias_deg": fixed[s], "source": "reference_emitters", **s1[s]}
        elif s in s2 and s2[s]["bias_deg"] is not None and s2[s]["n_epochs"] >= MIN_EPOCHS                 and s2[s]["inlier_frac"] >= MIN_INLIER:
            est[s] = {"bias_deg": s2[s]["bias_deg"], "source": "drone_tracks", **s2[s], "stage1": s1.get(s)}
        else:
            est[s] = {"bias_deg": 0.0, "source": "unresolved", "stage1": s1.get(s), "stage2": s2.get(s)}
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["M0", "M5", "M1", "M6", "M2", "M7", "M8"])
    ap.add_argument("--seeds", default="51-60")
    ap.add_argument("--out", default="runs/s82/anchored_calib.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rep, summ = {}, {}
    for tag in a.tags:
        errs, raw, src = [], [], []
        for s in range(lo, hi + 1):
            d = SC / (f"kandla_s{s}_X" if tag == "N" else f"kandla_s{s}_X_{tag}")
            if not (d / "observed.duckdb").exists():
                continue
            est = calibrate(d)
            wv = d / "truth/world_variant.yaml"            # offline scoring only, after the estimate
            tb = {k: v["heading_bias_deg"] for k, v in yaml.safe_load(wv.read_text())["nodes"].items()} if wv.exists() else {}
            for k, v in est.items():
                if k in tb:
                    v["true_deg"] = round(tb[k], 3)
                    v["err_deg"] = round(float(wrap(v["bias_deg"] - tb[k])), 3)
                    errs.append(abs(v["err_deg"]))
                    raw.append(abs(tb[k]))
                    src.append(v["source"])
            rep[f"{tag}/s{s}"] = est
        if errs:
            e, r, src = np.array(errs), np.array(raw), np.array(src)
            summ[tag] = {"n_nodes": len(e), "median_abs_err_deg": round(float(np.median(e)), 3),
                         "p90_abs_err_deg": round(float(np.percentile(e, 90)), 3),
                         "median_abs_true_bias_deg": round(float(np.median(r)), 3),
                         "rms_after_deg": round(float(np.sqrt(np.mean(e ** 2))), 3),
                         "rms_before_deg": round(float(np.sqrt(np.mean(r ** 2))), 3),
                         "sources": {k: int((src == k).sum()) for k in set(src)},
                         "median_abs_err_by_source": {k: round(float(np.median(e[src == k])), 3) for k in set(src)}}
            print(tag, json.dumps(summ[tag]), flush=True)
    (REPO / a.out).write_text(json.dumps({"summary": summ, "per_scenario": rep, "params": {
        "FREQ_TOL": FREQ_TOL, "TOL": TOL, "MIN_SUP": MIN_SUP, "MIN_FRAC": MIN_FRAC, "SIGMA": SIGMA,
        "F_SCALE": F_SCALE, "PRIOR_FREE": PRIOR_FREE, "MIN_EPOCHS": MIN_EPOCHS, "MIN_INLIER": MIN_INLIER}}, indent=1))


if __name__ == "__main__":
    main()
