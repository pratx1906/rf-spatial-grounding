"""RF living-world gap report (D-031): offline analysis, never part of an env step.

    python scripts/gap_report.py [--policies expert sft_v11] [--band ism_2400]  -> runs/eval/gap_report.json

1. Paired world gap (non-DF areas): eval/frozen/v3w/<W>.jsonl row i is the same (seed, template, question) as N row i,
   so score differences W - N are paired by row; per-area mean and bootstrap 95% CI.
2. DF unpaired: per world, mean score of df_<W>.jsonl with a bootstrap CI, and W - N with an unpaired bootstrap CI;
   per-family means.
3. Map calibration: does the belief ensemble's ghost risk (nominal + B1-B3, what the agent sees) predict where the
   hidden worlds W1t-W3 actually put the strongest arrival off the geometric bearing? Compared with the nominal
   model alone (0/1 ghost), by Brier score and AUC, per height, over all sensors' node grids.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.rf_maps import GHOST_DEG, _load, _members, _strongest, node_metrics  # noqa: E402

SETS = REPO / "eval" / "frozen" / "v3w"
EV = REPO / "runs" / "eval"
SITE = REPO / "data" / "site_kandla"
WORLDS = ["W0", "W1t", "W1", "W2", "W3"]


def _ci(xs: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    m = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [round(m[int(0.025 * n)], 3), round(m[int(0.975 * n)], 3)]


def _scores(policy: str, name: str) -> dict[str, dict] | None:
    f = EV / f"{policy}_v3w_{name}.json"
    if not f.exists():
        return None
    return {r["task_id"]: r for r in json.loads(f.read_text())["rows"]}


def _ids(name: str) -> list[str]:
    return [json.loads(x)["task_id"] for x in (SETS / f"{name}.jsonl").read_text().splitlines()]


def paired(policy: str) -> dict:
    base, ids_n = _scores(policy, "N"), _ids("N")
    if base is None:
        return {}
    out = {"N": {"score": round(statistics.mean(base[t]["score"] for t in ids_n), 3)}}
    for w in WORLDS:
        sw = _scores(policy, w)
        if sw is None:
            continue
        by = defaultdict(list)
        for tn, tw in zip(ids_n, _ids(w)):
            d = sw[tw]["score"] - base[tn]["score"]
            by[base[tn]["area"]].append(d)
            by["_all"].append(d)
        out[w] = {"score": round(statistics.mean(sw[t]["score"] for t in _ids(w)), 3),
                  **{a: {"n": len(d), "diff": round(statistics.mean(d), 3), "ci95": _ci(d)} for a, d in sorted(by.items())}}
    return out


def df_unpaired(policy: str) -> dict:
    out, fams = {}, defaultdict(dict)
    sn = _scores(policy, "df_N")
    if sn is None:
        return {}
    xn = [r["score"] for r in sn.values()]
    for w in ["N"] + WORLDS:
        s = _scores(policy, f"df_{w}")
        if s is None:
            continue
        x = [r["score"] for r in s.values()]
        rng = random.Random(1)
        diffs = sorted(statistics.mean(rng.choices(x, k=len(x))) - statistics.mean(rng.choices(xn, k=len(xn)))
                       for _ in range(2000))
        out[w] = {"n": len(x), "score": round(statistics.mean(x), 3), "ci95": _ci(x),
                  "diff_vs_N": round(statistics.mean(x) - statistics.mean(xn), 3),
                  "diff_ci95": [round(diffs[50], 3), round(diffs[1949], 3)]}
        per = defaultdict(list)
        for r in s.values():
            per[r["template"]].append(r["score"])
        for f, v in per.items():
            fams[f][w] = round(statistics.mean(v), 3)
    return {"worlds": out, "families": dict(fams)}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks[order] = cum[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def calibration(band: str) -> dict:
    sensors = [s for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"] if s["type"] == "fixed_df"]
    acc = defaultdict(lambda: {"ens": [], "nom": [], "world": []})
    for s in sensors:
        mem = _members(s["id"], band)
        if not mem:
            continue
        g = mem[0]
        gx, gy = np.meshgrid(g["xs"], g["ys"])
        geo = np.degrees(np.arctan2(gx - s["pose"]["x_m"], gy - s["pose"]["y_m"])) % 360.0
        geo3 = np.broadcast_to(geo, (len(g["heights"]),) + geo.shape)
        ens = node_metrics(mem, geo3)["ghost_risk"]
        nom = node_metrics(mem[:1], geo3)["ghost_risk"]
        for w in ("W1t", "W1", "W2", "W3"):
            f = SITE / "world_cache" / "rf" / w / f"air7311_{s['id']}" / "spec_diff" / f"{band}.npz"
            if not f.exists():
                continue
            az, _, ok = _strongest(_load(f))
            ghost = (np.abs((az - geo3 + 180.0) % 360.0 - 180.0) > GHOST_DEG) & ok
            valid = ok & np.isfinite(ens) & np.isfinite(nom)
            for ih, h in enumerate(g["heights"]):
                v = valid[ih]
                k = int(round(float(h)))
                acc[k]["ens"].append(ens[ih][v])
                acc[k]["nom"].append(nom[ih][v])
                acc[k]["world"].append(ghost[ih][v])
    out = {}
    for h in sorted(acc):
        e, n, y = (np.concatenate(acc[h][k]) for k in ("ens", "nom", "world"))
        y = y.astype(bool)
        bins = {}
        for lo, hi, name in ((0, 0, "0"), (0.01, 0.49, "0.25"), (0.5, 0.74, "0.5"), (0.75, 1, ">=0.75")):
            m = (e >= lo) & (e <= hi)
            if m.any():
                bins[name] = {"cells": int(m.sum()), "world_ghost_rate": round(float(y[m].mean()), 3)}
        out[h] = {"cells": int(len(y)), "world_ghost_rate": round(float(y.mean()), 4),
                  "brier_ensemble": round(float(((e - y) ** 2).mean()), 4),
                  "brier_nominal": round(float(((n - y) ** 2).mean()), 4),
                  "auc_ensemble": round(_auc(e, y), 3), "auc_nominal": round(_auc(n, y), 3),
                  "by_ensemble_risk": bins}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["expert", "sft_v11"])
    ap.add_argument("--band", default="ism_2400")
    a = ap.parse_args()
    rep = {"paired": {p: paired(p) for p in a.policies}, "df_unpaired": {p: df_unpaired(p) for p in a.policies},
           "calibration": {a.band: calibration(a.band)}}
    (EV / "gap_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
