"""Heading-bias magnitude sweep (review W4; not pre-registered, exploratory). 2026-11-07.

Per policy (scripted expert, SFT v1.4 seeds 0 and 1) and per world sigma in {0.5 (M0 control), 1, 3, 5, 10, 15, 30} deg:
paired N-vs-M metrics on the 51s pair subset (pair_id), with cluster-bootstrap CIs over scenario seeds.
  - twin_fail: full task score ok on N, not ok on M (task key = the tracker's own key in each world)
  - world_twin_fail: comparable part correct vs TRUE emitters on N, wrong on M (world key)
  - world_ok_m: comparable part correct vs true emitters on M
  - key_wrong_m: the M world's own tracker key disagrees with the true emitters (the twin itself is wrong; policy-free)
  - flag_m: the policy abstains or flags (requires_human_confirm) on the M twin
Note: at 30 deg seed 58 yields no drone tracks at all, so it has no tasks; its 30 pairs are missing from M8 (survivor
bias: the 30-deg numbers are optimistic).
    python runs/s81/sweep_analysis.py  -> runs/s81/sweep_results.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.pairing import WorldKeys, _load, _seed_of_pair, flagged, key_agrees, part_ok_world  # noqa: E402

EV = REPO / "runs/eval/s81"
SW = [("M0c", 0.5), ("M5c", 1), ("M1c", 3), ("M6c", 5), ("M2c", 10), ("M7c", 15), ("M8c", 30)]
POL = ["expert", "sft14", "sft14s1"]
THR = 0.8
B = 10_000


def per_pair(pol, k, wk):
    nres, ntasks = _load(EV / f"{pol}_Npair.json")
    mres, mtasks = _load(EV / f"{pol}_{k}.json")

    def by_pair(res, tasks):
        return {tasks[r["task_id"]]["pair_id"]: (r, tasks[r["task_id"]]) for r in res["rows"]
                if tasks[r["task_id"]].get("pair_id") and r.get("sample", 0) == 0}

    pn, pm = by_pair(nres, ntasks), by_pair(mres, mtasks)
    out = {}
    te = wk.track_emitter
    for pid in sorted(set(pn) & set(pm)):
        (rn, tn), (rm, tm) = pn[pid], pm[pid]
        wn, wm = wk.key(tn), wk.key(tm)
        d = {"fam": tn["template"], "twin_fail": float(rn["score"] >= THR and rm["score"] < THR),
             "ok_n": float(rn["score"] >= THR), "ok_m": float(rm["score"] >= THR),
             "flag_m": float(flagged(rm.get("answer")))}
        if wn is not None and wm is not None:
            a, b = part_ok_world(tn, rn.get("answer"), wn, te), part_ok_world(tm, rm.get("answer"), wm, te)
            d.update(world_twin_fail=float(a and not b), world_ok_n=float(a), world_ok_m=float(b),
                     key_wrong_m=float(not key_agrees(tm, wm, te)), key_wrong_n=float(not key_agrees(tn, wn, te)))
        out[pid] = d
    return out


def boot(vals_by_pid, rng):
    cl = defaultdict(list)
    for pid, v in vals_by_pid.items():
        cl[_seed_of_pair(pid)].append(v)
    keys = sorted(cl)
    pt = float(np.mean([v for vs in cl.values() for v in vs]))
    bs = np.empty(B)
    arrs = {k: np.asarray(cl[k]) for k in keys}
    for i in range(B):
        ks = rng.choice(len(keys), len(keys))
        bs[i] = np.mean(np.concatenate([rng.choice(arrs[keys[j]], len(arrs[keys[j]])) for j in ks]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": round(pt, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(vals_by_pid),
            "clusters": len(keys)}


def main():
    rng = np.random.default_rng(20261107)
    wk = WorldKeys()
    res = {}
    cache = {(p, k): per_pair(p, k, wk) for p in POL for k, _ in SW}
    for metric in ["twin_fail", "world_twin_fail", "world_ok_m", "ok_m", "flag_m", "key_wrong_m"]:
        res[metric] = {}
        for p in POL + ["sft_mean"]:
            res[metric][p] = {}
            for k, sig in SW:
                if p == "sft_mean":
                    a, b = cache[("sft14", k)], cache[("sft14s1", k)]
                    v = {pid: (a[pid][metric] + b[pid][metric]) / 2 for pid in a
                         if pid in b and metric in a[pid] and metric in b[pid]}
                else:
                    v = {pid: d[metric] for pid, d in cache[(p, k)].items() if metric in d}
                res[metric][p][str(sig)] = boot(v, rng)
    # SFT minus expert, world twin fail (paired by pair id, seed-mean SFT)
    res["sft_minus_expert_world_twin_fail"] = {}
    for k, sig in SW:
        e, a, b = cache[("expert", k)], cache[("sft14", k)], cache[("sft14s1", k)]
        v = {pid: (a[pid]["world_twin_fail"] + b[pid]["world_twin_fail"]) / 2 - e[pid]["world_twin_fail"]
             for pid in e if pid in a and pid in b and all("world_twin_fail" in x[pid] for x in (e, a, b))}
        res["sft_minus_expert_world_twin_fail"][str(sig)] = boot(v, rng)
    # per family world twin fail, seed-mean SFT and expert
    fams = defaultdict(dict)
    for p in ["expert", "sft14", "sft14s1"]:
        for k, sig in SW:
            acc = defaultdict(list)
            for d in cache[(p, k)].values():
                if "world_twin_fail" in d:
                    acc[d["fam"]].append(d["world_twin_fail"])
            for f, v in acc.items():
                fams[f].setdefault(p, {})[str(sig)] = round(float(np.mean(v)), 3)
    res["per_family_world_twin_fail"] = fams
    out = REPO / "runs/s81/sweep_results.json"
    out.write_text(json.dumps(res, indent=1))
    for metric in ["world_twin_fail", "twin_fail", "world_ok_m", "key_wrong_m", "flag_m"]:
        print(f"\n{metric}")
        print("policy     " + "  ".join(f"{s:>16}" for _, s in SW))
        for p in POL + ["sft_mean"]:
            print(f"{p:10s} " + "  ".join(
                f"{r['mean']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]" for r in
                (res[metric][p][str(s)] for _, s in SW)))
    print("\nSFT-expert world_twin_fail:", {s: (r["mean"], r["ci95"]) for s, r in res["sft_minus_expert_world_twin_fail"].items()})


if __name__ == "__main__":
    main()
