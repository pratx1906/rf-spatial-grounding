"""C-PAIR analysis exactly as pre-registered (docs/paper/CONTRAST_PREREG.md section 4).

Per task: score / wrong-abstain averaged over the two training seeds; paired by task_id; two-level cluster bootstrap
(scenario seed, then tasks), 10,000 resamples; Holm over the two P1 tests. Output: runs/s80/prereg/cpair_results.json
    python runs/s80/cpair_analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "runs/eval/cpair"
B = 10_000
OTHER = ["locate_ghost", "weak_low_alt", "loiter_start"]


def rows(arm, s, set_):
    # cpair.sh hung after the seed-0 F group (17:17); only those three files come from runs/eval/cpair, everything
    # else from cpair_finish.sh (runs/eval/cpair_b)
    name = f"{arm}_s{s}_{set_}.json"
    f = EV / name if (s == 0 and set_ == "F") else REPO / "runs/eval/cpair_b" / name
    return {r["task_id"]: r for r in json.load(open(f))["rows"]}


def seed_of(tid):
    return int(re.match(r"k(\d+)", tid).group(1))


def per_task(arm, set_, fam=None, metric="score", seeds=(0, 1)):
    out = {}
    rs = [rows(arm, s, set_) for s in seeds]
    for tid in rs[0]:
        if fam and rs[0][tid]["template"] != fam:
            continue
        if not all(tid in r for r in rs):
            continue
        v = []
        for r in rs:
            x = r[tid]
            v.append(float(x["score"]) if metric == "score" else float(bool(x["abstained"]) and float(x["score"]) < 0.5))
        out[tid] = (np.mean(v), rs[0][tid]["template"])
    return out


def boot(diff: dict, macro_fams=None, rng=None):
    """diff: tid -> (value, family). Returns point, CI95, two-sided bootstrap p. macro over families if given."""
    tids = sorted(diff)
    cl = {}
    for t in tids:
        cl.setdefault(seed_of(t), []).append(t)
    keys = sorted(cl)

    def stat(sample):
        if macro_fams:
            return np.mean([np.mean([diff[t][0] for t in sample if diff[t][1] == f]) for f in macro_fams
                            if any(diff[t][1] == f for t in sample)])
        return np.mean([diff[t][0] for t in sample])

    point = stat(tids)
    bs = np.empty(B)
    for i in range(B):
        ks = rng.choice(keys, len(keys))
        sample = [t for k in ks for t in rng.choice(cl[k], len(cl[k]))]
        bs[i] = stat(sample)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"diff": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "p_boot": round(float(p), 4),
            "n": len(tids), "clusters": len(keys)}


def sub(a, b):
    return {t: (b[t][0] - a[t][0], a[t][1]) for t in a if t in b}


def main():
    rng = np.random.default_rng(20261106)
    res = {}
    wa = {arm: per_task(arm, "F", "near_object", "wrong_abstain") for arm in "ABC"}
    sc = {arm: per_task(arm, "F", "near_object") for arm in "ABC"}
    res["F_near_object_wrong_abstain_rate"] = {a: round(float(np.mean([v for v, _ in wa[a].values()])), 4) for a in "ABC"}
    res["F_near_object_score"] = {a: round(float(np.mean([v for v, _ in sc[a].values()])), 4) for a in "ABC"}
    res["P1_B_minus_A"] = boot(sub(wa["A"], wa["B"]), rng=rng)
    res["P1_B_minus_C"] = boot(sub(wa["C"], wa["B"]), rng=rng)
    ps = sorted([("P1_B_minus_A", res["P1_B_minus_A"]["p_boot"]), ("P1_B_minus_C", res["P1_B_minus_C"]["p_boot"])],
                key=lambda x: x[1])
    holm, stop = {}, False
    for i, (k, p) in enumerate(ps):
        ok = (not stop) and p <= 0.05 / (len(ps) - i)
        stop = stop or not ok
        holm[k] = ok
    per_seed = {}
    for s in (0, 1):
        w = {a: per_task(a, "F", "near_object", "wrong_abstain", seeds=(s,)) for a in "ABC"}
        m = {a: float(np.mean([v for v, _ in w[a].values()])) for a in "ABC"}
        per_seed[f"s{s}"] = {**{a: round(m[a], 4) for a in "ABC"}, "B<A": m["B"] < m["A"], "B<C": m["B"] < m["C"]}
    res["P1_per_seed"] = per_seed
    res["P1_pass"] = bool(all(holm.values()) and res["P1_B_minus_A"]["diff"] < 0 and res["P1_B_minus_C"]["diff"] < 0
                          and all(v["B<A"] and v["B<C"] for v in per_seed.values()))
    res["P1_holm"] = holm
    res["P2_B_minus_A_score"] = boot(sub(sc["A"], sc["B"]), rng=rng)
    res["P2_pass"] = bool(res["P2_B_minus_A_score"]["ci95"][0] > 0)
    oa, ob = per_task("A", "F"), per_task("B", "F")
    oa = {t: v for t, v in oa.items() if v[1] in OTHER}
    res["P3_other_macro_B_minus_A"] = boot(sub(oa, ob), macro_fams=OTHER, rng=rng)
    res["P3_pass"] = bool(res["P3_other_macro_B_minus_A"]["ci95"][0] >= -0.03)
    va, vb, vc = per_task("A", "v6"), per_task("B", "v6"), per_task("C", "v6")
    res["v6_mean"] = {a: round(float(np.mean([v for v, _ in d.values()])), 4) for a, d in zip("ABC", (va, vb, vc))}
    res["P4_v6_B_minus_A"] = boot(sub(va, vb), rng=rng)
    res["P4_no_harm"] = bool(res["P4_v6_B_minus_A"]["ci95"][0] >= -0.03)
    res["P5_C_minus_A_wrong_abstain"] = boot(sub(wa["A"], wa["C"]), rng=rng)
    res["P5_C_minus_A_score"] = boot(sub(sc["A"], sc["C"]), rng=rng)
    res["v6_C_minus_A"] = boot(sub(va, vc), rng=rng)          # not pre-registered
    out = REPO / "runs/s80/prereg/cpair_results.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
