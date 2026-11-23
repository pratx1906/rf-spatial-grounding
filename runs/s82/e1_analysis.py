"""E1 SOUP analysis exactly as pre-registered (docs/paper/SOUP_PREREG.md). Output runs/s82/e1_results.json.
    python runs/s82/e1_analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "runs/eval/s82"
B = 10_000
SEEDS = ["s0", "s1", "s2"]
F_FAMS = ["near_object", "locate_ghost", "weak_low_alt", "loiter_start"]


def rows(m, set_):
    return {r["task_id"]: r for r in json.load(open(EV / f"e1_{m}_{set_}.json"))["rows"]}


def seed_of(tid):
    return int(re.match(r"k(\d+)", tid).group(1))


def val(r, metric):
    return float(r["score"]) if metric == "score" else float(bool(r["abstained"]) and float(r["score"]) < 0.5)


def per_task(models, set_, metric="score"):
    rs = [rows(m, set_) for m in models]
    return {t: (float(np.mean([val(r[t], metric) for r in rs])), rs[0][t]["template"]) for t in rs[0] if all(t in r for r in rs)}


def boot(diff, fams=None, rng=None):
    cl = {}
    for t in sorted(diff):
        cl.setdefault(seed_of(t), []).append(t)
    keys = sorted(cl)

    def stat(s):
        if fams:
            return np.mean([np.mean([diff[t][0] for t in s if diff[t][1] == f]) for f in fams if any(diff[t][1] == f for t in s)])
        return np.mean([diff[t][0] for t in s])

    pt = stat(sorted(diff))
    bs = np.array([stat([t for k in rng.choice(keys, len(keys)) for t in rng.choice(cl[k], len(cl[k]))]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": round(float(pt), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(diff), "clusters": len(keys)}


def sub(a, b):
    return {t: (b[t][0] - a[t][0], a[t][1]) for t in a if t in b}


def macro(d, fams):
    return float(np.mean([np.mean([v for v, f in d.values() if f == fa]) for fa in fams if any(f == fa for _, f in d.values())]))


def main():
    rng = np.random.default_rng(20261107)
    res = {}
    soupF, meanF = per_task(["soup"], "F"), per_task(SEEDS, "F")
    res["F_macro"] = {m: round(macro(per_task([m], "F"), F_FAMS), 4) for m in SEEDS + ["soup"]}
    res["F_macro"]["seed_mean"] = round(macro(meanF, F_FAMS), 4)
    res["F_family"] = {m: {f: round(float(np.mean([v for v, ff in per_task([m], "F").values() if ff == f])), 3) for f in F_FAMS}
                       for m in SEEDS + ["soup"]}
    res["S1_F_macro_soup_minus_seedmean"] = boot(sub(meanF, soupF), F_FAMS, rng)
    res["S1_pass"] = bool(res["S1_F_macro_soup_minus_seedmean"]["ci95"][0] > 0)
    best = max(SEEDS, key=lambda m: res["F_macro"][m])
    res["S2_best_seed"] = best
    res["S2_F_macro_soup_minus_best"] = boot(sub(per_task([best], "F"), soupF), F_FAMS, rng)
    res["S2_pass"] = bool(res["S2_F_macro_soup_minus_best"]["ci95"][0] >= -0.03)
    wa_s, wa_m = per_task(["soup"], "F", "wa"), per_task(SEEDS, "F", "wa")
    res["F_wrong_abstain_rate"] = {m: round(float(np.mean([v for v, _ in per_task([m], "F", "wa").values()])), 4) for m in SEEDS + ["soup"]}
    res["S3_F_wrong_abstain_soup_minus_seedmean"] = boot(sub(wa_m, wa_s), rng=rng)
    res["S3_pass"] = bool(res["S3_F_wrong_abstain_soup_minus_seedmean"]["ci95"][1] < 0)
    v6s, v6m = per_task(["soup"], "v6"), per_task(SEEDS, "v6")
    res["v6_mean"] = {m: round(float(np.mean([v for v, _ in per_task([m], "v6").values()])), 4) for m in SEEDS + ["soup"]}
    res["S4_v6_soup_minus_seedmean"] = boot(sub(v6m, v6s), rng=rng)
    res["S4_pass"] = bool(res["S4_v6_soup_minus_seedmean"]["ci95"][0] >= -0.02)
    hs, hm = per_task(["soup"], "heldout"), per_task(SEEDS, "heldout")
    hf = sorted({f for _, f in hm.values()})
    res["heldout_macro"] = {m: round(macro(per_task([m], "heldout"), hf), 4) for m in SEEDS + ["soup"]}
    res["S5_heldout_macro_soup_minus_seedmean"] = round(macro(hs, hf) - macro(hm, hf), 4)
    res["S5_direction_pass"] = bool(res["S5_heldout_macro_soup_minus_seedmean"] > 0)
    (REPO / "runs/s82/e1_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
