"""E4 final one-session table + GRPO14 pre-registered tests (docs/paper/GRPO14_PREREG.md). Output runs/s82/e4_results.json.
    python runs/s82/e4_analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "runs/eval/s82"
B = 10_000
MODELS = ["v11", "s0", "s1", "s2", "soup", "rlb3", "grpo14"]
SETS = ["F", "heldout", "v6", "W4", "W5"]
F_FAMS = ["near_object", "locate_ghost", "weak_low_alt", "loiter_start"]


def rows(m, s):
    f = EV / f"e4_{m}_{s}.json"
    return {r["task_id"]: r for r in json.load(open(f))["rows"]} if f.exists() else None


def seed_of(tid):
    return int(re.match(r"k(\d+)", tid).group(1))


def v(r, metric):
    return float(r["score"]) if metric == "score" else float(bool(r["abstained"]) and float(r["score"]) < 0.5)


def boot(a, b, metric="score", fams=None, rng=None):
    tids = [t for t in a if t in b]
    d = {t: (v(b[t], metric) - v(a[t], metric), a[t]["template"]) for t in tids}
    cl = {}
    for t in tids:
        cl.setdefault(seed_of(t), []).append(t)
    ks = sorted(cl)

    def stat(s):
        if fams:
            return np.mean([np.mean([d[t][0] for t in s if d[t][1] == f]) for f in fams if any(d[t][1] == f for t in s)])
        return np.mean([d[t][0] for t in s])

    pt = stat(tids)
    bs = np.array([stat([t for k in rng.choice(ks, len(ks)) for t in rng.choice(cl[k], len(cl[k]))]) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": round(float(pt), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(tids), "clusters": len(ks)}


def macro(r, fams, metric="score"):
    return float(np.mean([np.mean([v(x, metric) for x in r.values() if x["template"] == f]) for f in fams
                          if any(x["template"] == f for x in r.values())]))


def main():
    rng = np.random.default_rng(20261107)
    R = {(m, s): rows(m, s) for m in MODELS for s in SETS}
    res = {"table": {}, "errors": {}}
    for m in MODELS:
        res["table"][m] = {}
        for s in SETS:
            r = R[(m, s)]
            if r is None:
                continue
            res["table"][m][s] = round(float(np.mean([x["score"] for x in r.values()])), 4)
            res["errors"][f"{m}/{s}"] = sum(1 for x in r.values() if x.get("error"))
        if R[(m, "F")]:
            res["table"][m]["F_macro"] = round(macro(R[(m, "F")], F_FAMS), 4)
            res["table"][m]["F_wrong_abstain"] = round(float(np.mean([v(x, "wa") for x in R[(m, "F")].values()])), 4)
            res["table"][m]["F_family"] = {f: round(float(np.mean([x["score"] for x in R[(m, "F")].values() if x["template"] == f])), 3) for f in F_FAMS}
        if R[(m, "heldout")]:
            hf = sorted({x["template"] for x in R[(m, "heldout")].values()})
            res["table"][m]["heldout_macro"] = round(macro(R[(m, "heldout")], hf), 4)
    if R[("grpo14", "F")]:
        a, g = R[("s0", "F")], R[("grpo14", "F")]
        res["G1_F_wrong_abstain_grpo_minus_v14"] = boot(a, g, "wa", rng=rng)
        res["G1_pass"] = bool(res["G1_F_wrong_abstain_grpo_minus_v14"]["ci95"][1] < 0)
        res["G2_F_macro_grpo_minus_v14"] = boot(a, g, "score", F_FAMS, rng)
        res["G2_pass"] = bool(res["G2_F_macro_grpo_minus_v14"]["ci95"][0] > 0)
        res["G3_v6_grpo_minus_v14"] = boot(R[("s0", "v6")], R[("grpo14", "v6")], rng=rng)
        res["G3_pass"] = bool(res["G3_v6_grpo_minus_v14"]["ci95"][0] >= -0.02)
        res["G4_v6_grpo_minus_rlb3"] = boot(R[("rlb3", "v6")], R[("grpo14", "v6")], rng=rng)
        res["G4_direction_pass"] = bool(res["G4_v6_grpo_minus_rlb3"]["diff"] > 0)
        # not pre-registered: W4, W5, old held-out, vs v1.1
        res["x_grpo_minus_v14"] = {s: boot(R[("s0", s)], R[("grpo14", s)], rng=rng) for s in ("heldout", "W4", "W5")}
        res["x_grpo_minus_v11_v6"] = boot(R[("v11", "v6")], R[("grpo14", "v6")], rng=rng)
    (REPO / "runs/s82/e4_results.json").write_text(json.dumps(res, indent=1))
    print("model    " + "  ".join(f"{s:>7}" for s in SETS) + "  F_macro  F_wrongabst  heldout_macro")
    for m in MODELS:
        t = res["table"][m]
        print(f"{m:8s} " + "  ".join(f"{t.get(s, float('nan')):7.3f}" for s in SETS)
              + f"  {t.get('F_macro', float('nan')):.3f}    {t.get('F_wrong_abstain', float('nan')):.3f}       {t.get('heldout_macro', float('nan')):.3f}")
    for k in [k for k in res if k.startswith(("G", "x_"))]:
        print(k, res[k])
    print("errors", {k: e for k, e in res["errors"].items() if e})


if __name__ == "__main__":
    main()
