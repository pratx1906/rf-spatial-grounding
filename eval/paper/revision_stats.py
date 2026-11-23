"""Statistics added for the 2026-11-02 revision (reviewer requests; existing evaluation rows only, CPU).
    python -m eval.paper.revision_stats [--out runs/paper/results/P9full/revision_stats.json]
1. Held-out families: per-family mean score for the expert, SFT v1.1, SFT v1.4 (two seeds), GRPO (two seeds), and the
   expert-normalised retention (model / expert) per family.
2. v6 by seed group: seeds 15-16 (original cast) vs 51-60 (varied cast) for every main policy.
3. Exact sign-flip randomisation tests over cluster-level paired differences (valid with few clusters; all 2^k sign
   patterns when k <= 20): RL vs SFT and RL vs its initialisation on v6 (12 clusters), twin excess M2/M1 vs control
   (10 clusters), unseen-area d (20 pairs, from the per-pair values), fix-study reductions (10 clusters).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")


def _rows(p: Path) -> dict:
    return {r["task_id"]: r for r in json.loads(p.read_text())["rows"] if int(r.get("sample") or 0) == 0} if p.exists() else {}


def _seed(t: str) -> int:
    return int(re.match(r"k(\d+)", t).group(1))


def signflip(diffs: list[float]) -> dict:
    """Two-sided exact sign-flip test of mean(diffs) = 0 over clusters (all 2^k patterns; Monte Carlo above 20)."""
    d = np.asarray(diffs, float)
    k = len(d)
    obs = abs(d.mean())
    if k <= 20:
        pats = np.array(list(itertools.product([1, -1], repeat=k)))
    else:
        pats = np.random.default_rng(0).choice([1, -1], size=(200000, k))
    stats = np.abs((pats * d).mean(1))
    return {"k": k, "mean": round(float(d.mean()), 4), "p": round(float((stats >= obs - 1e-12).mean()), 5),
            "min_p": round(1 / len(pats), 6)}


def cluster_means(a: dict, b: dict) -> list[float]:
    by = defaultdict(list)
    for t in set(a) & set(b):
        by[_seed(t)].append(float(a[t]["score"]) - float(b[t]["score"]))
    return [float(np.mean(v)) for _, v in sorted(by.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/P9full/revision_stats.json")
    a = ap.parse_args()
    rep: dict = {}
    # 1. held-out per family
    runs = {"expert": "P1/expert_heldout_v1", "sft11": "P1/sft11_heldout_v1", "sft14": "P1/sft14_heldout_v1",
            "sft14_s1": "P5/sft14s1_heldout_v1", "rl_b3": "P5/rl_b3_heldout_v1", "rl_b3s1": "P5/rl_b3s1_heldout_v1"}
    rows = {m: _rows(R / p / "eval.json") for m, p in runs.items()}
    fam = defaultdict(dict)
    for m, rr in rows.items():
        by = defaultdict(list)
        for r in rr.values():
            by[r["template"]].append(float(r["score"]))
        for f, v in by.items():
            fam[f][m] = round(float(np.mean(v)), 3)
            fam[f]["n"] = len(v)
    for f, d in fam.items():
        for m in ("sft11", "sft14", "sft14_s1", "rl_b3", "rl_b3s1"):
            if d.get("expert") and m in d:
                d[f"{m}_retention"] = round(d[m] / d["expert"], 3)
    rep["heldout_per_family"] = dict(sorted(fam.items()))
    # 2. v6 by seed group
    v6 = {"expert": "P1/expert_v6", "sft11": "P1/sft11_v6", "sft14": "P1/sft14_v6", "sft14_s1": "P5/sft14s1_v6",
          "rl_b3": "P5/rl_b3_v6", "rl_b3s1": "P5/rl_b3s1_v6", "base": "P1/base_v6"}
    v6r = {m: _rows(R / p / "eval.json") for m, p in v6.items()}
    grp = {}
    for m, rr in v6r.items():
        g = defaultdict(list)
        for t, r in rr.items():
            g["s15_16" if _seed(t) in (15, 16) else "s51_60"].append(float(r["score"]))
        grp[m] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 3)} for k, v in g.items()}
    rep["v6_by_seed_group"] = grp
    # 3. sign-flip tests
    sf = {}
    sf["rl_b3_minus_sft14_v6"] = signflip(cluster_means(v6r["rl_b3"], v6r["sft14"]))
    sf["rl_b3_minus_sft11_v6"] = signflip(cluster_means(v6r["rl_b3"], v6r["sft11"]))
    sf["rl_b3s1_minus_sft11_v6"] = signflip(cluster_means(v6r["rl_b3s1"], v6r["sft11"]))
    sf["sft14_minus_base_v6"] = signflip(cluster_means(v6r["sft14"], v6r["base"]))
    # twin excess (10 clusters): per-seed twin-fail(M) - twin-fail(M0) from the 51-60 pair sets
    def pairs(m: str, M: str) -> dict:
        n, mm = _rows(R / f"P5/{m}_v6_Npair_51s/eval.json"), _rows(R / f"P5/{m}_v6_{M}_51s/eval.json")
        tn = {json.loads(x)["task_id"]: json.loads(x)["pair_id"] for x in open("eval/frozen/v6_Npair_51s/tasks.jsonl")}
        tm = {json.loads(x)["pair_id"]: json.loads(x)["task_id"] for x in open(f"eval/frozen/v6_{M}_51s/tasks.jsonl")}
        out = {}
        for t, r in n.items():
            p = tn.get(t)
            if p in tm and tm[p] in mm:
                out[p] = (_seed(t), float(float(r["score"]) >= 0.8 and float(mm[tm[p]]["score"]) < 0.8))
        return out
    for m in ("sft14", "sft11", "rl_b3"):
        p0 = pairs(m, "M0c")
        for M in ("M1c", "M2c"):
            pm = pairs(m, M)
            by = defaultdict(list)
            for p in set(p0) & set(pm):
                by[pm[p][0]].append(pm[p][1] - p0[p][1])
            sf[f"twin_excess_{M}_{m}"] = signflip([float(np.mean(v)) for _, v in sorted(by.items())])
    # unseen-area d per pair (expert-normalised difference per pair)
    na = {k: _rows(R / f"NA/{k}/eval.json") for k in ("sft14_naU_v1", "sft14_naF_v1", "expert_naU_v1", "expert_naF_v1")}
    def per_pair(model: str, arm: str) -> dict:
        by = defaultdict(list)
        for t, r in na[f"{model}_na{arm}_v1"].items():
            by[_seed(t) % 100].append(float(r["score"]))
        return {k: float(np.mean(v)) for k, v in by.items()}
    mU, mF, eU, eF = per_pair("sft14", "U"), per_pair("sft14", "F"), per_pair("expert", "U"), per_pair("expert", "F")
    ds = [mU[k] / eU[k] - mF[k] / eF[k] for k in sorted(set(mU) & set(mF) & set(eU) & set(eF)) if eU[k] and eF[k]]
    sf["unseen_area_d_sft14_per_pair"] = signflip(ds)
    sf["unseen_area_d_sft14_per_pair"]["note"] = "two-sided test of d = 0; H1 is non-inferiority (CI lower bound > -0.03)"
    # one-sided non-inferiority: H0 mean(d) <= -0.03 vs H1 mean(d) > -0.03 (sign-flip on d + 0.03)
    sh = np.asarray(ds) + 0.03
    pats = np.random.default_rng(1).choice([1, -1], size=(200000, len(sh)))
    sf["unseen_area_noninferiority_sft14"] = {"k": len(sh), "mean_d": round(float(np.mean(ds)), 4), "margin": -0.03,
                                              "p_one_sided": round(float(((pats * sh).mean(1) >= sh.mean()).mean()), 5)}
    # fine-tuned parser + expert baseline (2026-11-02): agent vs parser, parse accuracy
    pft = {k: _rows(R / f"P1/parser_ft_{k}/eval.json") for k in ("v6", "heldout_v1")}
    if pft["v6"]:
        sf["sft14_minus_parser_ft_v6"] = signflip(cluster_means(v6r["sft14"], pft["v6"]))
        hel = _rows(R / "P1/sft14_heldout_v1/eval.json")
        sf["sft14_minus_parser_ft_heldout"] = signflip(cluster_means(hel, pft["heldout_v1"]))
        acc = {}
        for k, rr in pft.items():
            T = {json.loads(x)["task_id"]: json.loads(x) for x in open(f"eval/frozen/{k}/tasks.jsonl")}
            fam = sum((r.get("parsed") or {}).get("family") == T[t]["template"] for t, r in rr.items())
            full = sum((r.get("parsed") or {}).get("family") == T[t]["template"]
                       and (r.get("parsed") or {}).get("params") == T[t].get("params") for t, r in rr.items())
            acc[k] = {"n": len(rr), "family_exact": fam, "family_and_slots_exact": full,
                      "family_exact_share": round(fam / max(len(rr), 1), 3)}
        rep["parser_ft_accuracy"] = acc
    rep["signflip"] = sf
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({"v6_by_seed_group": grp, "signflip": sf}, indent=1)[:4000])


if __name__ == "__main__":
    main()
