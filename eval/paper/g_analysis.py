"""Analysis of the extra evals G1-G5 (2026-10-31). Cluster bootstrap over scenario seeds (10,000), keys only.
    python -m eval.paper.g_analysis [--out runs/paper/results/P5/G]
G1  twin-induced failure (ok on nominal, not ok on mismatch, score >= 0.8) on 300 pairs of seeds 51-60 for M0c
    (control), M1c, M2c; M - M0 difference; next to the 248-pair seeds 15-16 numbers.
G1b pre-registered contrast (iii) on seeds 51-60: RL B3 twin failure with minus without sensor_consistency (M2c),
    paired by pair_id; Holm over the three contrasts with (iii) replaced by the 10-seed estimate (both reported).
G2  para_v1 (rule-based rewording of 150 v6 questions): paired by paraphrase_of, reworded minus original score.
G4  RL B3 seed 1 on the hidden worlds next to seed 0 and SFT v1.1.
G5  RL B3 v6 run twice: mean of each run, questions whose score changed.
Writes g_summary.json and g_report.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
B = 10000


def _rows(p: Path) -> dict:
    return {r["task_id"]: r for r in json.loads(p.read_text())["rows"] if int(r.get("sample") or 0) == 0} if p.exists() else {}


def _tasks(s: str) -> dict:
    return {t["task_id"]: t for t in (json.loads(x) for x in Path(f"eval/frozen/{s}/tasks.jsonl").read_text().splitlines() if x.strip())}


def _seed(tid: str) -> int:
    return int(re.match(r"k(\d+)", tid).group(1))


def _cboot(vals: dict, n: int = B, seed: int = 0) -> list[float]:
    """vals: cluster -> list of per-item values; CI of the overall mean."""
    rng = np.random.default_rng(seed)
    ks = list(vals)
    s = np.array([sum(vals[k]) for k in ks]); c = np.array([len(vals[k]) for k in ks])
    idx = rng.integers(0, len(ks), (n, len(ks)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _twin(m: str, M: str, suffix: str = "", abl: str = "") -> dict:
    """pair_id -> (seed, twin_fail) for model m on Npair vs M (suffix '_51s' or '')."""
    a = f"_{abl}" if abl else ""
    n = _rows(R / f"P5/{m}_v6_Npair{suffix}{a}/eval.json"); mm = _rows(R / f"P5/{m}_v6_{M}{suffix}{a}/eval.json")
    tn, tm = _tasks(f"v6_Npair{suffix}"), _tasks(f"v6_{M}{suffix}")
    pm = {tm[t]["pair_id"]: t for t in tm if tm[t].get("pair_id") and t in mm}
    out = {}
    for t, r in n.items():
        p = tn.get(t, {}).get("pair_id")
        if p and p in pm:
            out[p] = (_seed(t), float(float(r["score"]) >= 0.8 and float(mm[pm[p]]["score"]) < 0.8))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/P5/G")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    # G1
    g1 = {}
    for m in ("sft11", "sft14", "rl_b3"):
        g1[m] = {}
        tw = {M: _twin(m, M, "_51s") for M in ("M0c", "M1c", "M2c")}
        for M, d in tw.items():
            if not d:
                continue
            cl = {}
            for s, v in d.values():
                cl.setdefault(s, []).append(v)
            g1[m][M] = {"pairs": len(d), "seeds": len(cl), "twin_fail_rate": round(float(np.mean([v for _, v in d.values()])), 4),
                        "ci95": _cboot(cl)}
        for M in ("M1c", "M2c"):
            if tw.get(M) and tw.get("M0c"):
                ids = sorted(set(tw[M]) & set(tw["M0c"]))
                cl = {}
                for p in ids:
                    cl.setdefault(tw[M][p][0], []).append(tw[M][p][1] - tw["M0c"][p][1])
                g1[m][f"{M}_minus_M0c"] = {"delta": round(float(np.mean([v for vs in cl.values() for v in vs])), 4),
                                           "ci95": _cboot(cl)}
        old = {}
        for M in ("M0c", "M1c", "M2c"):
            j = R / f"P5/pairing/{m}_Npair_vs_{M}.json"
            if j.exists():
                d = json.loads(j.read_text()); old[M] = {"pairs": d["pairs"], "twin_fail_rate": d["twin_fail_rate"]}
        g1[m]["seeds_15_16"] = old
    rep["G1_twin_failure_51_60"] = g1

    # G1b
    on, off = _twin("rl_b3", "M2c", "_51s", "with_consistency"), _twin("rl_b3", "M2c", "_51s")
    ids = sorted(set(on) & set(off))
    if ids:
        cl = {}
        for p in ids:
            cl.setdefault(on[p][0], []).append(on[p][1] - off[p][1])
        dvals = np.array([v for vs in cl.values() for v in vs])
        rng = np.random.default_rng(1); ks = list(cl)
        bs = np.array([np.mean([v for k in rng.choice(ks, len(ks)) for v in cl[k]]) for _ in range(B)])
        p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
        if dvals.std() == 0:
            p = 1.0
        rep["G1b_consistency_51_60"] = {"pairs": len(ids), "seeds": len(cl),
                                        "twin_fail_with": round(float(np.mean([on[i][1] for i in ids])), 4),
                                        "twin_fail_without": round(float(np.mean([off[i][1] for i in ids])), 4),
                                        "delta": round(float(dvals.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                                                          round(float(np.percentile(bs, 97.5)), 4)],
                                        "p": round(max(p, 1 / B), 5), "pairs_changed": int((dvals != 0).sum())}
        h = json.loads((R / "P9full/holm.json").read_text())["contrasts"]
        ps = {"i": h["i_bestRL_vs_bestSFT_v6"]["p"], "ii": h["ii_bestRL_vs_bestPrompted_v6"]["p"],
              "iii_51_60": rep["G1b_consistency_51_60"]["p"]}
        order = sorted(ps, key=ps.get); prev = 0.0; holm = {}
        for j, k in enumerate(order):
            prev = max(prev, min(1.0, (len(order) - j) * ps[k])); holm[k] = round(prev, 5)
        rep["Holm_with_iii_on_51_60"] = {"p": ps, "p_holm": holm, "reject_0.05": {k: v < 0.05 for k, v in holm.items()}}

    # G2
    para = _tasks("para_v1")
    src = {"sft11": R / "P1/sft11_v6/eval.json", "sft14": R / "P1/sft14_v6/eval.json", "rl_b3": R / "P5/rl_b3_v6/eval.json",
           "rl_b3s1": R / "P5/rl_b3s1_v6/eval.json"}
    g2 = {}
    for m, p in src.items():
        rp, ro = _rows(R / f"P5/{m}_para_v1/eval.json"), _rows(p)
        cl = {}
        for t, r in rp.items():
            o = para[t].get("paraphrase_of")
            if o in ro:
                cl.setdefault(_seed(t), []).append(float(r["score"]) - float(ro[o]["score"]))
        if cl:
            vals = [v for vs in cl.values() for v in vs]
            g2[m] = {"n": len(vals), "reworded_minus_original": round(float(np.mean(vals)), 4), "ci95": _cboot(cl),
                     "reworded_mean": round(float(np.mean([float(rp[t]["score"]) for t in rp])), 4),
                     "changed_items": int(sum(abs(v) > 1e-9 for v in vals))}
    rep["G2_paraphrase"] = g2

    # G4
    g4 = {}
    for s in ("v6_W4", "v6_W5", "v6_W4_51", "v6_W5_51"):
        row = {}
        for m in ("sft11", "sft14", "rl_b3", "rl_b3s1"):
            j = R / f"P5/{m}_{s}/summary.json"
            if j.exists():
                d = json.loads(j.read_text()); row[m] = {"score": d["score"], "ci95": d.get("ci95")}
        g4[s] = row
    rep["G4_hidden_worlds_seed_spread"] = g4

    # G5
    r1, r2 = _rows(R / "P5/rl_b3_v6/eval.json"), _rows(R / "P5/rl_b3_v6_r2/eval.json")
    ids = sorted(set(r1) & set(r2))
    if ids:
        rep["G5_rl_b3_v6_rerun"] = {"n": len(ids), "run1": round(float(np.mean([float(r1[t]["score"]) for t in ids])), 4),
                                    "run2": round(float(np.mean([float(r2[t]["score"]) for t in ids])), 4),
                                    "questions_changed": int(sum(abs(float(r1[t]["score"]) - float(r2[t]["score"])) > 1e-9 for t in ids))}
        s14 = (_rows(R / "P1/sft14_v6/eval.json"), _rows(R / "P1/sft14_v6_r2/eval.json"))
        rep["G5_rl_b3_v6_rerun"]["sft14_reference"] = {"run1": round(float(np.mean([float(r["score"]) for r in s14[0].values()])), 4),
                                                       "run2": round(float(np.mean([float(r["score"]) for r in s14[1].values()])), 4)}
    # H2/H4 (2026-11-01): per-tool logging (tool_names) on the with-consistency reruns; twin failure with vs without
    tl = {}
    for m in ("rl_b3", "sft14"):
        on2, off2 = _twin(m, "M2c", "_51s", "with_consistency_tl"), _twin(m, "M2c", "_51s")
        rows = {**_rows(R / f"P5/{m}_v6_Npair_51s_with_consistency_tl/eval.json"),
                **_rows(R / f"P5/{m}_v6_M2c_51s_with_consistency_tl/eval.json")}
        if not rows or not all("tool_names" in r for r in rows.values()):
            continue
        used = [r for r in rows.values() if "sensor_consistency" in r["tool_names"]]
        cnt: dict = {}
        for r in rows.values():
            for t in r["tool_names"]:
                cnt[t] = cnt.get(t, 0) + 1
        e = {"episodes": len(rows), "episodes_calling_sensor_consistency": len(used),
             "share_calling": round(len(used) / len(rows), 4),
             "sensor_consistency_calls": cnt.get("sensor_consistency", 0), "tool_call_counts": dict(sorted(cnt.items(), key=lambda x: -x[1]))}
        ids = sorted(set(on2) & set(off2))
        if ids:
            cl = {}
            for p_ in ids:
                cl.setdefault(on2[p_][0], []).append(on2[p_][1] - off2[p_][1])
            e["twin_fail_with"] = round(float(np.mean([on2[i][1] for i in ids])), 4)
            e["twin_fail_without"] = round(float(np.mean([off2[i][1] for i in ids])), 4)
            e["delta"] = round(float(np.mean([v for vs in cl.values() for v in vs])), 4)
            e["ci95"] = _cboot(cl)
            e["pairs"] = len(ids)
        tl[m] = e
    rep["H_consistency_tool_use_logged"] = tl
    (out / "g_summary.json").write_text(json.dumps(rep, indent=1))
    L = ["# Extra evaluations G1-G5", "", "Generated by `python -m eval.paper.g_analysis`; CIs are cluster bootstrap over scenario seeds.", "",
         "```", json.dumps(rep, indent=1), "```"]
    (out / "g_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
