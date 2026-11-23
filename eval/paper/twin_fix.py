"""T-FIX analysis, exactly as pre-registered in docs/paper/TWIN_FIX_PREREG.md (sections 4-5).
    python -m eval.paper.twin_fix [--out runs/paper/results/TFIX]
Twin failure per pair_id: nominal score >= 0.8 and mismatch score < 0.8. Excess E(arm, M) = mean over pairs of
[fail(M) - fail(M0)]. F1: R = 1 - E(arm, M2) / E(T0, M2) >= 0.5 and the CI of E(T0, M2) - E(arm, M2) excludes 0
(paired by pair_id, cluster bootstrap over seeds, 10,000). F2: v6 score(arm) - v6(T0) lower CI >= -0.02 and
fail(M0) <= 0.03. F3: F1 rule on M1. F4: CI of the M3/M4 excess reduction excludes 0. F5: share of track/DF episodes
calling sensor_consistency >= 0.8. Holm over the F1 p-values of the arms run. Convergence: 5 seeds (51-55) vs 10.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
B = 10000
THR = 0.8
C = "with_consistency"
ARMS = {  # arm -> (run-dir pattern by set, nominal run, mismatch set suffix)
    "T0": {"pat": "P5/sft14_{s}_" + C + "_tl", "alt": "TFIX/sft14_{s}_" + C + "_tl", "v6": "P1/sft14_v6", "k": ""},
    "T1": {"pat": "TFIX/sft14_{s}_" + C + "+fix_line", "v6": "TFIX/sft14_v6_" + C + "+fix_line", "k": ""},
    "T2": {"pat": "TFIX/sft14_{s}_" + C + "+fix_line+health_gate", "v6": "TFIX/sft14_v6_" + C + "+fix_line+health_gate",
           "k": ""},
    "T3": {"pat": "TFIX/sft14_{s}_" + C + "+fix_line+health_gate", "v6": "TFIX/sft14_v6_" + C + "+fix_line+health_gate",
           "k": "k", "nominal_from": "T2"},
    "T3x": {"pat": "TFIX/sft14_{s}_" + C + "_tl", "v6": "P1/sft14_v6", "k": "k", "nominal_from": "T0",
            "exploratory": True},
    "T4a": {"pat": "TFIX/sft15a_{s}_" + C + "+fix_line", "v6": "TFIX/sft15a_v6_" + C + "+fix_line", "k": "",
            "exploratory": True},
    "T4b": {"pat": "TFIX/sft15b_{s}_" + C + "+fix_line", "v6": "TFIX/sft15b_v6_" + C + "+fix_line", "k": "",
            "exploratory": True},
}
MS = ("M0", "M1", "M2", "M3", "M4")


def _rows(p: Path) -> dict:
    return {r["task_id"]: r for r in json.loads(p.read_text())["rows"] if int(r.get("sample") or 0) == 0} \
        if p.exists() else {}


def _tasks(s: str) -> dict:
    p = Path(f"eval/frozen/{s}/tasks.jsonl")
    return {t["task_id"]: t for t in (json.loads(x) for x in p.read_text().splitlines() if x.strip())} if p.exists() else {}


def _seed(tid: str) -> int:
    return int(re.match(r"k(\d+)", tid).group(1))


def _run(arm: str, s: str) -> dict:
    a = ARMS[arm]
    if a.get("k") and arm != "T3" and "nominal_from" in a and s == "v6_Npair_51s":
        a = ARMS[a["nominal_from"]]
    for pat in (a["pat"], a.get("alt")):
        if pat and (R / pat.format(s=s) / "eval.json").exists():
            return _rows(R / pat.format(s=s) / "eval.json")
    return {}


def _by_pair(arm: str, s: str) -> dict:
    """pair_id -> (seed, score) for an arm on a set."""
    rows, tk = _run(arm, s), _tasks(s)
    return {tk[t]["pair_id"]: (_seed(t), float(r["score"])) for t, r in rows.items() if tk.get(t, {}).get("pair_id")}


def _fails(arm: str) -> dict:
    """M -> {pair_id: (seed, fail)} for the arm (nominal from the arm itself, or from nominal_from)."""
    a = ARMS[arm]
    nom = _by_pair(a.get("nominal_from", arm), "v6_Npair_51s")
    out = {}
    for m in MS:
        mm = _by_pair(arm, f"v6_{m}c{a['k']}_51s")
        f = {p: (nom[p][0], float(nom[p][1] >= THR and mm[p][1] < THR)) for p in nom if p in mm}
        if f:
            out[m] = f
    return out


def _boot(cl: dict, seed: int = 0, seeds=None) -> tuple[float, list[float], float]:
    """mean, 95 % CI, two-sided p (H0: mean 0) with clusters = seeds."""
    ks = [k for k in cl if seeds is None or k in seeds]
    if not ks:
        return float("nan"), [float("nan")] * 2, float("nan")
    s = np.array([sum(cl[k]) for k in ks]); c = np.array([len(cl[k]) for k in ks])
    idx = np.random.default_rng(seed).integers(0, len(ks), (B, len(ks)))
    m = s[idx].sum(1) / c[idx].sum(1)
    mean = float(s.sum() / c.sum())
    p = 1.0 if np.all(m == m[0]) else float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    return round(mean, 4), [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)], max(p, 1 / B)


def _excess(f: dict, m: str) -> dict:
    cl = {}
    for p, (sd, v) in f.get(m, {}).items():
        if p in f.get("M0", {}):
            cl.setdefault(sd, []).append(v - f["M0"][p][1])
    return cl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/TFIX")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    F = {arm: _fails(arm) for arm in ARMS}
    F = {k: v for k, v in F.items() if v}
    base = F.get("T0", {})
    rep = {"prereg": "docs/paper/TWIN_FIX_PREREG.md", "threshold": THR, "arms": {}}
    p_f1 = {}
    for arm, f in F.items():
        e = {}
        for m in MS:
            if m in f:
                cl = {}
                for p, (sd, v) in f[m].items():
                    cl.setdefault(sd, []).append(v)
                e[f"twin_fail_{m}"] = _boot(cl)[:2]
                if m != "M0":
                    e[f"excess_{m}"] = _boot(_excess(f, m))[:2]
        r = {"pairs": {m: len(f[m]) for m in f}, **e}
        # raw accuracies (twin failure is conditional on a correct nominal answer: if the nominal accuracy drops,
        # twin failure falls mechanically, so the mean scores are reported next to it)
        nom = _run(ARMS[arm].get("nominal_from", arm), "v6_Npair_51s")
        r["mean_score"] = {"nominal": round(float(np.mean([float(x["score"]) for x in nom.values()])), 4) if nom else None}
        for m in MS:
            rows = _run(arm, f"v6_{m}c{ARMS[arm]['k']}_51s")
            if rows:
                r["mean_score"][m] = round(float(np.mean([float(x["score"]) for x in rows.values()])), 4)
                r.setdefault("abstain_rate", {})[m] = round(float(np.mean([bool(x.get("abstained")) for x in rows.values()])), 4)
        if arm != "T0" and base:
            for m in ("M1", "M2", "M3", "M4"):
                ea, eb = _excess(f, m), _excess(base, m)
                cl = {}
                for sd in set(ea) & set(eb):
                    pa = {p: f[m][p][1] - f["M0"][p][1] for p in f.get(m, {}) if p in f.get("M0", {})}
                    pb = {p: base[m][p][1] - base["M0"][p][1] for p in base.get(m, {}) if p in base.get("M0", {})}
                    cl[sd] = [pb[p] - pa[p] for p in pa if p in pb and f[m][p][0] == sd]
                if not any(cl.values()):
                    continue
                mean, ci, p = _boot(cl, seed=1)
                e0 = _boot(eb)[0]
                red = {"reduction": mean, "ci95": ci, "p": p,
                       "R": round(1 - (e0 - mean) / e0, 4) if e0 and e0 == e0 else None,
                       "convergence_5_seeds": _boot(cl, seed=1, seeds=range(51, 56))[:2]}
                r[f"reduction_{m}"] = red
                if m == "M2" and not ARMS[arm].get("exploratory"):     # Holm family = pre-registered arms only
                    p_f1[arm] = p
            v6a, v6b = _rows(R / ARMS[arm]["v6"] / "eval.json"), _rows(R / ARMS["T0"]["v6"] / "eval.json")
            ids = set(v6a) & set(v6b)
            if arm == "T3":
                ids = {t for t in ids if _seed(t) != 52}      # deviation: nominal s52 flagged (TWIN_FIX.md)
            if ids:
                cl = {}
                for t in ids:
                    cl.setdefault(_seed(t), []).append(float(v6a[t]["score"]) - float(v6b[t]["score"]))
                r["v6_delta"] = _boot(cl, seed=2)[:2]
                r["v6_n"] = len(ids)
            # F5: share of track/DF episodes calling the tool (nominal + M2 runs)
            eps = list(_run(arm, "v6_Npair_51s").values()) + list(_run(arm, f"v6_M2c{ARMS[arm]['k']}_51s").values())
            eps = [x for x in eps if "tool_names" in x and x.get("area") in ("operator_query", "df_ambiguity")]
            if eps:
                r["F5_share_calling"] = round(sum("sensor_consistency" in x["tool_names"] for x in eps) / len(eps), 4)
            red2 = r.get("reduction_M2", {})
            r["F1"] = bool(red2 and red2["R"] is not None and red2["R"] >= 0.5 and red2["ci95"][0] > 0)
            r["F2"] = bool(r.get("v6_delta") and r["v6_delta"][1][0] >= -0.02 and
                           e.get("twin_fail_M0", [1])[0] <= 0.03)
            red1 = r.get("reduction_M1", {})
            r["F3"] = bool(red1 and red1["R"] is not None and red1["R"] >= 0.5 and red1["ci95"][0] > 0)
            r["F4"] = {m: bool(r.get(f"reduction_{m}", {}).get("ci95", [0])[0] > 0) for m in ("M3", "M4")
                       if f"reduction_{m}" in r}
            if "F5_share_calling" in r:
                r["F5"] = r["F5_share_calling"] >= 0.8
        rep["arms"][arm] = r
    order = sorted(p_f1, key=p_f1.get); prev = 0.0; holm = {}
    for j, k in enumerate(order):
        prev = max(prev, min(1.0, (len(order) - j) * p_f1[k])); holm[k] = round(prev, 5)
    rep["holm_F1"] = holm
    (out / "twin_fix.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
