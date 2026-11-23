"""Paired unseen-area experiment: the pre-registered analysis of docs/paper/UNSEEN_AREA.md (H1-H4, leakage).
    python -m eval.paper.na_paired [--out runs/paper/results/NA]
Inputs: eval/frozen/naU_v1, naF_v1 (arm U seeds 101-120, arm F 201-220; pair = seed mod 100), the eval outputs
runs/paper/results/NA/<model>_na{U,F}_v1/eval.json, physics features runs/paper/results/P9/features_na{U,F}_v1.parquet,
truth tables of the scenarios (offline, leakage section only).
Cells are (pair, family) present in both arms with all models answered, so both arms ask the same families of the
same pairs. CIs: cluster bootstrap over pairs (10,000). Writes paired.json, paired_report.md, convergence.png,
per_area.png.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
MODELS = ("sft14", "rl_b3")
MARGIN = -0.03
B = 10000


def _rows(name: str) -> dict[str, dict]:
    p = R / "NA" / name / "eval.json"
    return {r["task_id"]: r for r in json.loads(p.read_text())["rows"]} if p.exists() else {}


def _tasks(arm: str) -> dict[str, dict]:
    return {t["task_id"]: t for t in (json.loads(x) for x in
            Path(f"eval/frozen/na{arm}_v1/tasks.jsonl").read_text().splitlines() if x.strip())}


def _pair(tid: str) -> int:
    return int(re.match(r"k(\d+)", tid).group(1)) % 100


def _cells(tasks: dict, rows: dict[str, dict]) -> dict[tuple, list[str]]:
    c = defaultdict(list)
    for tid, t in tasks.items():
        if all(tid in r for r in rows.values()):
            c[(_pair(tid), t["template"])].append(tid)
    return c


def _boot(fn, pairs: list[int], rng) -> np.ndarray:
    return np.array([fn(list(rng.choice(pairs, len(pairs)))) for _ in range(B)])


def _ci(a: np.ndarray) -> list[float]:
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/NA")
    a = ap.parse_args()
    out = Path(a.out)
    T = {"U": _tasks("U"), "F": _tasks("F")}
    names = ("expert",) + MODELS
    rows = {arm: {m: _rows(f"{m}_na{arm}_v1") for m in names} for arm in "UF"}
    cells = {arm: _cells(T[arm], rows[arm]) for arm in "UF"}
    keys = sorted(set(cells["U"]) & set(cells["F"]))
    pairs = sorted({p for p, _ in keys})
    sel = {arm: [tid for k in keys for tid in cells[arm][k]] for arm in "UF"}
    rep: dict = {"pairs": len(pairs), "cells": len(keys),
                 "questions": {arm: len(sel[arm]) for arm in "UF"},
                 "families": len({f for _, f in keys}), "margin": MARGIN}

    # per-pair sums so a bootstrap resample is a cheap reindex
    def sums(arm, m, use=None):
        s = defaultdict(float); n = defaultdict(int)
        for tid in sel[arm]:
            if use is not None and tid not in use:
                continue
            p = _pair(tid); s[p] += float(rows[arm][m][tid]["score"]); n[p] += 1
        return s, n

    def mean_over(sn, ps):
        s, n = sn
        tot = sum(n[p] for p in ps)
        return sum(s[p] for p in ps) / tot if tot else float("nan")

    rng = np.random.default_rng(0)
    res = {}
    for m in MODELS:
        S = {(arm, k): sums(arm, k) for arm in "UF" for k in (m, "expert")}

        def d_fn(ps):
            rU = mean_over(S[("U", m)], ps) / mean_over(S[("U", "expert")], ps)
            rF = mean_over(S[("F", m)], ps) / mean_over(S[("F", "expert")], ps)
            return rU - rF

        def raw_fn(ps, k=m):
            return mean_over(S[("U", k)], ps) - mean_over(S[("F", k)], ps)

        bd = _boot(d_fn, pairs, rng)
        r = {"U_mean": round(mean_over(S[("U", m)], pairs), 4), "F_mean": round(mean_over(S[("F", m)], pairs), 4),
             "U_expert": round(mean_over(S[("U", "expert")], pairs), 4),
             "F_expert": round(mean_over(S[("F", "expert")], pairs), 4)}
        r["r_U"] = round(r["U_mean"] / r["U_expert"], 4); r["r_F"] = round(r["F_mean"] / r["F_expert"], 4)
        r["d"] = round(d_fn(pairs), 4); r["d_ci95"] = _ci(bd)
        r["H1_noninferior"] = bool(r["d_ci95"][0] > MARGIN)
        r["raw_delta_U_minus_F"] = round(raw_fn(pairs), 4); r["raw_ci95"] = _ci(_boot(raw_fn, pairs, rng))
        # H2 convergence: first n pairs
        conv = []
        for n in (5, 10, 15, len(pairs)):
            ps = pairs[:n]
            conv.append({"pairs": n, "d": round(d_fn(ps), 4), "ci95": _ci(_boot(d_fn, ps, rng)[:4000])})
        r["convergence"] = conv
        r["H2_stable_from_10"] = all(r["d_ci95"][0] <= c["d"] <= r["d_ci95"][1] for c in conv if c["pairs"] >= 10)
        # per area
        pa = {}
        for area in sorted({T["U"][t]["area"] for t in sel["U"]}):
            def am(arm, k):
                v = [float(rows[arm][k][t]["score"]) for t in sel[arm] if T[arm][t]["area"] == area]
                return round(float(np.mean(v)), 3) if v else None
            pa[area] = {"U": am("U", m), "F": am("F", m), "U_expert": am("U", "expert"), "F_expert": am("F", "expert")}
        r["per_area"] = pa
        res[m] = r
    exp = {"U": rep and None}
    res["expert"] = {"U_mean": round(float(np.mean([float(rows["U"]["expert"][t]["score"]) for t in sel["U"]])), 4),
                     "F_mean": round(float(np.mean([float(rows["F"]["expert"][t]["score"]) for t in sel["F"]])), 4)}
    rep["results"] = res

    # H4 tool use / harness
    def rate(arm, m, f):
        v = [f(rows[arm][m][t]) for t in sel[arm]]
        return round(float(np.mean(v)), 4)
    H4 = {}
    probes = {"tool_calls": lambda r: float(r.get("tool_calls") or 0),
              "verifier_ok": lambda r: float(str(r.get("verifier_ok")) == "True"),
              "submitted": lambda r: float(r.get("end_reason") == "submitted"),
              "format_failed": lambda r: float(r.get("end_reason") == "format_failed"),
              "max_turns": lambda r: float(r.get("end_reason") in ("max_turns", "budget", "wallclock")),
              "abstained": lambda r: float(str(r.get("abstained")) == "True"),
              "any_timeout": lambda r: float(int(r.get("n_timeouts") or 0) > 0),
              "error": lambda r: float(bool(r.get("error")) and str(r.get("error")) != "None"),
              "latency_s": lambda r: float(r.get("latency_s") or 0)}
    for m in MODELS:
        H4[m] = {}
        for k, f in probes.items():
            def dfn(ps, k=k, f=f):
                u = [f(rows["U"][m][t]) for t in sel["U"] if _pair(t) in set(ps)]
                g = [f(rows["F"][m][t]) for t in sel["F"] if _pair(t) in set(ps)]
                return float(np.mean(u) - np.mean(g))
            bt = np.array([dfn(list(rng.choice(pairs, len(pairs)))) for _ in range(1000)])
            H4[m][k] = {"U": rate("U", m, f), "F": rate("F", m, f), "delta": round(dfn(pairs), 4), "ci95": _ci(bt)}
        H4[m]["end_reasons_U"] = dict(Counter(rows["U"][m][t]["end_reason"] for t in sel["U"]))
        H4[m]["end_reasons_F"] = dict(Counter(rows["F"][m][t]["end_reason"] for t in sel["F"]))
        H4[m]["H4_same_behaviour"] = all(abs(v["delta"]) <= 0.05 for k, v in H4[m].items()
                                         if isinstance(v, dict) and "delta" in v and k not in ("tool_calls", "latency_s"))
    rep["H4"] = H4

    # H3 physics-controlled regression (tasks with physics features)
    try:
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        fe = pd.concat([pd.read_parquet(R / f"P9/features_na{arm}_v1.parquet").assign(arm_u=int(arm == "U"))
                        for arm in "UF"])
        cols = ["ghost_frac_label", "gdop_m_per_deg", "nearest_sensor_m", "emitter_z_m", "map_nlos_frac",
                "n_tracks_active"]
        H3 = {}
        for m in MODELS:
            df = fe[fe.task_id.isin(set(sel["U"]) | set(sel["F"]))].dropna(subset=cols).copy()
            df["y"] = [float(rows["U" if a_ else "F"][m][t]["score"]) >= 0.8 for t, a_ in zip(df.task_id, df.arm_u)]
            df["pair"] = [_pair(t) for t in df.task_id]
            X = pd.concat([df[["arm_u"]], (df[cols] - df[cols].mean()) / df[cols].std(),
                           pd.get_dummies(df.band.fillna("none"), prefix="b", drop_first=True, dtype=float),
                           pd.get_dummies(df.template, prefix="f", drop_first=True, dtype=float)], axis=1).astype(float)
            fit = lambda Xs, ys: LogisticRegression(C=1.0, max_iter=2000).fit(Xs, ys).coef_[0][0]
            coef = fit(X.values, df.y.values)
            bs = []
            P = df.pair.values
            for _ in range(300):
                pick = rng.choice(pairs, len(pairs))
                idx = np.concatenate([np.where(P == p)[0] for p in pick])
                if len(set(df.y.values[idx])) == 2:
                    bs.append(fit(X.values[idx], df.y.values[idx]))
            H3[m] = {"n": len(df), "arm_u_logodds": round(float(coef), 4), "ci95": _ci(np.array(bs)),
                     "odds_ratio": round(float(np.exp(coef)), 3), "controls": cols + ["band", "family"],
                     "H3_no_unseen_penalty": bool(np.percentile(bs, 2.5) > -1e9 and not np.percentile(bs, 97.5) < 0)}
        rep["H3"] = H3
    except Exception as e:                         # noqa: BLE001 - reported, never hides the other results
        rep["H3"] = {"error": repr(e)}

    # sensitivity: drop questions whose text equals a training question
    tq = {json.loads(x)["question"] for x in Path("data/tasks/x3/train.jsonl").read_text().splitlines() if x.strip()}
    keep = {t for arm in "UF" for t in sel[arm] if T[arm][t]["question"] not in tq}
    rep["sensitivity_drop_train_text"] = {"dropped": {arm: sum(T[arm][t]["question"] in tq for t in sel[arm]) for arm in "UF"}}
    for m in MODELS:
        S2 = {(arm, k): sums(arm, k, keep) for arm in "UF" for k in (m, "expert")}
        d2 = (mean_over(S2[("U", m)], pairs) / mean_over(S2[("U", "expert")], pairs)
              - mean_over(S2[("F", m)], pairs) / mean_over(S2[("F", "expert")], pairs))
        rep["sensitivity_drop_train_text"][m] = round(d2, 4)

    # robustness (added 2026-10-31 21:00 after the independent audit; not pre-registered, reported as such):
    # coverage confound (each sensor's traced area is 1.5 km: arm U is heard by fewer sensors), headline mask,
    # scorer v2, track tasks only
    import pandas as pd

    from tasks.base import Task
    from tasks.scoring import is_headline, score_versioned
    fe = pd.concat([pd.read_parquet(R / f"P9/features_na{arm}_v1.parquet") for arm in "UF"]).set_index("task_id")
    nsens = fe["n_sensors"].to_dict()
    task_obj = {arm: {t: Task.from_row(T[arm][t]) for t in sel[arm]} for arm in "UF"}
    rep["coverage"] = {arm: {"track_tasks": int(sum(t in nsens and nsens[t] == nsens[t] for t in sel[arm])),
                             "mean_n_sensors": round(float(np.nanmean([nsens.get(t, np.nan) for t in sel[arm]])), 3),
                             "share_ge2_sensors": round(float(np.nanmean([float(nsens[t] >= 2) for t in sel[arm]
                                                                          if t in nsens and nsens[t] == nsens[t]])), 3)}
                       for arm in "UF"}

    def v2(arm, m, t):
        r = rows[arm][m][t]
        a_ = r.get("answer")
        if isinstance(a_, str):
            try:
                import ast
                a_ = ast.literal_eval(a_)
            except (ValueError, SyntaxError):
                a_ = None
        try:
            return float(score_versioned(task_obj[arm][t], a_, version="v2")) if a_ is not None else float(r["score"])
        except Exception:                                          # noqa: BLE001 - keep v1 for unscorable rows
            return float(r["score"])

    variants = {
        "coverage_ge2_sensors": (lambda arm, m, t: float(rows[arm][m][t]["score"]),
                                 lambda arm, t: t in nsens and nsens[t] == nsens[t] and nsens[t] >= 2),
        "track_tasks_only": (lambda arm, m, t: float(rows[arm][m][t]["score"]),
                             lambda arm, t: t in nsens and nsens[t] == nsens[t]),
        "headline_mask": (lambda arm, m, t: float(rows[arm][m][t]["score"]), lambda arm, t: is_headline(task_obj[arm][t])),
        "scorer_v2": (v2, lambda arm, t: True),
        # audit problem 2: own-mount diffraction (S5 on building_b030) labelled as reflection in arm F only, which
        # touches the bearing-family keys; drop the df_ambiguity area
        "no_df_ambiguity": (lambda arm, m, t: float(rows[arm][m][t]["score"]),
                            lambda arm, t: T[arm][t]["area"] != "df_ambiguity"),
    }
    rob = {}
    for vn, (sc, inc) in variants.items():
        rob[vn] = {}
        for m in MODELS:
            S3 = {}
            for arm in "UF":
                for k in (m, "expert"):
                    s = defaultdict(float); n = defaultdict(int)
                    for t in sel[arm]:
                        if inc(arm, t):
                            p = _pair(t); s[p] += sc(arm, k, t); n[p] += 1
                    S3[(arm, k)] = (s, n)
            ps_ok = [p for p in pairs if S3[("U", m)][1][p] and S3[("F", m)][1][p]]

            def dv(ps, S3=S3, m=m):
                return (mean_over(S3[("U", m)], ps) / mean_over(S3[("U", "expert")], ps)
                        - mean_over(S3[("F", m)], ps) / mean_over(S3[("F", "expert")], ps))
            bt = np.array([dv(list(rng.choice(ps_ok, len(ps_ok)))) for _ in range(4000)])
            rob[vn][m] = {"pairs": len(ps_ok), "n_U": sum(S3[("U", m)][1].values()), "n_F": sum(S3[("F", m)][1].values()),
                          "U": round(mean_over(S3[("U", m)], ps_ok), 4), "F": round(mean_over(S3[("F", m)], ps_ok), 4),
                          "U_expert": round(mean_over(S3[("U", "expert")], ps_ok), 4),
                          "F_expert": round(mean_over(S3[("F", "expert")], ps_ok), 4),
                          "d": round(dv(ps_ok), 4), "ci95": _ci(bt), "noninferior": bool(np.percentile(bt, 2.5) > MARGIN)}
    rep["robustness"] = rob
    # H3 again with sensor-count controls
    try:
        from sklearn.linear_model import LogisticRegression
        fe2 = pd.concat([pd.read_parquet(R / f"P9/features_na{arm}_v1.parquet").assign(arm_u=int(arm == "U")) for arm in "UF"])
        cols = ["ghost_frac_label", "gdop_m_per_deg", "nearest_sensor_m", "emitter_z_m", "map_nlos_frac",
                "n_tracks_active", "n_sensors", "n_bearings"]
        H3b = {}
        for m in MODELS:
            df = fe2[fe2.task_id.isin(set(sel["U"]) | set(sel["F"]))].dropna(subset=cols).copy()
            df["y"] = [float(rows["U" if a_ else "F"][m][t]["score"]) >= 0.8 for t, a_ in zip(df.task_id, df.arm_u)]
            P = np.array([_pair(t) for t in df.task_id])
            X = pd.concat([df[["arm_u"]], (df[cols] - df[cols].mean()) / df[cols].std().replace(0, 1),
                           pd.get_dummies(df.band.fillna("none"), prefix="b", drop_first=True, dtype=float),
                           pd.get_dummies(df.template, prefix="f", drop_first=True, dtype=float)], axis=1).astype(float).values
            fit = lambda Xs, ys: LogisticRegression(C=1.0, max_iter=3000).fit(Xs, ys).coef_[0][0]
            bs = []
            for _ in range(300):
                idx = np.concatenate([np.where(P == p)[0] for p in rng.choice(pairs, len(pairs))])
                if len(set(df.y.values[idx])) == 2:
                    bs.append(fit(X[idx], df.y.values[idx]))
            H3b[m] = {"n": len(df), "arm_u_logodds": round(float(fit(X, df.y.values)), 4), "ci95": _ci(np.array(bs)),
                      "controls": cols + ["band", "family"]}
        rep["H3_with_sensor_controls"] = H3b
    except Exception as e:                                       # noqa: BLE001
        rep["H3_with_sensor_controls"] = {"error": repr(e)}

    (out / "paired.json").write_text(json.dumps(rep, indent=1))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, m in enumerate(MODELS):
        c = res[m]["convergence"]
        x = np.array([v["pairs"] for v in c]) + (i - 0.5) * 0.4
        y = [v["d"] for v in c]
        ax.errorbar(x, y, yerr=[[v["d"] - v["ci95"][0] for v in c], [v["ci95"][1] - v["d"] for v in c]], fmt="o-",
                    capsize=3, label={"sft14": "SFT v1.4", "rl_b3": "RL (B3)"}[m])
    ax.axhline(0, color="#888", lw=0.8); ax.axhline(MARGIN, color="#c0392b", ls="--", lw=1, label="non-inferiority margin")
    ax.set_xlabel("scenario pairs"); ax.set_ylabel("unseen minus familiar\n(expert-normalised score)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "convergence.png", dpi=150); plt.close(fig)
    areas = list(res["sft14"]["per_area"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.2
    for j, (lab, key) in enumerate([("SFT v1.4 familiar", "F"), ("SFT v1.4 unseen", "U"),
                                    ("expert familiar", "F_expert"), ("expert unseen", "U_expert")]):
        ax.bar(np.arange(len(areas)) + (j - 1.5) * w, [res["sft14"]["per_area"][a_][key] or 0 for a_ in areas], w, label=lab)
    ax.set_xticks(range(len(areas))); ax.set_xticklabels(areas, rotation=20, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean score"); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(out / "per_area.png", dpi=150); plt.close(fig)

    # report
    L = ["# Paired unseen-area experiment: results", "", f"Pairs {rep['pairs']}, cells (pair x family) {rep['cells']}, "
         f"questions U {rep['questions']['U']} / F {rep['questions']['F']}, families {rep['families']}. "
         f"Margin {MARGIN}. Generated by `python -m eval.paper.na_paired`.", "",
         "| model | familiar | unseen | expert familiar | expert unseen | r_F | r_U | d = r_U - r_F [95 % CI] | H1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        v = res[m]
        L.append(f"| {m} | {v['F_mean']:.3f} | {v['U_mean']:.3f} | {v['F_expert']:.3f} | {v['U_expert']:.3f} | "
                 f"{v['r_F']:.3f} | {v['r_U']:.3f} | {v['d']:+.3f} {v['d_ci95']} | {'PASS' if v['H1_noninferior'] else 'FAIL'} |")
    L += ["", "Convergence (H2):", ""]
    for m in MODELS:
        L.append(f"- {m}: " + "; ".join(f"{c['pairs']} pairs {c['d']:+.3f} {c['ci95']}" for c in res[m]["convergence"])
                 + f" -> stable from 10: {res[m]['H2_stable_from_10']}")
    L += ["", "Physics-controlled regression (H3):", "", "```", json.dumps(rep["H3"], indent=1), "```", "",
          "Tool use and harness (H4):", "", "```", json.dumps(H4, indent=1), "```", "",
          "Sensitivity (drop questions worded like a training question):", "",
          "```", json.dumps(rep["sensitivity_drop_train_text"], indent=1), "```"]
    (out / "paired_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
