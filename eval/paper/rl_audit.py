"""P9 (c, e): tool-use behaviour before vs after RL and the reward-hacking audit, from the GRPO trajectory logs.
    python -m eval.paper.rl_audit --arms B3_s0 B3_s1 A3_s0 --out runs/paper/results/P9full
Reads runs/paper/results/P4/<arm>/traj/{val_iter_*,iter_*}.jsonl.gz (written by the RL loop; keys and counters only).
Val files are greedy runs on the same val400 tasks, so iteration 1 vs later is a paired before/after comparison.
Outputs: rl_behaviour.csv (per arm x val iteration), rl_tools.csv (tool calls per episode), rl_bigrams.csv,
rl_train_counters.csv (per arm x training iteration: reward parts and hacking counters), rl_audit.json (paired deltas).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

P4 = Path("runs/paper/results/P4")
FLAGS = ["submit_first", "zero_evidence_submit", "unsupported_refs", "conf_extreme", "abstain", "wrong_abstain",
         "repeat_calls", "verifier_rejects", "format_failures", "truth_guard_hits", "max_turns_hit"]


def _read(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _episode(r: dict) -> dict:
    c = r.get("counters") or {}
    a = r.get("answer") if isinstance(r.get("answer"), dict) else {}
    names = [k["name"] for k in r.get("calls") or []]
    return {"task_id": r["task_id"], "area": r.get("area"), "score": float(r.get("score") or 0),
            "reward": float(r.get("reward") or 0), "parts": r.get("reward_parts") or {},
            "tools": names, "n_calls": sum(n != "submit" for n in names),
            "distinct": len({n for n in names if n != "submit"}), "turns": int(r.get("turns") or 0),
            "conf": a.get("confidence"), "n_refs": len(a.get("evidence_refs") or []),
            **{k: float(bool(c.get(k))) for k in FLAGS}}


def _load(path: str) -> tuple[str, list[dict]]:
    return path, [_episode(r) for r in _read(Path(path))]


def _boot(d: np.ndarray, n: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B3_s0", "B3_s1", "A3_s0"])
    ap.add_argument("--out", default="runs/paper/results/P9full")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = [str(p) for arm in a.arms for p in sorted((P4 / arm / "traj").glob("*.jsonl.gz"))]
    with ProcessPoolExecutor(a.workers) as ex:
        data = dict(ex.map(_load, files))

    beh, tools, bigr, train, audit = [], [], [], [], {}
    for arm in a.arms:
        best = json.loads((P4 / arm / "best.json").read_text())["iter"]
        val = {int(Path(f).name[9:12]): eps for f, eps in data.items() if f"/{arm}/" in f.replace("\\", "/")
               and Path(f).name.startswith("val_")}
        for it, eps in sorted(val.items()):
            row = {"arm": arm, "iter": it, "best": it == best, "n": len(eps),
                   "score": np.mean([e["score"] for e in eps])}
            for k in ["n_calls", "distinct", "turns", "n_refs"] + FLAGS:
                row[k] = round(float(np.mean([e[k] for e in eps])), 4)
            cs = [e["conf"] for e in eps if isinstance(e["conf"], (int, float))]
            row["conf_mean"] = round(float(np.mean(cs)), 4) if cs else None
            beh.append(row)
            tc = Counter(t for e in eps for t in e["tools"])
            tools += [{"arm": arm, "iter": it, "tool": t, "per_episode": round(n / len(eps), 4)} for t, n in tc.items()]
            bc = Counter((x, y) for e in eps for x, y in zip(["<s>"] + e["tools"], e["tools"]))
            bigr += [{"arm": arm, "iter": it, "bigram": f"{x}>{y}", "per_episode": round(n / len(eps), 4)}
                     for (x, y), n in bc.most_common(25)]
        # paired before (iter 1) vs best, same tasks
        e1 = {e["task_id"]: e for e in val[1]}; eb = {e["task_id"]: e for e in val[best]}
        ids = sorted(set(e1) & set(eb))
        audit[arm] = {"best_iter": best, "n_paired": len(ids), "deltas": {}}
        for k in ["score", "n_calls", "distinct", "n_refs"] + FLAGS:
            d = np.array([eb[i][k] - e1[i][k] for i in ids], float)
            audit[arm]["deltas"][k] = {"before": round(float(np.mean([e1[i][k] for i in ids])), 4),
                                      "after": round(float(np.mean([eb[i][k] for i in ids])), 4),
                                      "delta": round(float(d.mean()), 4), "ci95": _boot(d)}
        # training iterations: reward vs answer part (a reward that rises while the answer part does not = hacking)
        tr = {int(Path(f).name[5:8]): eps for f, eps in data.items() if f"/{arm}/" in f.replace("\\", "/")
              and Path(f).name.startswith("iter_")}
        for it, eps in sorted(tr.items()):
            row = {"arm": arm, "iter": it, "n": len(eps), "reward": round(float(np.mean([e["reward"] for e in eps])), 4),
                   "score": round(float(np.mean([e["score"] for e in eps])), 4)}
            for p in ["answer", "dist", "ground", "valid", "step", "calib", "twin", "missed", "false_alarm",
                      "wrong_abstain", "pen"]:
                row[f"r_{p}"] = round(float(np.mean([float(e["parts"].get(p) or 0) for e in eps])), 4)
            for k in FLAGS:
                row[k] = round(float(np.mean([e[k] for e in eps])), 4)
            train.append(row)
        rs = np.array([r["reward"] for r in train if r["arm"] == arm])
        ss = np.array([r["score"] for r in train if r["arm"] == arm])
        nonans = rs - np.array([r["r_answer"] for r in train if r["arm"] == arm])
        audit[arm]["train"] = {"corr_reward_score": round(float(np.corrcoef(rs, ss)[0, 1]), 4),
                               "nonanswer_reward_first_last": [round(float(nonans[:3].mean()), 4),
                                                               round(float(nonans[-3:].mean()), 4)],
                               "truth_guard_hits_total": int(sum(r["truth_guard_hits"] * r["n"] for r in train
                                                                 if r["arm"] == arm))}

    import csv
    for name, rows in [("rl_behaviour", beh), ("rl_tools", tools), ("rl_bigrams", bigr), ("rl_train_counters", train)]:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "rl_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: {"best": v["best_iter"], "score": v["deltas"]["score"], "train": v["train"]}
                      for k, v in audit.items()}, indent=1))


if __name__ == "__main__":
    main()
