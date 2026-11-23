"""Confidence calibration and cost per policy (C1 + C2 of the overnight plan, 2026-11-01). CPU, keys only.
    python -m eval.paper.calib_latency [--out runs/paper/results/P9full/calib_latency.json]
Correct = score >= 0.8. Confidence = the submitted answer's stated confidence (no answer: 0). ECE over 10 equal-width
bins; Brier; AURC = area under the risk-coverage curve when answers are ranked by confidence (lower is better);
selective accuracy at 80 % coverage. Cost: latency p50/p95 (s), turns, tool calls and generated tokens per episode.
CIs: cluster bootstrap over scenario seeds (2,000) for ECE, Brier and AURC.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
RUNS = {"base": "P1/base_{s}", "sft11": "P1/sft11_{s}", "sft14": "P1/sft14_{s}", "sft14_s1": "P5/sft14s1_{s}",
        "rl_b3": "P5/rl_b3_{s}", "rl_b3s1": "P5/rl_b3s1_{s}", "rl_a3": "P5/rl_a3_{s}", "qwen3-8b": "P2/qwen3-8b/{s}_dir",
        "expert": "P1/expert_{s}"}
SETS = ("v6", "heldout_v1")


def _rows(p: Path) -> list[dict]:
    return [r for r in json.loads(p.read_text())["rows"] if int(r.get("sample") or 0) == 0] if p.exists() else []


def _metrics(c: np.ndarray, y: np.ndarray) -> dict:
    bins = np.minimum((c * 10).astype(int), 9)
    ece = sum(abs(y[bins == b].mean() - c[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    order = np.argsort(-c, kind="stable")
    risk = np.cumsum(1 - y[order]) / np.arange(1, len(y) + 1)
    k80 = max(1, int(round(0.8 * len(y))))
    return {"ece": float(ece), "brier": float(np.mean((c - y) ** 2)), "aurc": float(risk.mean()),
            "sel_acc_80": float(1 - risk[k80 - 1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/P9full/calib_latency.json")
    a = ap.parse_args()
    rep = {}
    rng = np.random.default_rng(0)
    for m, pat in RUNS.items():
        for s in SETS:
            f = R / pat.format(s=s) / "eval.json"
            if m == "qwen3-8b":
                f = R / f"P2/qwen3-8b/{s}.json"
            rows = _rows(f)
            if not rows:
                continue
            c = np.array([float((r.get("answer") or {}).get("confidence") or r.get("confidence") or 0.0) for r in rows])
            y = np.array([float(float(r["score"]) >= 0.8) for r in rows])
            seed = np.array([int(re.match(r"k(\d+)", r["task_id"]).group(1)) for r in rows])
            base = _metrics(c, y)
            ks = np.unique(seed)
            boots = []
            for _ in range(2000):
                pick = rng.choice(ks, len(ks))
                idx = np.concatenate([np.where(seed == k)[0] for k in pick])
                boots.append(_metrics(c[idx], y[idx]))
            ci = {k: [round(float(np.percentile([b[k] for b in boots], q)), 4) for q in (2.5, 97.5)] for k in base}
            lat = np.array([float(r.get("latency_s") or 0) for r in rows])
            rep[f"{m}/{s}"] = {"n": len(rows), "accuracy": round(float(y.mean()), 4),
                               **{k: round(v, 4) for k, v in base.items()}, "ci95": ci,
                               "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
                               "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
                               "turns_mean": round(float(np.mean([r.get("turns") or 0 for r in rows])), 2),
                               "tool_calls_mean": round(float(np.mean([r.get("tool_calls") or 0 for r in rows])), 2),
                               "gen_tokens_mean": round(float(np.mean([r.get("gen_tokens") or 0 for r in rows])), 1)}
            print(m, s, rep[f"{m}/{s}"]["ece"], rep[f"{m}/{s}"]["latency_p50_s"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
