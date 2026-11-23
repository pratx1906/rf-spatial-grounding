"""Exploratory (not pre-registered; replaces the planned tool-necessity probe set): does the model reach for the tool
the expert used on the same question? Needs runs with the per-call `tool_names` field (2026-11-01 onward).
    python -m eval.paper.tool_recall --expert runs/paper/results/TFIX/expert_v6_tl/eval.json \
        --model name=path ... [--out runs/paper/results/TFIX/tool_recall.json]
Per tool: episodes where the expert called it, recall = share of those where the model also called it, and the
model's own call share. Per episode: Jaccard of the two tool sets. Control tools (submit, declare_intent, ask_user)
are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTROL = {"submit", "declare_intent", "ask_user"}


def _rows(p: str) -> dict:
    return {r["task_id"]: r for r in json.loads(Path(p).read_text())["rows"] if "tool_names" in r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--model", nargs="+", required=True, help="name=path/to/eval.json")
    ap.add_argument("--out", default="runs/paper/results/TFIX/tool_recall.json")
    a = ap.parse_args()
    ex = _rows(a.expert)
    rep = {"expert": a.expert, "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            continue
        mo = _rows(path)
        ids = sorted(set(ex) & set(mo))
        if not ids:
            continue
        per, jac = {}, []
        for t in ids:
            e = set(ex[t]["tool_names"]) - CONTROL
            m = set(mo[t]["tool_names"]) - CONTROL
            jac.append(len(e & m) / len(e | m) if e | m else 1.0)
            for tool in e | m:
                d = per.setdefault(tool, {"expert_eps": 0, "model_eps": 0, "both": 0})
                d["expert_eps"] += tool in e
                d["model_eps"] += tool in m
                d["both"] += tool in e and tool in m
        for d in per.values():
            d["recall"] = round(d["both"] / d["expert_eps"], 3) if d["expert_eps"] else None
        rep["models"][name] = {"episodes": len(ids), "mean_jaccard": round(sum(jac) / len(jac), 4),
                               "distinct_tools_model": sum(d["model_eps"] > 0 for d in per.values()),
                               "distinct_tools_expert": sum(d["expert_eps"] > 0 for d in per.values()),
                               "per_tool": dict(sorted(per.items(), key=lambda kv: -kv[1]["expert_eps"]))}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: {x: v[x] for x in ("episodes", "mean_jaccard", "distinct_tools_model", "distinct_tools_expert")}
                      for k, v in rep["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
