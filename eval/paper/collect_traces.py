"""P9: run a policy on a frozen set and keep each episode's tool sequence (normalised episode jsonl for behaviour.py).

    SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr python -m eval.paper.collect_traces --policy expert \
        --set eval/frozen/v6/tasks.jsonl --out runs/paper/results/P9/traces_expert_v6.jsonl

Uses eval.run.run_one in one thread and reads the episode's harness Tracer events from that thread's env (CPU for
expert / trivial / classical policies; `--policy model` needs the LLM server and belongs in the GPU queue).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    from eval import run as ev
    from tasks.base import Task

    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert")
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="expert")
    ap.add_argument("--served", default="scenescout")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in Path(REPO / a.set).read_text().splitlines() if x.strip()]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(REPO / a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for r in rows:
            t = Task.from_row(r)
            row = ev.run_one(t, a.policy, a.model_id, 0.0, 0, a.served)
            env = ev._LOCAL.envs[(t.site, "none")]
            evs = [e for e in env.tracer.events if "tool" in e]
            tools = []
            for e in evs:
                name = e["tool"]
                if name == "fit_candidates" and (e.get("args") or {}).get("method") == "rt_match":
                    name = "fit_candidates:rt_match"
                tools.append(name)
            f.write(json.dumps({"task_id": t.task_id, "template": t.template, "area": t.area, "tools": tools,
                                "ok_flags": [e.get("ok") for e in evs], "score": row["score"],
                                "end_reason": row["end_reason"], "turns": row["turns"], "policy": a.policy}) + "\n")
            n += 1
    print(json.dumps({"episodes": n, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
