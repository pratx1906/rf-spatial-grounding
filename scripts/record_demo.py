"""Record the demo questions with the trained adapter and with the untrained base model, for the console's
"Compare with the untrained model" view and the recorded-replay fallback.

    python scripts/record_demo.py [--who sft,scenescout]   (vLLM must serve both: python -m serve.llm --lora sft=...)
    -> data/demo_replays/<task_id>__<who>.json  {"events": [...], "summary": {...}}
    python scripts/record_demo.py --obs v2 --label "..."   (v2 demo set on kandla_s0_X, scripts/demo_set.py --obs v2)
    -> data/demo_replays_v2/<task_id>__<who>.json   (the gateway defaults to the v2 set once these exist)

Events are exactly what the gateway streams (no transcript). The summary keeps the measured wall time, end reason
and the eval score of the run (offline, from the answer key). Recordings are labelled as such in the UI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from serve.agent import run_task  # noqa: E402
from tasks.base import Task  # noqa: E402

SETS = {"v1": (REPO / "data" / "demo_tasks_kandla.json", REPO / "data" / "demo_replays"),
        "v2": (REPO / "data" / "demo_tasks_kandla_v2.json", REPO / "data" / "demo_replays_v2"),
        "v2b": (REPO / "data" / "demo_tasks_kandla_v2b.json", REPO / "data" / "demo_replays_v2b")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", default="sft,scenescout")
    ap.add_argument("--label", default="", help="model label stored with the recording, e.g. 'Qwen3-4B + SceneScout SFT v1.1'")
    ap.add_argument("--obs", choices=sorted(SETS), default="v1", help="demo set (v2: RF phase 2, kandla_s0_X)")
    ap.add_argument("--only", default="", help="comma-separated task ids to (re)record")
    a = ap.parse_args()
    src, OUT = SETS[a.obs]
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [Task.from_row(r) for r in json.loads(src.read_text())]
    if a.only:
        tasks = [t for t in tasks if t.task_id in a.only.split(",")]
    for who in a.who.split(","):
        for t in tasks:
            env = AkshaEnv(REPO / t.site, model_id=who, trace_dir=REPO / "runs" / "ui_traces")
            t0 = time.perf_counter()
            events = [e for e in run_task(env, t, model=who) if e.get("type") != "transcript"]
            wall = round(time.perf_counter() - t0, 1)
            summary = {"task_id": t.task_id, "who": who, "template": t.template, "wall_s": wall,
                       "end_reason": getattr(env, "end_reason", None), "turns": env.turn,
                       "score": round(float(getattr(env, "answer_score", 0.0) or 0.0), 3) if env.done else 0.0,
                       "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "demo_set": a.obs,
                       "label": a.label if who != "scenescout" else "4B base, untrained"}
            (OUT / f"{t.task_id}__{who}.json").write_text(json.dumps({"events": events, "summary": summary},
                                                                     default=str))
            print(f"{who:11s} {t.template:16s} score {summary['score']:.2f} {summary['end_reason']} {wall:5.1f} s",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
