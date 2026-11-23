"""Rehearse the demo: run every demo question through the model and score it against its answer key.

    python scripts/demo_check.py [--repeats 3]

Needs the LLM server (python -m serve.llm). Prints per-question score, turns, latency, and the trace on failure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from serve.agent import run_model  # noqa: E402
from tasks.base import Task  # noqa: E402
from tasks.scoring import score_answer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--only", default=None, help="substring filter on the question")
    a = p.parse_args()
    tasks = [Task.from_row(r) for r in json.loads((REPO / "data" / "demo_tasks.json").read_text())]
    env = AkshaEnv(REPO / "data" / "stub_site", model_id="scenescout")
    rows = []
    for t in tasks:
        if a.only and a.only not in t.question:
            continue
        for _ in range(a.repeats):
            t0 = time.time()
            trace = list(run_model(env, t.question, temperature=a.temperature, clarification=t.clarification))
            s = score_answer(t, env.answer) if env.answer is not None else 0.0
            rows.append((t.template, s))
            calls = [f"{e['name']}" + ("" if e["result"]["ok"] else f"!{e['result']['error']['code']}")
                     for e in trace if e["type"] == "tool_result"]
            print(f"[{s:.2f}] {time.time() - t0:5.1f}s turns={env.turn} end={env.end_reason}  {t.question[:70]}")
            print(f"        calls: {' > '.join(calls)}")
            if s < 1:
                calls_args = [e for e in trace if e["type"] == "tool_call"]
                results = [e for e in trace if e["type"] == "tool_result"]
                for c, r in zip(calls_args, results, strict=False):
                    if not r["result"]["ok"]:
                        err = r["result"]["error"]
                        print(f"        BAD {c['name']} {json.dumps(c['arguments'])[:220]}")
                        print(f"            -> {err['code']}: {json.dumps(err.get('fields', err.get('message')))[:220]}")
                texts = [e["text"][:160] for e in trace if e["type"] == "assistant" and e["text"]]
                if texts:
                    print(f"        text: {texts[:2]}")
                ans = env.answer.model_dump(mode='json') if env.answer is not None else None
                print(f"        answer: {json.dumps(ans)[:300]}")
                print(f"        key:    {json.dumps(t.answer_key)[:300]}")
                errs = [e for e in trace if e["type"] == "error"]
                if errs:
                    print(f"        errors: {errs[:2]}")
    by = {}
    for k, s in rows:
        by.setdefault(k, []).append(s)
    print("\nmean score by template:", {k: round(statistics.mean(v), 2) for k, v in by.items()})
    print("overall:", round(statistics.mean(s for _, s in rows), 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
