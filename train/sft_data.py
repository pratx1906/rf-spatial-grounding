"""SFT data (SPEC 9.3): scripted-expert trajectories, rejection-sampled, recorded exactly as AkshaEnv renders them.

    python -m train.sft_data --split train --out data/sft/train.jsonl [--per-area 600]

Each record: {"task_id", "area", "template", "score", "messages"} with OpenAI-style tool calls (arguments as dicts,
which the Qwen chat template serialises). Kept only if the episode was submitted, the verifier accepted it and the
answer scored >= MIN_SCORE. `tokenize_with_mask` renders with the SAME chat template + tool schemas used for
serving (harness.prompt.static_prefix) and masks everything except assistant turns.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from tasks.base import Task  # noqa: E402

MIN_SCORE = 0.9


def record(env: AkshaEnv, task: Task) -> dict:
    """Run the expert on one task and return the conversation as the model would have seen and produced it."""
    from tasks.expert_kandla import solve

    obs = env.reset(task)
    msgs = [dict(m) for m in obs["messages"]]
    orig = env.step
    n = [0]

    def step(action):
        calls = action.get("tool_calls") or []
        tcs = []
        for c in calls:
            n[0] += 1
            args = c.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            tcs.append({"id": f"call_{n[0]}", "type": "function", "function": {"name": c["name"], "arguments": args}})
        msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
        out = orig(action)
        tool_msgs = [m for m in out[0] if m["role"] == "tool"]
        for tc, m in zip(tcs, tool_msgs, strict=False):
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": m["content"]})
        for m in out[0]:
            if m["role"] == "user":
                msgs.append(dict(m))
        return out

    env.step = step
    try:
        solve(env, task)
    finally:
        env.step = orig
    while msgs and msgs[-1]["role"] == "tool":           # the submit acknowledgement is not part of the policy
        msgs.pop()
    ok = env.done and getattr(env, "end_reason", "") == "submitted" and bool(env.verifier_result
                                                                              and env.verifier_result.get("ok"))
    return {"task_id": task.task_id, "area": task.area, "template": task.template,
            "score": float(env.answer_score) if ok else 0.0, "messages": msgs}


def tokenize_with_mask(tok, messages: list[dict], tools: list[dict], max_len: int = 12288):
    """input_ids and labels (-100 outside assistant turns), using incremental chat-template renders."""
    ids: list[int] = []
    labels: list[int] = []
    prev = ""
    for i in range(len(messages)):
        if messages[i]["role"] != "assistant":
            continue
        before = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        after = tok.apply_chat_template(messages[: i + 1], tools=tools, tokenize=False)
        assert before.startswith(prev) and after.startswith(before), "chat template is not prefix-stable"
        ctx = tok(before[len(prev):], add_special_tokens=False)["input_ids"]
        gen = tok(after[len(before):], add_special_tokens=False)["input_ids"]
        ids += ctx + gen
        labels += [-100] * len(ctx) + gen
        prev = after
    return ids[:max_len], labels[:max_len]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-area", type=int, default=0, help="cap per area (balance), 0 = all")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (REPO / "data" / "tasks" / f"{a.split}.jsonl").read_text().splitlines()]
    rng = random.Random(0)
    rng.shuffle(rows)
    if a.per_area:
        by = defaultdict(list)
        for r in rows:
            by[r["area"]].append(r)
        rows = [r for v in by.values() for r in v[: a.per_area]]
    out = Path(a.out or REPO / "data" / "sft" / f"{a.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    envs: dict[str, AkshaEnv] = {}
    kept = defaultdict(int)
    seen = defaultdict(int)
    with open(out, "w") as f:
        for r in rows:
            t = Task.from_row(r)
            env = envs.setdefault(t.site, AkshaEnv(REPO / t.site))
            try:
                rec = record(env, t)
            except Exception as e:  # noqa: BLE001
                print("skip", t.task_id, type(e).__name__, e)
                continue
            seen[t.area] += 1
            if rec["score"] >= MIN_SCORE:
                kept[t.area] += 1
                f.write(json.dumps(rec) + "\n")
    print(json.dumps({"kept": dict(kept), "seen": dict(seen), "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
