"""SFT data for the fine-tuned parser + scripted expert baseline (independent review, 2026-11-02).
    python -m train.parser_data [--per-family 80]
The parser sees the question only and must return {"family": ..., "params": {...}} exactly as the task generator
wrote them; the scripted expert then runs on those slots (eval/parser_expert.py, --parser-examples none). Training
questions come from the train split (seeds 1-12, 17-28; held-out families are absent from it by construction), the
validation questions from the val split (seeds 13-14). Prompt = eval.parser_expert.messages_for(question, []).
Writes data/sft/parser_train.jsonl and data/sft/parser_val.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.parser_expert import _excluded, messages_for

REPO = Path(__file__).resolve().parents[1]


def build(src: Path, per_family: int, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    excl = _excluded()
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r["template"] in excl:
            continue
        by[r["template"]].append(r)
    out = []
    for fam in sorted(by):
        rows = by[fam]
        rng.shuffle(rows)
        for r in rows[:per_family]:
            target = json.dumps({"family": r["template"], "params": r.get("params") or {}})
            out.append({"task_id": r["task_id"], "area": r["area"], "template": r["template"],
                        "messages": messages_for(r["question"], []) + [{"role": "assistant", "content": target}]})
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=80)
    a = ap.parse_args()
    rng = random.Random(20261102)
    tr = build(REPO / "data/tasks/x3/train.jsonl", a.per_family, rng)
    va = build(REPO / "data/tasks/x3/val.jsonl", 4, rng)
    for name, rows in (("parser_train", tr), ("parser_val", va)):
        (REPO / f"data/sft/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"train": len(tr), "val": len(va), "families": len({r['template'] for r in tr})}))


if __name__ == "__main__":
    main()
