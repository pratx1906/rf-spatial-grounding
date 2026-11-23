"""SFT v2 data: the balanced subset with paraphrased questions mixed in, and a held-out-family validation file.

    python -m train.sft_augment --para data/tasks/train_para.jsonl --mix 0.6

The expert solves from the task's structured slots, not its wording, so a recorded trajectory stays correct when
the user message is replaced by a faithful paraphrase (scripts/paraphrase.py guards the facts). Each example keeps
its original question with probability 1 - mix, else takes one of its paraphrases (paraphrase groups never cross
splits: paraphrases are made from train tasks only).
Writes data/sft/train_v2.jsonl and data/sft/val_heldout.jsonl (validation episodes of held-out families + an
equal number of seen-family episodes, so the loss shows generalisation, not just fit).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    from scripts.make_tasks import HELD_OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sft/train_bal.jsonl")
    ap.add_argument("--para", default="data/tasks/train_para.jsonl")
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--noise", type=float, default=0.0, help="probability of operator-style surface noise "
                    "(scripts/augment_text.py) on top of the chosen wording")
    a = ap.parse_args()
    rng = random.Random(3)
    para = defaultdict(list)
    if (REPO / a.para).exists():
        for line in (REPO / a.para).read_text().splitlines():
            r = json.loads(line)
            para[r["paraphrase_of"]].append(r["question"])
    from scripts.augment_text import noisy

    out, swapped, noised = [], 0, 0
    for line in (REPO / a.base).read_text().splitlines():
        rec = json.loads(line)
        alts = para.get(rec["task_id"], [])
        if alts and rng.random() < a.mix:
            rec["messages"][1] = {"role": "user", "content": rng.choice(alts)}
            rec["paraphrased"] = True
            swapped += 1
        if a.noise and rng.random() < a.noise:
            rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
            rec["noised"] = True
            noised += 1
        out.append(json.dumps(rec))
    (REPO / "data/sft/train_v2.jsonl").write_text("\n".join(out) + "\n")
    val = [json.loads(x) for x in (REPO / a.val).read_text().splitlines()]
    held = [v for v in val if v["template"] in HELD_OUT]
    seen = [v for v in val if v["template"] not in HELD_OUT]
    rng.shuffle(seen)
    vh = held[:60] + seen[: min(60, len(held[:60]) or 60)]
    (REPO / "data/sft/val_heldout.jsonl").write_text("\n".join(json.dumps(v) for v in vh) + "\n")
    print(json.dumps({"train_v2": len(out), "paraphrased": swapped, "noised": noised, "tasks_with_paraphrases": len(para),
                      "val_heldout": len(vh), "held_out_in_val": len(held[:60])}))


if __name__ == "__main__":
    main()
