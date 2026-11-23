"""SFT v1.1 data: a short continuation of SFT v1 aimed at its measured weaknesses (frozen eval v3, docs/RESULTS/M2_M3.md).

    python -m train.sft_v11_data        -> data/sft/v11_base.jsonl, data/tasks/v11_tasks.jsonl
    python scripts/paraphrase.py --src data/tasks/v11_tasks.jsonl --out data/tasks/v11_para.jsonl --k 1
    python -m train.sft_augment --base data/sft/v11_base.jsonl --para data/tasks/v11_para.jsonl --mix 0.6
        (writes data/sft/train_v2.jsonl; copied to data/sft/train_v11.jsonl)

Half the examples come from weak *seen* families (never the held-out ones, so the held-out score stays an honest
generalisation measure); half are replay across every other family so v1's strengths are not forgotten. Varied-plan
trajectories (train_bal_v2) are preferred; the full train trajectories top up families that have too few.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEAK = ["first_into_zone", "count_in_zone", "tracks_in_band", "locate", "reject_ghosts", "direct_all", "entered_between"]
PER_WEAK, REPLAY = 35, 235


def main() -> None:
    from scripts.make_tasks import HELD_OUT

    assert not set(WEAK) & HELD_OUT
    rng = random.Random(11)
    bal = [json.loads(x) for x in (REPO / "data/sft/train_bal_v2.jsonl").read_text().splitlines()]
    full = [json.loads(x) for x in (REPO / "data/sft/train.jsonl").read_text().splitlines()]
    tasks = {json.loads(x)["task_id"]: x for x in (REPO / "data/tasks/train.jsonl").read_text().splitlines()}
    by = defaultdict(list)
    for r in bal:
        by[r["template"]].append(r)
    full_by = defaultdict(list)
    for r in full:
        full_by[r["template"]].append(r)
    picked, used = [], set()
    for fam in WEAK:
        pool = by[fam][:]
        rng.shuffle(pool)
        take = pool[:PER_WEAK]
        if len(take) < PER_WEAK:
            extra = [r for r in full_by[fam] if r["task_id"] not in {t["task_id"] for t in take}]
            rng.shuffle(extra)
            take += extra[: PER_WEAK - len(take)]
        picked += take
    others = [f for f in by if f not in WEAK and f not in HELD_OUT]
    per = max(1, REPLAY // len(others))
    for fam in others:
        pool = by[fam][:]
        rng.shuffle(pool)
        picked += pool[:per]
    rng.shuffle(picked)
    out, trows = [], []
    for r in picked:
        if r["task_id"] in used:
            continue
        used.add(r["task_id"])
        out.append(json.dumps(r))
        if r["task_id"] in tasks:
            trows.append(tasks[r["task_id"]])
    (REPO / "data/sft/v11_base.jsonl").write_text("\n".join(out) + "\n")
    (REPO / "data/tasks/v11_tasks.jsonl").write_text("\n".join(trows) + "\n")
    fams = defaultdict(int)
    for r in out:
        fams[json.loads(r)["template"]] += 1
    print(json.dumps({"examples": len(out), "tasks_for_paraphrase": len(trows),
                      "weak": {f: fams[f] for f in WEAK}, "replay_families": len(others)}))


if __name__ == "__main__":
    main()
