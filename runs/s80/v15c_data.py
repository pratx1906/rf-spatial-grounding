"""C-PAIR control arm data (docs/paper/CONTRAST_PREREG.md): same size as train_v15b (167 new + the SAME 300 replay),
new expert episodes from seen families EXCLUDING the six contrast families and every HELD_OUT family."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.make_tasks import HELD_OUT  # noqa: E402
from train.sft_v12_data import _record  # noqa: E402

WORK = REPO / "runs/s80/v15c"
B_FAMS = {"nearest_drone", "which_zone", "zone_incursion", "entity_not_found", "near_object_ambiguous",
          "zone_incursion_ambiguous"}
N_NEW = 167
NOISE = 0.2


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    rng = random.Random(1506)
    WORK.mkdir(parents=True, exist_ok=True)
    used = {r["task_id"] for r in rows(REPO / "data/sft/train_v14.jsonl")}
    used |= {r["task_id"] for r in rows(REPO / "data/sft/train_v15b.jsonl")}
    pool = [r for r in rows(REPO / "data/tasks/x3/train.jsonl")
            if r["task_id"] not in used and r["template"] not in B_FAMS | HELD_OUT]
    pick = rng.sample(pool, 230)
    res = _record(pick, WORK / "rec_new.jsonl")
    print(json.dumps(res, indent=1), flush=True)
    from scripts.augment_text import noisy
    recs = rows(WORK / "rec_new.jsonl")
    order = {r["task_id"]: i for i, r in enumerate(pick)}
    recs.sort(key=lambda r: order[r["task_id"]])
    new = []
    for rec in recs[:N_NEW]:
        rec.pop("site", None)
        if rng.random() < NOISE:
            rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
            rec["noised"] = True
        rec["source"] = "v15c_control"
        new.append(rec)
    assert len(new) == N_NEW, len(new)
    rep = [r for r in rows(REPO / "data/sft/train_v15b.jsonl") if r["source"] == "v14_replay"]
    assert len(rep) == 300
    allrec = new + rep
    rng.shuffle(allrec)
    assert not {r["template"] for r in allrec} & HELD_OUT
    assert not {r["template"] for r in new} & B_FAMS
    out = REPO / "data/sft/train_v15c.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in allrec))
    print(json.dumps({"train": len(allrec), "new_by_family": dict(Counter(r["template"] for r in new))}, indent=1))


if __name__ == "__main__":
    main()
