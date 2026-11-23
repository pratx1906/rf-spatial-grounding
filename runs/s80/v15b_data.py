"""SFT v1.5b data: corrective mix on top of SFT v1.4 (near_object over-abstain fix, 2026-11-06).

Diagnosis (runs/eval/sft_v1{1,4}_heldout_t7.json): v1.4 abstains on 13/40 held-out near_object tasks (v1.1: 4), all
wrong. The trained family near_object_ambiguous asks "within N m of THE jetty" (8 jetties -> clarify); held-out
near_object asks "within N m of Jetty J2" (answerable). v1.4 learned "within N m of ... -> clarify" with no contrast.
Fix without touching any HELD_OUT family: more expert episodes of seen families that name ONE specific object and
must be ANSWERED (nearest_drone, zone_incursion, which_zone), next to the abstain families (near_object_ambiguous,
zone_incursion_ambiguous, entity_not_found), plus replay of the v1.4 mix. NOTE: the family choice was informed by the
held-out failure analysis (disclosed in the report).

    python runs/s80/v15b_data.py   (WSL, SCENESCOUT_DATA_VERSION=v3 SCENESCOUT_RF_GEOM=rfr)
"""
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

WORK = REPO / "runs/s80/v15b"
NEW = {"nearest_drone": 80, "zone_incursion": 40, "which_zone": 40,
       "near_object_ambiguous": 15, "zone_incursion_ambiguous": 15, "entity_not_found": 15}
N_REPLAY = 300
NOISE = 0.2


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    rng = random.Random(1505)
    WORK.mkdir(parents=True, exist_ok=True)
    assert not set(NEW) & HELD_OUT
    used = {r["task_id"] for r in rows(REPO / "data/sft/train_v14.jsonl")}
    train = [r for r in rows(REPO / "data/tasks/x3/train.jsonl") if r["task_id"] not in used]
    pick = []
    for fam, n in NEW.items():
        pool = [r for r in train if r["template"] == fam]
        pick += rng.sample(pool, min(n, len(pool)))
    (WORK / "tasks_new.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pick))
    res = _record(pick, WORK / "rec_new.jsonl")
    print(json.dumps(res, indent=1), flush=True)

    from scripts.augment_text import noisy
    new = []
    for rec in rows(WORK / "rec_new.jsonl"):
        rec.pop("site", None)
        if rng.random() < NOISE:
            rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
            rec["noised"] = True
        rec["source"] = "v15b_contrast"
        new.append(rec)
    rep = rng.sample(rows(REPO / "data/sft/train_v14.jsonl"), N_REPLAY)
    for r in rep:
        r["source"] = "v14_replay"
    allrec = new + rep
    rng.shuffle(allrec)
    assert not {r["template"] for r in allrec} & HELD_OUT
    out = REPO / "data/sft/train_v15b.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in allrec))
    print(json.dumps({"train": len(allrec), "by_source": dict(Counter(r["source"] for r in allrec)),
                      "new_by_family": dict(Counter(r["template"] for r in new)), "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
