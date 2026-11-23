"""The textually disjoint held-out subset (paper Revision 2, AUDIT_CODE C5). CPU, offline, keys only.

    python scripts/heldout_disjoint.py [--src eval/frozen/heldout_v1] [--out eval/frozen/heldout_v1_disjoint]

A held-out question is kept only if its masked text (track ids, numbers and other digits masked; case and spacing
normalised) matches NO masked question of any training source: every data/tasks/**/train*.jsonl, the SFT task lists
(data/tasks/{train,train_bal_tasks,v11_tasks,v12_tasks,v11_para,v12_para,v13_extra}.jsonl) and the user turns of
every data/sft/train*.jsonl (the SFT v1-v1.4 data). Several held-out families are difficulty strata of trained
families with the same phrasings (locate_ghost = locate, ds_wide_prior = ds_hover): they drop out here.

The manifest records the sources (with sha256), the per-family kept / dropped counts, and near_object split into
answerable (a non-empty key) and empty-key ("no drone") tasks, the latter being a "say no" test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SFT_TASK_LISTS = ("train", "train_bal_tasks", "v11_tasks", "v12_tasks", "v11_para", "v12_para", "v13_extra")
_TRK = re.compile(r"\b(trk|track|sensor|zone|obj|evt|hyp|fp|ras|nodecand)[:_ ]?#?[\w\-]*\d[\w\-]*", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def mask(q: str) -> str:
    """Question text with ids and numbers masked (ids before numbers so 'trk:2738' -> 'trk:<id>')."""
    q = _TRK.sub(lambda m: m.group(1).lower() + ":<id>", q)
    q = _NUM.sub("#", q)
    return re.sub(r"\s+", " ", q.strip().lower())


def training_sources() -> list[Path]:
    t = REPO / "data" / "tasks"
    ps = sorted(t.glob("**/train*.jsonl")) + [t / f"{n}.jsonl" for n in SFT_TASK_LISTS]
    ps += sorted((REPO / "data" / "sft").glob("train*.jsonl"))
    return [p for p in dict.fromkeys(ps) if p.exists()]


def questions_of(p: Path):
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("question"):
            yield r["question"]
        for m in r.get("messages") or []:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                yield m["content"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def disjoint(rows: list[dict], train_masked: set[str]) -> tuple[list[dict], list[dict]]:
    keep, drop = [], []
    for r in rows:
        (drop if mask(r["question"]) in train_masked else keep).append(r)
    return keep, drop


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="eval/frozen/heldout_v1")
    ap.add_argument("--out", default="eval/frozen/heldout_v1_disjoint")
    a = ap.parse_args(argv)
    from tasks.base import Task
    from tasks.scoring import empty_scores_full

    src = REPO / a.src / "tasks.jsonl"
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    srcs = training_sources()
    train_masked: set[str] = set()
    for p in srcs:
        train_masked |= {mask(q) for q in questions_of(p)}
    keep, drop = disjoint(rows, train_masked)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    tf = out / "tasks.jsonl"
    tf.write_text("".join(json.dumps(r) + "\n" for r in keep))
    fam_all, fam_keep = Counter(r["template"] for r in rows), Counter(r["template"] for r in keep)
    no = [r for r in rows if r["template"] == "near_object"]
    no_empty = [r["task_id"] for r in no if empty_scores_full(Task.from_row(r))]
    man = {"version": out.name, "source_set": f"{a.src}/tasks.jsonl", "source_sha256": sha256(src),
           "tasks_sha256": sha256(tf), "n": len(keep), "n_source": len(rows), "n_dropped": len(drop),
           "rule": "masked question (ids + numbers) not equal to any masked training question (scripts/heldout_disjoint.py)",
           "per_family": {f: {"source": fam_all[f], "kept": fam_keep.get(f, 0)} for f in sorted(fam_all)},
           "per_area": dict(sorted(Counter(r["area"] for r in keep).items())),
           "per_seed": dict(sorted(Counter(str(r.get("scenario_seed")) for r in keep).items())),
           "near_object": {"n": len(no), "empty_key_ids": no_empty,
                           "answerable_ids": [r["task_id"] for r in no if r["task_id"] not in set(no_empty)],
                           "note": "empty key = no drone near the object at t (an always-'no' answer scores 1.0)"},
           "dropped_examples": [{"task_id": r["task_id"], "masked": mask(r["question"])} for r in drop[:10]],
           "training_sources": [{"path": str(p.relative_to(REPO)).replace("\\", "/"), "sha256": sha256(p)}
                                for p in srcs],
           "n_training_masked_questions": len(train_masked)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("n", "n_source", "per_family")}, indent=1))
    print(json.dumps({"near_object_empty": len(no_empty), "near_object_answerable": len(no) - len(no_empty)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
