"""SPEC 8.2 split rules on the generated task files: fail the build on any violation."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "data" / "tasks"
pytestmark = pytest.mark.skipif(not (TASKS / "train.jsonl").exists(), reason="tasks not generated")


def _rows():
    out = {}
    for sp in ("train", "val", "test"):
        f = TASKS / f"{sp}.jsonl"
        out[sp] = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
    return out


def test_no_scenario_seed_shared_across_splits():
    rows = _rows()
    seeds = {sp: {r["scenario_seed"] for r in rs} for sp, rs in rows.items()}
    assert not (seeds["train"] & seeds["val"]) and not (seeds["train"] & seeds["test"]) \
        and not (seeds["val"] & seeds["test"]), seeds


def test_held_out_families_never_in_train():
    from scripts.make_tasks import HELD_OUT

    rows = _rows()
    assert not {r["template"] for r in rows["train"]} & HELD_OUT
    assert {r["template"] for r in rows["test"]} & HELD_OUT, "the test split must contain held-out families"


def test_no_duplicate_question_scenario_pairs_across_splits():
    rows = _rows()
    seen = defaultdict(set)
    for sp, rs in rows.items():
        for r in rs:
            seen[(r["question"], r["scenario_seed"])].add(sp)
    assert all(len(v) == 1 for v in seen.values())


def test_answer_keys_never_leak_into_questions():
    for rs in _rows().values():
        for r in rs[:2000]:
            q = r["question"]
            assert "los_hyp_refs" not in q and "answer_key" not in q
            for ref in (r["answer_key"].get("los_hyp_refs") or []):
                assert ref not in q
