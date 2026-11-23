"""Audit M9 (plan 2026-10-28 T5): no held-out or compositional family in any val / selection set."""

import json
from pathlib import Path

import pytest

from scripts.make_tasks import COMPOSITIONAL, HELD_OUT, NOT_FOR_SELECTION, selection_rows, write_clean_val

REPO = Path(__file__).resolve().parents[2]
VAL_CLEAN = REPO / "data" / "tasks" / "x3" / "val_clean.jsonl"


def _rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def test_selection_set_covers_held_out_and_compositional():
    assert NOT_FOR_SELECTION == HELD_OUT | COMPOSITIONAL and HELD_OUT <= NOT_FOR_SELECTION


@pytest.mark.skipif(not VAL_CLEAN.exists(), reason="data/tasks/x3/val_clean.jsonl not built")
def test_val_clean_has_no_held_out_families():
    rows = _rows(VAL_CLEAN)
    assert rows and not {r["template"] for r in rows} & NOT_FOR_SELECTION
    assert all(r["split"] == "val" for r in rows)
    src = REPO / "data" / "tasks" / "x3" / "val.jsonl"
    if src.exists():                                   # exactly the filtered source, nothing regenerated
        assert rows == [r for r in _rows(src) if r["template"] not in NOT_FOR_SELECTION]


def test_write_clean_val_filters_and_never_overwrites(tmp_path):
    rows = [{"task_id": f"t{i}", "template": f, "split": "val"}
            for i, f in enumerate(["locate", "near_object", "ds_wide_prior", "zone_incursion"])]
    src, dst = tmp_path / "val.jsonl", tmp_path / "val_clean.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = write_clean_val(src, dst)
    assert [r["template"] for r in _rows(dst)] == ["locate", "zone_incursion"] and rep["kept"] == 2
    assert rep["dropped"] == {"ds_wide_prior": 1, "near_object": 1}
    with pytest.raises(SystemExit):
        write_clean_val(src, dst)
    assert selection_rows(rows) == _rows(dst)


def test_generator_strips_held_out_from_val_too():
    import inspect

    import scripts.make_tasks as mt

    src = inspect.getsource(mt.main)
    assert 'NOT_FOR_SELECTION and sp in ("train", "val")' in src
