"""Truth guard extensions (paper Revision 2, AUDIT_CODE minor 6): other seeds' truth/ dirs are forbidden while an
episode runs, and AkshaEnv.call_tools is guarded when called directly (the TRL adapter path)."""

from pathlib import Path

import pytest

from harness import guard
from harness.guard import TruthAccessError, other_truth, truth_guard

REPO = Path(__file__).resolve().parents[2]


def test_other_truth_matches_any_truth_dir_under_data():
    d = guard._DATA_REAL
    import os

    assert other_truth(os.path.join(d, "scenarios_v3", "kandla_s16_X", "truth", "truth.duckdb"))
    assert other_truth(os.path.join(d, "scenarios_v9", "not_built_yet", "truth"))
    assert other_truth(os.path.join(d, "site_kandla", "truth", "x.parquet"))
    assert not other_truth(os.path.join(d, "scenarios_v3", "kandla_s16_X", "observed.duckdb"))
    assert not other_truth(os.path.join(d, "tasks", "x3", "train.jsonl"))
    assert not other_truth(str(REPO / "truth" / "x"))                    # outside data/: roots handle the site


def test_guard_blocks_another_seeds_truth(tmp_path):
    target = REPO / "data" / "scenarios_v3" / "kandla_s16_X" / "truth" / "truth.duckdb"
    with truth_guard(tmp_path):
        with pytest.raises(TruthAccessError):
            open(target, "rb")
    if target.exists():                                                  # outside the guard: offline reads work
        open(target, "rb").close()


def test_call_tools_is_guarded_on_its_own(stub_dir, monkeypatch):
    from harness.env import AkshaEnv
    from harness.executor import Executor
    from tasks.base import Task

    seen = []
    orig = Executor.run

    def run(self, calls, turn):
        seen.append(guard._roots())
        return orig(self, calls, turn)

    monkeypatch.setattr(Executor, "run", run)
    env = AkshaEnv(stub_dir)
    env.reset(Task("t-1", "operator_query", "zone_incursion", "q", {"track_refs": [], "boolean": False}))
    assert guard._roots() == ()
    env.call_tools([{"name": "site_resolve", "arguments": {"query": "Cargo Area"}}])     # direct, outside step()
    assert seen and seen[0] and any(r.endswith("truth") for r in seen[0])
    assert guard._roots() == ()                                          # restored afterwards
