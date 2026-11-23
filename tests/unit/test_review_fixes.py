"""Regression tests for the final-review findings (2026-10-18)."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"


def test_no_process_randomised_hash_in_seeded_code():
    """Builtin hash() of str/tuple changes per process (PYTHONHASHSEED): it must never seed simulation or tasks."""
    for f in ("sim/obs_model/observe_air7311.py", "tasks/kandla/gen.py", "sim/world/kandla_scenario.py",
              "sim/obs_model/observe_rt.py"):
        tree = ast.parse((REPO / f).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "hash"]
        assert not calls, f"{f} uses builtin hash()"


def test_stable_seed_is_identical_across_processes():
    code = "from sim.geometry import stable_seed; print(stable_seed('S1', 'ism_2400', 7))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                           env={"PYTHONHASHSEED": s, "PYTHONPATH": str(REPO)}).stdout.strip() for s in ("1", "2")}
    assert len(outs) == 1 and outs.pop().isdigit()


@pytest.mark.skipif(not (REPO / "data" / "tasks" / "train.jsonl").exists(), reason="tasks not generated")
def test_no_question_with_its_key_is_shared_between_train_and_eval_splits():
    def keys(sp):
        f = REPO / "data" / "tasks" / f"{sp}.jsonl"
        return {(json.loads(x)["question"], json.dumps(json.loads(x)["answer_key"], sort_keys=True))
                for x in f.read_text().splitlines()}
    train = keys("train")
    assert not (train & keys("test")) and not (train & keys("val"))


def _drone_env():
    from harness.env import AkshaEnv
    from tasks.base import Task

    pay = {"target_track": "trk:0000", "t0_s": 0.0, "start": {"x_m": 0, "y_m": 0, "z_m": 80}, "budget_moves": 2,
           "prior": {"x_m": 500, "y_m": 500, "a_m": 200, "b_m": 100, "theta_deg": 30},
           "waypoints": [{"id": f"wp:0{i}", "x_m": 500 + 300 * i, "y_m": 0, "z_m": 80, "bearing_deg": 0.0,
                          "snr_db": 20} for i in range(1, 4)]}
    env = AkshaEnv(KANDLA)
    env.reset(Task("t", "drone_search", "ds_hover", "q", {}, drone_payload=pay))
    return env


@pytest.mark.skipif(not (KANDLA / "meta.yaml").exists(), reason="Kandla site not built")
def test_parallel_stateful_calls_respect_the_budget():
    env = _drone_env()
    calls = [{"name": "move_drone", "arguments": {"wp_ref": f"wp:0{i}"}} for i in (1, 2, 3)]
    res = env.step({"tool_calls": calls})[3]["results"]
    assert [r["ok"] for r in res] == [True, True, False] and res[2]["error"]["code"] == "BUDGET_EXCEEDED"
    assert env.executor.ctx.episode["drone"].moves == ["wp:01", "wp:02"]


@pytest.mark.skipif(not (KANDLA / "meta.yaml").exists(), reason="Kandla site not built")
def test_reflying_a_waypoint_is_refused():
    env = _drone_env()
    env.step({"tool_calls": [{"name": "move_drone", "arguments": {"wp_ref": "wp:01"}}]})
    env.step({"tool_calls": [{"name": "drone_state", "arguments": {}}]})          # bumps nothing; harmless
    r = env.step({"tool_calls": [{"name": "move_drone", "arguments": {"wp_ref": "wp:01"}}]})[3]["results"][0]
    assert not r["ok"] and r["error"]["code"] == "INVALID_ARGS" and "already" in r["error"]["message"]


@pytest.mark.skipif(not (KANDLA / "meta.yaml").exists(), reason="Kandla site not built")
def test_plan_waypoints_accepts_a_bare_track_id():
    env = _drone_env()
    r = env.step({"tool_calls": [{"name": "plan_waypoints", "arguments": {"target_ref": "0000", "k": 1}}]})
    assert r[3]["results"][0]["ok"]


@pytest.mark.skipif(not (KANDLA / "rf_cache" / "coverage").exists(), reason="coverage not built")
def test_weak_zones_outside_the_model_is_out_of_coverage():
    from sim.belief.site_view import SiteView
    from tools.errors import ToolError
    from tools.registry import REGISTRY, ToolContext, load_all

    load_all()
    ctx = ToolContext(SiteView(KANDLA))
    spec = REGISTRY["weak_zones"]
    with pytest.raises(ToolError) as e:
        spec.fn(ctx, spec.args.model_validate({"metric": "snr_db", "band": "ism_2400", "threshold": 10,
                                               "region": {"type": "circle", "center": {"x_m": 9000, "y_m": 9000},
                                                          "radius_m": 200}}))
    assert e.value.code == "OUT_OF_COVERAGE"
