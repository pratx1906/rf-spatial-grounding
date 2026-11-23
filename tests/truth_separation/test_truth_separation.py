"""SPEC 12.1 / 15.3: no agent-visible surface may reach truth. Never weaken these tests."""

import ast
import json
from pathlib import Path

import pytest

from harness.guard import TruthAccessError, truth_guard

REPO = Path(__file__).resolve().parents[2]
AGENT_SIDE = ["tools", "harness", "serve", "sim/belief"]
FORBIDDEN_IMPORTS = ("sim.world", "sim.obs_model", "sim.stub_site", "tasks.truth_access", "tasks.m1_tasks")
TRUTH_TOKENS = ("emt_", "reflect:", "path_type", "friendly_drone", "decoy", "hyp_labels", "track_labels",
                "emitter_points", "truth.duckdb")


def _py_files():
    for d in AGENT_SIDE:
        p = REPO / d
        if p.exists():
            yield from p.rglob("*.py")


def test_agent_side_never_imports_world_or_truth_modules():
    bad = []
    for f in _py_files():
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [f"{f.relative_to(REPO)}: {n}" for n in names if n.startswith(FORBIDDEN_IMPORTS)]
    assert not bad, bad


def _code_strings(tree: ast.AST) -> list[str]:
    """String literals used as values, excluding docstrings (documentation may *name* truth/, code may not)."""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_agent_side_has_no_truth_paths():
    for f in _py_files():
        if f.name == "guard.py":
            continue                         # the guard names the forbidden dirs on purpose
        for s in _code_strings(ast.parse(f.read_text())):
            assert not any(tok in s for tok in ("truth/", "truth.duckdb", "world_cache")), (f, s)
            assert s != "truth", (f, s)


def test_guard_blocks_python_level_access(stub_dir):
    with truth_guard(stub_dir):
        with pytest.raises(TruthAccessError):
            open(stub_dir / "truth" / "truth.duckdb", "rb")
        with pytest.raises(TruthAccessError):
            list((stub_dir / "truth").iterdir())
        (stub_dir / "meta.yaml").read_text()          # belief files stay readable
    open(stub_dir / "truth" / "truth.duckdb", "rb").close()   # guard is scoped


def test_episodes_never_show_truth(env, tasks):
    from tasks.expert import solve

    for t in tasks:
        obs = env.reset(t)
        text = [json.dumps(obs["messages"])]
        orig_step = env.step

        def step(action, _orig=orig_step, _text=text):
            out = _orig(action)
            _text.append(json.dumps(out[0]))
            return out
        env.step = step
        solve(env, t)
        env.step = orig_step
        blob = "\n".join(text)
        for tok in TRUTH_TOKENS:
            assert tok not in blob, (t.task_id, tok)
        for k in ("los_hyp_refs", "all_hyp_refs"):
            assert k not in blob


def test_answer_key_never_in_prompt(env, tasks):
    for t in tasks:
        obs = env.reset(t)
        prompt = json.dumps(obs)
        assert "answer_key" not in prompt and "clarification" not in prompt and "drone_payload" not in prompt
        if t.clarification:
            assert t.clarification not in prompt
