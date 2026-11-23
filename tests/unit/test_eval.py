"""Eval runner pieces: summary metrics, paired bootstrap, and the model loop on a scripted fake LLM."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.run import paired_bootstrap, summarise

REPO = Path(__file__).resolve().parents[2]


def _row(tid, area, score, code=None):
    return {"task_id": tid, "area": area, "score": score, "verifier_code": code, "abstained": False, "turns": 3,
            "latency_s": 1.0, "error": None, "intent": "replay"}


def test_paired_bootstrap_detects_a_clear_improvement():
    a = [_row(f"t{i}", "replay", 0.2) for i in range(50)]
    b = [_row(f"t{i}", "replay", 0.8) for i in range(50)]
    r = paired_bootstrap(a, b)
    assert r["replay"]["diff"] == 0.6 and r["replay"]["ci95"][0] > 0.5


def test_summarise_reports_macro_score_and_ungrounded():
    s = summarise([_row("a", "replay", 1.0), _row("b", "siting", 0.0, "UNGROUNDED")])
    assert s["_all"]["macro_score"] == 0.5 and s["_all"]["ungrounded"] == 1


class _FakeLLM:
    """Replays a fixed sequence of tool calls as if they came from the model server."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        calls = self.turns.pop(0)
        tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
               for i, (n, a) in enumerate(calls)]
        msg = SimpleNamespace(content="", tool_calls=tcs)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(completion_tokens=20))


def test_run_task_keeps_the_answer_key_for_scoring():
    site = REPO / "data" / "site_kandla"
    if not (site / "meta.yaml").exists():
        pytest.skip("Kandla site not built")
    from harness.env import AkshaEnv
    from serve.agent import run_task
    from tasks.base import Task

    env = AkshaEnv(site)
    t = Task("t1", "unanswerable", "actuation", "Jam the drone now.", {"reason": "outside_tool_scope"})
    llm = _FakeLLM([[("submit", {"answer": {"intent": "abstain", "reason": "outside_tool_scope",
                                            "evidence_refs": [], "confidence": 0.9}})]])
    events = list(run_task(env, t, client=llm))
    assert events[-1]["type"] == "done" and env.answer_score == 1.0
