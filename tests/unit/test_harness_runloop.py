"""demo-v2 S5.3 / S5.5: the serve run loop (serve/agent.py) on a scripted fake LLM: wall-clock budget, LLM timeout and
errors always end the episode with an answer event, verifier-failed answers are delivered flagged, and every
question writes an audit record."""

import json
import time
from types import SimpleNamespace

from harness.audit import AuditLog, answer_hash
from tasks.base import Task

Q = Task("t-1", "operator_query", "zone_incursion", "Is any drone inside the Cargo Area at t=100 s?",
         {"track_refs": [], "boolean": False})
ABSTAIN = [("submit", {"answer": {"intent": "abstain", "reason": "insufficient_evidence", "evidence_refs": [],
                                  "confidence": 0.3}})]
UNGROUNDED = [("submit", {"answer": {"intent": "operator_query", "track_refs": ["trk:9999"],
                                     "evidence_refs": ["trk:9999"], "confidence": 0.9}})]
RESOLVE = [("site_resolve", {"query": "Cargo Area"})]


class FakeLLM:
    def __init__(self, turns, delay_s=0.0, exc=None):
        self.turns, self.delay_s, self.exc, self.kw = list(turns), delay_s, exc, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kw.append(kw)
        if self.exc is not None:
            raise self.exc
        time.sleep(self.delay_s)
        calls = self.turns.pop(0)
        tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
               for i, (n, a) in enumerate(calls)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=tcs))],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=20))


class APITimeoutError(Exception):
    pass


def _run(env, llm, **kw):
    from serve.agent import run_task

    return [e for e in run_task(env, Q, client=llm, **kw) if e["type"] != "transcript"]


def test_llm_error_ends_the_episode_with_an_answer_event(env):
    ev = _run(env, FakeLLM([], exc=ConnectionError("refused")))
    err = next(e for e in ev if e["type"] == "error")
    assert err["code"] == "llm_unreachable" and "LLM server" in err["message"]
    ans = next(e for e in ev if e["type"] == "answer")
    assert ans["end_reason"] == "llm_error" and ans["answer"] is None and ans["status"] == "none"
    assert ev[-1]["type"] == "done"
    assert env.tracer.events[-1]["role"] == "end" and env.tracer.events[-1]["reason"] == "llm_error"


def test_llm_timeout_has_its_own_code(env):
    ev = _run(env, FakeLLM([], exc=APITimeoutError("read timed out")))
    assert next(e for e in ev if e["type"] == "error")["code"] == "llm_timeout"


def test_wallclock_budget_is_enforced_and_bounds_the_llm_timeout(env):
    llm = FakeLLM([RESOLVE, RESOLVE, RESOLVE, ABSTAIN], delay_s=0.3)
    ev = _run(env, llm, max_wallclock_s=0.5)
    ans = next(e for e in ev if e["type"] == "answer")
    assert ans["end_reason"] == "wallclock" and ev[-1]["type"] == "done"
    assert next(e for e in ev if e["type"] == "error")["code"] == "wallclock"
    assert 0 < llm.kw[0]["timeout"] <= 0.5 and llm.kw[1]["timeout"] < llm.kw[0]["timeout"]


def test_no_wallclock_by_default_but_a_client_timeout_always(env):
    llm = FakeLLM([ABSTAIN])
    ev = _run(env, llm)
    assert next(e for e in ev if e["type"] == "answer")["end_reason"] == "submitted"
    assert llm.kw[0]["timeout"] > 0                         # never the OpenAI client's 600 s default


def test_verifier_failed_answer_is_delivered_flagged(env):
    ev = _run(env, FakeLLM([UNGROUNDED, UNGROUNDED]))
    ans = next(e for e in ev if e["type"] == "answer")
    assert ans["end_reason"] == "verifier_failed"
    assert ans["status"] == "flagged" and ans["flagged"] is True and ans["flags"] == ["UNGROUNDED"]
    assert ans["gate_required"] is False


def test_verified_answer_carries_its_hash(env):
    ev = _run(env, FakeLLM([ABSTAIN]))
    ans = next(e for e in ev if e["type"] == "answer")
    assert ans["status"] == "verified" and ans["flagged"] is False and ans["confidence"] == 0.3
    assert ans["answer_hash"] == answer_hash(env.episode_id, env.site.version, ans["answer"])
    assert any(r["rule"] == "grounding" for r in ans["verifier_rules"])


def test_every_question_writes_an_audit_record(env, tmp_path):
    log = AuditLog(tmp_path)
    _run(env, FakeLLM([RESOLVE, ABSTAIN]), audit=log, request={"question": Q.question, "client": "c1",
                                                               "request_id": "r1"})
    rec = log.trace(env.episode_id)["question"]
    assert rec["request"]["question"] == Q.question and rec["request"]["request_id"] == "r1"
    assert rec["model"]["served"] and "adapter" in rec["model"]
    assert rec["tokens"] == {"prompt": 2000, "completion": 40} and len(rec["turns"]) == 2
    assert rec["turns"][0]["tools"] == ["site_resolve"]
    assert rec["end_reason"] == "submitted" and rec["status"] == "verified"
    assert {r["rule"] for r in rec["verifier_rules"]} >= {"grounding", "collateral_complete", "rf_ambiguity"}
    assert rec["operator_decision"] is None
    assert AuditLog.verify(next(tmp_path.glob("*.jsonl"))) == (True, None)


def test_eval_rows_log_confidence():
    """S5.4: eval rows carry the answer's stated confidence (calibration needs it)."""
    from pathlib import Path

    import pytest

    repo = Path(__file__).resolve().parents[2]
    frozen = repo / "eval" / "frozen" / "v3" / "tasks.jsonl"
    if not frozen.exists() or not (repo / "data" / "scenarios" / "kandla_s15" / "meta.yaml").exists():
        pytest.skip("frozen v3 set or Kandla seeds missing")
    from eval.run import run_one

    rows = [json.loads(x) for x in frozen.read_text().splitlines()]
    t = Task.from_row(next(r for r in rows if r["template"] == "zone_incursion"))
    row = run_one(t, "expert", "expert", 0.0, 0, "scenescout")
    assert isinstance(row["confidence"], float) and 0.0 <= row["confidence"] <= 1.0
    assert "flags" in row
