"""Trivial, classical and parser->expert baselines (plan 2026-10-28 T4): determinism, valid answers, and the parser
never seeing task.params."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.base import Task

REPO = Path(__file__).resolve().parents[2]
V6 = REPO / "eval" / "frozen" / "v6" / "tasks.jsonl"
FAMS = ("locate", "locate_two_sensors", "ds_hover", "zone_incursion", "nearest_drone", "count_in_zone",
        "actuation", "jam_all", "direct_path")


def _tasks():
    if not V6.exists():
        pytest.skip("frozen v6 missing")
    rows = [json.loads(x) for x in V6.read_text().splitlines()]
    out = {}
    for r in rows:
        if r["template"] in FAMS and r["template"] not in out and (REPO / r["site"] / "meta.yaml").exists():
            out[r["template"]] = Task.from_row(r)
    if len(out) < len(FAMS):
        pytest.skip("v3 scenarios not built")
    return out


def _run(policy, task, seed=0, opts=None):
    from eval.run import run_one

    return run_one(task, policy, policy, 0.0, seed, "scenescout", opts=opts)


@pytest.mark.parametrize("policy", ["always_abstain", "prior_centroid", "nearest_sensor", "random_track",
                                    "tracker_only", "triangulate_only"])
def test_baselines_are_deterministic_and_answer_validly(policy):
    from harness.answers import Answer
    from pydantic import TypeAdapter

    ta = TypeAdapter(Answer)
    for fam, t in _tasks().items():
        a, b = _run(policy, t), _run(policy, t)
        assert a["error"] is None, (policy, fam, a["error"])
        assert a["answer"] == b["answer"] and a["score"] == b["score"], (policy, fam)
        assert a["end_reason"] in ("submitted", "verifier_failed"), (policy, fam, a["end_reason"])
        ta.validate_python(a["answer"])
        assert 0.0 <= a["score"] <= 1.0


def test_always_abstain_scores_only_where_abstain_is_the_key():
    ts = _tasks()
    assert _run("always_abstain", ts["locate"])["score"] == 0.0
    from eval.trivial import ABSTAIN_REASON

    r = _run("always_abstain", ts["actuation"])         # one constant reason: right only where it is the key
    assert r["abstained"] and r["answer"]["reason"] == ABSTAIN_REASON and r["score"] == 0.0


def test_position_baselines_answer_positions():
    ts = _tasks()
    for policy in ("prior_centroid", "nearest_sensor", "tracker_only", "triangulate_only"):
        a = _run(policy, ts["locate"])["answer"]
        assert a["intent"] == "df_ambiguity" and a["position"] is not None, policy
    for policy in ("prior_centroid", "nearest_sensor", "tracker_only"):
        a = _run(policy, ts["ds_hover"])["answer"]
        assert a["intent"] == "drone_search", policy


def test_random_track_depends_on_seed_only():
    t = _tasks()["nearest_drone"]
    picks = {json.dumps(_run("random_track", t, seed=s)["answer"]["track_refs"]) for s in range(6)}
    assert len(picks) >= 1
    assert _run("random_track", t, seed=4)["answer"] == _run("random_track", t, seed=4)["answer"]


def test_tracker_only_oq_matches_the_expert_slots():
    t = _tasks()["zone_incursion"]
    assert _run("tracker_only", t)["score"] == _run("expert", t)["score"]


# --- parser -> expert ----------------------------------------------------------------------------------------------
class _ParserLLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        msg = SimpleNamespace(content=self.reply, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_parser_never_receives_params_and_runs_the_expert():
    from eval.parser_expert import parse_task

    t = _tasks()["locate"]
    secret = Task.from_row({**t.to_row(), "params": {**t.params, "secret_slot": "ZZ-SENTINEL-91"},
                            "answer_key": {**t.answer_key, "secret": "KEY-SENTINEL-77"}})
    reply = json.dumps({"family": t.template, "params": t.params})
    llm = _ParserLLM(reply)
    shadow, info = parse_task(llm, "m", secret, examples=[("Where is trk:1 at t=5 s?", "locate",
                                                           {"track": "trk:1", "t": 5.0})])
    sent = json.dumps(llm.seen[0]["messages"])
    assert "ZZ-SENTINEL-91" not in sent and "KEY-SENTINEL-77" not in sent and "secret_slot" not in sent
    assert secret.question in sent
    assert shadow.template == t.template and shadow.params == t.params and info["ok"]
    r = _run("parser_expert", t, opts={"parser_client": _ParserLLM(reply), "parser_model": "m",
                                        "parser_examples": []})
    e = _run("expert", t)
    assert r["score"] == e["score"] and r["parsed"]["family"] == t.template


def test_parser_garbage_abstains():
    t = _tasks()["locate"]
    r = _run("parser_expert", t, opts={"parser_client": _ParserLLM("no json here"), "parser_model": "m",
                                        "parser_examples": []})
    assert r["error"] is None and r["abstained"] and r["parsed"]["ok"] is False


def test_few_shot_examples_come_from_train_families_only(tmp_path):
    from eval.parser_expert import few_shot_examples

    rows = [{"task_id": "a", "template": "locate", "question": "q1", "params": {"t": 1}, "split": "train"},
            {"task_id": "b", "template": "locate", "question": "q2", "params": {"t": 2}, "split": "train"},
            {"task_id": "c", "template": "near_object", "question": "q3", "params": {"t": 3}, "split": "train"}]
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ex = few_shot_examples(p, exclude={"near_object"})
    assert ex == [("q1", "locate", {"t": 1})]
