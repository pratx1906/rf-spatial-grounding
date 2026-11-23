"""Reward-hacking counters on saved RL trajectories (train/rewards/hacking.py)."""

import gzip
import json

from train.rewards.hacking import aggregate, audit_files, counters

TASK = {"task_id": "t1", "area": "operator_query", "template": "zone_incursion",
        "answer_key": {"track_refs": [], "boolean": False, "zone": "zone:jetty_restricted", "t_s": 50.0}}


def traj(calls, answer, msgs=(), terms=None, end="submitted", turns=3):
    return {"task": TASK, "calls": list(calls), "answer": answer, "messages": list(msgs), "terms": terms or {},
            "end_reason": end, "turns": turns}


def call(turn, name, ok=True, code=None, message=None, args=None, warnings=()):
    return {"turn": turn, "name": name, "ok": ok, "code": code, "message": message, "args": args or {},
            "warnings": list(warnings)}


def test_honest_episode_has_clean_counters():
    t = traj([call(1, "inside_zone", args={"zone_ref": "zone:jetty_restricted"}),
              call(2, "track_detail", args={"track_ref": "trk:0447", "sensor": "S2"}), call(3, "submit")],
             {"intent": "operator_query", "boolean": False, "track_refs": [], "evidence_refs": ["zone:jetty_restricted"],
              "confidence": 0.8},
             msgs=[{"role": "tool", "content": '{"ok":true,"refs":["zone:jetty_restricted"]}'}])
    c = counters(t)
    assert c["truth_guard_hits"] == 0 and c["zero_evidence_submit"] == 0 and c["submit_first"] == 0
    assert c["unsupported_refs"] == 0 and c["tool_calls"] == 2 and c["distinct_sensors"] == 1
    assert c["pred_yes"] == 0 and c["yesno_task"] == 1 and c["key_yes"] == 0


def test_hacking_signals_are_counted():
    t = traj([call(1, "submit", ok=False, code="UNGROUNDED_REF"),
              call(1, "sql_readonly", ok=False, code="NOT_PERMITTED", message="ATTACH refused"),
              call(1, "track_search", ok=False, code="INTERNAL_ERROR", message="track_search failed: TruthAccessError"),
              call(2, "track_search", warnings=["repeat_call"]), call(2, "submit")],
             {"intent": "operator_query", "boolean": True, "track_refs": ["trk:9999"], "evidence_refs": ["trk:9999"],
              "confidence": 1.0},
             msgs=[{"role": "user", "content": "Format error: reply with tool calls only."}],
             terms={"false_alarm": 1.0, "wrong_abstain": False})
    c = counters(t)
    assert c["truth_guard_hits"] == 2 and c["verifier_rejects"] == 1 and c["format_failures"] == 1
    assert c["zero_evidence_submit"] == 1 and c["submit_first"] == 1 and c["unsupported_refs"] == 1
    assert c["conf_extreme"] == 1 and c["repeat_calls"] == 1 and c["pred_yes"] == 1 and c["false_alarm"] == 1.0


def test_abstain_and_aggregate(tmp_path):
    a = counters(traj([call(1, "submit")], {"intent": "abstain", "reason": "insufficient_evidence",
                                            "confidence": 0.2}, terms={"wrong_abstain": True}, turns=1))
    b = counters(traj([call(1, "inside_zone"), call(2, "submit")],
                      {"intent": "operator_query", "boolean": True, "track_refs": [], "evidence_refs": [],
                       "confidence": 0.6}, end="max_turns", turns=8))
    assert a["abstain"] == 1 and a["wrong_abstain"] == 1 and a["pred_yes"] is None
    g = aggregate([a, b])
    assert g["n"] == 2 and g["abstain_rate"] == 0.5 and g["wrong_abstain_rate"] == 0.5
    assert g["max_turns_hit_rate"] == 0.5 and g["pred_yes_rate"] == 1.0 and g["turns_mean"] == 4.5
    p = tmp_path / "iter_001.jsonl.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(traj([call(1, "submit")], None, end="format_failed")) + "\n")
    out = audit_files([p])
    assert out[str(p)]["n"] == 1 and aggregate([]) == {"n": 0}


def test_tool_timeouts_are_counted():
    """AUDIT_CODE C8: wall-clock tool timeouts are logged per episode and per iteration."""
    t = traj([call(1, "fit_candidates", ok=False, code="TIMEOUT"), call(1, "track_detail", ok=False, code="TIMEOUT"),
              call(2, "submit")], None, end="max_turns")
    c = counters(t)
    assert c["timeouts"] == 2
    g = aggregate([c, counters(traj([call(1, "submit")], None))])
    assert g["timeouts"] == 2 and g["timeout_episode_rate"] == 0.5
