"""demo-v2 S5.1: server-side operator gate (two-step confirm) and the append-only, hash-chained audit log."""

import json

import pytest

from harness.audit import AuditLog, OperatorGate, answer_hash

ANS = {"intent": "collateral", "fp_ref": "fp:1", "violations": ["obj:a"], "requires_human_confirm": True,
       "evidence_refs": ["fp:1"], "confidence": 0.75}


@pytest.fixture()
def gate(tmp_path):
    g = OperatorGate(AuditLog(tmp_path), min_confirm_delay_s=0.0)
    h = answer_hash("ep1", "v1", ANS)
    g.register("ep1", h, gate_required=True)
    g.register("ep2", answer_hash("ep2", "v1", {"intent": "abstain"}), gate_required=False)
    return g, h


def _req(h, step="review", decision="acknowledge", op="op_7", **kw):
    return {"episode_id": "ep1", "answer_hash": h, "operator_id": op, "decision": decision, "step": step, **kw}


def test_answer_hash_is_canonical_and_binds_episode_and_data():
    a = answer_hash("ep1", "v1", ANS)
    assert a == answer_hash("ep1", "v1", dict(reversed(list(ANS.items()))))
    assert a != answer_hash("ep2", "v1", ANS) and a != answer_hash("ep1", "v2", ANS) and len(a) == 64


def test_review_then_confirm_records_the_decision(gate, tmp_path):
    g, h = gate
    code, r1 = g.submit(_req(h), client="c1")
    assert code == 200 and r1["step"] == "review" and r1["confirm_token"]
    code, r2 = g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"]), client="c1")
    assert code == 200 and r2["recorded"] and r2["decision"] == "acknowledge" and r2["record_id"]
    recs = [json.loads(x) for f in sorted(tmp_path.glob("*.jsonl")) for x in f.read_text().splitlines()]
    steps = [(r["kind"], r["step"]) for r in recs]
    assert steps == [("gate", "review"), ("gate", "confirm")]
    assert recs[1]["answer_hash"] == h and recs[1]["operator_id"] == "op_7" and recs[1]["episode_id"] == "ep1"
    assert g.status("ep1")["decision"] == "acknowledge"
    code, _ = g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"]), client="c1")
    assert code == 409                                                   # a decision is recorded once


def test_confirm_without_review_or_with_wrong_token_is_rejected(gate):
    g, h = gate
    assert g.submit(_req(h, step="confirm", confirm_token="x"), client="c1")[0] == 409
    _, r1 = g.submit(_req(h), client="c1")
    assert g.submit(_req(h, step="confirm", confirm_token="wrong"), client="c1")[0] == 403
    assert g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"], decision="dispute"),
                    client="c1")[0] == 409                               # confirm must match the reviewed decision
    assert g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"], op="op_8"), client="c1")[0] == 409


def test_bad_requests(gate):
    g, h = gate
    assert g.submit(_req("0" * 64), client="c1")[0] == 409               # answer changed / wrong hash
    assert g.submit({**_req(h), "episode_id": "nope"}, client="c1")[0] == 404
    assert g.submit(_req(h, decision="fire"), client="c1")[0] == 422
    assert g.submit(_req(h, op="bad id!"), client="c1")[0] == 422
    assert g.submit(_req(h, step="3"), client="c1")[0] == 422
    h2 = answer_hash("ep2", "v1", {"intent": "abstain"})
    assert g.submit({**_req(h2), "episode_id": "ep2"}, client="c1")[0] == 409   # no gate for this answer


def test_confirm_too_soon_is_rejected(tmp_path):
    g = OperatorGate(AuditLog(tmp_path), min_confirm_delay_s=30.0)
    h = answer_hash("ep1", "v1", ANS)
    g.register("ep1", h, gate_required=True)
    _, r1 = g.submit(_req(h), client="c1")
    code, body = g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"]), client="c1")
    assert code == 409 and body["code"] == "too_soon"


def test_audit_chain_verifies_and_detects_an_edit(tmp_path):
    log = AuditLog(tmp_path)
    for i in range(3):
        log.append({"kind": "question", "episode_id": f"e{i}", "n": i})
    f = next(tmp_path.glob("*.jsonl"))
    assert AuditLog.verify(f) == (True, None)
    lines = f.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["n"] = 99
    lines[1] = json.dumps(rec)
    f.write_text("\n".join(lines) + "\n")
    assert AuditLog.verify(f) == (False, 1)


def test_audit_log_is_append_only_across_instances(tmp_path):
    AuditLog(tmp_path).append({"kind": "question", "episode_id": "a"})
    AuditLog(tmp_path).append({"kind": "question", "episode_id": "b"})      # a new process continues the chain
    f = next(tmp_path.glob("*.jsonl"))
    assert len(f.read_text().splitlines()) == 2 and AuditLog.verify(f) == (True, None)


def test_trace_joins_question_and_operator_decision(gate, tmp_path):
    g, h = gate
    g.audit.append({"kind": "question", "episode_id": "ep1", "answer_hash": h})
    _, r1 = g.submit(_req(h), client="c1")
    g.submit(_req(h, step="confirm", confirm_token=r1["confirm_token"]), client="c1")
    tr = g.audit.trace("ep1")
    assert tr["question"]["answer_hash"] == h and tr["operator_decision"]["decision"] == "acknowledge"
