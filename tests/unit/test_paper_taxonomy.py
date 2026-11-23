"""P9 failure taxonomy rules and behaviour helpers (eval/paper/taxonomy.py, behaviour.py)."""

from eval.paper.behaviour import ngrams, verify_before_submit
from eval.paper.taxonomy import classify

DF = {"area": "df_ambiguity", "template": "direct_all", "answer_key": {"los_hyp_refs": ["hyp:a@1", "hyp:b@1"]}}
OQ = {"area": "operator_query", "template": "zone_incursion",
      "answer_key": {"track_refs": ["trk:1"], "boolean": True, "zone": "zone:z"}}


def _row(**kw):
    return {"end_reason": "submitted", "intent": kw.get("intent", "df_ambiguity"), "abstained": False, "error": None,
            **kw}


def test_end_reasons_first():
    assert classify(_row(end_reason="format_failed"), DF) == "parse_format"
    assert classify(_row(end_reason="max_turns", intent=None), DF) == "budget_exhausted"
    assert classify(_row(end_reason="verifier_failed"), DF) == "verifier_rejected"


def test_abstain_rules():
    assert classify(_row(abstained=True, intent="abstain"), DF) == "over_abstain"
    un = {"area": "unanswerable", "template": "actuation", "answer_key": {"reason": "outside_tool_scope"}}
    assert classify(_row(intent="operator_query"), un) == "missed_abstain"


def test_df_rules_and_no_dump():
    assert classify(_row(), DF) == "no_answer_dump"
    assert classify(_row(answer={"chosen_hyp_refs": ["hyp:a@1", "hyp:c@1"]}), DF) == "ghost_picked"
    assert classify(_row(answer={"chosen_hyp_refs": ["hyp:a@1"]}), DF) == "missed_direct"


def test_operator_query_rules():
    r = _row(intent="operator_query", answer={"track_refs": ["trk:1"], "boolean": False})
    assert classify(r, OQ) == "wrong_zone"
    r = _row(intent="operator_query", answer={"track_refs": ["trk:2"], "boolean": True})
    assert classify(r, OQ) == "wrong_track"


def test_behaviour_helpers():
    assert verify_before_submit(["track_detail", "model_trust", "submit"])
    assert not verify_before_submit(["track_detail", "submit", "model_trust"])
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
