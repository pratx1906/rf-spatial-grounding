"""Scoring (answer key only) and the answer gate for every area (SPEC 6.3, 7.4, 9.6)."""

import pytest
from pydantic import ValidationError
from shapely.geometry import box

from harness.ledger import Ledger
from harness.verifier import verify
from tasks.base import Task
from tasks.scoring import score_answer
from tools.control import SubmitArgs


def ans(**kw):
    return SubmitArgs.model_validate({"answer": {"confidence": 0.8, **kw}}).answer


def task(area, key):
    return Task("t", area, "x", "q", key)


def test_drone_search_scores_by_distance():
    t = task("drone_search", {"position": {"x_m": 100.0, "y_m": 200.0}, "tau_m": 30.0})
    good = ans(intent="drone_search", final_position={"x_m": 100.0, "y_m": 200.0},
               uncertainty={"kind": "ellipse", "a_m": 20, "b_m": 10, "theta_deg": 0}, evidence_refs=["wp:01"])
    far = ans(intent="drone_search", final_position={"x_m": 190.0, "y_m": 200.0},
              uncertainty={"kind": "ellipse", "a_m": 20, "b_m": 10, "theta_deg": 0}, evidence_refs=["wp:01"])
    assert score_answer(t, good) == 1.0 and 0.0 < score_answer(t, far) < 0.1


def test_siting_scores_zone_overlap_and_node_hits():
    t = task("siting", {"zones_wkt": [box(0, 0, 100, 100).wkt], "node_refs": ["nodecand:b1", "nodecand:b2"]})
    a = ans(intent="siting", zones=[{"type": "ref", "ref": "ras:wz0_aa"}], node_candidates=["nodecand:b1"],
            evidence_refs=["ras:wz0_aa"])
    resolved = {"ras:wz0_aa": box(0, 0, 100, 100)}
    assert score_answer(t, a, resolved) == 1.0
    half = {"ras:wz0_aa": box(0, 0, 50, 100)}
    assert 0.6 < score_answer(t, a, half) < 0.9
    nodes_only = task("siting", {"node_refs": ["nodecand:b1"]})
    assert score_answer(nodes_only, ans(intent="siting", node_candidates=["nodecand:b9"], evidence_refs=["x:1"])) == 0


def test_collateral_recall_weighted():
    t = task("collateral", {"violations": ["obj:a", "fe:b", "zone:c"], "fp_any": True})
    full = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a", "fe:b", "zone:c"], evidence_refs=["fp:1"])
    miss = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a"], evidence_refs=["fp:1"])
    extra = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a", "fe:b", "zone:c", "obj:z"],
                evidence_refs=["fp:1"])
    assert score_answer(t, full) == 1.0
    assert score_answer(t, miss) < score_answer(t, extra) < 1.0          # missing a victim costs more


def test_replay_scores_events_and_time():
    t = task("replay", {"evt_refs": ["evt:00012"], "time_s": 120.0})
    a = ans(intent="replay", evt_refs=["evt:00012"], time_s=120.0, evidence_refs=["evt:00012"])
    late = ans(intent="replay", evt_refs=["evt:00012"], time_s=150.0, evidence_refs=["evt:00012"])
    assert score_answer(t, a) == 1.0 and 0.5 < score_answer(t, late) < 0.6


def test_abstain_on_answerable_scores_zero_everywhere():
    ab = ans(intent="abstain", reason="insufficient_evidence")
    for area, key in [("drone_search", {"position": {"x_m": 0, "y_m": 0}}), ("siting", {"node_refs": ["nodecand:x"]}),
                      ("collateral", {"violations": []}), ("replay", {"evt_refs": ["evt:1"]})]:
        assert score_answer(task(area, key), ab) == 0.0


def test_collateral_without_human_confirm_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        ans(intent="collateral", fp_ref="fp:1", violations=[], evidence_refs=["fp:1"], requires_human_confirm=False)


def test_verifier_rejects_violations_not_checked_in_the_episode():
    led = Ledger()
    led.register(["fp:1", "obj:a", "obj:b"], "collateral_check", 1)
    a = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a", "obj:b"], evidence_refs=["fp:1"])
    res = verify(a, led, lambda *_: {}, episode={"collateral": {"fp:1": ["obj:a"]}})
    assert not res.ok and res.code == "INCONSISTENT"
    ok = verify(ans(intent="collateral", fp_ref="fp:1", violations=["obj:a"], evidence_refs=["fp:1"]), led,
                lambda *_: {}, episode={"collateral": {"fp:1": ["obj:a"]}})
    assert ok.ok


def test_verifier_rejects_drone_position_far_from_the_belief():
    from tools.drone import DroneSim

    sim = DroneSim({"target_track": "trk:1", "t0_s": 0, "start": {"x_m": 0, "y_m": 0, "z_m": 80},
                    "prior": {"x_m": 500, "y_m": 500, "a_m": 60, "b_m": 40, "theta_deg": 0}, "waypoints": []})
    led = Ledger()
    led.register(["trk:1"], "drone_state", 1)
    far = ans(intent="drone_search", final_position={"x_m": 900, "y_m": 900},
              uncertainty={"kind": "ellipse", "a_m": 60, "b_m": 40, "theta_deg": 0}, evidence_refs=["trk:1"])
    res = verify(far, led, lambda *_: {}, episode={"drone": sim})
    assert not res.ok and res.code == "INCONSISTENT"
