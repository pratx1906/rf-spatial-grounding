"""RL reward v2 (paper plan P3, Revision 1 M5/B3): v1 unchanged, missed violation vs false alarm, wrong abstain only on
answerable tasks with consistent evidence, wrong-twin abstain credit, distance bonus, step cost, and adversarial
policies (always-yes, always-abstain, zero-tool, submit-first) scoring below a real expert trace."""

import math
import statistics

import pytest

from train.rewards.episode import (EpisodeStats, RewardConfig, episode_reward, evidence_inconsistent, task_terms)

V2 = RewardConfig(version="v2")


def stats(task, answer, *, verified=True, turns=4, evidence_calls=3, score=None, tool_results=None, cfg=V2,
          reason="submitted"):
    """EpisodeStats the way stats_from_env builds them, for a hand-made episode."""
    if score is None:
        score = _score(task, answer)
    terms = task_terms(task, answer, score, tool_results, cfg)
    answered = answer is not None and reason == "submitted"
    return EpisodeStats(turns=turns, max_turns=8, generated_tokens=300, max_tokens_per_turn=512,
                        valid_call_fraction=1.0, repeat_calls=0, answered=answered,
                        grounded=verified, verifier_failed_twice=reason == "verifier_failed",
                        format_failed_twice=False, unanswerable=task["area"] == "unanswerable",
                        abstained=terms["abstained"], confidence=(answer or {}).get("confidence"),
                        answer_score=score, correct=terms["correct"], position_error_m=terms["position_error_m"],
                        missed_violation=terms["missed_violation"], false_alarm=terms["false_alarm"],
                        wrong_abstain=terms["wrong_abstain"], twin_flag_credit=terms["twin_flag_credit"],
                        evidence_calls=evidence_calls)


def _score(task, answer):
    """The task score for dict answers (the env scores pydantic answers with tasks.scoring.score_answer)."""
    from harness.answers import Answer
    from pydantic import TypeAdapter

    from tasks.base import Task
    from tasks.scoring import score_answer

    if answer is None:
        return 0.0
    a = TypeAdapter(Answer).validate_python(answer)
    return score_answer(Task.from_row(task), a)


def zi(boolean, tracks=("trk:0447",), site="data/scenarios_v3/kandla_s1_X", fam="zone_incursion"):
    return {"task_id": f"t-{fam}-{boolean}", "area": "operator_query", "template": fam,
            "question": "Any UAV within the jetty zone at t=50 s?", "site": site,
            "answer_key": {"track_refs": list(tracks) if boolean else [], "boolean": boolean,
                           "zone": "zone:jetty_restricted", "t_s": 50.0}}


def oq(boolean, tracks=(), conf=0.8, confirm=False):
    return {"intent": "operator_query", "boolean": boolean, "track_refs": list(tracks), "evidence_refs": ["trk:0447"],
            "confidence": conf, "requires_human_confirm": confirm}


def abstain(reason="insufficient_evidence"):
    return {"intent": "abstain", "reason": reason, "confidence": 0.3}


DS = {"task_id": "t-ds", "area": "drone_search", "template": "ds_hover", "question": "Fly the DF drone...",
      "site": "data/scenarios_v3/kandla_s2_X",
      "answer_key": {"position": {"x_m": 1000.0, "y_m": 1000.0}, "full_m": 20.0, "tau_m": 40.0}}
UNANS = {"task_id": "t-un", "area": "unanswerable", "template": "entity_not_found", "question": "Where is jetty J99?",
         "site": "data/scenarios_v3/kandla_s2_X", "answer_key": {"reason": "entity_not_found"}}
COLL = {"task_id": "t-c", "area": "collateral", "template": "jam_protected", "question": "Which protected assets...",
        "site": "data/scenarios_v3/kandla_s2_X", "answer_key": {"violations": ["obj:jetty_j1", "obj:jetty_j2"]}}


def ds_answer(dx):
    return {"intent": "drone_search", "final_position": {"x_m": 1000.0 + dx, "y_m": 1000.0},
            "uncertainty": {"kind": "ellipse", "a_m": 30.0, "b_m": 20.0, "theta_deg": 0.0},
            "evidence_refs": ["wp:1"], "confidence": 0.7}


def coll(viol):
    return {"intent": "collateral", "violations": list(viol), "fp_ref": "fp:1", "evidence_refs": ["fp:1"],
            "confidence": 0.8, "requires_human_confirm": True}


# --- v1 unchanged ---------------------------------------------------------------------------------------------
def test_v1_is_the_default_and_unchanged():
    assert RewardConfig().version == "v1"
    s = EpisodeStats(turns=4, max_turns=8, generated_tokens=512, max_tokens_per_turn=512, valid_call_fraction=0.5,
                     repeat_calls=1, answered=True, grounded=True, verifier_failed_twice=False,
                     format_failed_twice=False, unanswerable=False, abstained=False, confidence=0.9,
                     answer_score=0.7)
    r = episode_reward(s)
    exp = {"answer": 0.7, "ground": 0.2, "valid": 0.05, "step": -0.025, "lat": -0.0025, "repeat": -0.02,
           "calib": round(0.1 * (1 - 0.2 ** 2), 4), "pen": 0.0}
    for k, v in exp.items():
        assert r[k] == pytest.approx(v, abs=1e-4), k
    assert r["total"] == pytest.approx(sum(exp.values()), abs=2e-4)
    assert set(r) == set(exp) | {"total"}          # no v2 parts leak into v1


def test_unknown_version_rejected():
    s = EpisodeStats(1, 8, 0, 512, 1.0, 0, True, True, False, False, False, False, None, 1.0)
    with pytest.raises(ValueError):
        episode_reward(s, RewardConfig(version="v9"))


# --- violations ------------------------------------------------------------------------------------------------
def test_missed_violation_costs_more_than_false_alarm():
    miss = episode_reward(stats(zi(True), oq(False)), V2)
    fa = episode_reward(stats(zi(False), oq(True, ["trk:0447"])), V2)
    assert miss["missed"] == -2.0 and fa["false_alarm"] == -1.0
    assert miss["total"] < fa["total"] < 0
    ok = episode_reward(stats(zi(True), oq(True, ["trk:0447"])), V2)
    assert ok["missed"] == 0 and ok["false_alarm"] == 0 and ok["answer"] == 1.0


def test_no_answer_on_an_incursion_counts_as_missed():
    r = episode_reward(stats(zi(True), None, verified=False, reason="max_turns"), V2)
    assert r["missed"] == -2.0


def test_collateral_partial_miss_and_false_alarm():
    t = task_terms(COLL, coll(["obj:jetty_j1", "obj:tank_farm_tf1"]), 0.5)
    assert t["missed_violation"] == 0.5 and t["false_alarm"] == 0.5
    assert task_terms(COLL, abstain(), 0.0)["missed_violation"] == 1.0


# --- abstain -----------------------------------------------------------------------------------------------------
def test_wrong_abstain_only_on_answerable():
    r = episode_reward(stats(zi(False), abstain()), V2)
    assert r["wrong_abstain"] == -0.5 and r["answer"] == 0 and r["ground"] == 0
    ok = episode_reward(stats(UNANS, abstain("entity_not_found"), evidence_calls=1), V2)
    assert ok["wrong_abstain"] == 0 and ok["answer"] == 1.0 and ok["pen"] == 0
    # an evidence-free correct abstain (actuation / future time) keeps its grounding credit
    assert episode_reward(stats(UNANS, abstain("entity_not_found"), evidence_calls=0), V2)["ground"] == 0.2


def test_wrong_abstain_costs_more_than_v1():
    s = stats(zi(False), abstain())
    assert episode_reward(s, V2)["total"] < episode_reward(s, RewardConfig())["total"] - 0.3


def test_ambiguous_family_answered_after_clarification_keeps_full_credit():
    t = zi(False, fam="zone_incursion_ambiguous")
    r = episode_reward(stats(t, oq(False)), V2)
    assert r["answer"] == 1.0 and r["wrong_abstain"] == 0 and r["false_alarm"] == 0


SUSPECT = [{"ok": True, "tool": "sensor_consistency", "data": {"suspect_refs": ["sensor:S4"]}}]


def test_wrong_twin_abstain_gets_credit_only_with_inconsistent_evidence():
    m2 = zi(True, site="data/scenarios_v3/kandla_s3_X_M2")
    r = episode_reward(stats(m2, abstain(), tool_results=SUSPECT), V2)
    assert r["twin"] == 0.5 and r["wrong_abstain"] == 0 and r["missed"] == 0
    # same world, no inconsistency shown: an ordinary wrong abstain (and a missed incursion)
    r2 = episode_reward(stats(m2, abstain(), tool_results=[]), V2)
    assert r2["twin"] == 0 and r2["wrong_abstain"] == -0.5 and r2["missed"] == -2.0
    # the M0 null control and nominal worlds never give twin credit
    m0 = zi(True, site="data/scenarios_v3/kandla_s3_X_M0")
    assert not evidence_inconsistent(m0, SUSPECT)
    # a flagged (requires_human_confirm) answer on an inconsistent twin task also earns the credit
    assert episode_reward(stats(m2, oq(False, confirm=True), tool_results=SUSPECT), V2)["twin"] == 0.5
    # an explicit task flag wins over the episode's evidence
    assert evidence_inconsistent({**m2, "evidence_inconsistent": False}, SUSPECT) is False


def test_twin_credit_is_below_a_correct_answer():
    m2 = zi(True, site="data/scenarios_v3/kandla_s3_X_M2")
    flag = episode_reward(stats(m2, abstain(), tool_results=SUSPECT), V2)["total"]
    right = episode_reward(stats(m2, oq(True, ["trk:0447"]), tool_results=SUSPECT), V2)["total"]
    assert right > flag > 0


# --- distance and step -------------------------------------------------------------------------------------------
def test_distance_bonus_and_acc50():
    near = episode_reward(stats(DS, ds_answer(40.0)), V2)
    assert near["answer"] == 1.0 and near["dist"] == pytest.approx(0.3 * math.exp(-40 / 80), abs=1e-4)
    far = episode_reward(stats(DS, ds_answer(100.0)), V2)
    assert far["answer"] == 0.0 and far["dist"] == pytest.approx(0.3 * math.exp(-100 / 80), abs=1e-4)
    off = episode_reward(stats(DS, ds_answer(40.0)), RewardConfig(version="v2", dist_bonus=False))
    assert off["dist"] == 0.0 and off["answer"] == 1.0
    # an unverified answer earns neither
    assert episode_reward(stats(DS, ds_answer(0.0), verified=False), V2)["dist"] == 0.0


def test_step_cost_per_turn():
    a = episode_reward(stats(zi(True), oq(True, ["trk:0447"]), turns=2), V2)
    b = episode_reward(stats(zi(True), oq(True, ["trk:0447"]), turns=6), V2)
    assert a["step"] == -0.02 and b["step"] == -0.06
    assert episode_reward(stats(zi(True), oq(True, ["trk:0447"]), turns=6),
                          RewardConfig(version="v2", step_cost=0.05))["step"] == -0.3


def test_reward_config_round_trips_to_dict():
    d = V2.to_dict()
    assert d["version"] == "v2" and d["penalty_missed_violation"] == 2.0 and d["w_dist"] == 0.3
    assert RewardConfig(**d) == V2


# --- adversarial policies vs a real expert trace -------------------------------------------------------------------
TASKS = [zi(True), zi(False), zi(True, tracks=("trk:1681",)), zi(False), DS, UNANS, COLL]


def expert(t):
    if t is DS:
        return stats(t, ds_answer(15.0), turns=5)
    if t is UNANS:
        return stats(t, abstain("entity_not_found"), turns=2, evidence_calls=1)
    if t is COLL:
        return stats(t, coll(["obj:jetty_j1", "obj:jetty_j2"]), turns=3)
    k = t["answer_key"]
    return stats(t, oq(k["boolean"], k["track_refs"]), turns=4)


def always_yes(t):
    if t["area"] == "operator_query":
        return stats(t, oq(True, ["trk:0447"]), turns=3)
    return expert(t)                                   # generous: perfect elsewhere


def always_abstain(t):
    return stats(t, abstain("entity_not_found" if t is UNANS else "insufficient_evidence"), turns=1,
                 evidence_calls=0)


def zero_tool(t):
    """Submits a plausible answer with no evidence call: the verifier rejects it twice (refs never returned)."""
    ans = {"intent": "operator_query", "boolean": True, "track_refs": ["trk:0447"], "evidence_refs": ["trk:0447"],
           "confidence": 1.0} if t["area"] == "operator_query" else expert_answer(t)
    return stats(t, ans, verified=False, turns=2, evidence_calls=0, reason="verifier_failed")


def submit_first(t):
    """Guesses the answer in turn 1 (correct key, but unverifiable): same verifier outcome as zero-tool."""
    return stats(t, expert_answer(t), verified=False, turns=2, evidence_calls=0, reason="verifier_failed")


def expert_answer(t):
    if t is DS:
        return ds_answer(15.0)
    if t is UNANS:
        return abstain("entity_not_found")
    if t is COLL:
        return coll(["obj:jetty_j1", "obj:jetty_j2"])
    k = t["answer_key"]
    return oq(k["boolean"], k["track_refs"])


@pytest.mark.parametrize("policy", [always_yes, always_abstain, zero_tool, submit_first])
def test_adversarial_policies_score_below_expert(policy):
    exp = statistics.mean(episode_reward(expert(t), V2)["total"] for t in TASKS)
    adv = statistics.mean(episode_reward(policy(t), V2)["total"] for t in TASKS)
    assert exp > 1.0
    assert adv < exp - 0.4, (policy.__name__, adv, exp)


def test_always_abstain_is_negative_on_answerable():
    rs = [episode_reward(always_abstain(t), V2)["total"] for t in TASKS if t is not UNANS]
    assert max(rs) < 0


# --- AUDIT_CODE C2 / C7 ------------------------------------------------------------------------------------------
LOC = {"task_id": "t-loc", "area": "df_ambiguity", "template": "locate", "question": "Where is trk:0447 at t=50 s?",
       "site": "data/scenarios_v3/kandla_s2_X",
       "answer_key": {"los_hyp_refs": ["hyp:S1-1@50"], "position": {"x_m": 500.0, "y_m": 500.0}, "full_m": 20.0,
                      "tau_m": 25.0}}


def loc(pos=None):
    a = {"intent": "df_ambiguity", "chosen_hyp_refs": ["hyp:S1-1@50"], "evidence_refs": ["hyp:S1-1@50"],
         "confidence": 0.8}
    if pos is not None:
        a["position"] = {"x_m": 500.0 + pos, "y_m": 500.0}
    return a


def test_locate_without_position_is_wrong_even_when_hyps_are_right():
    s_none = stats(LOC, loc(None))
    assert s_none.answer_score >= 0.99                 # tasks.scoring still gives hyp F1 alone (C2)
    r = episode_reward(s_none, V2)
    assert r["answer"] == 0.0 and r["dist"] == 0.0
    good = episode_reward(stats(LOC, loc(10.0)), V2)
    poor = episode_reward(stats(LOC, loc(200.0)), V2)
    assert good["answer"] == 1.0 and poor["answer"] == 0.0 and poor["dist"] > 0
    assert good["total"] > poor["total"] > r["total"]  # leaving the position out never beats giving a poor one


def test_verifier_rejected_answer_gets_no_correctness_credit():
    k = zi(True)["answer_key"]
    s = stats(zi(True), oq(True, k["track_refs"]), verified=False, reason="verifier_failed")
    assert s.answer_score == 1.0                       # the env still scores the rejected answer (C7)
    r = episode_reward(s, V2)
    assert r["answer"] == 0.0 and r["ground"] == 0.0 and r["pen"] == -1.0 and r["total"] < 0
    d = episode_reward(stats(DS, ds_answer(0.0), verified=False, reason="verifier_failed"), V2)
    assert d["answer"] == 0.0 and d["dist"] == 0.0
