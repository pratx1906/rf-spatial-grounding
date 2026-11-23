"""S5.2 / S5.4 verifier rules (demo-v2 plan): collateral completeness, region/zone/position grounding, chosen hyps vs
fit_candidates explained_as, RF-ambiguity abstention policy, per-rule results for the audit trace."""

from harness.ledger import Ledger
from harness.verifier import verify
from tools.control import SubmitArgs


def ans(**kw):
    return SubmitArgs.model_validate({"answer": {"confidence": 0.8, **kw}}).answer


def noop(*_):
    return {}


# --- collateral completeness (P0 #2) -----------------------------------------------------------------------------
def test_collateral_under_reporting_is_rejected_with_every_missing_ref():
    led = Ledger()
    led.register(["fp:1", "obj:a", "fe:b"], "collateral_check", 1)
    a = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a"], evidence_refs=["fp:1"])
    res = verify(a, led, noop, episode={"collateral": {"fp:1": ["obj:a", "fe:b"]}})
    assert not res.ok and res.code == "INCOMPLETE_COLLATERAL"
    assert res.details["missing"] == ["fe:b"]


def test_collateral_empty_list_is_rejected_when_the_check_found_violations():
    led = Ledger()
    led.register(["fp:1", "obj:a"], "collateral_check", 1)
    a = ans(intent="collateral", fp_ref="fp:1", violations=[], evidence_refs=["fp:1"])
    res = verify(a, led, noop, episode={"collateral": {"fp:1": ["obj:a"]}})
    assert not res.ok and res.code == "INCOMPLETE_COLLATERAL"


def test_collateral_no_20_item_cap_in_the_verdict():
    """collateral_check shows the model 20 rows; the verdict uses the full list and names every missing ref."""
    full = [f"obj:t{i:02d}" for i in range(25)]
    led = Ledger()
    led.register(["fp:1"] + full[:20], "collateral_check", 1)           # what the capped tool output registered
    none = ans(intent="collateral", fp_ref="fp:1", violations=[], evidence_refs=["fp:1"])
    res = verify(none, led, noop, episode={"collateral": {"fp:1": full}})
    assert res.code == "INCOMPLETE_COLLATERAL" and res.details["missing"] == full
    allv = ans(intent="collateral", fp_ref="fp:1", violations=full, evidence_refs=["fp:1"])
    assert verify(allv, led, noop, episode={"collateral": {"fp:1": full}}).ok      # refs past 20 count as grounded


def test_collateral_complete_and_empty_answers_pass():
    led = Ledger()
    led.register(["fp:1", "obj:a"], "collateral_check", 1)
    ok = ans(intent="collateral", fp_ref="fp:1", violations=["obj:a"], evidence_refs=["fp:1"])
    assert verify(ok, led, noop, episode={"collateral": {"fp:1": ["obj:a"]}}).ok
    empty = ans(intent="collateral", fp_ref="fp:1", violations=[], evidence_refs=["fp:1"])
    assert verify(empty, led, noop, episode={"collateral": {"fp:1": []}}).ok


# --- grounding of region, zone and position (P1 #5) ----------------------------------------------------------------
def test_operator_query_region_ref_must_be_grounded():
    led = Ledger()
    led.register(["trk:1"], "track_search", 1)
    a = ans(intent="operator_query", track_refs=["trk:1"], region={"type": "ref", "ref": "zone:invented"},
            evidence_refs=["trk:1"])
    res = verify(a, led, noop)
    assert not res.ok and res.code == "UNGROUNDED" and "zone:invented" in res.details["refs"]


def test_siting_zone_refs_must_be_grounded():
    led = Ledger()
    led.register(["nodecand:a"], "suggest_node", 1)
    a = ans(intent="siting", zones=[{"type": "ref", "ref": "ras:made_up"}], node_candidates=["nodecand:a"],
            evidence_refs=["nodecand:a"])
    res = verify(a, led, noop)
    assert not res.ok and res.code == "UNGROUNDED" and "ras:made_up" in res.details["refs"]


def _df_ledger(chosen_kind: str = "reflection", pos=(100.0, 100.0)):
    led = Ledger()
    hyps = ["hyp:S1_a@100", "hyp:S1_b@100", "hyp:S2_c@100"]
    led.register(["trk:1"] + hyps + ["cand:0"], "df_hypotheses", 1)
    led.record("df_hypotheses", {"track_ref": "trk:1", "at_time_s": 100}, {"ok": True, "data": {
        "track": "trk:1", "t_s": 100, "sensors": [
            {"sensor": "sensor:S1", "hyps": [{"ref": hyps[0], "bearing_deg": 10}, {"ref": hyps[1], "bearing_deg": 50}]},
            {"sensor": "sensor:S2", "hyps": [{"ref": hyps[2], "bearing_deg": 200}]}]}})
    led.record("fit_candidates", {"hyp_refs": hyps, "method": "rt_match"}, {"ok": True, "data": {
        "method": "rt_match", "t_s": 100, "band": "ism_2400", "candidates": [
            {"ref": "cand:0", "position": {"x_m": pos[0], "y_m": pos[1]},
             "uncertainty": {"kind": "ellipse", "a_m": 30.0, "b_m": 20.0, "theta_deg": 0.0},
             "hyps": [{"ref": hyps[0], "residual_deg": 1.0, "explained_as": chosen_kind},
                      {"ref": hyps[1], "residual_deg": 2.0, "explained_as": "direct"},
                      {"ref": hyps[2], "residual_deg": 1.5, "explained_as": "direct"}]}]}})
    return led


def test_chosen_reflection_over_a_direct_hyp_on_the_same_node_is_inconsistent():
    led = _df_ledger()
    bad = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_a@100", "hyp:S2_c@100"],
              rejected_hyp_refs=["hyp:S1_b@100"], evidence_refs=["trk:1"])
    res = verify(bad, led, noop)
    assert not res.ok and res.code == "INCONSISTENT" and res.details["refs"] == ["hyp:S1_a@100"]
    good = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100", "hyp:S2_c@100"],
               rejected_hyp_refs=["hyp:S1_a@100"], evidence_refs=["trk:1"])
    assert verify(good, led, noop).ok


def test_chosen_hyp_without_a_direct_alternative_passes():
    """Fallback (no direct bearing on that node): choosing the best-residual reflection is not contradicted."""
    led = _df_ledger(chosen_kind="unexplained")
    a = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_a@100"], rejected_hyp_refs=[], evidence_refs=["trk:1"])
    assert verify(a, led, noop).ok


def test_df_position_must_be_near_a_fitted_candidate():
    led = _df_ledger(chosen_kind="direct")
    far = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100"], position={"x_m": 900.0, "y_m": 900.0},
              evidence_refs=["trk:1"])
    res = verify(far, led, noop)
    assert not res.ok and res.code == "INCONSISTENT" and "position" in res.message
    near = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100"], position={"x_m": 120.0, "y_m": 90.0},
               evidence_refs=["trk:1"])
    assert verify(near, led, noop).ok


# --- RF-ambiguity abstention policy (S5.4) -------------------------------------------------------------------------
def _rows(s1=(0.3, 0.5), s2=(0.9, 0.8)):
    return lambda track, t: [{"sensor": "sensor:S1", "trust": s1[0], "ghost_risk": s1[1]},
                             {"sensor": "sensor:S2", "trust": s2[0], "ghost_risk": s2[1]},
                             {"sensor": "sensor:S3", "trust": 0.95, "ghost_risk": 0.0}]   # did not hear it


def test_rf_ambiguous_df_answer_must_abstain_or_widen():
    led = _df_ledger(chosen_kind="direct")
    sure = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100", "hyp:S2_c@100"],
               position={"x_m": 100.0, "y_m": 100.0},
               uncertainty={"kind": "ellipse", "a_m": 30, "b_m": 20, "theta_deg": 0}, evidence_refs=["trk:1"])
    res = verify(sure, led, noop, rf_rows=_rows())
    assert not res.ok and res.code == "RF_AMBIGUOUS"
    assert "abstain" in res.details["hint"].lower() and res.details["nodes"] == ["sensor:S1", "sensor:S2"]
    wide = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100", "hyp:S2_c@100"],
               position={"x_m": 100.0, "y_m": 100.0}, confidence=0.45,
               uncertainty={"kind": "ellipse", "a_m": 150, "b_m": 100, "theta_deg": 0}, evidence_refs=["trk:1"])
    assert verify(wide, led, noop, rf_rows=_rows()).ok
    narrow = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100"], position={"x_m": 100.0, "y_m": 100.0},
                 confidence=0.45, uncertainty={"kind": "ellipse", "a_m": 40, "b_m": 20, "theta_deg": 0},
                 evidence_refs=["trk:1"])
    assert verify(narrow, led, noop, rf_rows=_rows()).code == "RF_AMBIGUOUS"
    no_pos = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100"], confidence=0.5, evidence_refs=["trk:1"])
    assert verify(no_pos, led, noop, rf_rows=_rows()).ok


def test_rf_rule_quiet_when_one_hearing_node_is_trustworthy():
    led = _df_ledger(chosen_kind="direct")
    sure = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:S1_b@100", "hyp:S2_c@100"], evidence_refs=["trk:1"])
    assert verify(sure, led, noop, rf_rows=_rows(s2=(0.9, 0.5))).ok
    assert verify(sure, led, noop, rf_rows=lambda *_: []).ok            # no maps: no ruling


def test_rf_ambiguity_function_thresholds():
    from harness.abstain import rf_ambiguity

    rows = [{"sensor": "S1", "trust": 0.39, "ghost_risk": 0.0}, {"sensor": "sensor:S2", "trust": 0.8, "ghost_risk": 0.75}]
    r = rf_ambiguity(rows, {"S1", "S2"})
    assert r["ambiguous"] and r["nodes"] == ["sensor:S1", "sensor:S2"]
    assert not rf_ambiguity(rows, {"S1", "S2", "S4"})["ambiguous"]       # S4 heard it but has no map value
    assert not rf_ambiguity([{"sensor": "S1", "trust": 0.4, "ghost_risk": 0.74}], {"S1"})["ambiguous"]


# --- per-rule results (S5.5 audit) -----------------------------------------------------------------------------------
def test_every_rule_reports_a_result():
    led = Ledger()
    led.register(["fp:1", "obj:a"], "collateral_check", 1)
    a = ans(intent="collateral", fp_ref="fp:1", violations=[], evidence_refs=["fp:1"])
    res = verify(a, led, noop, episode={"collateral": {"fp:1": ["obj:a"]}})
    by = {r["rule"]: r for r in res.rules}
    assert by["grounding"]["ok"] and not by["collateral_complete"]["ok"]
    assert {"grounding", "recompute_inside_zone", "df_consistency", "collateral_complete", "drone_belief",
            "rf_ambiguity", "reference_resolution", "human_confirm"} <= set(by)
    assert "rules" not in res.to_dict()          # the model-visible error envelope is unchanged
