"""Paper Revision 2 (AUDIT_CODE C2, C3, C5-C8 and minors): versioned scoring, offline re-scoring, pairing keys,
deterministic tool budget, set hashes, fair baselines, truth guard extensions."""

import json
import time
from pathlib import Path

import pytest

from tasks.base import Task
from tasks.scoring import (empty_scores_full, is_headline, score_answer, score_answer_v2, score_versioned)
from tools.control import SubmitArgs

REPO = Path(__file__).resolve().parents[2]


def ans(**kw):
    return SubmitArgs.model_validate({"answer": {"confidence": 0.8, **kw}}).answer


DF_KEY = {"los_hyp_refs": ["hyp:a", "hyp:b"], "position": {"x_m": 0.0, "y_m": 0.0}, "full_m": 20.0, "tau_m": 25.0}


def df_task(key=DF_KEY):
    return Task("k15X3-locate-00001", "df_ambiguity", "locate", "q", key)


# --- C2 -------------------------------------------------------------------------------------------------------
def test_v2_zeroes_the_position_half_of_a_positionless_locate_answer():
    t = df_task()
    no_pos = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:a", "hyp:b"], evidence_refs=["hyp:a"])
    assert score_answer(t, no_pos) == 1.0                  # v1: omitting the position scored full
    assert score_answer_v2(t, no_pos) == 0.5               # v2: error = infinity on the position half
    assert score_versioned(t, no_pos, version="v1") == 1.0 and score_versioned(t, no_pos, version="v2") == 0.5
    far = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:a", "hyp:b"], evidence_refs=["hyp:a"],
              position={"x_m": 60.0, "y_m": 0.0})
    assert score_answer_v2(t, far) == score_answer(t, far) > score_answer_v2(t, no_pos)
    with pytest.raises(ValueError):
        score_versioned(t, far, version="v9")


def test_v2_equals_v1_without_a_position_key():
    t = df_task({"los_hyp_refs": ["hyp:a"]})
    a = ans(intent="df_ambiguity", chosen_hyp_refs=["hyp:a"], evidence_refs=["hyp:a"])
    assert score_answer_v2(t, a) == score_answer(t, a) == 1.0


# --- C6 -------------------------------------------------------------------------------------------------------
def test_headline_mask_drops_empty_keys_and_tool_fidelity_areas():
    empty = Task("t", "operator_query", "near_object", "q", {"track_refs": [], "boolean": False})
    pos = Task("t", "operator_query", "near_object", "q", {"track_refs": ["trk:1"], "boolean": True})
    rep = Task("t", "replay", "entered_between", "q", {"evt_refs": []})
    coll = Task("t", "collateral", "jam_links", "q", {"violations": ["obj:a"]})
    assert empty_scores_full(empty) and not is_headline(empty)
    assert not empty_scores_full(pos) and is_headline(pos)
    assert empty_scores_full(rep) and not is_headline(rep)
    assert not is_headline(coll)
    assert is_headline(df_task()) and is_headline(Task("t", "unanswerable", "actuation", "q",
                                                       {"reason": "outside_tool_scope"}))


# --- eval/rescore ---------------------------------------------------------------------------------------------
def _row(tid, score, answer="missing", **kw):
    r = {"task_id": tid, "score": score, "area": "df_ambiguity", "template": "locate", "end_reason": "submitted",
         "intent": "df_ambiguity", "verifier_code": None, "verifier_ok": True, **kw}
    if answer != "missing":
        r["answer"] = answer
    return r


def test_rescore_rows_v2_strict_and_bounds():
    from eval.rescore import _block, rescore_rows

    tasks = {"k15X3-locate-00001": df_task(), "k16X3-locate-00002": df_task()}
    tasks["k16X3-locate-00002"].task_id = "k16X3-locate-00002"
    a = {"intent": "df_ambiguity", "chosen_hyp_refs": ["hyp:a", "hyp:b"], "evidence_refs": ["hyp:a"],
         "confidence": 0.8}
    rows = [_row("k15X3-locate-00001", 1.0, a),
            _row("k16X3-locate-00002", 1.0, a, end_reason="verifier_failed", verifier_code="UNGROUNDED")]
    rs = rescore_rows(rows, tasks)
    assert [x["v1"] for x in rs] == [1.0, 1.0] and [x["v2"] for x in rs] == [0.5, 0.5]
    assert [x["v2_strict"] for x in rs] == [0.5, 0.0] and rs[0]["c2_applied"] and rs[1]["verifier_rejected"]
    assert [x["seed"] for x in rs] == ["15", "16"]
    b = _block(rs)
    assert b["score_v1"]["mean"] == 1.0 and b["score_v2"]["mean"] == 0.5
    assert b["score_v2_verifier_strict"]["mean"] == 0.25 and b["score_v2"]["n_clusters"] == 2
    # rows saved without answers: v2 unknown where C2 may apply -> bounds [0.5 v1, v1]
    old = rescore_rows([_row("k15X3-locate-00001", 0.8)], tasks)
    assert old[0]["v2"] is None and (old[0]["v2_lo"], old[0]["v2_hi"]) == (0.4, 0.8)
    ob = _block(old)
    assert ob["score_v2"] is None and ob["score_v2_bounds"] == [0.4, 0.8]


def test_rescore_verifier_off_rows_count_as_rejected():
    from eval.rescore import _verifier_rejected

    assert _verifier_rejected({"end_reason": "submitted", "intent": "operator_query", "verifier_code": "UNGROUNDED"})
    assert not _verifier_rejected({"end_reason": "submitted", "intent": "operator_query", "verifier_code": None})
    assert not _verifier_rejected({"end_reason": "max_turns", "intent": None, "verifier_code": "UNGROUNDED"})


def test_nominal_partner_by_name(tmp_path):
    from eval.rescore import nominal_partner, out_path

    (tmp_path / "expert_v6_Npair").mkdir()
    (tmp_path / "expert_v6_Npair" / "eval.json").write_text("{}")
    (tmp_path / "expert_v6_M2c").mkdir()
    p = tmp_path / "expert_v6_M2c" / "eval.json"
    assert nominal_partner(p, {}) == tmp_path / "expert_v6_Npair" / "eval.json"
    assert nominal_partner(tmp_path / "expert_v6" / "eval.json", {}) is None
    assert out_path(p).name == "rescore.json" and out_path(tmp_path / "a_t7.json").name == "a_t7.rescore.json"


# --- eval.run minors + C8 -------------------------------------------------------------------------------------
def test_paired_bootstrap_keeps_every_sample_and_reports_drops():
    from eval.run import paired_bootstrap, paired_cluster_bootstrap

    a = [{"task_id": "k1-x-1", "area": "a", "score": 0.0, "sample": s} for s in range(3)]
    b = [{"task_id": "k1-x-1", "area": "a", "score": 1.0, "sample": s} for s in range(2)]
    pb = paired_bootstrap(a, b, n=50)
    assert pb["_all"]["n"] == 2 and pb["_all"]["n_dropped"] == 1 and pb["_all"]["diff"] == 1.0
    assert paired_cluster_bootstrap(a, b, n=50)["_all"]["n_dropped"] == 1
    one = [{"task_id": f"k1-x-{i}", "area": "a", "score": i % 2} for i in range(6)]     # no sample field: as before
    assert paired_bootstrap(one, one, n=50)["_all"]["n"] == 6


def test_set_hash_check(tmp_path, capsys):
    from eval.run import check_set_hash, set_sha256

    p = tmp_path / "tasks.jsonl"
    p.write_text("{}\n")
    h = set_sha256(p)
    assert check_set_hash(p, {"tasks_sha256": h})["status"] == "ok"
    assert check_set_hash(p, {"tasks_sha256": "0" * 64})["status"] == "mismatch"
    assert "WARNING" in capsys.readouterr().err
    assert check_set_hash(p, {})["status"] == "no_manifest_hash"
    with pytest.raises(SystemExit):
        check_set_hash(p, {"tasks_sha256": "0" * 64}, enforce=True)


def test_frozen_paper_sets_carry_their_hash():
    from eval.run import set_sha256

    for n in ("v6", "v6_W4", "v6_W5", "heldout_v1"):
        d = REPO / "eval" / "frozen" / n
        assert json.loads((d / "manifest.json").read_text())["tasks_sha256"] == set_sha256(d / "tasks.jsonl")


def test_eval_config_deterministic_timeouts():
    from eval.run import DETERMINISTIC_TIMEOUT_MS, eval_config

    d = eval_config()
    assert d.timeout_ms("track_search") == 250                        # default unchanged
    hi = eval_config("none", DETERMINISTIC_TIMEOUT_MS)
    assert {hi.timeout_ms(k) for k in ("track_search", "fit_candidates", "anything")} == {DETERMINISTIC_TIMEOUT_MS}
    assert eval_config("no_rt", 5000).fit_methods_disabled == ["rt_match"]


def _slow_run(delay):
    from harness.executor import Executor

    orig = Executor._run

    def run(self, spec, args):
        time.sleep(delay)
        return orig(self, spec, args)
    return run


def test_executor_logs_timeouts(stub_dir, monkeypatch):
    from eval.run import eval_config, timeout_summary
    from harness.env import AkshaEnv
    from harness.executor import Executor

    q = Task("t-1", "operator_query", "zone_incursion", "q", {"track_refs": [], "boolean": False})
    monkeypatch.setattr(Executor, "_run", _slow_run(0.4))
    env = AkshaEnv(stub_dir)                                       # default 250 ms: times out
    env.reset(q)
    _, _, _, info = env.step({"tool_calls": [{"name": "site_resolve", "arguments": {"query": "Cargo Area"}}]})
    assert info["results"][0]["error"]["code"] == "TIMEOUT"
    assert env.executor.timeouts == [{"tool": "site_resolve", "turn": 1, "timeout_ms": 250}]
    det = AkshaEnv(stub_dir, config=eval_config("none", 60_000))   # deterministic mode: no timeout
    det.reset(q)
    _, _, _, info = det.step({"tool_calls": [{"name": "site_resolve", "arguments": {"query": "Cargo Area"}}]})
    assert info["results"][0]["ok"] and det.executor.timeouts == []
    s = timeout_summary([{"n_timeouts": 1, "timeouts": env.executor.timeouts}, {"n_timeouts": 0, "timeouts": []},
                         {"score": 1}])
    assert s == {"n_rows_logged": 2, "n_timeouts": 1, "rows_with_timeout": 1, "per_tool": {"site_resolve": 1}}


# --- fair baselines -------------------------------------------------------------------------------------------
def test_always_abstain_oracle_reason(monkeypatch):
    import eval.trivial as tv

    got = []
    monkeypatch.setattr(tv, "abstain", lambda env, reason=tv.ABSTAIN_REASON: got.append(reason))
    for tmpl, area in (("actuation", "unanswerable"), ("future_time", "unanswerable"),
                       ("unknown_track", "unanswerable"), ("locate", "df_ambiguity")):
        tv.solve(None, Task("t", area, tmpl, "q", {"reason": "WRONG"}), "always_abstain_oracle_reason")
    assert got == ["outside_tool_scope", "no_sensor_coverage", "entity_not_found", "insufficient_evidence"]
    from eval.run import CLASSICAL, TRIVIAL

    assert "always_abstain_oracle_reason" in TRIVIAL and "residual_check" in CLASSICAL


def test_residual_check_flags_on_suspects(monkeypatch):
    import eval.classical as cl

    calls, out = [], []
    sus = {"v": ["sensor:S2"]}

    def fake_call(env, name, why="", **args):
        calls.append((name, args))
        return {"ok": True, "data": {"suspect_refs": sus["v"]}}

    monkeypatch.setattr(cl, "_call", fake_call)
    monkeypatch.setattr(cl, "abstain", lambda env, *a: out.append("abstain"))
    monkeypatch.setattr(cl, "tracker_only", lambda env, task: out.append("tracker"))
    t = Task("t", "operator_query", "zone_incursion", "q", {"track_refs": []}, params={"t": 90.0})
    cl.solve(None, t, "residual_check")
    sus["v"] = []
    cl.solve(None, t, "residual_check")
    assert out == ["abstain", "tracker"] and calls[0] == ("sensor_consistency", {"at_time_s": 90.0})


# --- C3 + act/abstain (eval/pairing) --------------------------------------------------------------------------
def _pair_sets(m_flag: bool):
    def task(tid, world, key_b):
        return {"task_id": tid, "template": "zone_incursion", "area": "operator_query", "pair_id": f"s{tid[1:3]}-zi-000",
                "pair_part": "boolean", "world_variant": world, "answer_key": {"track_refs": [], "boolean": key_b},
                "site": "x", "params": {"t": 30.0}}

    nt = {"k15N": task("k15N", "N", True), "k16N": task("k16N", "N", True)}
    mt = {"k15M": task("k15M", "M2", False), "k16M": task("k16M", "M2", False)}
    nt["k16N"]["pair_id"] = mt["k16M"]["pair_id"] = "s16-zi-000"
    nt["k15N"]["pair_id"] = mt["k15M"]["pair_id"] = "s15-zi-000"
    yes = {"intent": "operator_query", "track_refs": ["trk:1"], "boolean": True}
    m_ans = {"intent": "abstain", "reason": "insufficient_evidence"} if m_flag else \
        {"intent": "operator_query", "track_refs": [], "boolean": False}
    nr = {"rows": [{"task_id": k, "score": 1.0, "template": "zone_incursion", "answer": yes} for k in nt]}
    mr = {"rows": [{"task_id": k, "score": 0.0 if m_flag else 1.0, "template": "zone_incursion", "answer": m_ans}
                   for k in mt]}
    return nr, nt, mr, mt


def test_act_abstain_pairs():
    from eval.pairing import flagged, paired_report

    assert flagged({"intent": "abstain"}) and flagged({"intent": "operator_query", "requires_human_confirm": True})
    assert not flagged({"intent": "collateral", "requires_human_confirm": True}) and not flagged(None)
    rep = paired_report(*_pair_sets(m_flag=True))
    aa = rep["act_abstain"]
    assert aa["pairs_ok"] == 2 and aa["rate"] == 1.0 and aa["m_twin_world"] and aa["n_clusters"] == 2
    assert rep["twin_fail"] == 2                         # the old count is unchanged (score-based)
    rep2 = paired_report(*_pair_sets(m_flag=False))
    assert rep2["act_abstain"]["pairs_ok"] == 0 and rep2["twin_fail"] == 0 and "world_key" not in rep2


def test_world_key_judges_on_true_emitters():
    from eval.pairing import key_agrees, paired_report, part_ok_world

    class FakeWorld:
        def key(self, task):                             # one true drone inside in both worlds
            return {"emitters": ["e1"], "boolean": True, "count": 1, "nearest": "e1"}

        def track_emitter(self, site, ref):
            return "e1"

    nr, nt, mr, mt = _pair_sets(m_flag=False)            # M key says "no" (tracker lost the drone); answer "no"
    rep = paired_report(nr, nt, mr, mt, world=FakeWorld())
    wk = rep["world_key"]
    assert rep["part_twin_fail"] == 0 and wk["part_twin_fail"] == 2      # hidden by the tracker key, seen here
    assert wk["key_disagree_m"] == 2 and wk["key_disagree_n"] == 0       # the M tracker key lost the drone
    t = nt["k15N"]
    assert part_ok_world(t, {"intent": "operator_query", "boolean": True}, FakeWorld().key(t), None)
    assert not key_agrees(mt["k15M"], FakeWorld().key(t), None)


# --- C5 -------------------------------------------------------------------------------------------------------
def test_heldout_mask_and_disjoint():
    from scripts.heldout_disjoint import disjoint, mask

    assert mask("Task D1 to pin down trk:2738; report  its position.") == \
        mask("Task D1 to pin down trk:91; report its position.")
    assert mask("Within 500 m of Tank farm TF-4 at t=410 s?") == mask("Within 200 m of Tank farm TF-6 at t=30 s?")
    train = {mask("Where is trk:12 at t=30 s?")}
    keep, drop = disjoint([{"question": "Where is trk:99 at t=130 s?"}, {"question": "When did trk:3 start?"}], train)
    assert [r["question"] for r in drop] == ["Where is trk:99 at t=130 s?"] and len(keep) == 1


# --- minors: pins, geometry label -----------------------------------------------------------------------------
def test_model_pin_recorded():
    from scripts.model_pins import revision

    assert revision("Qwen/Qwen3-4B-Instruct-2507") == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert revision("nope/none") is None


def test_geometry_label_is_blender_for_v2_stacks():
    from envbuild.site_kandla import geometry_label

    assert geometry_label("v1", "x").startswith("OpenStreetMap footprints")
    assert geometry_label("v2", "data/site_kandla/rf_scene_blender_rfr").startswith("Blender world")
    import yaml

    for p in sorted((REPO / "data" / "scenarios_v3").glob("*/meta.yaml"))[:5]:
        assert yaml.safe_load(p.read_text())["fidelity"]["geometry"].startswith("Blender world")
