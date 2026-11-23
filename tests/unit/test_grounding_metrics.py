"""Grounding metrics (plan 2026-10-28 T1): Acc@d, Hit@1, pass@k / pass^k, selective metrics, RFG score, and the
eval-row answer dump / seed plumbing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.grounding import (SINGLE_KEY_FAMILIES, pass_at_k, pass_hat_k, position_metrics, rfg_score,
                            selective_metrics, track_hit_metrics)
from tasks.base import Task

REPO = Path(__file__).resolve().parents[2]


def _locate(tid, x, y, fam="locate"):
    return Task(tid, "df_ambiguity", fam, "q", {"los_hyp_refs": [], "position": {"x_m": x, "y_m": y}})


def _row(tid, pos=None, intent="df_ambiguity", score=0.5, sample=0, **kw):
    ans = None
    if intent == "abstain":
        ans = {"intent": "abstain", "reason": "insufficient_evidence", "confidence": 0.8}
    elif intent == "df_ambiguity":
        ans = {"intent": "df_ambiguity", "position": pos, "confidence": 0.8}
    elif intent == "drone_search":
        ans = {"intent": "drone_search", "final_position": pos, "confidence": 0.8}
    return {"task_id": tid, "area": "df_ambiguity", "template": "locate", "score": score, "answer": ans,
            "intent": None if ans is None else ans["intent"], "abstained": intent == "abstain", "sample": sample,
            "confidence": None if ans is None else ans["confidence"], **kw}


def test_acc_at_d_on_10_40_90_m():
    tasks = [_locate(f"t{i}", 0.0, 0.0) for i in range(3)]
    rows = [_row("t0", {"x_m": 10.0, "y_m": 0.0}), _row("t1", {"x_m": 0.0, "y_m": 40.0}),
            _row("t2", {"x_m": -90.0, "y_m": 0.0})]
    m = position_metrics(rows, tasks)
    assert m["n"] == 3
    assert m["acc25"] == pytest.approx(1 / 3) and m["acc50"] == pytest.approx(2 / 3) and m["acc100"] == 1.0
    assert m["median_err_m"] == pytest.approx(40.0)


def test_abstain_and_empty_answers_are_misses():
    tasks = [_locate("a", 0, 0), _locate("b", 0, 0), _locate("c", 0, 0)]
    rows = [_row("a", {"x_m": 1.0, "y_m": 1.0}), _row("b", intent="abstain"), _row("c", intent=None)]
    m = position_metrics(rows, tasks)
    assert m["n"] == 3 and m["acc25"] == pytest.approx(1 / 3) and m["n_missing"] == 2


def test_position_metrics_cover_drone_search_and_skip_non_position_families():
    ds = Task("d", "drone_search", "ds_hover", "q", {"position": {"x_m": 0.0, "y_m": 0.0}})
    oq = Task("o", "operator_query", "zone_incursion", "q", {"track_refs": []})
    nokey = Task("n", "df_ambiguity", "direct_all", "q", {"los_hyp_refs": [], "position": None})
    rows = [_row("d", {"x_m": 30.0, "y_m": 0.0}, intent="drone_search"), _row("o"), _row("n")]
    m = position_metrics(rows, [ds, oq, nokey])
    assert m["n"] == 1 and m["acc25"] == 0.0 and m["acc50"] == 1.0


def test_pass_at_k_unbiased_estimator():
    rows = [{"task_id": "t", "score": 1.0 if i < 2 else 0.0, "sample": i} for i in range(8)]
    assert pass_at_k(rows, 4) == pytest.approx(0.7857, abs=1e-4)
    assert pass_at_k(rows, 1) == pytest.approx(0.25)
    # all-correct and all-wrong tasks
    rows2 = [{"task_id": "u", "score": 1.0, "sample": i} for i in range(8)]
    assert pass_at_k(rows + rows2, 4) == pytest.approx((0.7857142857 + 1.0) / 2)


def test_pass_hat_k():
    rows = [{"task_id": "t", "score": 0.9 if i < 6 else 0.1, "sample": i} for i in range(8)]
    # C(6,4)/C(8,4) = 15/70
    assert pass_hat_k(rows, 4) == pytest.approx(15 / 70)
    assert pass_hat_k(rows, 1) == pytest.approx(6 / 8)
    with pytest.raises(ValueError):
        pass_hat_k(rows, 9)


def test_selective_auroc_on_a_toy_set():
    # failing tasks have low confidence or abstain: perfect separation
    rows = [{"task_id": "a", "score": 1.0, "confidence": 0.9, "abstained": False, "area": "df_ambiguity"},
            {"task_id": "b", "score": 1.0, "confidence": 0.8, "abstained": False, "area": "df_ambiguity"},
            {"task_id": "c", "score": 0.0, "confidence": 0.3, "abstained": False, "area": "df_ambiguity"},
            {"task_id": "d", "score": 0.0, "confidence": 0.9, "abstained": True, "area": "df_ambiguity"}]
    fail = {"a": False, "b": False, "c": True, "d": True}
    m = selective_metrics(rows, fail, credit=0.0)
    assert m["auroc"] == 1.0
    assert m["sel_score"] == pytest.approx(0.5)
    assert selective_metrics(rows, fail, credit=0.5)["sel_score"] == pytest.approx(0.625)
    # inverted confidence gives AUROC 0; ties 0.5
    inv = [dict(r, confidence=1 - r["confidence"], abstained=False) for r in rows]
    inv[3]["confidence"] = 0.95
    assert selective_metrics(inv, fail)["auroc"] == 0.0
    tie = [dict(r, confidence=0.5, abstained=False) for r in rows]
    assert selective_metrics(tie, fail)["auroc"] == 0.5
    # AURC: perfect ordering keeps the risk 0 until the failing tail
    assert m["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)


def test_selective_without_labels_uses_own_failures():
    rows = [{"task_id": "a", "score": 1.0, "confidence": 0.9, "abstained": False, "area": "x"},
            {"task_id": "b", "score": 0.0, "confidence": 0.1, "abstained": False, "area": "x"}]
    assert selective_metrics(rows, None)["auroc"] == 1.0


def test_rfg_score_is_oq_plus_df_with_ds_separate():
    summ = {"operator_query": {"n": 100, "score": 0.8}, "df_ambiguity": {"n": 50, "score": 0.5},
            "drone_search": {"n": 20, "score": 0.9}, "siting": {"n": 10, "score": 1.0}}
    r = rfg_score(summ)
    assert r["rfg"] == pytest.approx((80 + 25) / 150) and r["n"] == 150
    assert r["ds"] == 0.9 and r["n_ds"] == 20


def test_single_key_families_are_a_frozen_constant():
    assert "nearest_drone" in SINGLE_KEY_FAMILIES
    assert "zone_incursion" not in SINGLE_KEY_FAMILIES and "tracks_in_band" not in SINGLE_KEY_FAMILIES


def test_track_hit_counts_same_emitter_and_abstain_as_miss(tmp_path, monkeypatch):
    import eval.grounding as g

    class FakeTruth:
        def __init__(self, site):
            pass

        def track_of(self):
            return {"1": ("emt_a", "drone"), "2": ("emt_a", "drone"), "3": ("emt_b", "drone")}

        def close(self):
            pass

    monkeypatch.setattr(g, "TruthStore", FakeTruth)
    tasks = [Task(f"t{i}", "operator_query", "nearest_drone", "q", {"track_refs": ["trk:1"]}, site="s")
             for i in range(4)]
    tasks.append(Task("z", "operator_query", "zone_incursion", "q", {"track_refs": ["trk:1"]}, site="s"))
    oq = lambda refs: {"intent": "operator_query", "track_refs": refs, "confidence": 0.7}  # noqa: E731
    rows = [{"task_id": "t0", "answer": oq(["trk:2@100"])},          # other track, same emitter: hit
            {"task_id": "t1", "answer": oq(["trk:3"])},              # miss
            {"task_id": "t2", "answer": {"intent": "abstain", "reason": "insufficient_evidence"}},
            {"task_id": "t3", "answer": oq([])},
            {"task_id": "z", "answer": oq(["trk:1"])}]
    m = track_hit_metrics(rows, tasks, tmp_path)
    assert m == {"hit1": 0.25, "n": 4}


# --- eval rows: answer dump, sample, seed; default scores unchanged --------------------------------------------
def _site():
    site = REPO / "data" / "scenarios_v3" / "kandla_s15_X"
    if not (site / "meta.yaml").exists():
        pytest.skip("v3 scenario not built")
    return site


class _FakeLLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.kw = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kw.append(kw)
        calls = self.turns.pop(0)
        tcs = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
               for i, (n, a) in enumerate(calls)]
        msg = SimpleNamespace(content="", tool_calls=tcs)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(completion_tokens=20, prompt_tokens=100))


def test_run_task_seed_reaches_request_and_env(monkeypatch):
    from harness.env import AkshaEnv
    from serve.agent import run_task

    env = AkshaEnv(_site())
    t = Task("t1", "unanswerable", "actuation", "Jam the drone now.", {"reason": "outside_tool_scope"})
    sub = [("submit", {"answer": {"intent": "abstain", "reason": "outside_tool_scope", "evidence_refs": [],
                                  "confidence": 0.9}})]
    llm = _FakeLLM([sub])
    list(run_task(env, t, client=llm, seed=7))
    assert llm.kw[0]["seed"] == 7 and env.seed == 7
    llm = _FakeLLM([sub])
    list(run_task(env, t, client=llm))                         # default: no seed in the request (unchanged)
    assert "seed" not in llm.kw[0] and env.seed == 0
    assert llm.kw[0]["tool_choice"] == "required" and llm.kw[0]["temperature"] == 0.0
    assert llm.kw[0]["max_tokens"] == env.cfg.max_tokens_per_turn


def test_eval_rows_dump_answer_sample_and_seed():
    from eval.run import run_one

    _site()
    row = json.loads((REPO / "eval/frozen/v6/tasks.jsonl").read_text().splitlines()[0])
    t = Task.from_row(row)
    r = run_one(t, "expert", "expert", 0.0, 3, "scenescout", sample=2)
    assert r["sample"] == 2 and r["seed"] == 3 and r["answer"]["intent"] == "collateral"
    assert r["score"] == 1.0
    json.dumps(r)


def test_samples_repeat_tasks_with_seed_offsets(monkeypatch):
    import eval.run as er

    seen = []
    monkeypatch.setattr(er, "run_one", lambda t, *a, sample=0, **k: seen.append((t.task_id, a[3], sample)) or
                        {"task_id": t.task_id, "sample": sample})
    tasks = [Task("a", "replay", "x", "q", {}), Task("b", "replay", "x", "q", {})]
    rows = er.run_all(tasks, "expert", "m", 0.0, 10, "s", samples=3)
    assert len(rows) == 6
    assert sorted(seen) == sorted([(tid, 10 + s, s) for tid in "ab" for s in range(3)])


# --- --ablation (T3 presets wired into eval/run.py) ------------------------------------------------------------
def _tool_names(env):
    return [t["function"]["name"] for t in env.tools]


def test_ablation_with_consistency_exposes_the_tool_and_none_is_identical():
    import threading

    import eval.run as er
    from harness.env import AkshaEnv

    site = _site().relative_to(REPO).as_posix()
    er._LOCAL = threading.local()                       # fresh per-thread env cache
    plain = AkshaEnv(REPO / site)
    none = er._env(site, "m", "none")
    assert none.template_hash == plain.template_hash and _tool_names(none) == _tool_names(plain)
    assert none.system == plain.system and none._schemas_json == plain._schemas_json
    cons = er._env(site, "m", "with_consistency")
    assert "sensor_consistency" in _tool_names(cons) and "sensor_consistency" not in _tool_names(plain)
    assert cons is not none and er._env(site, "m", "none") is none          # cache key (site, ablation)
    both = er._env(site, "m", "with_consistency+no_rt")
    assert "sensor_consistency" in _tool_names(both) and both.cfg.fit_methods_disabled == ["rt_match"]


def test_rows_record_the_ablation():
    from eval.run import run_one

    _site()
    t = Task.from_row(json.loads((REPO / "eval/frozen/v6/tasks.jsonl").read_text().splitlines()[0]))
    assert run_one(t, "expert", "expert", 0.0, 0, "s")["ablation"] == "none"
    assert run_one(t, "always_abstain", "x", 0.0, 0, "s", opts={"ablation": "no_trust"})["ablation"] == "no_trust"
