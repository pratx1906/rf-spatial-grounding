import pytest

from harness.answers import AnswerOperatorQuery
from harness.config import EnvConfig
from harness.ledger import Ledger
from harness.verifier import verify
from tasks.base import Task

Q = Task("t-1", "operator_query", "zone_incursion", "Is any drone inside the Cargo Area at t=100 s?",
         {"track_refs": [], "boolean": False})


def test_context_budget_invariant():
    EnvConfig()                                           # defaults fit (D-002)
    with pytest.raises(ValueError, match="context budget"):
        EnvConfig(max_obs_tokens=6000)                    # the original SPEC value overflowed 8192


def test_invalid_args_give_unit_hint(env):
    env.reset(Q)
    _, _, _, info = env.step({"tool_calls": [{"name": "region_contents", "arguments": {
        "region": {"type": "circle", "center": {"x_m": 0, "y_m": 0}, "radius_km": 1}}}]})
    err = info["results"][0]["error"]
    assert err["code"] == "INVALID_ARGS" and any("radius_m" in f["hint"] or "_m (metres)" in f["hint"]
                                                 for f in err["fields"])


def test_repeat_calls_are_cached_and_counted(env):
    env.reset(Q)
    call = {"name": "site_resolve", "arguments": {"query": "Cargo Area"}}
    env.step({"tool_calls": [call]})
    _, _, _, info = env.step({"tool_calls": [call]})
    assert "repeat_call" in info["results"][0]["warnings"] and env.executor.repeat_calls == 1


def test_parallel_call_limit(env):
    env.reset(Q)
    calls = [{"name": "site_describe", "arguments": {"ref": f"obj:berth_{i}"}} for i in range(1, 5)]
    _, _, _, info = env.step({"tool_calls": calls})
    assert [r["ok"] for r in info["results"][:3]] == [True] * 3
    assert info["results"][3]["error"]["code"] == "BUDGET_EXCEEDED"


def test_ledger_delta_is_appended_once(env):
    env.reset(Q)
    msgs, _, _, _ = env.step({"tool_calls": [{"name": "site_resolve", "arguments": {"query": "Cargo Area"}}]})
    assert "LEDGER+ resolved:" in msgs[-1]["content"]
    msgs, _, _, _ = env.step({"tool_calls": [{"name": "site_describe", "arguments": {"ref": "obj:berth_1"}}]})
    assert "LEDGER+" not in msgs[-1]["content"]           # nothing new worth recording


def test_format_error_retry_then_terminate(env):
    env.reset(Q)
    msgs, _, done, _ = env.step({"text": "I think there is a drone."})
    assert not done and "Format error" in msgs[0]["content"]
    _, parts, done, _ = env.step({"text": "Still no tool call."})
    assert done and parts["pen"] <= -0.5


def test_ask_user_once(env):
    env.reset(Q)
    ask = {"name": "ask_user", "arguments": {"question": "Which zone?"}}
    _, _, _, i1 = env.step({"tool_calls": [ask]})
    _, _, _, i2 = env.step({"tool_calls": [ask]})
    assert i1["results"][0]["ok"] and i2["results"][0]["error"]["code"] == "BUDGET_EXCEEDED"


def test_ungrounded_answer_rejected_then_penalised(env):
    env.reset(Q)
    sub = {"name": "submit", "arguments": {"answer": {"intent": "operator_query", "track_refs": ["trk:1234"],
                                                      "evidence_refs": ["trk:1234"], "confidence": 0.9}}}
    _, _, done, info = env.step({"tool_calls": [sub]})
    assert not done and info["results"][0]["error"]["code"] == "UNGROUNDED"
    _, parts, done, _ = env.step({"tool_calls": [sub]})
    assert done and parts["pen"] <= -1.0


def test_verifier_off_accepts_but_logs(stub_dir):
    from harness.env import AkshaEnv

    env = AkshaEnv(stub_dir, config=EnvConfig(verifier_off=True))
    env.reset(Q)
    sub = {"name": "submit", "arguments": {"answer": {"intent": "operator_query", "track_refs": ["trk:1234"],
                                                      "evidence_refs": ["trk:1234"], "confidence": 0.9}}}
    _, _, done, info = env.step({"tool_calls": [sub]})
    assert done and info["results"][0]["ok"] and env.verifier_result["code"] == "UNGROUNDED"


def test_no_tool_answer_fails_grounding():
    led = Ledger()
    ans = AnswerOperatorQuery(intent="operator_query", track_refs=[], evidence_refs=["zone:cargo_area"],
                              confidence=0.5)
    assert verify(ans, led, lambda *_: {}).code == "UNGROUNDED"


def test_submit_must_be_alone(env):
    env.reset(Q)
    _, _, _, info = env.step({"tool_calls": [
        {"name": "site_resolve", "arguments": {"query": "Cargo Area"}},
        {"name": "submit", "arguments": {"answer": {"intent": "abstain", "reason": "insufficient_evidence",
                                                    "confidence": 0.2}}}]})
    assert info["results"][1]["error"]["code"] == "INVALID_ARGS" and not env.done


def test_always_abstain_scores_below_answering(env, tasks):
    """Anti-hacking (SPEC 9.6): abstaining on answerable questions must not pay."""
    from tasks.expert import solve

    answerable = [t for t in tasks if t.area != "unanswerable"][:10]
    abstain, expert = [], []
    for t in answerable:
        env.reset(t)
        env.step({"tool_calls": [{"name": "submit", "arguments": {"answer": {
            "intent": "abstain", "reason": "insufficient_evidence", "confidence": 0.5}}}]})
        abstain.append(env.reward_parts["total"])
        env.reset(t)
        solve(env, t)
        expert.append(env.reward_parts["total"])
    assert sum(abstain) / len(abstain) < sum(expert) / len(expert) - 0.5


def test_stateful_call_invalidates_the_dedupe_cache():
    """move_drone changes the world: the same plan_waypoints call afterwards must be recomputed, not replayed."""
    from pathlib import Path

    import pytest

    from harness.env import AkshaEnv
    from tasks.base import Task

    site = Path(__file__).resolve().parents[2] / "data" / "site_kandla"
    if not (site / "meta.yaml").exists():
        pytest.skip("Kandla site not built")
    pay = {"target_track": "trk:0000", "t0_s": 0.0, "start": {"x_m": 0, "y_m": 0, "z_m": 80}, "budget_moves": 3,
           "prior": {"x_m": 500, "y_m": 500, "a_m": 200, "b_m": 100, "theta_deg": 30},
           "waypoints": [{"id": f"wp:0{i}", "x_m": 500 + 300 * i, "y_m": 0, "z_m": 80, "bearing_deg": 0.0,
                          "snr_db": 20} for i in range(1, 4)]}
    env = AkshaEnv(site)
    env.reset(Task("t", "drone_search", "ds_hover", "q", {}, drone_payload=pay))
    plan = {"name": "plan_waypoints", "arguments": {"target_ref": "trk:0000", "k": 1}}
    first = env.step({"tool_calls": [plan]})[3]["results"][0]["data"]["waypoints"][0]["ref"]
    env.step({"tool_calls": [{"name": "move_drone", "arguments": {"wp_ref": first}}]})
    again = env.step({"tool_calls": [plan]})[3]["results"][0]
    assert "repeat_call" not in again["warnings"] and again["data"]["waypoints"][0]["ref"] != first


def test_a_crashing_tool_returns_an_error_instead_of_killing_the_episode(env, monkeypatch):
    """S2.5: noise_floor raised TypeError on v2 data and the exception killed the whole eval process."""
    from tools.registry import REGISTRY

    def boom(ctx, args):
        raise TypeError("'<' not supported between instances of 'float' and 'NoneType'")

    import dataclasses

    monkeypatch.setitem(REGISTRY, "site_resolve", dataclasses.replace(REGISTRY["site_resolve"], fn=boom))
    env.reset(Q)
    _, _, _, info = env.step({"tool_calls": [{"name": "site_resolve", "arguments": {"query": "Cargo Area"}}]})
    r = info["results"][0]
    assert r["ok"] is False and r["error"]["code"] == "INTERNAL_ERROR"
