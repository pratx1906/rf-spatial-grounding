"""Ablation presets (plan 2026-10-28 T3, audit M13): eval/ablations.py maps a name to EnvConfig switches."""

import pytest

from eval.ablations import ABLATIONS, apply_ablation
from harness.config import EnvConfig
from harness.prompt import SYSTEM_PROMPT, static_prefix


def _names(cfg):
    return {t["function"]["name"] for t in static_prefix(cfg)[1]}


def test_every_preset_builds_a_valid_prefix():
    base = EnvConfig.load()
    for name in ABLATIONS:
        static_prefix(apply_ablation(base, name))


def test_presets():
    base = EnvConfig.load()
    assert apply_ablation(base, "none") == base
    no_rt = apply_ablation(base, "no_rt")
    assert no_rt.fit_methods_disabled == ["rt_match"] and static_prefix(no_rt)[0] == SYSTEM_PROMPT
    assert "model_trust" not in _names(apply_ablation(base, "no_trust"))
    assert {"signal_detail", "pilot_search", "tdoa_fix", "noise_floor"}.isdisjoint(
        _names(apply_ablation(base, "no_signals")))
    po = apply_ablation(base, "prompt_only_no_trust")
    assert "model_trust" in _names(po) and "model_trust" not in static_prefix(po)[0]
    on = apply_ablation(base, "with_consistency")
    assert "sensor_consistency" in _names(on)
    assert "sensor_consistency" not in _names(apply_ablation(on, "no_consistency"))
    both = apply_ablation(base, "with_consistency+no_rt")
    assert both.tools_extra == ["sensor_consistency"] and both.fit_methods_disabled == ["rt_match"]
    assert base.tools_extra == [] and base.fit_methods_disabled == []           # input never mutated


def test_no_verifier_preset():
    base = EnvConfig.load()
    off = apply_ablation(base, "no_verifier+no_rt")
    assert off.verifier_off is True and off.fit_methods_disabled == ["rt_match"] and base.verifier_off is False
    assert static_prefix(off)[0] == static_prefix(base)[0]                    # prompt unchanged


def test_unknown_preset():
    with pytest.raises(KeyError):
        apply_ablation(EnvConfig.load(), "no_such")


def test_lenient_submit_preset_and_off_intent_drop():
    from harness.answers import drop_off_intent

    cfg = apply_ablation(EnvConfig.load(), "lenient_submit")
    assert cfg.submit_drop_off_intent and cfg.max_tokens_per_turn == 1024 and cfg.max_seq_len == 19456
    assert static_prefix(cfg)[0] == SYSTEM_PROMPT                       # diagnostic only: same prompt and tools
    # the Qwen3-1.7B pattern (2026-10-29): every flat-schema field filled for a collateral answer
    raw = {"intent": "collateral", "evidence_refs": ["fp:1"], "confidence": 0.4, "fp_ref": "fp:1",
           "violations": [], "track_refs": [], "boolean": False, "protocol": "wifi", "region": {"type": "ref"},
           "position": {}, "reason": "no_sensor_coverage", "time_s": 0}
    kept, dropped = drop_off_intent(raw)
    assert set(kept) == {"intent", "evidence_refs", "confidence", "fp_ref", "violations"}
    assert sorted(dropped) == ["boolean", "position", "protocol", "reason", "region", "time_s", "track_refs"]
    assert drop_off_intent({"intent": "nonsense", "x": 1}) == ({"intent": "nonsense", "x": 1}, [])   # left to the validator


def test_env_lenient_submit_accepts_off_intent_fields():
    from harness.env import AkshaEnv
    from tasks.base import Task

    import json
    from pathlib import Path
    row = json.loads(Path("eval/frozen/v6/tasks.jsonl").read_text().splitlines()[0])
    for name, ok in (("none", False), ("lenient_submit", True)):
        env = AkshaEnv("data/site_kandla", config=apply_ablation(EnvConfig.load(), name))
        env.reset(Task.from_row(row), seed=0)
        res = env._control({"name": "submit", "arguments": {"answer": {"intent": "abstain", "reason": "insufficient_evidence",
                                                 "confidence": 0.2, "track_refs": [], "boolean": False}}}, solo=True)
        assert ("unknown field" not in json.dumps(res)) is ok
