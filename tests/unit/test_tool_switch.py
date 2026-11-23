"""Opt-in / knockout tool switches (plan 2026-10-28 T3): the default prefix must stay byte-identical, disabled tools
vanish from the schemas and answer NOT_PERMITTED, prompt-only edits keep the tool, rt_match can be knocked out alone."""

import hashlib
import json

import pytest

from harness.config import EnvConfig
from harness.prompt import SYSTEM_PROMPT, static_prefix

# Pinned 2026-10-28 from the prefix BEFORE the switch code existed (31 tools). Every trained adapter saw these bytes.
PREFIX_SHA = "486e593785bda2a3a256432bb736c6adb66a253a3e7ae90a843463c8797fdffd"      # system + schemas_json
TOOLS_SHA = "320933cb142ccebfb6983604f5970f50755715aefd5a96787ff95d85572eb8de"       # tool list as sent to vLLM


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _hashes(system, tools, schemas):
    return _sha(system + schemas), _sha(json.dumps(tools, separators=(",", ":"))), len(tools)


def test_default_prefix_byte_identical():
    assert _hashes(*static_prefix()) == (PREFIX_SHA, TOOLS_SHA, 31)
    assert _hashes(*static_prefix(EnvConfig.load())) == (PREFIX_SHA, TOOLS_SHA, 31)
    assert _hashes(*static_prefix(EnvConfig())) == (PREFIX_SHA, TOOLS_SHA, 31)


def test_default_env_prefix_byte_identical(env):
    assert _hashes(env.system, env.tools, env._schemas_json) == (PREFIX_SHA, TOOLS_SHA, 31)
    assert "sensor_consistency" not in {t["function"]["name"] for t in env.tools}


def _names(tools):
    return {t["function"]["name"] for t in tools}


def _call(env, task, name, args):
    env.reset(task)
    _, _, _, info = env.step({"tool_calls": [{"name": name, "arguments": args}]})
    return info["results"][0]


def test_disabled_tool_absent_and_not_permitted(stub_dir, tasks):
    from harness.env import AkshaEnv

    env = AkshaEnv(stub_dir, config=EnvConfig.load(tools_disabled=["model_trust"]))
    assert "model_trust" not in _names(env.tools) and len(env.tools) == 30
    assert "model_trust" not in env.system
    assert "Coverage / siting: weak_zones" in env.system          # only the clause naming the tool is dropped
    assert env.template_hash != _sha(SYSTEM_PROMPT)[:16]
    e = _call(env, tasks[0], "model_trust", {"point": {"x_m": 0, "y_m": 0}, "band": "ism_2400"})
    assert not e["ok"] and e["error"]["code"] == "NOT_PERMITTED" and e["error"]["hint"]


def test_disabled_tool_line_dropped_when_recipe_names_it():
    system, tools, _ = static_prefix(EnvConfig.load(tools_disabled=["effect_footprint", "collateral_check"]))
    assert "Collateral of an effector" not in system and "effect_footprint" not in system
    assert {"effect_footprint", "collateral_check"}.isdisjoint(_names(tools))
    # nothing outside the playbook section changes
    head, tail = SYSTEM_PROMPT.split("Playbooks:\n")[0], SYSTEM_PROMPT.split("submit.answer fields")[1]
    assert system.startswith(head) and system.endswith(tail)


def test_prompt_only_keeps_tool(stub_dir, tasks):
    from harness.env import AkshaEnv

    env = AkshaEnv(stub_dir, config=EnvConfig.load(playbook_disabled=["model_trust"]))
    assert "model_trust" in _names(env.tools) and len(env.tools) == 31
    assert "model_trust" not in env.system
    e = _call(env, tasks[0], "model_trust", {"point": {"x_m": 0, "y_m": 0}, "band": "ism_2400"})
    assert e["ok"] or e["error"]["code"] != "NOT_PERMITTED"


def test_rt_match_rejected_triangulate_kept(stub_dir, tasks):
    from harness.env import AkshaEnv

    env = AkshaEnv(stub_dir, config=EnvConfig.load(fit_methods_disabled=["rt_match"]))
    assert "fit_candidates" in _names(env.tools)
    assert _hashes(env.system, env.tools, env._schemas_json) == (PREFIX_SHA, TOOLS_SHA, 31)   # tool-level knockout only
    e = _call(env, tasks[0], "fit_candidates", {"hyp_refs": ["hyp:a", "hyp:b"], "method": "rt_match"})
    assert not e["ok"] and e["error"]["code"] == "NOT_PERMITTED" and "triangulate" in e["error"]["hint"]
    e = _call(env, tasks[0], "fit_candidates", {"hyp_refs": ["hyp:a", "hyp:b"], "method": "triangulate"})
    assert not e["ok"] and e["error"]["code"] != "NOT_PERMITTED"         # fake refs, but the method is allowed


def test_opt_in_tool_off_by_default_on_when_listed(stub_dir, tasks, env):
    from harness.env import AkshaEnv

    e = _call(env, tasks[0], "sensor_consistency", {})
    assert not e["ok"] and e["error"]["code"] == "NOT_PERMITTED"
    on = AkshaEnv(stub_dir, config=EnvConfig.load(tools_extra=["sensor_consistency"]))
    assert "sensor_consistency" in _names(on.tools) and len(on.tools) == 32
    assert "sensor_consistency(" in on.system and "sensor_consistency" not in SYSTEM_PROMPT
    # prompt-only control for an opt-in tool: exposed, but no playbook line
    po = AkshaEnv(stub_dir, config=EnvConfig.load(tools_extra=["sensor_consistency"],
                                                  playbook_disabled=["sensor_consistency"]))
    assert "sensor_consistency" in _names(po.tools) and po.system == SYSTEM_PROMPT


@pytest.mark.parametrize("kw", [{"tools_disabled": ["no_such_tool"]}, {"tools_extra": ["model_trust"]},
                                {"tools_disabled": ["submit"]}, {"fit_methods_disabled": ["magic"]},
                                {"playbook_disabled": ["no_such_tool"]}])
def test_bad_switches_rejected(kw):
    with pytest.raises(ValueError):
        static_prefix(EnvConfig.load(**kw))
