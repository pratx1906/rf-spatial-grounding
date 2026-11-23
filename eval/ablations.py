"""Test-time tool ablations (plan 2026-10-28 T3 / E9; audit M13). Each preset edits EnvConfig switches only:

    cfg = apply_ablation(EnvConfig.load(), "no_trust")          # or "with_consistency+no_rt"
    env = AkshaEnv(site_dir, config=cfg)

- no_rt: fit_candidates(method=rt_match) answers NOT_PERMITTED; triangulate is kept and the prompt is unchanged
  (the playbook still names rt_match, so the agent meets the knockout through the error hint).
- no_trust / no_signals: tools removed from the schemas, playbook clauses naming them dropped.
- prompt_only_<x>: the control for <x>: the same playbook clauses dropped, tools kept (separates "lost the tool"
  from "saw a prompt it was not trained on"). The family S signal tools are not named in the playbook, so
  prompt_only_no_signals equals the default prompt; no_rt has no prompt edit, so it has no prompt-only variant.
- with_consistency / no_consistency: expose / hide the opt-in sensor_consistency tool (and its playbook line);
  prompt_only_with_consistency exposes the tool without the playbook line.
- no_verifier: the submit-time evidence verifier still runs and is logged (verifier_code), but a failed check no
  longer rejects the answer (EnvConfig.verifier_off; paper plan Revision 1, M8). The prompt is unchanged.
- lenient_submit: DIAGNOSTIC, not a benchmark setting (P2, 2026-10-29). Submit fields the declared intent does not
  define are dropped instead of rejected, and a turn may use 1024 tokens (context 19456; the server needs
  --max-model-len >= 19456). Separates "cannot follow the flat submit schema" (Qwen3-1.7B fills all 21 fields:
  unknown-field rejections, submits cut at 512 tokens) from "cannot do the task". Prompt and tools are unchanged.
These are test-time knockouts, not Urbex-style training-time ablations: the adapters were trained with every tool.
"""

from __future__ import annotations

from harness.config import EnvConfig

SIGNAL_TOOLS = ["signal_detail", "pilot_search", "tdoa_fix", "noise_floor"]

# name -> {config key: (add, remove)}
ABLATIONS: dict[str, dict[str, tuple[list[str], list[str]]]] = {
    "none": {},
    "no_rt": {"fit_methods_disabled": (["rt_match"], [])},
    "no_trust": {"tools_disabled": (["model_trust"], [])},
    "no_signals": {"tools_disabled": (SIGNAL_TOOLS, [])},
    "with_consistency": {"tools_extra": (["sensor_consistency"], [])},
    "no_consistency": {"tools_extra": ([], ["sensor_consistency"]),
                       "playbook_disabled": ([], ["sensor_consistency"])},
    "no_verifier": {"verifier_off": True},
    # T-FIX (docs/paper/TWIN_FIX_PREREG.md): T1 = with_consistency+fix_line, T2 = T1+health_gate
    "fix_line": {"consistency_line": "fix"},
    "health_gate": {"require_health_check": True},
    "lenient_submit": {"submit_drop_off_intent": True, "max_tokens_per_turn": 1024, "max_seq_len": 19456},
    # step budget (P7 test-time compute): the default is 8 turns; each extra turn needs max_tokens_per_turn (512) more
    # context (serve with --max-model-len >= max_seq_len)
    "turns4": {"max_turns": 4},
    "turns12": {"max_turns": 12, "max_seq_len": 17408},
    "turns16": {"max_turns": 16, "max_seq_len": 19456},
    "prompt_only_no_trust": {"playbook_disabled": (["model_trust"], [])},
    "prompt_only_no_signals": {"playbook_disabled": (SIGNAL_TOOLS, [])},
    "prompt_only_with_consistency": {"tools_extra": (["sensor_consistency"], []),
                                     "playbook_disabled": (["sensor_consistency"], [])},
}


def apply_ablation(cfg: EnvConfig, name: str) -> EnvConfig:
    """A new EnvConfig with the preset(s) applied; `name` may join presets with '+'. Raises KeyError if unknown."""
    data = cfg.model_dump()
    for part in name.split("+"):
        for key, spec in ABLATIONS[part.strip()].items():
            if not isinstance(spec, tuple):          # scalar switch (e.g. verifier_off): set it
                data[key] = spec
                continue
            add, remove = spec
            cur = [t for t in data[key] if t not in remove]
            data[key] = cur + [t for t in add if t not in cur]
    return EnvConfig(**data)
