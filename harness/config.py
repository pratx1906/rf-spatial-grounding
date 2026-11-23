"""Episode configuration with the context-budget invariant (SPEC 7.1, D-002) enforced at construction."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

REPO = Path(__file__).resolve().parents[1]


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    site: str = "stub_site"
    max_turns: int = 8
    max_tokens_per_turn: int = 512
    max_obs_tokens: int = 3000
    per_tool_output_tokens: int = 1000
    max_question_tokens: int = 200
    template_overhead_tokens: int = 400
    static_prefix_tokens: int = 6900
    max_seq_len: int = 15360
    parallel_tool_calls: int = 3
    max_ask_user: int = 1
    max_format_retries: int = 1
    max_verifier_retries: int = 1
    tool_timeout_ms: dict[str, int] = {"default": 250, "sql_readonly": 2000, "fit_candidates": 1500, "weak_zones": 800,
                                       "suggest_node": 800, "effect_footprint": 800, "coverage_query": 800,
                                       "sensor_consistency": 1500}
    max_wallclock_s: float = 15.0
    chars_per_token: float = 3.5
    # Tool switches for ablations (plan 2026-10-28 T3). All empty = the byte-identical default prefix. Names are
    # checked against the registry when the prefix is built (harness.prompt.static_prefix).
    tools_disabled: list[str] = []          # removed from the schemas + their playbook clauses; calls -> NOT_PERMITTED
    tools_extra: list[str] = []             # opt-in tools to expose (e.g. sensor_consistency) + their playbook line
    playbook_disabled: list[str] = []       # prompt-only control: drop the playbook clauses, keep the tool
    fit_methods_disabled: list[str] = []    # fit_candidates methods that answer NOT_PERMITTED (e.g. rt_match)
    verifier_off: bool = False              # ablation (paper Revision 1, M8): the verifier still runs and is logged,
                                            # but a failed check no longer rejects the submit
    # T-FIX (docs/paper/TWIN_FIX_PREREG.md): T1 rewrites the sensor_consistency playbook line ("fix": exclude or
    # de-bias, then answer; default "abstain" is the historical line); T2 asks for a sensor-health check before a
    # track or DF answer is accepted (one redirecting hint per episode) and memoises the check per scenario
    consistency_line: str = "abstain"
    require_health_check: bool = False
    submit_drop_off_intent: bool = False    # diagnostic (P2, 2026-10-29): drop submit fields the declared intent
                                            # does not define instead of rejecting them (harness.answers)

    @model_validator(mode="after")
    def _context_budget(self):
        total = (self.static_prefix_tokens + self.max_question_tokens + self.template_overhead_tokens
                 + self.max_obs_tokens + self.max_turns * self.max_tokens_per_turn)
        if total > self.max_seq_len:
            raise ValueError(f"context budget {total} exceeds max_seq_len {self.max_seq_len} (SPEC 7.1, D-002)")
        return self

    def tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 1

    def timeout_ms(self, tool: str) -> int:
        return self.tool_timeout_ms.get(tool, self.tool_timeout_ms["default"])

    @classmethod
    def load(cls, path: Path | None = None, **overrides) -> "EnvConfig":
        path = path or REPO / "configs" / "env" / "default.yaml"
        data = yaml.safe_load(path.read_text())["env"]
        data.update(overrides)
        return cls(**data)
