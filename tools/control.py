"""Family H: control tools (SPEC 6.2 H). Executed by the env, not by the executor (they change episode state)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from harness.answers import Answer
from tools.registry import register_control
from tools.types import Strict

Intent = Literal["operator_query", "df_ambiguity", "drone_search", "siting", "collateral", "replay", "compound",
                 "out_of_scope"]


class DeclareIntentArgs(Strict):
    intent: Intent


class AskUserArgs(Strict):
    question: Annotated[str, Field(min_length=3, max_length=200)]
    options: Annotated[list[Annotated[str, Field(max_length=80)]], Field(max_length=6)] | None = None


class SubmitArgs(Strict):
    answer: Answer

    @model_validator(mode="before")
    @classmethod
    def _lift(cls, v):
        """Accept answer fields placed next to `answer` (e.g. confidence) by moving them inside it."""
        if isinstance(v, dict) and set(v) - {"answer"}:
            inner = dict(v.get("answer") or {})
            for k, val in v.items():
                if k != "answer":
                    inner.setdefault(k, val)
            v = {"answer": inner}
        return v


register_control("declare_intent", "Declare the question's capability area (optional; used for routing/logs).",
                 DeclareIntentArgs)
register_control("ask_user", "Ask the operator one clarifying question (max 1 per episode).", AskUserArgs)
# The Pydantic union inlines to a very large schema. The model sees one FLAT object listing every answer field
# once (per-intent usage is in the system prompt). It must be complete: under constrained decoding (SPEC 10.2) the
# model can only emit fields the schema declares. Validation still uses the full union with field-level hints.
_REFS = {"type": "array", "items": {"type": "string"}}
_PT = {"type": "object", "properties": {"x_m": {"type": "number"}, "y_m": {"type": "number"}}}
SUBMIT_SCHEMA = {
    "type": "object", "required": ["answer"],
    "properties": {"answer": {"type": "object", "required": ["intent", "evidence_refs", "confidence"], "properties": {
        "intent": {"enum": ["operator_query", "df_ambiguity", "drone_search", "siting", "collateral", "replay",
                            "abstain"]},
        "evidence_refs": {**_REFS, "description": "refs from tool results that support the answer"},
        "confidence": {"type": "number", "description": "0..1"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "track_refs": {**_REFS, "description": "THE ANSWER for operator_query: every track that matches"},
        "boolean": {"type": "boolean", "description": "yes/no answer when the question asks one"},
        "protocol": {"enum": ["ocusync_like", "wifi", "lora", "fm_video", "gfsk"], "description": "radio protocol class when the question asks one"},
        "band": {"enum": ["sub_ghz_868_915", "ism_2400", "ism_5800"], "description": "band when the question asks one"},
        "region": {"type": "object", "properties": {"type": {"enum": ["ref", "circle"]}, "ref": {"type": "string"}}},
        "chosen_hyp_refs": _REFS, "rejected_hyp_refs": _REFS, "position": _PT,
        "reason": {"enum": ["no_sensor_coverage", "entity_not_found", "ambiguous_unresolved", "outside_tool_scope",
                            "insufficient_evidence"]},
        "evt_refs": _REFS, "time_s": {"type": "number"}, "fp_ref": {"type": "string"}, "violations": _REFS,
        "node_candidates": _REFS, "final_position": _PT, "waypoints_used": _REFS,
        "requires_human_confirm": {"type": "boolean"}}}}}
register_control("submit", "Submit the final answer with evidence refs and confidence. Ends the episode.",
                 SubmitArgs, schema_override=SUBMIT_SCHEMA)
