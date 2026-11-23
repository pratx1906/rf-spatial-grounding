"""Terminal answer schema (SPEC 6.3). One discriminated union on `intent`."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from tools.types import Ellipse, Point, Region, Strict


class Common(Strict):
    evidence_refs: Annotated[list[str], Field(max_length=30)] = []
    assumptions: Annotated[list[Annotated[str, Field(max_length=120)]], Field(max_length=5)] = []
    confidence: Annotated[float, Field(ge=0, le=1)]
    requires_human_confirm: bool = False

    @model_validator(mode="after")
    def _evidence(self):
        if getattr(self, "intent", None) != "abstain" and not self.evidence_refs:
            raise ValueError("evidence_refs must cite at least one ref returned by a tool in this episode")
        return self


class AnswerOperatorQuery(Common):
    intent: Literal["operator_query"]
    track_refs: list[str] = []
    region: Region | None = None
    boolean: bool | None = None
    protocol: Literal["ocusync_like", "wifi", "lora", "fm_video", "gfsk"] | None = None   # RF phase 2 (v2 sets)
    band: Literal["sub_ghz_868_915", "ism_2400", "ism_5800"] | None = None


class AnswerDFResolution(Common):
    intent: Literal["df_ambiguity"]
    chosen_hyp_refs: Annotated[list[str], Field(min_length=1)]
    rejected_hyp_refs: list[str] = []
    position: Point | None = None
    uncertainty: Ellipse | None = None


class AnswerDroneSearch(Common):
    intent: Literal["drone_search"]
    final_position: Point
    uncertainty: Ellipse
    waypoints_used: list[str] = []


class AnswerSiting(Common):
    intent: Literal["siting"]
    zones: list[Region] = []
    node_candidates: list[str] = []


class AnswerCollateral(Common):
    intent: Literal["collateral"]
    violations: list[str] = []
    fp_ref: str
    requires_human_confirm: Literal[True] = True


class AnswerReplay(Common):
    intent: Literal["replay"]
    evt_refs: list[str] = []
    time_s: float | None = None


class Abstain(Common):
    intent: Literal["abstain"]
    reason: Literal["no_sensor_coverage", "entity_not_found", "ambiguous_unresolved", "outside_tool_scope",
                    "insufficient_evidence"]


Answer = Annotated[AnswerOperatorQuery | AnswerDFResolution | AnswerDroneSearch | AnswerSiting | AnswerCollateral
                   | AnswerReplay | Abstain, Field(discriminator="intent")]

_BY_INTENT = {cls.model_fields["intent"].annotation.__args__[0]: cls
              for cls in (AnswerOperatorQuery, AnswerDFResolution, AnswerDroneSearch, AnswerSiting, AnswerCollateral,
                          AnswerReplay, Abstain)}


def drop_off_intent(answer: dict) -> tuple[dict, list[str]]:
    """(answer without the flat-schema fields its intent does not define, dropped field names). Diagnostic
    lenient-submit mode only (eval/ablations.py "lenient_submit"): small models fill every field of the flat submit
    schema. An unknown intent is returned unchanged for the validator to reject."""
    cls = _BY_INTENT.get(answer.get("intent")) if isinstance(answer, dict) else None
    if cls is None:
        return answer, []
    keep = set(cls.model_fields)
    return {k: v for k, v in answer.items() if k in keep}, [k for k in answer if k not in keep]
