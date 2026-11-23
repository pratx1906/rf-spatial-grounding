"""Episode reward (SPEC 9.6, 15.2), with D-004 (token latency proxy) and D-005 (single unanswerable penalty).

Two versions, selected by `RewardConfig.version`:
- "v1" (default; the env, eval and every earlier run use it, unchanged): continuous answer score + grounding +
  validity - normalised step / latency / repeats + Brier calibration - penalties.
- "v2" (paper plan 2026-10-28 P3 + Revision 1 M5/B3; the GRPO / RFT loop): binarised correctness (task score >=
  `correct_threshold`; Acc@`acc_radius_m` for position answers, and a locate answer without a position is wrong:
  AUDIT_CODE C2) counted only for a verified answer (a verifier-rejected answer earns nothing: C7), plus the smooth
  distance bonus `w_dist * exp(-d / dist_scale_m)` for position answers (d = error to the true emitter; `dist_bonus`
  False drops it for the ablation), per-step cost, missed-violation AND false-alarm penalties (so "always yes" is not
  cheap), a wrong-abstain penalty that applies only to answerable tasks whose evidence is consistent, and credit for
  abstaining / flagging on a wrong-twin task (mismatch world M1-M4) whose evidence is inconsistent. Grounding and
  validity credit need at least one evidence tool call (a zero-tool episode earns neither). The Brier calibration term
  is off by default in v2 (`w_calib_v2`; gameable together with abstain, audit M5).
The task-dependent v2 terms are computed by `task_terms` from the task key, the answer and the episode's tool results
(agent-visible only; no truth access); `stats_from_env` rebuilds the episode stats from a finished AkshaEnv.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

VIOLATION_FAMILIES = frozenset({"zone_incursion", "zone_incursion_ambiguous", "swarm_in_zone", "near_object",
                                "near_object_ambiguous"})
CONTROL_TOOLS = frozenset({"declare_intent", "ask_user", "submit"})
TWIN_WORLD = re.compile(r"_M([1-9])$")        # hardware-mismatch worlds M1-M4 (M0 = the null control)


@dataclass(frozen=True)
class RewardConfig:
    w_ans: float = 1.0
    w_ground: float = 0.2
    w_valid: float = 0.1
    w_step: float = 0.05
    w_lat: float = 0.02
    w_rep: float = 0.02
    w_calib: float = 0.1
    penalty_missed_violation: float = 2.0
    penalty_hallucinated_ref: float = 1.0
    penalty_unconfirmed_mitigation: float = 1.0
    penalty_unanswerable_answered: float = 1.0
    penalty_format: float = 0.5
    # --- v2 (RL reward); ignored by v1 ---
    version: str = "v1"
    correct_threshold: float = 0.8           # binarised correctness on the task's own [0, 1] score
    acc_radius_m: float = 50.0               # position answers: correct within this error (Acc@50)
    dist_bonus: bool = True                  # False: binarised-only ablation
    w_dist: float = 0.3
    dist_scale_m: float = 80.0
    step_cost: float = 0.01                  # per turn (replaces v1's normalised step term)
    penalty_false_alarm: float = 1.0         # < penalty_missed_violation: a miss costs more than a false alarm
    penalty_wrong_abstain: float = 0.5       # answerable task, consistent evidence, abstained
    w_twin_flag: float = 0.5                 # wrong-twin task, inconsistent evidence, abstained / flagged
    w_calib_v2: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeStats:
    turns: int
    max_turns: int
    generated_tokens: int
    max_tokens_per_turn: int
    valid_call_fraction: float
    repeat_calls: int
    answered: bool
    grounded: bool
    verifier_failed_twice: bool
    format_failed_twice: bool
    unanswerable: bool
    abstained: bool
    confidence: float | None
    answer_score: float
    # v2 task terms (task_terms); defaults leave v1 unchanged
    correct: bool | None = None
    position_error_m: float | None = None
    missed_violation: float = 0.0
    false_alarm: float = 0.0
    wrong_abstain: bool = False
    twin_flag_credit: bool = False
    evidence_calls: int | None = None        # non-control tool calls; None = unknown (counted as > 0)


def episode_reward(s: EpisodeStats, cfg: RewardConfig = RewardConfig()) -> dict[str, float]:
    if cfg.version == "v2":
        return _reward_v2(s, cfg)
    if cfg.version != "v1":
        raise ValueError(f"unknown reward version {cfg.version!r}")
    parts = {
        "answer": cfg.w_ans * s.answer_score,
        "ground": cfg.w_ground * float(s.answered and s.grounded),
        "valid": cfg.w_valid * s.valid_call_fraction,
        "step": -cfg.w_step * s.turns / s.max_turns,
        "lat": -cfg.w_lat * min(1.0, s.generated_tokens / (s.max_turns * s.max_tokens_per_turn)),
        "repeat": -cfg.w_rep * s.repeat_calls,
        # Brier-style calibration against the continuous answer score
        "calib": cfg.w_calib * (1 - (s.confidence - s.answer_score) ** 2) if s.confidence is not None else 0.0,
    }
    pen = 0.0
    if s.verifier_failed_twice:
        pen += cfg.penalty_hallucinated_ref
    if s.format_failed_twice:
        pen += cfg.penalty_format
    if s.unanswerable and s.answered and not s.abstained:
        pen += cfg.penalty_unanswerable_answered
    parts["pen"] = -pen
    parts["total"] = round(sum(parts.values()), 4)
    return {k: round(v, 4) for k, v in parts.items()}


def _reward_v2(s: EpisodeStats, cfg: RewardConfig) -> dict[str, float]:
    verified = s.answered and s.grounded            # submitted and accepted by the evidence verifier
    correct = bool(s.correct) if s.correct is not None else s.answer_score >= cfg.correct_threshold
    # a correct abstain on an unanswerable question (actuation, future time) may need no evidence tool
    has_evidence = s.evidence_calls is None or s.evidence_calls > 0 or (s.unanswerable and s.abstained)
    d = s.position_error_m
    parts = {
        "answer": cfg.w_ans * float(verified and correct),
        "dist": (cfg.w_dist * math.exp(-max(0.0, d) / cfg.dist_scale_m)
                 if cfg.dist_bonus and verified and d is not None else 0.0),
        "ground": cfg.w_ground * float(verified and has_evidence and not s.wrong_abstain),
        "valid": cfg.w_valid * s.valid_call_fraction * float(has_evidence),
        "step": -cfg.step_cost * s.turns,
        "lat": -cfg.w_lat * min(1.0, s.generated_tokens / (s.max_turns * s.max_tokens_per_turn)),
        "repeat": -cfg.w_rep * s.repeat_calls,
        "calib": (cfg.w_calib_v2 * (1 - (s.confidence - float(correct)) ** 2)
                  if cfg.w_calib_v2 and s.confidence is not None else 0.0),
        "twin": cfg.w_twin_flag * float(s.twin_flag_credit and not (verified and correct)),
        "missed": -cfg.penalty_missed_violation * s.missed_violation,
        "false_alarm": -cfg.penalty_false_alarm * s.false_alarm,
        "wrong_abstain": -cfg.penalty_wrong_abstain * float(s.wrong_abstain),
    }
    pen = 0.0
    if s.verifier_failed_twice:
        pen += cfg.penalty_hallucinated_ref
    if s.format_failed_twice:
        pen += cfg.penalty_format
    if s.unanswerable and s.answered and not s.abstained:
        pen += cfg.penalty_unanswerable_answered
    parts["pen"] = -pen
    parts["total"] = round(sum(parts.values()), 4)
    return {k: round(v, 4) for k, v in parts.items()}


# --- task-dependent v2 terms ------------------------------------------------------------------------------------
def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _base(ref: str) -> str:
    return str(ref).split("@", 1)[0]


def _xy(p) -> tuple[float, float] | None:
    if p is None:
        return None
    x, y = _get(p, "x_m"), _get(p, "y_m")
    return None if x is None or y is None else (float(x), float(y))


def position_error(task, answer) -> float | None:
    """Error (m) of a position answer to the key position (the true emitter), or None if not a position answer."""
    key = _get(task, "answer_key") or {}
    area, intent = _get(task, "area"), _get(answer, "intent")
    if answer is None or intent != area or not key.get("position"):
        return None
    got = _xy(_get(answer, "final_position" if area == "drone_search" else "position"))
    want = _xy(key["position"])
    if got is None or want is None:
        return None
    return math.hypot(got[0] - want[0], got[1] - want[1])


def is_twin_world(task) -> bool:
    if _get(task, "world_variant"):
        return bool(re.fullmatch(r"M[1-9]", str(_get(task, "world_variant"))))
    return bool(TWIN_WORLD.search(str(_get(task, "site") or "").rstrip("/")))


def evidence_inconsistent(task, tool_results: list[dict] | None) -> bool:
    """Wrong-twin evidence: an explicit task flag `evidence_inconsistent` when the builder wrote one; otherwise, on a
    mismatch world (M1-M4) only, a sensor_consistency result in this episode that named suspect nodes."""
    flag = _get(task, "evidence_inconsistent")
    if flag is not None:
        return bool(flag)
    if not is_twin_world(task):
        return False
    for r in tool_results or []:
        if r.get("tool") == "sensor_consistency" and r.get("ok") and (r.get("data") or {}).get("suspect_refs"):
            return True
    return False


def _pred_bool(answer) -> bool | None:
    if answer is None or _get(answer, "intent") != "operator_query":
        return None
    b = _get(answer, "boolean")
    return bool(_get(answer, "track_refs") or []) if b is None else bool(b)


def violation_terms(task, answer, exempt: bool = False) -> tuple[float, float]:
    """(missed, false_alarm) in [0, 1]. Boolean incursion families: key yes and answer not yes (incl. abstain or no
    answer) = 1 missed; key no and answer yes = 1 false alarm. Collateral: missed = share of key violations not
    named, false alarm = share of named violations not in the key. `exempt`: a credited wrong-twin abstain/flag."""
    key = _get(task, "answer_key") or {}
    area, fam = _get(task, "area"), _get(task, "template")
    if exempt:
        return 0.0, 0.0
    if area == "operator_query" and fam in VIOLATION_FAMILIES and key.get("boolean") is not None:
        pred = bool(_pred_bool(answer))
        if key["boolean"] and not pred:
            return 1.0, 0.0
        if not key["boolean"] and pred:
            return 0.0, 1.0
        return 0.0, 0.0
    if area == "collateral" and key.get("violations") is not None:
        gold = {_base(r) for r in key["violations"]}
        got = ({_base(r) for r in (_get(answer, "violations") or [])}
               if _get(answer, "intent") == "collateral" else set())
        missed = len(gold - got) / len(gold) if gold else 0.0
        fa = len(got - gold) / len(got) if got else 0.0
        return missed, fa
    return 0.0, 0.0


def task_terms(task, answer, answer_score: float, tool_results: list[dict] | None = None,
               cfg: RewardConfig = RewardConfig()) -> dict:
    """The task-dependent v2 terms. `answer`: the pydantic answer or its dict; `tool_results`: the episode's tool
    result envelopes (agent-visible). Keys: correct, position_error_m, missed_violation, false_alarm, wrong_abstain,
    twin_flag_credit, evidence_inconsistent, flagged, abstained."""
    area = _get(task, "area")
    intent = _get(answer, "intent")
    abstained = intent == "abstain"
    d = position_error(task, answer)
    key = _get(task, "answer_key") or {}
    if d is not None:
        correct = d <= cfg.acc_radius_m
    elif key.get("position") and area in ("df_ambiguity", "drone_search") and intent == area:
        # AUDIT_CODE C2: a locate answer without a position is wrong (error = infinity), whatever its hyp F1 /
        # tasks.scoring gives it; it also earns no distance bonus
        correct = False
    else:
        correct = answer is not None and answer_score >= cfg.correct_threshold
    inconsistent = evidence_inconsistent(task, tool_results)
    flagged = abstained or (bool(_get(answer, "requires_human_confirm")) and intent not in (None, "collateral"))
    twin = inconsistent and answer is not None and flagged
    missed, fa = violation_terms(task, answer, exempt=twin)
    return {"correct": bool(correct), "position_error_m": None if d is None else round(d, 2),
            "missed_violation": round(missed, 4), "false_alarm": round(fa, 4),
            "wrong_abstain": bool(abstained and area != "unanswerable" and not inconsistent),
            "twin_flag_credit": bool(twin), "evidence_inconsistent": bool(inconsistent), "flagged": bool(flagged),
            "abstained": abstained}


def evidence_call_count(tool_results: list[dict] | None) -> int:
    return sum(1 for r in tool_results or [] if r.get("tool") not in CONTROL_TOOLS)


def stats_from_env(env, tool_results: list[dict] | None = None, cfg: RewardConfig = RewardConfig(),
                   task_row: dict | None = None) -> tuple[EpisodeStats, dict]:
    """EpisodeStats of a finished AkshaEnv episode, mirroring harness.env.AkshaEnv._finish, plus the v2 task terms.
    `task_row`: the raw task row (keeps fields Task drops, e.g. world_variant / evidence_inconsistent)."""
    task = task_row or env.task
    answer, reason = env.answer, env.end_reason
    ex = env.executor
    score = float(getattr(env, "answer_score", 0.0) or 0.0)
    terms = task_terms(task, answer, score, tool_results, cfg)
    s = EpisodeStats(
        turns=env.turn, max_turns=env.cfg.max_turns, generated_tokens=env.gen_tokens,
        max_tokens_per_turn=env.cfg.max_tokens_per_turn,
        valid_call_fraction=(ex.valid_calls / ex.calls) if ex.calls else 0.0, repeat_calls=ex.repeat_calls,
        answered=answer is not None and reason == "submitted",
        grounded=bool(env.verifier_result and env.verifier_result.get("ok")),
        verifier_failed_twice=reason == "verifier_failed", format_failed_twice=reason == "format_failed",
        unanswerable=env.task.area == "unanswerable", abstained=terms["abstained"],
        confidence=_get(answer, "confidence"), answer_score=score,
        correct=terms["correct"], position_error_m=terms["position_error_m"],
        missed_violation=terms["missed_violation"], false_alarm=terms["false_alarm"],
        wrong_abstain=terms["wrong_abstain"], twin_flag_credit=terms["twin_flag_credit"],
        evidence_calls=evidence_call_count(tool_results) if tool_results is not None else None)
    return s, terms
