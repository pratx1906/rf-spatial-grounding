"""Per-area answer scores in [0, 1] from the answer key only (SPEC 9.6). No truth file access."""

from __future__ import annotations

import math

from tasks.base import Task


def _base(ref: str) -> str:
    return ref.split("@", 1)[0]


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def _fbeta(pred: set, gold: set, beta: float) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _iou(answer, key_wkts: list[str], resolved: dict | None) -> float:
    from shapely import wkt
    from shapely.ops import unary_union

    gold = unary_union([wkt.loads(w) for w in key_wkts])
    got = [resolved.get(z.ref) for z in answer.zones if getattr(z, "type", None) == "ref"] if resolved else []
    got = [g for g in got if g is not None]
    if not got:
        return 0.0
    pred = unary_union(got)
    union = pred.union(gold).area
    return float(pred.intersection(gold).area / union) if union > 0 else 0.0


def _dist_score(err: float, full_m: float, tau_m: float) -> float:
    """1 within full_m (the achievable accuracy of the sensors), then exp decay with tau_m."""
    return 1.0 if err <= full_m else math.exp(-(err - full_m) / tau_m)


def score_answer(task: Task, answer, resolved: dict | None = None) -> float:
    """`resolved`: ref -> geometry for episode-scoped refs (ras: weak zones), supplied by the env at scoring time."""
    key = task.answer_key
    if answer is None or task.area == "free":
        return 0.0                 # free-form UI questions have no answer key
    if task.area == "unanswerable":
        return 1.0 if answer.intent == "abstain" and answer.reason == key["reason"] else 0.0
    if answer.intent == "abstain":
        return 0.0                 # answerable question abstained: the always-abstain policy must lose (SPEC 9.6)
    if task.area == "operator_query":
        if answer.intent != "operator_query":
            return 0.0
        tracks = _f1({_base(r) for r in answer.track_refs}, {_base(r) for r in key["track_refs"]})
        if key.get("region_ref"):
            got = getattr(answer.region, "ref", None) if answer.region is not None else None
            return round(0.5 * float(got == key["region_ref"]) + 0.5 * tracks, 3)
        for cat in ("protocol", "band"):                     # RF phase 2 categorical answers
            if key.get(cat) is not None:
                return round(0.5 * float(getattr(answer, cat, None) == key[cat]) + 0.5 * tracks, 3)
        if key.get("boolean") is None:
            return round(tracks, 3)
        ok_bool = float(answer.boolean is not None and answer.boolean == key["boolean"])
        return round(0.5 * ok_bool + 0.5 * tracks, 3)
    if task.area == "df_ambiguity":
        if answer.intent != "df_ambiguity":
            return 0.0
        hyp = _f1({_base(h) for h in answer.chosen_hyp_refs}, {_base(h) for h in key["los_hyp_refs"]})
        if key.get("position") is None or answer.position is None:
            return round(hyp, 3)
        err = math.hypot(answer.position.x_m - key["position"]["x_m"], answer.position.y_m - key["position"]["y_m"])
        return round(0.5 * hyp + 0.5 * _dist_score(err, key.get("full_m", 0.0), key.get("tau_m", 25.0)), 3)
    if answer.intent != task.area:
        return 0.0
    if task.area == "drone_search":
        err = math.hypot(answer.final_position.x_m - key["position"]["x_m"],
                         answer.final_position.y_m - key["position"]["y_m"])
        return round(_dist_score(err, key.get("full_m", 0.0), key.get("tau_m", 30.0)), 3)
    if task.area == "siting":
        parts = []
        if key.get("zones_wkt"):
            parts.append(min(1.0, _iou(answer, key["zones_wkt"], resolved) / 0.7))   # IoU >= 0.7 counts as full
        if key.get("node_refs"):
            got = [_base(r) for r in answer.node_candidates]
            parts.append(float(bool(got) and got[0] in set(key["node_refs"])) if key.get("top1")
                         else (sum(g in set(key["node_refs"]) for g in got) / len(got) if got else 0.0))
        return round(sum(parts) / len(parts), 3) if parts else 0.0
    if task.area == "collateral":
        return round(_fbeta({_base(r) for r in answer.violations}, {_base(r) for r in key["violations"]}, 2.0), 3)
    if task.area == "replay":
        parts = []
        if "evt_refs" in key:
            parts.append(_f1({_base(r) for r in answer.evt_refs}, {_base(r) for r in key["evt_refs"]}))
        if key.get("time_s") is not None:
            parts.append(0.0 if answer.time_s is None else math.exp(-abs(answer.time_s - key["time_s"]) / 10.0))
        return round(sum(parts) / len(parts), 3) if parts else 0.0
    raise NotImplementedError(f"no scoring for area {task.area!r}")


# --- Versioned scoring (paper Revision 2; AUDIT_CODE C2, C6, C7) --------------------------------------------------
# score_v1 = score_answer above: the scorer every stored number used, and still the one the env (rewards, eval.run
# rows) uses. score_v2 is applied offline by eval/rescore.py from the saved answers; old numbers stay labelled v1.
SCORE_VERSIONS = ("v1", "v2")
HEADLINE_EXCLUDED_AREAS = ("collateral", "siting")      # tool-use fidelity, not grounding (C6)


def score_answer_v2(task: Task, answer, resolved: dict | None = None) -> float:
    """v2 = v1 except (C2): a df_ambiguity answer that leaves out the position on a key with a position counts as
    error = infinity, so the position half scores 0 (v1 scored the hyp F1 alone, i.e. omitting beat a poor position)."""
    key = task.answer_key or {}
    if (answer is not None and task.area == "df_ambiguity" and getattr(answer, "intent", None) == "df_ambiguity"
            and key.get("position") is not None and getattr(answer, "position", None) is None):
        hyp = _f1({_base(h) for h in answer.chosen_hyp_refs}, {_base(h) for h in key["los_hyp_refs"]})
        return round(0.5 * hyp, 3)
    return score_answer(task, answer, resolved)


def score_versioned(task: Task, answer, resolved: dict | None = None, version: str = "v1") -> float:
    if version == "v1":
        return score_answer(task, answer, resolved)
    if version == "v2":
        return score_answer_v2(task, answer, resolved)
    raise ValueError(f"unknown score version {version!r}; one of {SCORE_VERSIONS}")


def empty_answer(area: str):
    """The 'always empty / always no' answer of an area (duck-typed for the scorer), or None where the schema has
    no empty form (df_ambiguity needs >= 1 hyp, drone_search a position, unanswerable is an abstain test)."""
    from types import SimpleNamespace as NS

    return {"operator_query": NS(intent="operator_query", track_refs=[], region=None, boolean=False, protocol=None,
                                 band=None),
            "replay": NS(intent="replay", evt_refs=[], time_s=None),
            "collateral": NS(intent="collateral", violations=[]),
            "siting": NS(intent="siting", zones=[], node_candidates=[])}.get(area)


def empty_scores_full(task: Task) -> bool:
    """True if the empty / 'no' answer scores 1.0 on this task's key (C6: _f1(empty, empty) = 1)."""
    e = empty_answer(task.area)
    if e is None or (task.area == "siting" and (task.answer_key or {}).get("zones_wkt")):
        return False                       # no empty form / an empty zone list has IoU 0
    return score_answer_v2(task, e) >= 0.999


def is_headline(task: Task) -> bool:
    """C6 headline mask: grounding tasks only; collateral / siting areas and empty-key-scores-1.0 tasks are out."""
    return task.area not in HEADLINE_EXCLUDED_AREAS and task.area != "free" and not empty_scores_full(task)
