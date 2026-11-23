"""T-FIX arm T4 expert (docs/paper/TWIN_FIX_PREREG.md): check -> exclude the suspect node -> answer.

Wraps the scripted expert (tasks.expert_kandla). Needs the opt-in sensor_consistency tool (ablation
with_consistency+fix_line). For track and DF questions it first runs the health check at the question time (whole
scenario if that window has no epochs).
- No suspect node: the normal expert answers (the check is the only extra call).
- Zone-type operator questions with a suspect node: the candidate drones around the zone are re-located from the
  strongest bearing of each NON-suspect node (fit_candidates triangulate) and tested with inside_zone on that point;
  candidates with fewer than two clean nodes keep the tracker position. The answer names the suspect node in its
  assumptions and lowers the confidence.
- Other families with a suspect node: the normal expert answers, then the assumption and confidence are adjusted.
Belief side only (tools), like every expert.
"""

from __future__ import annotations

import math
import os

from harness.env import AkshaEnv
from tasks.expert import ZONE_KINDS, _call, _resolve_zone
from tasks.expert_kandla import MINC, _parallel, _submit
from tasks.expert_kandla import solve as base_solve

ZONE_FAMILIES = ("zone_incursion", "count_in_zone", "zone_incursion_ambiguous", "band_in_zone", "swarm_in_zone")
MAX_REFIT = 3
# Gate G-D (2026-11-01): the clean-node re-fit LOWERED the expert's zone score on M2 training seeds (0.672 -> 0.571,
# runs/paper/results/TFIX/gd), so the pre-registered T4 was skipped; the exploratory T4x expert runs with TFIX_REFIT=0
# (check -> caveat + lower confidence, the normal investigation otherwise)
REFIT = os.environ.get("TFIX_REFIT", "1") == "1"
NEAR_M = 300.0


def health(env: AkshaEnv, t: float | None) -> list[str]:
    r = _call(env, "sensor_consistency", "sensor health before trusting tracks", **({"at_time_s": t} if t is not None else {}))
    if not r["ok"] and t is not None:
        r = _call(env, "sensor_consistency", "sensor health over the whole scenario")
    return list(r["data"]["suspect_refs"]) if r["ok"] else []


def _note(suspects: list[str]) -> str:
    return f"suspect DF heading bias at {', '.join(suspects)} (sensor_consistency); fixes use the other nodes"


def _refit_inside(env: AkshaEnv, zone: str, t: float, cands: list[dict], suspects: set[str]) -> list[str]:
    """inside_zone on the clean-node re-fit of up to MAX_REFIT candidates (tracker position for the rest)."""
    refit, keep = cands[:MAX_REFIT], cands[MAX_REFIT:]
    hy = _parallel(env, [{"name": "df_hypotheses", "arguments": {"track_ref": c["ref"], "at_time_s": t}} for c in refit])
    fits, fit_of = [], {}
    for c, r in zip(refit, hy, strict=False):
        if not r["ok"]:
            keep.append(c)
            continue
        best = [max(s["hyps"], key=lambda h: h["power_db"])["ref"] for s in r["data"]["sensors"]
                if s["sensor"] not in suspects and s["hyps"]]
        if len(best) >= 2:
            fit_of[c["ref"]] = len(fits)
            fits.append({"name": "fit_candidates", "arguments": {"hyp_refs": best, "method": "triangulate"}})
        else:
            keep.append(c)
    fr = _parallel(env, fits)
    subjects = []
    for c in refit:
        i = fit_of.get(c["ref"])
        if i is not None and i < len(fr) and fr[i]["ok"]:
            subjects.append((c["ref"], fr[i]["data"]["candidates"][0]["position"]))
        elif c not in keep:
            keep.append(c)
    subjects += [(c["ref"], f"{c['ref']}@{t:g}") for c in keep]
    res = _parallel(env, [{"name": "inside_zone", "arguments": {"subject": s, "zone_ref": zone}} for _, s in subjects])
    return [ref for (ref, _), r in zip(subjects, res, strict=False) if r["ok"] and r["data"]["inside"]]


def _zone_answer(env: AkshaEnv, task, suspects: list[str]) -> bool:
    f, p = task.template, task.params
    t = p.get("t")
    rz = _call(env, "site_resolve", "resolve the zone", query=p["zone_phrase"], kinds=ZONE_KINDS)
    if rz["ok"]:
        z = rz["data"]["candidates"][0]
    elif (rz.get("error") or {}).get("code") == "AMBIGUOUS_ENTITY":
        ref, _ = _resolve_zone(env, p["zone_phrase"])          # asks the operator, as the base expert does
        z = next((c for c in rz["error"].get("candidates", []) if c.get("ref") == ref and "bbox_m" in c), None)
        if z is None:
            return False
    else:
        return False
    bx = z["bbox_m"]
    radius = math.hypot(bx[2] - bx[0], bx[3] - bx[1]) / 2 + NEAR_M
    filt = {"time_range_s": [t, t], "klass": "drone", "min_confidence": MINC,
            "region": {"type": "circle", "center": z["centroid"], "radius_m": radius}}
    if f == "band_in_zone":
        filt["band"] = p["band"]
    s = _call(env, "track_search", "drones near the zone (positions may carry the bias)", filters=filt, limit=20)
    if not s["ok"]:
        return False
    cx, cy = z["centroid"]["x_m"], z["centroid"]["y_m"]
    cands = sorted(s["data"]["items"], key=lambda it: math.hypot(it["pos"]["x_m"] - cx, it["pos"]["y_m"] - cy))
    inside = _refit_inside(env, z["ref"], t, cands, set(suspects))
    if f == "swarm_in_zone":
        groups = {}
        for ref in inside:
            d = _call(env, "track_detail", "swarm group of the track", track_ref=ref)
            if d["ok"] and d["data"].get("group_id"):
                groups.setdefault(d["data"]["group_id"], []).append(ref)
        inside = [r for g in groups.values() for r in g]
    _submit(env, {"intent": "operator_query", "track_refs": inside, "region": {"type": "ref", "ref": z["ref"]},
                  "boolean": None if f == "count_in_zone" else bool(inside),
                  "evidence_refs": [z["ref"]] + [c["ref"] for c in cands[:MAX_REFIT]] + suspects,
                  "assumptions": [_note(suspects), f"drone = classifier argmax drone with track confidence >= {MINC}"],
                  "confidence": 0.6})
    return True


def solve(env: AkshaEnv, task) -> None:
    if task.area not in ("operator_query", "df_ambiguity"):
        return base_solve(env, task)
    suspects = health(env, task.params.get("t"))
    if REFIT and suspects and task.template in ZONE_FAMILIES and task.params.get("t") is not None:
        if _zone_answer(env, task, suspects) or env.done:
            return None
    if not suspects:
        return base_solve(env, task)
    # other families: the normal investigation; the submit then carries the caveat and a lower confidence
    orig = env.step

    def step(action):
        for c in action.get("tool_calls") or []:
            if c.get("name") == "submit" and isinstance(c.get("arguments"), dict):
                ans = dict(c["arguments"].get("answer") or {})
                ans["assumptions"] = ([_note(suspects)] + list(ans.get("assumptions") or []))[:5]
                ans["confidence"] = min(float(ans.get("confidence", 0.8)), 0.6)
                ans["evidence_refs"] = list(ans.get("evidence_refs") or []) + [s for s in suspects
                                                                               if s not in (ans.get("evidence_refs") or [])]
                c["arguments"] = {**c["arguments"], "answer": ans}
        return orig(action)

    env.step = step
    try:
        base_solve(env, task)
    finally:
        env.step = orig
    return None
