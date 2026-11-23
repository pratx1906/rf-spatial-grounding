"""Trivial baselines for the RF grounding paper (plan 2026-10-28 T4, E2). Each runs through AkshaEnv with the real
tools (truth separation holds), reads the question's structured slots (`task.params`, oracle parsing) and submits one
answer. Tasks a policy has no rule for get an abstain (scored 0 on answerable tasks).

- always_abstain: abstain(ABSTAIN_REASON) on every task.
- always_abstain_oracle_reason: the fair abstain baseline (AUDIT_CODE minor 4): abstain on every task, with the
  reason the task's unanswerable *type* calls for (template -> reason, oracle parsing like `task.params`; the answer
  key is not read), so it is correct on every unanswerable task and 0 on answerable ones.
- prior_centroid: position families answer the mean key position of the train split per area (a dataset prior, no
  observation); the site's sensor centroid if the train file is missing.
- nearest_sensor: position families answer the position of the DF sensor with the strongest bearing on the track.
- random_track: operator queries answer one drone track drawn uniformly from track_search at the question time
  (deterministic in task id and seed).

The verifier may reject these answers (e.g. a prior far from the drone belief); they are submitted until the episode
ends so the answer is always scored (end_reason verifier_failed).
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from harness.env import AkshaEnv
from sim.geometry import stable_seed
from tasks.expert import _call

REPO = Path(__file__).resolve().parents[1]
POLICIES = ("always_abstain", "prior_centroid", "nearest_sensor", "random_track", "always_abstain_oracle_reason")
ABSTAIN_REASON = "insufficient_evidence"
# unanswerable template -> the abstain reason its question type calls for (tasks/kandla/gen*.py)
ORACLE_REASON = {"entity_not_found": "entity_not_found", "unknown_track": "entity_not_found",
                 "actuation": "outside_tool_scope", "future_time": "no_sensor_coverage",
                 "rf_ambiguous": "insufficient_evidence"}
PRIOR_TASKS = REPO / "data" / "tasks" / "x3" / "train.jsonl"
WIDE_M = 500.0                                 # stated 1-sigma-ish ellipse of the prior answers
_PRIOR: dict[str, dict] = {}


def is_position_task(task) -> bool:
    from eval.grounding import POSITION_FAMILIES

    return task.template in POSITION_FAMILIES or task.template.startswith("ds_")


def submit_final(env: AkshaEnv, answer: dict) -> None:
    """Submit and resubmit the same answer until the episode ends (verifier rejections then end it as
    verifier_failed with the answer scored)."""
    for _ in range(env.cfg.max_verifier_retries + 2):
        if env.done:
            return
        _call(env, "submit", "baseline answer", answer=answer)
    if not env.done:
        env.finish_external("baseline_stuck")


def abstain(env: AkshaEnv, reason: str = ABSTAIN_REASON) -> None:
    submit_final(env, {"intent": "abstain", "reason": reason, "evidence_refs": [], "confidence": 0.5})


def _ellipse(a_m: float) -> dict:
    return {"kind": "ellipse", "a_m": a_m, "b_m": a_m, "theta_deg": 0.0}


def position_answer(env: AkshaEnv, task, xy: tuple[float, float], unc: dict, hyps: list[str], track: str,
                    conf: float = 0.5) -> None:
    """Submit a position in the task's answer schema: df_ambiguity (with chosen hyps) or drone_search."""
    pos = {"x_m": round(float(xy[0]), 1), "y_m": round(float(xy[1]), 1)}
    if task.area == "df_ambiguity":
        if not hyps:
            return abstain(env)
        return submit_final(env, {"intent": "df_ambiguity", "chosen_hyp_refs": hyps, "rejected_hyp_refs": [],
                                  "position": pos, "uncertainty": unc, "evidence_refs": [track] + hyps[:5],
                                  "confidence": conf})
    return submit_final(env, {"intent": "drone_search", "final_position": pos, "uncertainty": unc,
                              "waypoints_used": [], "evidence_refs": [track], "confidence": conf})


def hyps_at(env: AkshaEnv, task, t: float | None = None) -> dict | None:
    """df_hypotheses on the task's track (at its time, else the track's latest estimate time); None if none."""
    p = task.params
    if t is None:
        t = p.get("t")
    if t is None:
        d = _call(env, "track_detail", "latest estimate time", track_ref=p["track"])
        if not d["ok"]:
            return None
        t = d["data"]["t_s"]
    args = {"track_ref": p["track"], "at_time_s": t}
    if p.get("sensors"):
        args["sensor_refs"] = [f"sensor:{s}" for s in p["sensors"]]
    h = _call(env, "df_hypotheses", "bearings on the track", **args)
    return h["data"] if h["ok"] and h["data"]["sensors"] else None


def _prior(env: AkshaEnv, area: str) -> tuple[float, float]:
    key = str(PRIOR_TASKS)
    if key not in _PRIOR:
        acc: dict[str, list] = {}
        if PRIOR_TASKS.exists():
            for line in PRIOR_TASKS.read_text().splitlines():
                r = json.loads(line)
                pos = (r.get("answer_key") or {}).get("position")
                if r.get("split", "train") == "train" and pos:
                    acc.setdefault(r["area"], []).append((pos["x_m"], pos["y_m"]))
        _PRIOR[key] = {a: (statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v)) for a, v in acc.items()}
    if area in _PRIOR[key]:
        return _PRIOR[key][area]
    poses = [s["pose"] for s in env.site.sensors.values()]
    return statistics.mean(p["x_m"] for p in poses), statistics.mean(p["y_m"] for p in poses)


def _ground_track(env: AkshaEnv, task) -> bool:
    """Make the track ref citable (drone_state for drone search, track_detail otherwise)."""
    r = _call(env, "drone_state", "target and prior") if task.template.startswith("ds_") else \
        _call(env, "track_detail", "the track", track_ref=task.params["track"])
    return bool(r["ok"])


def solve(env: AkshaEnv, task, policy: str, seed: int = 0) -> None:
    if policy == "always_abstain_oracle_reason":
        return abstain(env, ORACLE_REASON.get(task.template, ABSTAIN_REASON) if task.area == "unanswerable"
                       else ABSTAIN_REASON)
    if policy == "always_abstain" or task.area == "unanswerable":
        return abstain(env)
    if policy == "prior_centroid" and is_position_task(task):
        if task.area == "df_ambiguity":
            h = hyps_at(env, task)
            hyps = [x["ref"] for s in (h or {}).get("sensors", []) for x in s["hyps"]]
            return position_answer(env, task, _prior(env, task.area), _ellipse(WIDE_M), hyps, task.params["track"])
        if not _ground_track(env, task):
            return abstain(env)
        return position_answer(env, task, _prior(env, task.area), _ellipse(WIDE_M), [], task.params["track"])
    if policy == "nearest_sensor" and is_position_task(task):
        h = hyps_at(env, task)
        if h is None:
            return abstain(env)
        best = max(((s, x) for s in h["sensors"] for x in s["hyps"]), key=lambda sx: (sx[1]["snr_db"], sx[1]["ref"]))
        sp = best[0]["sensor_pos"]
        return position_answer(env, task, (sp["x_m"], sp["y_m"]), _ellipse(WIDE_M), [best[1]["ref"]],
                               task.params["track"])
    if policy == "random_track" and task.area == "operator_query":
        t = task.params.get("t")
        flt = {"time_range_s": [t, t], "klass": "drone"} if t is not None else {"klass": "drone"}
        r = _call(env, "track_search", "drone tracks at the time", filters=flt, limit=20)
        items = sorted(it["ref"] for it in r["data"]["items"]) if r["ok"] else []
        if not items:
            return abstain(env)
        pick = random.Random(stable_seed(task.task_id, seed, "random_track")).choice(items)
        return submit_final(env, {"intent": "operator_query", "track_refs": [pick], "boolean": True,
                                  "evidence_refs": [pick], "confidence": 0.5})
    return abstain(env)
