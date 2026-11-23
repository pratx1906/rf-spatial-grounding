"""Classical-pipeline baselines for the RF grounding paper (plan 2026-10-28 T4, E3; audit M5). They run the real
tools through AkshaEnv (truth separation holds) with the question's structured slots (`task.params`: oracle parsing,
so the OQ part is an upper bound for a slot-filling pipeline). No ray-traced matching (`rt_match`) is used.

- tracker_only: positions are the truth-free tracker's estimate (`track_detail`); for DF bearing choices, each
  sensor's hypothesis closest to the bearing towards that estimate; operator queries the tracker can answer
  (track_search + inside_zone / distance / track_detail) run the scripted expert on the oracle slots.
- triangulate_only: per sensor the strongest bearing (highest SNR), then `fit_candidates(method=triangulate)`;
  answers DF tasks only.
- residual_check (paper Revision 3.1): a wrong-twin detector without an LLM. It runs the bearing-residual
  self-calibration `sensor_consistency` (leave-one-out fixes vs each node's bearing; the opt-in tool, exposed by
  eval.run for this policy) in the question's time window (the whole scenario if the window has no epochs). If any
  node is suspect it abstains (insufficient_evidence: a flag for the operator); otherwise it answers as tracker_only.
Everything else abstains (scored 0 on answerable tasks).
"""

from __future__ import annotations

from eval.trivial import abstain, hyps_at, is_position_task, position_answer, submit_final
from harness.env import AkshaEnv
from sim.geometry import angle_diff_deg
from tasks.expert import _call

POLICIES = ("tracker_only", "triangulate_only", "residual_check")
# operator-query families answered from tracker outputs alone (no RF model, signals or geometry tools)
TRACKER_OQ = ("zone_incursion", "zone_incursion_ambiguous", "count_in_zone", "band_in_zone", "swarm_in_zone",
              "near_object", "near_object_ambiguous", "nearest_drone", "tracks_in_band", "which_zone",
              "swarm_members", "genuine_drone", "high_altitude")


def _df_answer(env: AkshaEnv, task, chosen: list[str], hyps: list[str], pos: dict | None, unc: dict | None) -> None:
    track = task.params["track"]
    if not chosen:
        return abstain(env)
    submit_final(env, {"intent": "df_ambiguity", "chosen_hyp_refs": chosen,
                       "rejected_hyp_refs": [h for h in hyps if h not in chosen],
                       "position": {"x_m": pos["x_m"], "y_m": pos["y_m"]} if pos else None,
                       "uncertainty": unc if pos else None, "evidence_refs": [track] + chosen[:5],
                       "confidence": 0.6})


def tracker_only(env: AkshaEnv, task) -> None:
    if task.area == "operator_query" and task.template in TRACKER_OQ:
        from tasks.expert_kandla import solve

        return solve(env, task)
    if task.area == "df_ambiguity" and "track" in task.params:
        h = hyps_at(env, task)
        if h is None:
            return abstain(env)
        hyps = [x["ref"] for s in h["sensors"] for x in s["hyps"]]
        chosen = [min(s["hyps"], key=lambda x: (abs(angle_diff_deg(x["bearing_deg"],
                                                                    s["bearing_to_track_estimate_deg"])), x["ref"]))
                  ["ref"] for s in h["sensors"]]
        pos = unc = None
        if is_position_task(task):
            d = _call(env, "track_detail", "tracker estimate", track_ref=task.params["track"],
                      at_time_s=h["t_s"])
            if d["ok"]:
                pos, unc = d["data"]["pos"], d["data"]["uncertainty"]
        return _df_answer(env, task, chosen, hyps, pos, unc)
    if task.area == "drone_search" and task.template.startswith("ds_"):
        d = _call(env, "track_detail", "tracker estimate", track_ref=task.params["track"])
        if not d["ok"]:
            return abstain(env)
        p = d["data"]["pos"]
        return position_answer(env, task, (p["x_m"], p["y_m"]), d["data"]["uncertainty"], [],
                               task.params["track"])
    return abstain(env)


def triangulate_only(env: AkshaEnv, task) -> None:
    if task.area != "df_ambiguity" or "track" not in task.params:
        return abstain(env)
    h = hyps_at(env, task)
    if h is None:
        return abstain(env)
    hyps = [x["ref"] for s in h["sensors"] for x in s["hyps"]]
    chosen = [max(s["hyps"], key=lambda x: (x["snr_db"], x["ref"]))["ref"] for s in h["sensors"]]
    pos = unc = None
    if len(chosen) >= 2:
        fit = _call(env, "fit_candidates", "triangulate the strongest bearings", hyp_refs=chosen[:15],
                    method="triangulate")
        if fit["ok"]:
            c = fit["data"]["candidates"][0]
            pos, unc = c["position"], c["uncertainty"]
    return _df_answer(env, task, chosen, hyps, pos if is_position_task(task) else None, unc)


def twin_suspects(env: AkshaEnv, task) -> list[str] | None:
    """Suspect DF nodes from sensor_consistency around the question time (else the whole scenario); None if the tool
    gave no verdict."""
    t = task.params.get("t")
    r = _call(env, "sensor_consistency", "bearing residuals vs the twin", **({"at_time_s": t} if t is not None else {}))
    if not r["ok"] and t is not None:
        r = _call(env, "sensor_consistency", "bearing residuals vs the twin (whole scenario)")
    return list(r["data"]["suspect_refs"]) if r["ok"] else None


def residual_check(env: AkshaEnv, task) -> None:
    sus = twin_suspects(env, task)
    if sus:
        return abstain(env)            # flag: the RF twin disagrees with the observations
    return tracker_only(env, task)


def solve(env: AkshaEnv, task, policy: str) -> None:
    if task.area == "unanswerable":
        return abstain(env)
    return {"tracker_only": tracker_only, "triangulate_only": triangulate_only,
            "residual_check": residual_check}[policy](env, task)
