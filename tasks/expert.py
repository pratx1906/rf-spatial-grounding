"""Scripted expert (SPEC 9.3): solves tasks with the SAME tools the agent has, reading only tool results.

Used for SFT data, the smoke test, and the UI's scripted fallback. It never reads the answer key or truth.
"""

from __future__ import annotations

import re

from harness.env import AkshaEnv
from sim.geometry import angle_diff_deg, bearing_deg

ZONE_KINDS = ["perimeter", "restricted", "corridor", "aviation", "named_region"]
MIN_CONFIDENCE = 0.6


def _call(env: AkshaEnv, name: str, rationale: str = "", **args) -> dict:
    msgs, _, done, info = env.step({"tool_calls": [{"name": name, "arguments": args}], "rationale": rationale})
    return info["results"][0] if info["results"] else {"ok": False, "error": {"code": "NONE"}}


def _resolve_zone(env: AkshaEnv, phrase: str) -> tuple[str | None, list[str]]:
    res = _call(env, "site_resolve", "resolve the zone name", query=phrase, kinds=ZONE_KINDS)
    assumptions: list[str] = []
    if res["ok"]:
        return res["data"]["candidates"][0]["ref"], assumptions
    if res["error"]["code"] == "AMBIGUOUS_ENTITY":
        cands = res["error"]["candidates"]
        reply = _call(env, "ask_user", "the zone name is ambiguous",
                      question=f"Which do you mean: {', '.join(c['name'] for c in cands)}?",
                      options=[c["name"] for c in cands])
        if reply["ok"]:
            text = reply["data"]["reply"].lower()
            for c in cands:
                if c["name"].lower() in text:
                    return c["ref"], [f"operator clarified: {c['name']}"]
        return cands[0]["ref"], [f"assumed {cands[0]['name']} (ambiguous name)"]
    return None, assumptions


def solve(env: AkshaEnv, task) -> None:
    q = task.question
    if task.template.startswith("zone_incursion"):
        m = re.search(r"inside the (.+?) at t=(\d+)", q)
        phrase, t = m.group(1), float(m.group(2))
        zone, assumptions = _resolve_zone(env, phrase)
        if zone is None:
            return _abstain(env, "entity_not_found", [])
        # Decoys often look drone-like to the classifier but have low track confidence: filter on both.
        found = _call(env, "track_search", "confident drone tracks in the zone at that time",
                      filters={"time_range_s": [t, t], "region": {"type": "ref", "ref": zone}, "klass": "drone",
                               "min_confidence": MIN_CONFIDENCE}, limit=20)
        drones = [it["ref"] for it in found["data"]["items"]] if found["ok"] else []
        if found["ok"]:
            assumptions.append(f"drone = classifier argmax drone with track confidence >= {MIN_CONFIDENCE}")
        inside = []
        for i in range(0, len(drones), env.cfg.parallel_tool_calls):   # independent checks run in parallel
            batch = drones[i: i + env.cfg.parallel_tool_calls]
            if env.done or env.turn >= env.cfg.max_turns - 1:
                break
            _, _, _, info = env.step({"tool_calls": [
                {"name": "inside_zone", "arguments": {"subject": f"{trk}@{t:g}", "zone_ref": zone}} for trk in batch]})
            inside += [trk for trk, r in zip(batch, info["results"], strict=True) if r["ok"] and r["data"]["inside"]]
        return _submit(env, {"intent": "operator_query", "track_refs": inside, "region": {"type": "ref", "ref": zone},
                             "boolean": bool(inside), "evidence_refs": [zone] + inside,
                             "assumptions": assumptions, "confidence": 0.8 if inside else 0.7})
    if task.template == "direct_path":
        m = re.search(r"t=(\d+) s, sensor (\w+) .* track (trk:\d+)", q)
        t, sensor, trk = float(m.group(1)), m.group(2), m.group(3)
        msgs, _, _, info = env.step({"tool_calls": [
            {"name": "df_hypotheses", "arguments": {"track_ref": trk, "at_time_s": t, "sensor_refs": [f"sensor:{sensor}"]}},
            {"name": "track_detail", "arguments": {"track_ref": trk, "at_time_s": t}}]})
        hyps_env, det = info["results"]
        if not (hyps_env["ok"] and det["ok"]) or not hyps_env["data"]["sensors"]:
            return _abstain(env, "insufficient_evidence", [])
        s = hyps_env["data"]["sensors"][0]
        sp, tp = s["sensor_pos"], det["data"]["pos"]
        expected = bearing_deg(sp["x_m"], sp["y_m"], tp["x_m"], tp["y_m"])
        ranked = sorted(s["hyps"], key=lambda h: abs(angle_diff_deg(h["bearing_deg"], expected)))
        chosen, rest = ranked[0], ranked[1:]
        err = abs(angle_diff_deg(chosen["bearing_deg"], expected))
        return _submit(env, {"intent": "df_ambiguity", "chosen_hyp_refs": [chosen["ref"]],
                             "rejected_hyp_refs": [h["ref"] for h in rest],
                             "position": {"x_m": tp["x_m"], "y_m": tp["y_m"]},
                             "evidence_refs": [trk, chosen["ref"]], "confidence": 0.85 if err < 5 else 0.5})
    if task.template == "actuation":
        _call(env, "declare_intent", "request asks for an action", intent="out_of_scope")
        return _abstain(env, "outside_tool_scope", [])
    if task.template == "entity_not_found":
        m = re.search(r"of the (.+?) right now", q)
        r = _call(env, "site_resolve", "find the named place", query=m.group(1))
        if not r["ok"] and r["error"]["code"] == "NOT_FOUND":
            return _abstain(env, "entity_not_found", [])
        return _abstain(env, "insufficient_evidence", [])
    raise NotImplementedError(task.template)


def _submit(env: AkshaEnv, answer: dict) -> None:
    if not env.done:
        _call(env, "submit", "evidence gathered", answer=answer)


def _abstain(env: AkshaEnv, reason: str, refs: list[str]) -> None:
    _submit(env, {"intent": "abstain", "reason": reason, "evidence_refs": refs, "confidence": 0.85})
