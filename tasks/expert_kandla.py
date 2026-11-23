"""Scripted expert for the Kandla task families (SPEC 9.3). Uses only tools through AkshaEnv and the question's
structured slots (`task.params`); never the answer key or truth. Produces SFT trajectories and the expert baseline.
"""

from __future__ import annotations

from harness.abstain import WIDE_A_M, WIDE_CONF_MAX, rf_ambiguity
from harness.env import AkshaEnv
from tasks.expert import _abstain, _call, _resolve_zone

MINC = 0.6


def _submit(env: AkshaEnv, answer: dict) -> None:
    """Submit; on the two verifier rulings an expert can satisfy from the rejection itself (demo-v2 S5.2/S5.4), fix
    the answer and resubmit once: INCOMPLETE_COLLATERAL (violations past the 20 rows collateral_check shows) and
    RF_AMBIGUOUS (low model trust at every hearing node: widen the uncertainty and lower the confidence)."""
    if env.done:
        return
    r = _call(env, "submit", "evidence gathered", answer=answer)
    if r.get("ok") or env.done:
        return
    err = r.get("error") or {}
    if err.get("code") == "INCOMPLETE_COLLATERAL":
        answer = {**answer, "violations": list(answer.get("violations") or []) + list(err.get("missing") or [])}
    elif err.get("code") == "RF_AMBIGUOUS":
        answer = {**answer, "confidence": min(float(answer["confidence"]), WIDE_CONF_MAX - 0.05),
                  "assumptions": (list(answer.get("assumptions") or []) +
                                  ["RF model trust is low at every node that heard it: uncertainty widened"])[-5:]}
        if answer.get("position"):
            u = answer.get("uncertainty") or {"kind": "ellipse", "a_m": 0.0, "b_m": 0.0, "theta_deg": 0.0}
            answer["uncertainty"] = {**u, "a_m": max(float(u["a_m"]), WIDE_A_M), "b_m": max(float(u["b_m"]), WIDE_A_M)}
    else:
        return
    _call(env, "submit", "fix what the verifier flagged", answer=answer)


def _step(env: AkshaEnv, calls: list[dict]) -> list[dict]:
    _, _, _, info = env.step({"tool_calls": calls})
    return info["results"]


def _parallel(env: AkshaEnv, calls: list[dict]) -> list[dict]:
    out = []
    for i in range(0, len(calls), env.cfg.parallel_tool_calls):
        if env.done or env.turn >= env.cfg.max_turns - 1:
            break
        out += _step(env, calls[i: i + env.cfg.parallel_tool_calls])
    return out


def _drones_in(env, zone: str, t: float) -> tuple[list[str], list[str]]:
    found = _call(env, "track_search", "confident drone tracks in the zone at that time",
                  filters={"time_range_s": [t, t], "region": {"type": "ref", "ref": zone}, "klass": "drone",
                           "min_confidence": MINC}, limit=20)
    cands = [it["ref"] for it in found["data"]["items"]] if found["ok"] else []
    res = _parallel(env, [{"name": "inside_zone", "arguments": {"subject": f"{c}@{t:g}", "zone_ref": zone}}
                          for c in cands])
    inside = [c for c, r in zip(cands, res, strict=False) if r["ok"] and r["data"]["inside"]]
    return inside, [f"drone = classifier argmax drone with track confidence >= {MINC}"]


def solve(env: AkshaEnv, task) -> None:
    # plan variety (SFT v2): about a third of episodes declare their intent first, an optional but valid step,
    # so the policy does not learn one rigid script per question type (drone search is turn-limited: skipped)
    from sim.geometry import stable_seed

    intent = {"operator_query": "operator_query", "df_ambiguity": "df_ambiguity", "siting": "siting",
              "collateral": "collateral", "replay": "replay"}.get(task.area)
    if intent and stable_seed(task.task_id, "intent") % 3 == 0:
        _call(env, "declare_intent", "state the question type", intent=intent)
    if solve_more(env, task):
        return None
    f, p = task.template, task.params
    if f == "reject_ghosts":
        f = "direct_all"
    if f in ("zone_incursion", "count_in_zone", "zone_incursion_ambiguous"):
        zone, assumptions = _resolve_zone(env, p["zone_phrase"])
        if zone is None:
            return _abstain(env, "entity_not_found", [])
        inside, a2 = _drones_in(env, zone, p["t"])
        return _submit(env, {"intent": "operator_query", "track_refs": inside, "region": {"type": "ref", "ref": zone},
                             "boolean": bool(inside) if f != "count_in_zone" else None,
                             "evidence_refs": [zone] + inside, "assumptions": assumptions + a2,
                             "confidence": 0.8})
    if f == "tracks_in_band":
        t = p["t"]
        r = _call(env, "track_search", "drones on that band at t",
                  filters={"time_range_s": [t, t], "band": p["band"], "klass": "drone", "min_confidence": MINC},
                  limit=20)
        items = r["data"]["items"] if r["ok"] else []
        refs = [it["ref"] for it in items]
        if not items:                                    # negative answer: cite what was seen at t instead
            r2 = _call(env, "track_search", "everything seen at t", filters={"time_range_s": [t, t]}, limit=5)
            items = r2["data"]["items"] if r2["ok"] else []
        return _submit(env, {"intent": "operator_query", "track_refs": refs, "region": None, "boolean": None,
                             "evidence_refs": [it["ref"] for it in items] or [f"sensor:S1"],
                             "assumptions": [f"drone with confidence >= {MINC}; band at t"], "confidence": 0.8})
    if f == "swarm_members":
        d = _call(env, "track_detail", "find the swarm group of the track", track_ref=p["track"])
        gid = d["data"].get("group_id") if d["ok"] else None
        if not gid:
            return _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "boolean": None,
                                 "evidence_refs": [p["track"]], "confidence": 0.4,
                                 "assumptions": ["no swarm group recorded for this track"]})
        r = _call(env, "track_search", "all tracks of the group", filters={"group_id": gid}, limit=20)
        refs = [it["ref"] for it in r["data"]["items"]] if r["ok"] else [p["track"]]
        return _submit(env, {"intent": "operator_query", "track_refs": refs, "boolean": None,
                             "evidence_refs": refs, "confidence": 0.8, "assumptions": [f"swarm = group {gid}"]})
    if f in ("near_object", "near_object_ambiguous"):
        c, _ = _resolve_any(env, p["obj_phrase"])
        if c is None:
            return _abstain(env, "entity_not_found", [])
        bx = c["bbox_m"]
        half = max(bx[2] - bx[0], bx[3] - bx[1]) / 2
        t = p["t"]
        s = _call(env, "track_search", "drones around the object", filters={
            "time_range_s": [t, t], "klass": "drone", "min_confidence": MINC,
            "region": {"type": "circle", "center": c["centroid"], "radius_m": p["radius_m"] + half + 30}}, limit=20)
        cands = [it["ref"] for it in s["data"]["items"]] if s["ok"] else []
        res = _parallel(env, [{"name": "spatial_relation", "arguments": {
            "a_ref": f"{x}@{t:g}", "b_ref": c["ref"], "relation": "within_m", "param_m": p["radius_m"]}} for x in cands])
        near = [x for x, q in zip(cands, res, strict=False) if q["ok"] and q["data"]["value"]]
        return _submit(env, {"intent": "operator_query", "track_refs": near, "boolean": bool(near),
                             "region": {"type": "ref", "ref": c["ref"]}, "evidence_refs": [c["ref"]] + near,
                             "confidence": 0.8})
    if f == "which_zone":
        d = _call(env, "track_detail", "position and zones at t", track_ref=p["track"], at_time_s=p["t"])
        if not d["ok"]:
            return _abstain(env, "no_sensor_coverage", [])
        zs = [z for z in d["data"]["zones"] if "restricted" in z]
        if not zs:
            return _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "boolean": False,
                                 "evidence_refs": [p["track"]], "confidence": 0.5})
        return _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "boolean": True,
                             "region": {"type": "ref", "ref": zs[0]}, "evidence_refs": [p["track"], zs[0]],
                             "confidence": 0.8})
    if f == "genuine_drone":
        d = _call(env, "track_detail", "class probabilities and confidence", track_ref=p["track"])
        if not d["ok"]:
            return _abstain(env, "insufficient_evidence", [])
        kp = d["data"]["klass_probs"]
        real = max(kp, key=kp.get) == "drone" and d["data"]["confidence"] >= MINC
        return _submit(env, {"intent": "operator_query", "track_refs": [p["track"]] if real else [], "boolean": real,
                             "evidence_refs": [p["track"]], "confidence": 0.75,
                             "assumptions": [f"genuine = argmax drone and confidence >= {MINC}"]})
    if f == "los_check":
        g = _call(env, "geometry_check", "line of sight from the node to the estimate",
                  **{"from": f"sensor:{p['sensor']}", "to": f"{p['track']}@{p['t']:g}", "mode": "los"})
        if not g["ok"]:
            return _abstain(env, "no_sensor_coverage", [])
        return _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "boolean": g["data"]["result"],
                             "evidence_refs": [f"sensor:{p['sensor']}", p["track"]] + g["data"].get("blockers", [])[:3],
                             "confidence": 0.7, "assumptions": ["track position = estimate; heights estimated"]})
    if f in ("direct_path", "direct_all", "locate", "locate_ghost"):
        args = {"track_ref": p["track"], "at_time_s": p["t"]}
        if f == "direct_path":
            args["sensor_refs"] = [f"sensor:{p['sensor']}"]
        h = _call(env, "df_hypotheses", "bearings on the track", **args)
        if not h["ok"] or not h["data"]["sensors"]:
            return _abstain(env, "insufficient_evidence", [])
        hyps = [x["ref"] for s in h["data"]["sensors"] for x in s["hyps"]]
        if f == "direct_path":
            hyps_all = list(hyps)
            more = _call(env, "df_hypotheses", "other sensors for geometry", track_ref=p["track"], at_time_s=p["t"])
            if more["ok"]:
                hyps_all = [x["ref"] for s in more["data"]["sensors"] for x in s["hyps"]]
            fit = _call(env, "fit_candidates", "test bearings against the site geometry", hyp_refs=hyps_all[:15],
                        method="rt_match")
        else:
            fit = _call(env, "fit_candidates", "test bearings against the site geometry", hyp_refs=hyps[:15],
                        method="rt_match")
        if not fit["ok"]:
            return _abstain(env, "insufficient_evidence", [])
        top = fit["data"]["candidates"][0]
        # one direct bearing per node: the direct-explained hyp with the smallest residual
        best: dict[str, dict] = {}
        for x in top["hyps"]:
            node = x["ref"].split(":", 1)[1].split("_", 1)[0]
            if x["explained_as"] == "direct" and (node not in best or x["residual_deg"] < best[node]["residual_deg"]):
                best[node] = x
        direct = {x["ref"] for x in best.values()}
        chosen = [x for x in hyps if x in direct]
        if f.startswith("locate") and len(best) >= 2:
            tri = _call(env, "fit_candidates", "triangulate the direct bearings", hyp_refs=sorted(direct),
                        method="triangulate")
            if tri["ok"]:
                top = {**top, "position": tri["data"]["candidates"][0]["position"],
                       "uncertainty": tri["data"]["candidates"][0]["uncertainty"]}
        if f == "direct_path" and len(chosen) != 1:
            # one direct per sensor: fall back to the bearing closest to the fitted position
            ranked = sorted(top["hyps"], key=lambda x: x["residual_deg"])
            chosen = [next((x["ref"] for x in ranked if x["ref"] in hyps), hyps[0])]
        if not chosen:
            chosen = [min(top["hyps"], key=lambda x: x["residual_deg"])["ref"]]
        rejected = [x for x in hyps if x not in chosen]
        pos = top["position"] if f.startswith("locate") else None
        return _submit(env, {"intent": "df_ambiguity", "chosen_hyp_refs": chosen, "rejected_hyp_refs": rejected,
                             "position": {"x_m": pos["x_m"], "y_m": pos["y_m"]} if pos else None,
                             "uncertainty": top["uncertainty"] if pos else None,
                             "evidence_refs": [p["track"], top["ref"]] + chosen[:5],
                             "confidence": 0.8 if top["residual_deg_rms"] < 5 else 0.55})
    if f.startswith("ds_"):
        st = _call(env, "drone_state", "start pose and prior")
        if not st["ok"]:
            return _abstain(env, "outside_tool_scope", [])
        used = []
        while not env.done and env.turn < env.cfg.max_turns - 2:
            plan = _call(env, "plan_waypoints", "best next waypoint", target_ref=p["track"], k=1)
            if not plan["ok"] or not plan["data"]["waypoints"] or plan["data"]["moves_left"] <= 0:
                break
            w = plan["data"]["waypoints"][0]["ref"]
            m = _call(env, "move_drone", "take a bearing there", wp_ref=w)
            if not m["ok"]:
                break
            used.append(w)
            st = {"ok": True, "data": {"estimate": m["data"]["estimate"]}}
            if m["data"]["moves_left"] <= 0 or m["data"]["estimate"]["uncertainty"]["a_m"] < 20:
                break
        est = st["data"]["estimate"]
        return _submit(env, {"intent": "drone_search", "final_position": {"x_m": est["position"]["x_m"],
                                                                          "y_m": est["position"]["y_m"]},
                             "uncertainty": est["uncertainty"], "waypoints_used": used,
                             "evidence_refs": [p["track"]] + used, "confidence": 0.8})
    if f in ("weak_df", "weak_detect", "weak_low_alt"):
        w = _call(env, "weak_zones", "weak coverage areas", metric=p["metric"], band=p["band"],
                  threshold=p["threshold"], region={"type": "ref", "ref": p["zone"]}, height_m=p["height_m"], limit=5)
        if not w["ok"]:
            return _abstain(env, "no_sensor_coverage", [])
        refs = [z["ref"] for z in w["data"]["zones"]]
        ans = {"intent": "siting", "zones": [{"type": "ref", "ref": r} for r in refs],
               "node_candidates": [], "evidence_refs": refs or [p["zone"]], "confidence": 0.75}
        if p.get("assumed"):                 # SFT v1.4 diverse questions: documented defaults for omitted values
            ans["assumptions"] = list(p["assumed"])
        return _submit(env, ans)
    if f in ("suggest_one", "suggest_two"):
        s = _call(env, "suggest_node", "rank mount sites", region={"type": "ref", "ref": p["zone"]}, band=p["band"],
                  objective="df_err", k=p["k"], height_m=15)
        if not s["ok"]:
            return _abstain(env, "no_sensor_coverage", [])
        refs = [c["ref"] for c in s["data"]["candidates"]]
        return _submit(env, {"intent": "siting", "zones": [], "node_candidates": refs, "evidence_refs": refs,
                             "confidence": 0.7, "assumptions": ["new-node coverage from a line-of-sight proxy"]})
    if f.startswith("jam_"):
        fp = _call(env, "effect_footprint", "predicted footprint", effector=p["effector"], band=p["band"],
                   eirp_dbm=p["eirp_dbm"], threshold_dbm=p["threshold_dbm"])
        if not fp["ok"]:
            return _abstain(env, "insufficient_evidence", [])
        args = {"fp_ref": fp["data"]["ref"]}
        if p.get("kinds"):
            args["kinds"] = p["kinds"]
        c = _call(env, "collateral_check", "what it would hit", **args)
        vio = [v["ref"] for v in c["data"]["violations"]] if c["ok"] else []
        return _submit(env, {"intent": "collateral", "fp_ref": fp["data"]["ref"], "violations": vio,
                             "requires_human_confirm": True, "evidence_refs": [fp["data"]["ref"]] + vio[:10],
                             "confidence": 0.75})
    if f in ("first_entry", "exit_time", "loiter_start", "band_change", "first_detection", "entered_between"):
        args = {"event_type": p["type"], "limit": 20}
        zone = None
        if "zone_phrase" in p:
            zone, _ = _resolve_zone(env, p["zone_phrase"])
            if zone is None:
                return _abstain(env, "entity_not_found", [])
        if f == "entered_between":
            args.update(subject_ref=zone, time_range_s=[p["t0"], p["t1"]])
        else:
            args["subject_ref"] = p["track"]
        e = _call(env, "event_search", "recorded events", **args)
        items = e["data"]["items"] if e["ok"] else []
        if zone and f != "entered_between":
            items = [i for i in items if i.get("zone_ref") == zone]
        if f == "first_entry":
            items = [i for i in items if i.get("detail") != "at_detection"]
        if not items and f != "entered_between":
            return _abstain(env, "insufficient_evidence", [])
        if f == "entered_between":
            refs = [i["ref"] for i in items if i.get("detail") != "at_detection"]
            return _submit(env, {"intent": "replay", "evt_refs": refs, "time_s": None,
                                 "evidence_refs": refs or [zone], "confidence": 0.8})
        first = items[0]
        return _submit(env, {"intent": "replay", "evt_refs": [first["ref"]], "time_s": first["t_s"],
                             "evidence_refs": [first["ref"]], "confidence": 0.85})
    if f == "actuation":
        _call(env, "declare_intent", "request asks for an action", intent="out_of_scope")
        return _abstain(env, "outside_tool_scope", [])
    if f == "entity_not_found":
        r = _call(env, "site_resolve", "find the named place", query=p["phrase"])
        return _abstain(env, "entity_not_found" if not r["ok"] and r["error"]["code"] == "NOT_FOUND"
                        else "insufficient_evidence", [])
    raise NotImplementedError(f)


def _resolve_any(env, phrase: str):
    """Resolve an object/zone name; on AMBIGUOUS_ENTITY ask the operator once and pick the named candidate."""
    r = _call(env, "site_resolve", "resolve the place", query=phrase)
    if r["ok"]:
        return r["data"]["candidates"][0], []
    if r["error"]["code"] == "AMBIGUOUS_ENTITY":
        cands = r["error"]["candidates"]
        reply = _call(env, "ask_user", "the name is ambiguous",
                      question=f"Which do you mean: {', '.join(c['name'] for c in cands[:6])}?",
                      options=[c["name"] for c in cands[:6]])
        text = reply["data"]["reply"].lower() if reply["ok"] else ""
        for c in sorted(cands, key=lambda c: -len(c["name"])):
            if c["name"].lower() in text:
                d = _call(env, "site_resolve", "resolve the clarified name", query=c["name"])
                if d["ok"]:
                    return d["data"]["candidates"][0], [f"operator clarified: {c['name']}"]
        named = text.replace("i mean", "").strip(" .").strip()          # the operator may name an unlisted one
        if named:
            d = _call(env, "site_resolve", "resolve the clarified name", query=named)
            if d["ok"]:
                return d["data"]["candidates"][0], [f"operator clarified: {named}"]
        return None, []
    return None, []


def _best_direct(hyps_top):
    best = {}
    for x in hyps_top:
        node = x["ref"].split(":", 1)[1].split("_", 1)[0]
        if x["explained_as"] == "direct" and (node not in best or x["residual_deg"] < best[node]["residual_deg"]):
            best[node] = x
    return best


def solve_more(env, task) -> bool:
    """New families (gen_more). Returns False if the family is not handled here."""
    f, p = task.template, task.params
    if f == "protocol_id":
        r = _call(env, "signal_detail", "burst-level signal features of the track", track_ref=p["track"])
        if not r["ok"]:
            _abstain(env, "insufficient_evidence", [])
            return True
        cls = r["data"]["protocol_estimate"]
        _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "protocol": cls if cls != "unknown" else None,
                      "evidence_refs": [p["track"]], "confidence": 0.7,
                      "assumptions": ["protocol from signal features (bandwidth, CP, preamble, chirp), not decoded"]})
        return True
    if f == "pilot_locate":
        r = _call(env, "pilot_search", "controllers co-active with the drone", near_track_ref=p["track"])
        c = r["data"]["candidates"] if r["ok"] else []
        if not c:
            _abstain(env, "insufficient_evidence", [])
            return True
        best = c[0]
        unc = best["uncertainty_m"] or 100.0
        _submit(env, {"intent": "drone_search", "final_position": best["pos"],
                      "uncertainty": {"kind": "ellipse", "a_m": unc, "b_m": unc, "theta_deg": 0.0},
                      "evidence_refs": [p["track"], best["ref"]], "confidence": 0.6,
                      "assumptions": ["pilot = best-scoring co-active ground controller track"]})
        return True
    if f == "noisy_band":
        res = {}
        for b in ("sub_ghz_868_915", "ism_2400", "ism_5800"):
            r = _call(env, "noise_floor", f"measured noise in {b}", sensor_ref=f"sensor:{p['sensor']}", band=b)
            if r["ok"]:
                res[b] = r["data"]["noise_dbm_per_mhz"]
        if not res:
            _abstain(env, "no_sensor_coverage", [])
            return True
        _submit(env, {"intent": "operator_query", "track_refs": [], "band": max(res, key=res.get),
                      "evidence_refs": [f"sensor:{p['sensor']}"], "confidence": 0.75})
        return True
    if f == "ghost_check":
        r = _call(env, "model_trust", "ensemble ghost risk at the drone estimate",
                  track_ref=f"{p['track']}@{p['t']:g}", band=p["band"], sensor_ref=f"sensor:{p['sensor']}",
                  per_sensor=True)
        rows = [x for x in (r["data"].get("per_sensor") or [])] if r["ok"] else []
        row = next((x for x in rows if x["sensor"] == f"sensor:{p['sensor']}"), None)
        if row is None:
            _abstain(env, "no_sensor_coverage", [])
            return True
        _submit(env, {"intent": "operator_query", "track_refs": [p["track"]], "boolean": row["ghost_risk"] > 0.5,
                      "evidence_refs": [f"sensor:{p['sensor']}", p["track"]], "confidence": 0.7,
                      "assumptions": ["ghost = more than half of the RF model ensemble puts the strongest arrival "
                                      "off the direct bearing at the track estimate"]})
        return True
    if f in ("band_in_zone", "swarm_in_zone"):
        zone, assumptions = _resolve_zone(env, p["zone_phrase"])
        if zone is None:
            _abstain(env, "entity_not_found", [])
            return True
        t = p["t"]
        flt = {"time_range_s": [t, t], "region": {"type": "ref", "ref": zone}, "klass": "drone", "min_confidence": MINC}
        if f == "band_in_zone":
            flt["band"] = p["band"]
        r = _call(env, "track_search", "candidates in the zone", filters=flt, limit=20)
        items = r["data"]["items"] if r["ok"] else []
        if f == "swarm_in_zone":
            items = [it for it in items if it.get("group_id")]
        cands = [it["ref"] for it in items]
        res = _parallel(env, [{"name": "inside_zone", "arguments": {"subject": f"{c}@{t:g}", "zone_ref": zone}}
                              for c in cands])
        inside = [c for c, q in zip(cands, res, strict=False) if q["ok"] and q["data"]["inside"]]
        _submit(env, {"intent": "operator_query", "track_refs": inside, "region": {"type": "ref", "ref": zone},
                      "boolean": bool(inside) if f == "swarm_in_zone" else None,
                      "evidence_refs": [zone] + inside, "assumptions": assumptions + ["swarm = track with a group_id"]
                      if f == "swarm_in_zone" else assumptions, "confidence": 0.75})
        return True
    if f == "nearest_drone":
        c, _ = _resolve_any(env, p["obj_phrase"])
        if c is None:
            _abstain(env, "entity_not_found", [])
            return True
        t = p["t"]
        cands = []
        for rad in (600.0, 1500.0):
            r = _call(env, "track_search", "drones around the object", filters={
                "time_range_s": [t, t], "klass": "drone", "min_confidence": MINC,
                "region": {"type": "circle", "center": c["centroid"], "radius_m": rad}}, limit=6)
            cands = [it["ref"] for it in r["data"]["items"]] if r["ok"] else []
            if cands:
                break
        if not cands:
            _submit(env, {"intent": "operator_query", "track_refs": [], "boolean": None,
                          "evidence_refs": [c["ref"]], "confidence": 0.5})
            return True
        res = _parallel(env, [{"name": "spatial_relation", "arguments": {"a_ref": f"{x}@{t:g}", "b_ref": c["ref"],
                                                                         "relation": "distance"}} for x in cands])
        ds = [(q["data"]["value_m"], x) for x, q in zip(cands, res, strict=False) if q["ok"]]
        best = min(ds)[1] if ds else cands[0]
        _submit(env, {"intent": "operator_query", "track_refs": [best], "boolean": None,
                      "region": {"type": "ref", "ref": c["ref"]}, "evidence_refs": [c["ref"], best], "confidence": 0.75})
        return True
    if f == "high_altitude":
        t = p["t"]
        r = _call(env, "track_search", "drones at t with altitude", filters={"time_range_s": [t, t], "klass": "drone",
                                                                          "min_confidence": MINC}, limit=20)
        items = r["data"]["items"] if r["ok"] else []
        refs = [it["ref"] for it in items if (it["pos"].get("z_m") or 0) > p["alt_m"]]
        _submit(env, {"intent": "operator_query", "track_refs": refs, "boolean": None,
                      "evidence_refs": [it["ref"] for it in items] or ["sensor:S1"], "confidence": 0.6,
                      "assumptions": ["altitude from the DF elevation estimate (coarse)"]})
        return True
    if f == "locate_two_sensors":
        h = _call(env, "df_hypotheses", "bearings from the two named sensors", track_ref=p["track"], at_time_s=p["t"],
                  sensor_refs=[f"sensor:{s}" for s in p["sensors"]])
        hyps = [x["ref"] for sv in h["data"]["sensors"] for x in sv["hyps"]] if h["ok"] else []
        if len(hyps) < 2:
            _abstain(env, "insufficient_evidence", [])
            return True
        fit = _call(env, "fit_candidates", "test against the site geometry", hyp_refs=hyps, method="rt_match")
        if not fit["ok"]:
            _abstain(env, "insufficient_evidence", [])
            return True
        top = fit["data"]["candidates"][0]
        best = _best_direct(top["hyps"])
        pos, unc = top["position"], top["uncertainty"]
        if len(best) >= 2:
            tri = _call(env, "fit_candidates", "triangulate the direct bearings",
                        hyp_refs=sorted(x["ref"] for x in best.values()), method="triangulate")
            if tri["ok"]:
                pos, unc = tri["data"]["candidates"][0]["position"], tri["data"]["candidates"][0]["uncertainty"]
        chosen = [x["ref"] for x in best.values()] or [min(top["hyps"], key=lambda x: x["residual_deg"])["ref"]]
        _submit(env, {"intent": "df_ambiguity", "chosen_hyp_refs": chosen,
                      "rejected_hyp_refs": [x for x in hyps if x not in chosen],
                      "position": {"x_m": pos["x_m"], "y_m": pos["y_m"]}, "uncertainty": unc,
                      "evidence_refs": [p["track"], top["ref"]] + chosen, "confidence": 0.6})
        return True
    if f == "first_into_zone":
        zone, _ = _resolve_zone(env, p["zone_phrase"])
        if zone is None:
            _abstain(env, "entity_not_found", [])
            return True
        e = _call(env, "event_search", "zone entries", event_type="zone_enter", subject_ref=zone, limit=20)
        items = [i for i in (e["data"]["items"] if e["ok"] else []) if i.get("detail") != "at_detection"]
        if not items:
            _abstain(env, "insufficient_evidence", [])
            return True
        first = items[0]
        _submit(env, {"intent": "replay", "evt_refs": [first["ref"]], "time_s": first["t_s"],
                      "evidence_refs": [first["ref"], first["track_ref"]], "confidence": 0.85})
        return True
    if f == "last_seen":
        d = _call(env, "track_detail", "track history", track_ref=p["track"])
        if not d["ok"]:
            _abstain(env, "entity_not_found", [])
            return True
        _submit(env, {"intent": "replay", "evt_refs": [], "time_s": d["data"]["history"]["last_seen"],
                      "evidence_refs": [p["track"]], "confidence": 0.85})
        return True
    if f == "rf_ambiguous":
        # bearings and the track's band in one turn, then the RF model's trust at the estimate for every node
        h, d = _step(env, [{"name": "df_hypotheses", "arguments": {"track_ref": p["track"], "at_time_s": p["t"]}},
                           {"name": "track_detail", "arguments": {"track_ref": p["track"], "at_time_s": p["t"]}}])
        if not h["ok"] or not h["data"]["sensors"] or not d["ok"]:
            _abstain(env, "insufficient_evidence", [])
            return True
        band = d["data"]["band"] if "band" in d["data"] else d["data"]["signal"]["band"]
        targs = {"track_ref": f"{p['track']}@{h['data']['t_s']:g}", "band": band, "per_sensor": True}
        tr = _call(env, "model_trust", "is the RF model reliable for the nodes that heard it", **targs)
        heard = {s["sensor"] for s in h["data"]["sensors"]}
        rows = (tr["data"].get("per_sensor") or []) if tr["ok"] else []
        if rf_ambiguity(rows, heard)["ambiguous"]:
            _submit(env, {"intent": "abstain", "reason": "insufficient_evidence", "confidence": 0.8,
                          "evidence_refs": [p["track"]] + sorted(heard),
                          "assumptions": ["every hearing node: model trust < 0.4 or ghost risk >= 0.75"]})
            return True
        hyps = [x["ref"] for s in h["data"]["sensors"] for x in s["hyps"]]
        fit = _call(env, "fit_candidates", "test bearings against the site geometry", hyp_refs=hyps[:15],
                    method="rt_match")
        if not fit["ok"]:
            _abstain(env, "insufficient_evidence", [])
            return True
        top = fit["data"]["candidates"][0]
        chosen = [x["ref"] for x in _best_direct(top["hyps"]).values()] or             [min(top["hyps"], key=lambda x: x["residual_deg"])["ref"]]
        _submit(env, {"intent": "df_ambiguity", "chosen_hyp_refs": chosen,
                      "rejected_hyp_refs": [x for x in hyps if x not in chosen],
                      "position": {"x_m": top["position"]["x_m"], "y_m": top["position"]["y_m"]},
                      "uncertainty": top["uncertainty"], "evidence_refs": [p["track"], top["ref"]] + chosen[:5],
                      "confidence": 0.6})
        return True
    if f in ("future_time", "unknown_track"):
        args = {"track_ref": p["track"]}
        if f == "future_time":
            args["at_time_s"] = p["t"]
        d = _call(env, "track_detail", "look the track up", **args)
        code = d["error"]["code"] if not d["ok"] else None
        _abstain(env, "no_sensor_coverage" if code == "OUT_OF_COVERAGE" else
                 "entity_not_found" if code in ("UNKNOWN_REF", "NOT_FOUND") else "insufficient_evidence", [])
        return True
    return False
