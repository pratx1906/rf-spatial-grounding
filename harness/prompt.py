"""Static, byte-identical prompt prefix (SPEC 7.5): system prompt + tool schemas. Nothing episode-varying here."""

from __future__ import annotations

import re

from tools.registry import REGISTRY, default_names, load_all, schemas_json, tool_schemas

SYSTEM_PROMPT = """You are SceneScout, an assistant for counter-UAS operators at Kandla port. Answer by calling tools, then submit one grounded answer.
Rules:
- Every turn is a tool call; the only way to answer is submit. Up to 3 independent calls per turn.
- Frame: local ENU metres, bearings deg clockwise from north, sim seconds, MHz, dBm.
- Cite only refs returned by tools in this episode (evidence_refs). Lower confidence when evidence is weak or trust < 0.5.
- AMBIGUOUS_ENTITY: ask_user once with the candidate names; never silently pick. NOT_FOUND: abstain entity_not_found.
- Requests to act (jam, take over, land, intercept): abstain outside_tool_scope. You recommend; you never act.
- Drones are tracks, not site objects. Decoys look drone-like with low track confidence (<0.6).
Playbooks:
- Drone in/near a place at t: site_resolve -> track_search(filters region/time/klass drone/min_confidence 0.6) -> inside_zone or spatial_relation per track -> submit operator_query (track_refs, region, boolean for yes/no).
- Swarm: track_search(filters group_id) or sql_readonly; count members.
- Which bearing is direct / where is the emitter: df_hypotheses(track, t) -> fit_candidates(all hyp refs, method rt_match) -> hyps explained_as direct are chosen, reflection/unexplained rejected -> submit df_ambiguity with position = top candidate. predict_aoa / geometry_check explain a ghost (via_ref).
- Drone search: drone_state -> plan_waypoints -> move_drone (best waypoint; repeat while moves_left and a_m is large) -> submit drone_search with the last estimate and waypoints_used.
- Coverage / siting: weak_zones(metric, band, threshold, region, height_m) and/or suggest_node(region, band) -> submit siting with zones as {"type":"ref","ref":"ras:..."} and node_candidates (nodecand: refs). coverage_query for numbers, model_trust for reliability.
- Collateral of an effector: effect_footprint(effector, band, eirp_dbm, threshold_dbm) -> collateral_check(fp_ref) -> submit collateral with fp_ref, violations = all refs it listed, requires_human_confirm=true.
- When/what happened: event_search(event_type, subject_ref, time_range_s) or track_history -> submit replay with evt_refs and time_s.
submit.answer fields by intent (all need evidence_refs and confidence 0..1; assumptions optional):
operator_query: track_refs[], region {"type":"ref","ref":"zone:..."} or null, boolean or null
df_ambiguity: chosen_hyp_refs[], rejected_hyp_refs[], position {x_m,y_m} or null
drone_search: final_position {x_m,y_m}, uncertainty {kind:"ellipse",a_m,b_m,theta_deg}, waypoints_used[]
siting: zones[], node_candidates[] | collateral: fp_ref, violations[] | replay: evt_refs[], time_s
abstain: reason in no_sensor_coverage|entity_not_found|ambiguous_unresolved|outside_tool_scope|insufficient_evidence"""

# One playbook line per opt-in tool, added only when EnvConfig.tools_extra exposes it (plan 2026-10-28 T3).
OPT_IN_PLAYBOOK = {
    "sensor_consistency": "- Biased DF node: sensor_consistency(track_ref or sensor_ref) -> a suspect node has a "
                          "bearing bias; lower confidence, or abstain insufficient_evidence if the answer rests on it.",
}
OPT_IN_PLAYBOOK_FIX = {
    "sensor_consistency": "- Sensor health (first, for any track, zone or DF question): sensor_consistency() -> a suspect "
                          "node has a heading bias; prefer tracks/fixes that do not rest on it (fit_candidates with the "
                          "other nodes' hyps), still answer, lower confidence and name the node in assumptions.",
}
CONTROL_TOOLS = ("declare_intent", "ask_user", "submit")
FIT_METHODS = ("triangulate", "rt_match")
_PB_HEAD, _PB_TAIL = "Playbooks:\n", "\nsubmit.answer fields"


def _names_tool(text: str, tools: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in tools)


def drop_playbook(system: str, tools: set[str]) -> str:
    """Remove playbook text that names any of `tools`. A line whose recipe (first sentence) names one is dropped
    whole; otherwise only the later sentences naming one are dropped (e.g. 'model_trust for reliability' leaves the
    siting recipe intact). Text outside the Playbooks section never changes."""
    if not tools:
        return system
    head, rest = system.split(_PB_HEAD, 1)
    body, tail = rest.split(_PB_TAIL, 1)
    out = []
    for line in body.split("\n"):
        sents = re.split(r"(?<=\.) ", line)
        if _names_tool(sents[0], tools):
            continue
        out.append(" ".join([sents[0]] + [s for s in sents[1:] if not _names_tool(s, tools)]))
    return head + _PB_HEAD + "\n".join(out) + _PB_TAIL + tail


def _check(cfg) -> None:
    known = set(REGISTRY)
    for key in ("tools_disabled", "tools_extra", "playbook_disabled"):
        bad = [t for t in getattr(cfg, key) if t not in known]
        if bad:
            raise ValueError(f"{key}: unknown tool(s) {bad}")
    if set(cfg.tools_disabled) & set(CONTROL_TOOLS):
        raise ValueError(f"tools_disabled: control tools {CONTROL_TOOLS} cannot be disabled")
    bad = [t for t in cfg.tools_extra if not REGISTRY[t].opt_in]
    if bad:
        raise ValueError(f"tools_extra: {bad} are default tools, not opt-in")
    bad = [m for m in cfg.fit_methods_disabled if m not in FIT_METHODS]
    if bad or len(set(cfg.fit_methods_disabled)) >= len(FIT_METHODS):
        raise ValueError(f"fit_methods_disabled: {cfg.fit_methods_disabled} (allowed: one of {FIT_METHODS})")


def enabled_tools(cfg=None) -> list[str]:
    """Tool names exposed under `cfg` (sorted; the default set when cfg is None or has no switches)."""
    load_all()
    if cfg is None:
        return default_names()
    _check(cfg)
    return sorted(n for n in REGISTRY if n not in cfg.tools_disabled
                  and (not REGISTRY[n].opt_in or n in cfg.tools_extra))


def static_prefix(cfg=None) -> tuple[str, list[dict], str]:
    """(system prompt, tool schemas, canonical JSON of the schemas). Identical bytes every episode.

    With no switches set in `cfg` (EnvConfig) this is exactly the historical prefix every adapter was trained on."""
    load_all()
    if cfg is None or not (cfg.tools_disabled or cfg.tools_extra or cfg.playbook_disabled):
        if cfg is not None:
            _check(cfg)
        return SYSTEM_PROMPT, tool_schemas(), schemas_json()
    names = enabled_tools(cfg)
    system = SYSTEM_PROMPT
    book = OPT_IN_PLAYBOOK_FIX if getattr(cfg, "consistency_line", "abstain") == "fix" else OPT_IN_PLAYBOOK
    extra = [book[t] for t in cfg.tools_extra if t in book]
    if extra:
        head, tail = system.split(_PB_TAIL, 1)
        system = head + "\n" + "\n".join(extra) + _PB_TAIL + tail
    system = drop_playbook(system, set(cfg.tools_disabled) | set(cfg.playbook_disabled))
    return system, tool_schemas(names), schemas_json(names)
