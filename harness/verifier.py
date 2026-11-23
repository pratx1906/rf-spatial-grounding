"""Answer gate (SPEC 7.4): grounding, consistency recompute, reference resolution, safety.

Every rule runs on every answer and reports {rule, ok, applied[, code]} in `VerifierResult.rules` (the per-question
audit trace, demo-v2 S5.5); the FIRST failing rule (in the order below) is the verdict the model sees, exactly as
before. Rules (demo-v2 S5.2 / S5.4 additions marked +):
  grounding               cited refs came from a tool (+ region.ref, siting zones; + collateral refs past the
                          20-row tool cap count as grounded: they are in the episode's full collateral list)
  recompute_inside_zone   listed tracks re-checked inside the zone
  df_consistency          hyp refs from df_hypotheses; not chosen and rejected; + a chosen hyp that fit_candidates
                          explains as a reflection while a direct hyp of the same node was rejected; + position near a
                          fitted candidate
  collateral_complete     + every violation collateral_check found for that footprint is listed (full list, no cap)
  drone_belief            final position within ~4 sigma of the fused belief; waypoints flown
  rf_ambiguity            + low model trust / high ghost risk at every hearing node: abstain or widen (harness/abstain)
  reference_resolution    "the emitter" with several matches
  human_confirm           collateral needs requires_human_confirm
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from harness.abstain import HINT as RF_HINT
from harness.abstain import rf_ambiguity, widened
from harness.ledger import Ledger

POS_TOL_SIGMA = 3.0
POS_TOL_MIN_M = 150.0


@dataclass
class VerifierResult:
    ok: bool
    code: str | None = None
    message: str = ""
    details: dict = field(default_factory=dict)
    rules: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"ok": self.ok}
        if not self.ok:
            d.update(code=self.code, message=self.message, **self.details)
        return d


def _cited(answer) -> list[str]:
    refs = list(answer.evidence_refs)
    for name in ("track_refs", "chosen_hyp_refs", "rejected_hyp_refs", "violations", "evt_refs", "node_candidates",
                 "waypoints_used"):
        refs += list(getattr(answer, name, []) or [])
    if getattr(answer, "fp_ref", None):
        refs.append(answer.fp_ref)
    region = getattr(answer, "region", None)
    if region is not None and getattr(region, "type", None) == "ref":
        refs.append(region.ref)
    for z in getattr(answer, "zones", None) or []:
        if getattr(z, "type", None) == "ref":
            refs.append(z.ref)
    return refs


def _node(hyp: str) -> str:
    return hyp.split(":", 1)[-1].split("_", 1)[0]


# --- rules: each returns None (not applicable), True (pass) or a failing VerifierResult -------------------------------
def _grounding(answer, ledger, episode, **_):
    collateral = set()
    if getattr(answer, "fp_ref", None):
        collateral = set(episode.get("collateral", {}).get(answer.fp_ref, []))
    missing = [r for r in _cited(answer) if not ledger.has(r) and Ledger.base(r) not in collateral]
    if missing:
        citable = sorted(r for r in ledger.refs if "@" not in r)[:20]
        return VerifierResult(False, "UNGROUNDED", "cited refs were not returned by any tool in this episode",
                              {"refs": missing[:5], "citable_refs": citable,
                               "hint": "Cite only refs exactly as tools returned them (e.g. trk:1234, zone:x). "
                                       "An abstain may have empty evidence_refs."})
    return True


def _recompute_inside_zone(answer, ledger, recompute, **_):
    # re-run exactly the inside_zone checks the answer rests on (same subject@t, same zone) and confirm every listed
    # track was found inside. Claims with no check are not recomputed.
    if not (answer.intent == "operator_query" and answer.region is not None
            and getattr(answer.region, "type", None) == "ref" and answer.region.ref.startswith("zone:")):
        return None
    listed = {Ledger.base(r) for r in answer.track_refs}
    for tool, args, _ in ledger.check_calls:
        subj = args.get("subject")
        if tool != "inside_zone" or args.get("zone_ref") != answer.region.ref or not isinstance(subj, str) \
                or Ledger.base(subj) not in listed:
            continue
        env = recompute("inside_zone", args)
        if env.get("ok") and not env["data"]["inside"]:
            return VerifierResult(False, "INCONSISTENT", "a listed track is outside the zone at the checked time",
                                  {"track": subj, "recomputed_inside": False,
                                   "hint": "List only tracks that inside_zone confirms are inside."})
    return True


def _df_consistency(answer, ledger, **_):
    if answer.intent != "df_ambiguity":
        return None
    known = {h for hs in ledger.hyps.values() for h in hs}
    known |= {Ledger.base(h) for h in known}
    stray = [h for h in answer.chosen_hyp_refs + answer.rejected_hyp_refs
             if h not in known and Ledger.base(h) not in known]
    if stray:
        return VerifierResult(False, "INCONSISTENT", "hypothesis refs did not come from df_hypotheses",
                              {"refs": stray[:5], "hint": "Call df_hypotheses and use its hyp: refs."})
    if set(answer.chosen_hyp_refs) & set(answer.rejected_hyp_refs):
        return VerifierResult(False, "INCONSISTENT", "a hypothesis is both chosen and rejected",
                              {"hint": "Each hyp: ref goes in exactly one list."})
    # chosen vs fit_candidates explained_as (top candidate of the latest rt_match fit that tested the hyp)
    chosen = set(answer.chosen_hyp_refs)
    bad = []
    for h in answer.chosen_hyp_refs:
        top = None
        for fit in reversed(ledger.fits):
            if fit.get("method") != "rt_match" or not fit.get("candidates"):
                continue
            c = fit["candidates"][0]
            if any(x["ref"] == h for x in c.get("hyps", [])):
                top = c
                break
        if top is None:
            continue
        me = next(x for x in top["hyps"] if x["ref"] == h)
        if me.get("explained_as") in ("direct", "unexplained", None):
            continue
        alt = [x["ref"] for x in top["hyps"] if _node(x["ref"]) == _node(h) and x.get("explained_as") == "direct"
               and x["ref"] not in chosen]
        if alt:
            bad.append(h)
    if bad:
        return VerifierResult(False, "INCONSISTENT",
                              "a chosen hypothesis is explained as a reflection while a direct one of the same node "
                              "was rejected", {"refs": bad[:5],
                                               "hint": "Choose the hyps fit_candidates explains as direct; reject the "
                                                       "reflections."})
    # position near a candidate that fit_candidates returned in this episode
    if answer.position is not None:
        cands = [c for fit in ledger.fits for c in fit.get("candidates", []) if c.get("position")]
        if cands:
            def gap(c):
                d = math.hypot(answer.position.x_m - c["position"]["x_m"], answer.position.y_m - c["position"]["y_m"])
                a_m = (c.get("uncertainty") or {}).get("a_m") or 0.0
                return d - max(POS_TOL_SIGMA * a_m, POS_TOL_MIN_M)
            best = min(cands, key=gap)
            if gap(best) > 0:
                return VerifierResult(False, "INCONSISTENT", "position is far from every fit_candidates candidate",
                                      {"hint": f"Use a fitted position, e.g. {best['ref']} at "
                                               f"({best['position']['x_m']}, {best['position']['y_m']})."})
    return True


def _collateral_complete(answer, episode, **_):
    if answer.intent != "collateral":
        return None
    checked = episode.get("collateral", {})
    if answer.fp_ref not in checked:
        return VerifierResult(False, "INCONSISTENT", "no collateral_check was run on this footprint",
                              {"hint": f"Call collateral_check(fp_ref={answer.fp_ref}) and list its violations."})
    found = list(dict.fromkeys(checked[answer.fp_ref]))
    stray = [v for v in answer.violations if Ledger.base(v) not in set(found)]
    if stray:
        return VerifierResult(False, "INCONSISTENT", "violations not reported by collateral_check for this fp",
                              {"refs": stray[:5], "hint": "List only refs from that collateral_check result."})
    listed = {Ledger.base(v) for v in answer.violations}
    missing = [v for v in found if v not in listed]
    if missing:                                     # the full list, not the 20 rows the tool shows
        return VerifierResult(False, "INCOMPLETE_COLLATERAL",
                              f"{len(missing)} violation(s) found by collateral_check are not listed",
                              {"missing": missing, "hint": "List every violation collateral_check found for this "
                                                           "footprint, including the missing refs."})
    return True


def _drone_belief(answer, episode, **_):
    if answer.intent != "drone_search" or episode.get("drone") is None:
        return None
    import numpy as np

    sim = episode["drone"]
    d = np.array([answer.final_position.x_m, answer.final_position.y_m]) - sim.mean
    m2 = float(d @ np.linalg.solve(sim.cov, d))
    if m2 > 16.0:                                            # outside ~4 sigma of the fused belief
        est = sim.estimate()["position"]
        return VerifierResult(False, "INCONSISTENT", "final_position is far from the fused estimate",
                              {"hint": f"Use the estimate from drone_state/move_drone ({est['x_m']}, {est['y_m']})."})
    unflown = [w for w in answer.waypoints_used if w not in sim.moves]
    if unflown:
        return VerifierResult(False, "INCONSISTENT", "waypoints_used lists waypoints that were not flown",
                              {"refs": unflown[:5], "hint": "List only waypoints passed to move_drone."})
    return True


def _rf_ambiguity(answer, ledger, rf_rows, **_):
    if answer.intent != "df_ambiguity" or rf_rows is None:
        return None
    chosen = set(answer.chosen_hyp_refs) | {Ledger.base(h) for h in answer.chosen_hyp_refs}
    keys = [k for k, hs in ledger.hyps.items() if chosen & (set(hs) | {Ledger.base(h) for h in hs})]
    if not keys:
        return None
    for key in keys:
        track, t = key.rsplit("@", 1)
        rows = rf_rows(track, float(t)) or []
        r = rf_ambiguity(rows, {_node(h) for h in ledger.hyps[key]})
        if r["ambiguous"] and not widened(answer):
            return VerifierResult(False, "RF_AMBIGUOUS", "the RF model is not trustworthy at this track for any "
                                                         "node that heard it",
                                  {"track": key, "nodes": r["nodes"], "per_node": r["per_node"], "hint": RF_HINT})
    return True


def _reference_resolution(answer, ledger, singular_reference, **_):
    # "the emitter" with several matches needs all listed, an assumption, or abstain.
    if not (singular_reference and answer.intent == "operator_query"):
        return None
    if (ledger.last_search_total or 0) > 1 and len(answer.track_refs) < ledger.last_search_total \
            and not answer.assumptions:
        return VerifierResult(False, "AMBIGUOUS_REFERENCE",
                              f"the question names one emitter but the search matched {ledger.last_search_total}",
                              {"hint": "List all matches, state an assumption, ask_user, or abstain."})
    return True


def _human_confirm(answer, **_):
    # collateral / mitigation answers always require human confirmation (the schema already enforces it)
    if answer.intent != "collateral":
        return None
    if not answer.requires_human_confirm:
        return VerifierResult(False, "NOT_PERMITTED", "collateral answers must set requires_human_confirm=true",
                              {"hint": "Set requires_human_confirm to true."})
    return True


RULES: list[tuple[str, Callable]] = [
    ("grounding", _grounding), ("recompute_inside_zone", _recompute_inside_zone), ("df_consistency", _df_consistency),
    ("collateral_complete", _collateral_complete), ("drone_belief", _drone_belief), ("rf_ambiguity", _rf_ambiguity),
    ("reference_resolution", _reference_resolution), ("human_confirm", _human_confirm)]


def verify(answer, ledger: Ledger, recompute: Callable[[str, dict], dict], singular_reference: bool = False,
           episode: dict | None = None, rf_rows: Callable[[str, float], list[dict]] | None = None) -> VerifierResult:
    """`recompute(tool, args)` re-runs a deterministic tool through the executor and returns its envelope.
    `rf_rows(track_ref, t_s)` returns per-node {sensor, trust, ghost_risk} rows at the track estimate (None: the
    RF-ambiguity rule is off)."""
    kw = {"ledger": ledger, "recompute": recompute, "singular_reference": singular_reference,
          "episode": episode or {}, "rf_rows": rf_rows}
    first: VerifierResult | None = None
    report = []
    for name, rule in RULES:
        try:
            out = rule(answer, **kw)
        except Exception as e:  # noqa: BLE001 - a broken rule must not crash the episode; it is logged as an error
            report.append({"rule": name, "ok": True, "applied": False, "error": f"{type(e).__name__}: {e}"})
            continue
        if isinstance(out, VerifierResult):
            report.append({"rule": name, "ok": False, "applied": True, "code": out.code})
            first = first or out
        else:
            report.append({"rule": name, "ok": True, "applied": out is True})
    res = first or VerifierResult(True)
    res.rules = report
    return res
