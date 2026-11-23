"""Diverse question generator for SFT v1.4 (PLAN docs/superpowers/plans/2026-10-23-rf-convergence/PLAN.md Task 7).

Why: v1.2/v1.3 lost the held-out family `weak_low_alt` (0.93 -> 0.13) because every training siting question spelled
out height and threshold the same way, so the model copied parameters (always 40 m; an invented 90 deg threshold
when it was omitted). The live stress pass (runs/s78/training_targets.md) adds wording misses: drone search with
"allotted / allowed / terse" budgets, passive-voice collateral questions abstained as outside_tool_scope, "can S2
see X directly" read as a DF direct-path question, and entry questions whose rewording lost the verb "entered".

Two mechanisms, both offline and key-safe:
- `reword(row, split, rng)`: a new wording of an existing task whose key does not depend on the wording (drone
  search, LOS, direct path, collateral, entered_between). Every slot (ref, number, sensor, place) comes from the
  task's structured `params` / payload / key, so the answer key and the expert trajectory stay valid.
- `siting_question(...)`: weak_df / weak_detect questions with and without height and threshold, in varied units.
  When a value is omitted the task uses the DOCUMENTED DEFAULT (threshold 10 deg / detection 0.5, height 40 m) and
  records it in `params["assumed"]`; the scripted expert passes the default to `weak_zones` and states the
  assumption in its answer. The key is computed by the caller (train.sft_v14_data) with the effective values.

Held out: `weak_low_alt` is never generated (no 15 m height, no "blind spot" / "low drone" wording), nor is any other
family in scripts.make_tasks.HELD_OUT. Each family has `train` wordings and disjoint `val` wordings: the val bank
builds the held-out-LIKE validation set (unseen wordings on the val seeds) used to pick the checkpoint.
"""

from __future__ import annotations

import random
import re

DEFAULTS = {"df_err_p90_deg": 10.0, "detect_prob": 0.5, "height_m": 40.0}
HEIGHTS = (5.0, 40.0, 80.0)                    # coverage-cache heights; never 15 m (held-out weak_low_alt)
DF_THRESHOLDS = (5.0, 10.0, 15.0, 20.0)        # plausible 2-node bearing error thresholds (deg)
DETECT_THRESHOLDS = (0.3, 0.5, 0.7)
SITING_FAMILIES = ("weak_df", "weak_detect")
BAND_PHRASE = {"ism_2400": ["2.4 GHz", "the 2.4 GHz band", "2400 MHz"],
               "ism_5800": ["5.8 GHz", "the 5.8 GHz band", "5800 MHz"],
               "sub_ghz_868_915": ["868/915 MHz", "the sub-GHz band", "900 MHz"]}     # = tasks.kandla.gen
_LOW_ALT = re.compile(r"(?i)blind spot|low drone|low-flying|\b15 ?m\b|\b15 metres\b|\b49 ft\b")

HEIGHT_STYLES = ("m", "m_tight", "metres", "ft", "agl")
DEG_STYLES = ("deg", "sym", "degrees")


def is_low_alt_wording(q: str) -> bool:
    """True for the held-out weak_low_alt wording (never generated)."""
    return bool(_LOW_ALT.search(q))


def fmt_height(h: float, style: str) -> str:
    if style == "ft":
        return f"{round(h / 0.3048)} ft"
    return {"m": f"{h:g} m", "m_tight": f"{h:g}m", "metres": f"{h:g} metres", "agl": f"{h:g} m AGL"}[style]


def fmt_deg(x: float, style: str) -> str:
    return {"deg": f"{x:g} deg", "sym": f"{x:g}°", "degrees": f"{x:g} degrees"}[style]


def fmt_prob(p: float, rng: random.Random) -> str:
    return rng.choice([f"{p * 100:g} %", f"{p * 100:g}%", f"{p:g}"])


# --- wording banks ------------------------------------------------------------------------------------------------
# Siting templates: {b} band, {z} zone, {thr} threshold clause ("" when omitted), {h} height clause ("" when omitted).
# weak_detect templates that say "probably" mean the default 0.5 and take no {thr}.
WORDINGS: dict[str, dict[str, list[str]]] = {
    "weak_df": {
        "train": ["Where is {b} DF accuracy weak inside the {z}{thr}{h}?",
                  "Where inside the {z} is {b} direction finding poor{thr}{h}?",
                  "Which parts of the {z} have weak {b} DF coverage{thr}{h}?",
                  "Find the areas in the {z} where {b} bearings are unreliable{thr}{h}.",
                  "{B}: where is DF weak in the {z}{h}{thr}?",
                  "Where would {b} DF be inaccurate over the {z}{h}{thr}?"],
        "val": ["In the {z}, where would {b} DF struggle to get a good bearing{thr}{h}?",
                "Map the weak {b} DF areas over the {z}{h}{thr}."]},
    "weak_detect": {
        "train": ["Where inside the {z} would a drone{h} on {b} probably go undetected?",
                  "Where in the {z} is {b} detection probability below {p}{h}?",
                  "Which parts of the {z} have weak {b} detection coverage{h}{thr}?",
                  "Where could a {b} drone{h} slip through undetected in the {z}?",
                  "{B}: where is detection weak inside the {z}{h}{thr}?"],
        "val": ["In the {z}, where would we likely miss a {b} drone{h}?",
                "Show the {b} detection gaps across the {z}{h}{thr}."]},
    "ds": {
        "train": ["Fly the DF drone to refine the position of {x}. You are allotted {n} moves.",
                  "Fly the DF drone to refine the position of {x}. You are allowed {n} moves.",
                  "DF drone to {x}, {n} moves.",
                  "Send the DF drone to localise {x}; you get {n} moves.",
                  "Reposition the drone to get a better fix on {x} (budget: {n} moves).",
                  "Use the airborne direction finder on {x}. Max {n} moves.",
                  "Get the DF drone over to {x} and pin it down, {n} moves allowed.",
                  "Task the drone-borne DF on {x}: refine its position in at most {n} moves."],
        "train_nobudget": ["Send the DF drone after {x} and tighten its position.",
                           "Use the DF drone to get a better fix on {x}.",
                           "DF drone to {x}, refine the position."],
        "val": ["Launch the DF UAV toward {x} to narrow down where it is; {n} moves available.",
                "{x}: refine with the drone, {n} moves max."]},
    "los_check": {
        "train": ["Can {s} see {x} directly at t={t} s, or is something in the way?",
                  "At t={t} s, is there a clear line of sight from {s} to {x}?",
                  "Is the view from {s} to {x} unobstructed at t={t} s?",
                  "Does {s} have a clear view of {x} at t={t} s?",
                  "LOS check: {s} to {x} at t={t} s. Clear?",
                  "At t={t} s, can {s} see {x} directly, with nothing in between?"],
        "val": ["At t={t} s, would {s} have an unblocked sightline to {x}?",
                "Is {x} visible from {s} without obstruction at t={t} s?"]},
    "direct_path": {
        "train": ["At t={t} s, track {x} is reported by {s} with {n} bearings. Which one is the direct path and "
                  "which are reflections?",
                  "{S} shows {n} bearings on {x} at t={t} s; which is the direct bearing and which are multipath?",
                  "Of the {n} bearings {s} has on {x} at t={t} s, which is direct?",
                  "For {s} only: {x} has {n} bearings at t={t} s. Pick the direct one, flag the reflections.",
                  "{x} at t={t} s: {s} sees {n} bearings. Which is the direct path?"],
        "val": ["{S} logs {n} bearings for {x} at t={t} s. Separate the true bearing from the reflections."]},
    "jam_all": {
        "train": ["What would be affected (receivers above {th} dBm) if {b} were jammed from {e} at {p} dBm EIRP?",
                  "If {e} jammed {b} at {p} dBm EIRP, which receivers above {th} dBm would be hit?",
                  "Which receivers would be hit above {th} dBm by a {p} dBm {b} jammer at {e}?"],
        "val": ["Suppose {e} radiates {b} jamming at {p} dBm EIRP: what gets affected above {th} dBm?"]},
    "jam_protected": {
        "train": ["Which protected assets would be disturbed above {th} dBm if {b} were jammed from {e} at {p} dBm?",
                  "If our {p} dBm {b} jammer at {e} were switched on, which protected assets would be affected "
                  "above {th} dBm?",
                  "Would any protected asset be exposed above {th} dBm to a {p} dBm {b} jammer at {e}?"],
        "val": ["Suppose {e} jams {b} at {p} dBm: any protected assets disturbed above {th} dBm?"]},
    "jam_links": {
        "train": ["Would any of our own links or civil comms be disrupted (threshold {th} dBm) by {b} jamming at "
                  "{p} dBm from {e}?",
                  "Are our links or civil comms exposed above {th} dBm to a {p} dBm {b} jammer at {e}?",
                  "If {e} jammed {b} at {p} dBm, would our own links or civil comms be hit above {th} dBm?"],
        "val": ["Suppose {e} jams {b} at {p} dBm: are our links or civil comms disrupted above {th} dBm?"]},
    "jam_aviation": {
        "train": ["Would the airport traffic zone be reached above {th} dBm by a {p} dBm {b} jammer at {e}?",
                  "Is the airport traffic zone exposed above {th} dBm if {e} jams {b} at {p} dBm?",
                  "If we jammed {b} from {e} at {p} dBm, would it affect the airport traffic zone above {th} dBm?"],
        "val": ["Suppose {e} jams {b} at {p} dBm: does it reach the airport traffic zone above {th} dBm?"]},
    "jam_fratricide": {
        "train": ["If our DF sensors are exposed to a {b} jamming signal from {e} at {p} dBm, which ones would be "
                  "affected?",
                  "Which of our DF nodes would be hit by our own {p} dBm {b} jammer at {e}?",
                  "Our {b} jammer at {e}, {p} dBm: which of our own DF sensors would be blinded?",
                  "Which own DF sensors are affected if {e} jams {b} at {p} dBm?"],
        "val": ["Suppose {e} jams {b} at {p} dBm: which of our own DF sensors would be affected?"]},
    "entered_between": {
        "train": ["Which tracks came into the {z} between t={t0} s and t={t1} s?",
                  "Between t={t0} s and t={t1} s, which tracks got into the {z}?",
                  "List the tracks that entered the {z} from t={t0} s to t={t1} s.",
                  "Tracks that entered the {z} between t={t0} s and t={t1} s?",
                  "Which tracks crossed into the {z} between t={t0} s and t={t1} s?"],
        "val": ["From t={t0} s to t={t1} s, which tracks made an entry into the {z}?"]},
}
DS_FAMILIES = ("ds_hover", "ds_hover_budget2", "ds_loiter", "ds_swarm")       # ds_wide_prior is held out
REWORD_FAMILIES = DS_FAMILIES + ("los_check", "direct_path", "jam_all", "jam_protected", "jam_links", "jam_aviation",
                                 "jam_fratricide", "entered_between")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s and not s.startswith("trk:") else s


def _num(x: float) -> str:
    return f"{x:.0f}" if float(x) == int(x) else f"{x:g}"


def _sensor(s: str, rng: random.Random) -> str:
    return rng.choice([s, f"sensor {s}"])


def siting_question(fam: str, band_phrase: str, zone_name: str, height: float | None, thr: float | None,
                    split: str, rng: random.Random) -> tuple[str, dict]:
    """A weak_df / weak_detect question. height / thr None = omitted (the documented default applies).
    Returns (question, params-delta {metric, threshold, height_m, assumed})."""
    metric = "df_err_p90_deg" if fam == "weak_df" else "detect_prob"
    tpls = WORDINGS[fam][split]
    if fam == "weak_detect" and thr not in (None, DEFAULTS["detect_prob"]):
        tpls = [t for t in tpls if "probably" not in t]            # "probably" means the default 0.5 only
    if fam == "weak_detect" and thr is None:
        tpls = [t for t in tpls if "{p}" not in t]                 # {p} needs an explicit probability
    tpl = rng.choice(tpls)
    assumed = []
    eff_thr = DEFAULTS[metric] if thr is None else float(thr)
    eff_h = DEFAULTS["height_m"] if height is None else float(height)
    if thr is None and "probably" not in tpl:
        assumed.append(f"threshold not given: used the default {_thr_text(metric, eff_thr)}")
    if height is None:
        assumed.append("height not given: used the default 40 m")
    if thr is None or "probably" in tpl:
        thr_c = ""
    elif metric == "df_err_p90_deg":
        d = fmt_deg(eff_thr, rng.choice(DEG_STYLES))
        thr_c = rng.choice([f" (bearing error above {d})", f" with bearing error over {d}",
                            f" where the 2-node bearing error exceeds {d}", f", error threshold {d}"])
    else:
        thr_c = rng.choice([f" (detection probability below {fmt_prob(eff_thr, rng)})",
                            f", detection below {fmt_prob(eff_thr, rng)}"])
    if height is None:
        h_c = ""
    else:
        hs = fmt_height(eff_h, rng.choice(HEIGHT_STYLES))
        h_c = rng.choice([f" at {hs}", f" flying at {hs}"]) if fam == "weak_detect" else \
            rng.choice([f" for drones at {hs}", f" at {hs} altitude", f" for a UAV flying at {hs}"])
    q = tpl.format(b=band_phrase, B=_cap(band_phrase), z=zone_name, thr=thr_c, h=h_c,
                   p=fmt_prob(eff_thr, rng) if "{p}" in tpl else "")
    q = re.sub(r"\s{2,}", " ", q)
    return q, {"metric": metric, "threshold": eff_thr, "height_m": eff_h, "assumed": assumed}


def _thr_text(metric: str, thr: float) -> str:
    return f"{thr:g} deg 2-node bearing error" if metric == "df_err_p90_deg" else f"detection probability {thr:g}"


def reword(row: dict, split: str, rng: random.Random) -> dict | None:
    """A new wording of a task whose key does not depend on the wording; None if the family is not covered."""
    fam = row["template"]
    if fam not in REWORD_FAMILIES:
        return None
    p = row.get("params") or {}
    if fam in DS_FAMILIES:
        n = int((row.get("drone_payload") or {}).get("budget_moves", 4))
        bank = WORDINGS["ds"][split]
        if split == "train" and n == 4 and rng.random() < 0.3:     # 4 = the default budget, may go unsaid
            bank = WORDINGS["ds"]["train_nobudget"]
        q = rng.choice(bank).format(x=p["track"], n=n)
    elif fam == "los_check":
        q = rng.choice(WORDINGS[fam][split]).format(s=_sensor(p["sensor"], rng), x=p["track"], t=_num(p["t"]))
    elif fam == "direct_path":
        s = _sensor(p["sensor"], rng)
        q = rng.choice(WORDINGS[fam][split]).format(s=s, S=_cap(s), x=p["track"], t=_num(p["t"]),
                                                    n=len(row["answer_key"]["all_hyp_refs"]))
    elif fam.startswith("jam_"):
        if not str(p.get("effector", "")).startswith("sensor:"):
            return None                                             # point jammers (jam_point) are held out
        sid = p["effector"].split(":", 1)[1]
        bands = [b for b in BAND_PHRASE[p["band"]] if not b.startswith("the ")]      # attributive: "a 33 dBm 2.4 GHz jammer"
        q = rng.choice(WORDINGS[fam][split]).format(e=_sensor(sid, rng), b=rng.choice(bands),
                                                    p=_num(p["eirp_dbm"]), th=_num(p["threshold_dbm"]))
    else:                                                           # entered_between
        z = p["zone_phrase"][4:] if p["zone_phrase"].lower().startswith("the ") else p["zone_phrase"]
        q = rng.choice(WORDINGS[fam][split]).format(z=z, t0=_num(p["t0"]), t1=_num(p["t1"]))
    q = _cap(re.sub(r"\s{2,}", " ", q))
    out = dict(row)
    out["question"] = q
    out["task_id"] = f"{row['task_id']}-d{rng.randrange(10 ** 6):06d}"
    out["diverse_of"] = row["task_id"]
    out["categories"] = list(row.get("categories") or []) + ["diverse"]
    return out
