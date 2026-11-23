"""P9 (d): rule-based failure taxonomy over failed episodes (score < thr) of an eval/run.py output.

    python -m eval.paper.taxonomy --eval runs/eval/sft_v11_v6_t7.json --set eval/frozen/v6/tasks.jsonl \
        --name sft_v11_v6 [--sample 100] [--out runs/paper/results/P9]

Rules (first match wins; keys only, no truth reads):
  parse_format       end_reason format_failed, or an error that names JSON/schema/parse
  budget_exhausted   end_reason max_turns / wallclock / budget, or no answer at all
  verifier_rejected  end_reason verifier_failed (the evidence verifier refused the final answer)
  over_abstain       abstained on an answerable task (the key has no abstain reason)
  missed_abstain     answered an unanswerable task (key reason) without abstaining
  wrong_abstain_reason  abstained on an unanswerable task with the wrong reason
  wrong_intent       answer intent differs from the task area
  ghost_picked       df_ambiguity: a chosen hyp is not one of the key's LOS hyps
  missed_direct      direct_all / reject_ghosts: only LOS hyps chosen, but not all of them
  position_off       df/drone_search: answer position further than tau_m from the key (or missing)
  position_imprecise  the same, within tau_m but below the full-credit radius (partial credit < thr)
  wrong_zone         operator_query with a zone/region in the key: wrong region ref, or wrong in-zone boolean
  wrong_track        operator_query: track set differs from the key
  wrong_protocol / wrong_boolean   operator_query: protocol or (non-zone) yes/no differs from the key
  wrong_event        replay: wrong evt refs or time
  incomplete_collateral / wrong_siting   collateral or siting answer not matching the key
  other              anything else (partial credit below thr)
Rows from older eval runs carry no `answer` dump; answer-dependent rules then give `no_answer_dump`.
Writes <name>_taxonomy.csv (one row per failed episode), <name>_taxonomy_summary.json, and a stratified random sample
of `--sample` failures (<name>_taxonomy_sample.csv) with an empty `manual_class` column for the manual check.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = {"max_turns", "wallclock", "budget_exhausted", "max_tokens", "timeout"}


def _base(r: str) -> str:
    return r.split("@", 1)[0]


def _xy(p) -> tuple[float, float] | None:
    if isinstance(p, dict) and "x_m" in p and "y_m" in p:
        return float(p["x_m"]), float(p["y_m"])
    return None


def _region_ref(ans: dict) -> str | None:
    reg = ans.get("region")
    if isinstance(reg, dict):
        return reg.get("ref") or reg.get("zone_ref")
    return reg if isinstance(reg, str) else None


def classify(row: dict, task: dict) -> str:
    """One failure class for a failed episode (see module doc)."""
    end = row.get("end_reason") or ""
    err = (row.get("error") or "").lower()
    key = task.get("answer_key") or {}
    area = task.get("area")
    if end == "format_failed" or any(w in err for w in ("json", "schema", "parse")):
        return "parse_format"
    if end in BUDGET or (end not in ("submitted", "verifier_failed") and not row.get("intent")):
        return "budget_exhausted"
    if end == "verifier_failed":
        return "verifier_rejected"
    unanswerable = "reason" in key
    if row.get("abstained"):
        if not unanswerable:
            return "over_abstain"
        ans = row.get("answer")
        if ans is not None and ans.get("reason") != key["reason"]:
            return "wrong_abstain_reason"
        return "other"
    if unanswerable:
        return "missed_abstain"
    if row.get("intent") and row["intent"] != area:
        return "wrong_intent"
    ans = row.get("answer")
    if ans is None:
        return "no_answer_dump"
    if area == "df_ambiguity":
        los = {_base(h) for h in key.get("los_hyp_refs") or []}
        chosen = {_base(h) for h in ans.get("chosen_hyp_refs") or []}
        if chosen and los and not chosen <= los:
            return "ghost_picked"
        if los and chosen < los and task.get("template") in ("direct_all", "reject_ghosts"):
            return "missed_direct"
    if area in ("df_ambiguity", "drone_search") and _xy(key.get("position")):
        got = _xy(ans.get("position") or ans.get("final_position"))
        tau = float(key.get("tau_m") or 0)
        if got is None or math.dist(got, _xy(key["position"])) > tau:
            return "position_off"
        return "position_imprecise"                      # within tau_m, partial credit below thr
    if area == "operator_query":
        kz = key.get("region_ref") or (key.get("zone") if key.get("boolean") is not None else None)
        if key.get("region_ref") and _region_ref(ans) != key["region_ref"]:
            return "wrong_zone"
        if kz and key.get("boolean") is not None and ans.get("boolean") is not None \
                and bool(ans["boolean"]) != bool(key["boolean"]):
            return "wrong_zone"
        kt = {_base(t) for t in key.get("track_refs") or []}
        at = {_base(t) for t in ans.get("track_refs") or []}
        if kt != at:
            return "wrong_track"
        if key.get("protocol") and ans.get("protocol") != key["protocol"]:
            return "wrong_protocol"
        if key.get("boolean") is not None and ans.get("boolean") is not None                 and bool(ans["boolean"]) != bool(key["boolean"]):
            return "wrong_boolean"
    if area == "replay":
        ke = {_base(e) for e in key.get("evt_refs") or []}
        ae = {_base(e) for e in ans.get("evt_refs") or []}
        if ke != ae or (key.get("time_s") is not None and ans.get("time_s") is not None
                        and abs(float(ans["time_s"]) - float(key["time_s"])) > 5):
            return "wrong_event"
    if area == "collateral":
        return "incomplete_collateral"
    if area == "siting":
        return "wrong_siting"
    return "other"


def run(eval_json: str | Path, set_path: str | Path, thr: float = 0.8) -> list[dict]:
    tasks = {}
    for x in Path(REPO / set_path).read_text().splitlines():
        if x.strip():
            t = json.loads(x)
            tasks[t["task_id"]] = t
    rows = json.loads(Path(REPO / eval_json).read_text())["rows"]
    out = []
    for r in rows:
        if r["score"] >= thr or r["task_id"] not in tasks:
            continue
        out.append({"task_id": r["task_id"], "area": r["area"], "template": r["template"], "score": r["score"],
                    "end_reason": r.get("end_reason"), "turns": r.get("turns"), "tool_calls": r.get("tool_calls"),
                    "abstained": r.get("abstained"), "has_answer_dump": "answer" in r,
                    "class": classify(r, tasks[r["task_id"]])})
    return out


def summary(fails: list[dict], n_rows: int) -> dict:
    c = Counter(f["class"] for f in fails)
    by_area = {}
    for f in fails:
        by_area.setdefault(f["area"], Counter())[f["class"]] += 1
    return {"episodes": n_rows, "failed": len(fails),
            "class_share": {k: round(v / max(len(fails), 1), 3) for k, v in c.most_common()},
            "class_count": dict(c.most_common()), "by_area": {a: dict(v.most_common()) for a, v in by_area.items()}}


def stratified_sample(fails: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n failures, proportional per class (at least one per class), deterministic."""
    rng = random.Random(seed)
    by = {}
    for f in fails:
        by.setdefault(f["class"], []).append(f)
    if len(fails) <= n:
        return list(fails)
    out = []
    for k, v in sorted(by.items()):
        m = max(1, round(n * len(v) / len(fails)))
        out += rng.sample(v, min(m, len(v)))
    rng.shuffle(out)
    return out[:n]


def main(argv: list[str] | None = None) -> int:
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default="runs/paper/results/P9")
    a = ap.parse_args(argv)
    out = Path(REPO / a.out)
    out.mkdir(parents=True, exist_ok=True)
    fails = run(a.eval, a.set, a.thr)
    n_rows = len(json.loads(Path(REPO / a.eval).read_text())["rows"])
    s = summary(fails, n_rows)
    s.update({"eval": a.eval, "set": a.set, "thr": a.thr})
    (out / f"{a.name}_taxonomy_summary.json").write_text(json.dumps(s, indent=1))
    cols = ["task_id", "area", "template", "score", "end_reason", "turns", "tool_calls", "abstained",
            "has_answer_dump", "class"]
    for fn, rows, extra in ((f"{a.name}_taxonomy.csv", fails, []),
                            (f"{a.name}_taxonomy_sample.csv", stratified_sample(fails, a.sample), ["manual_class"])):
        with open(out / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols + extra)
            w.writeheader()
            for r in rows:
                w.writerow({**r, **{e: "" for e in extra}})
    print(json.dumps({k: s[k] for k in ("episodes", "failed", "class_share")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
