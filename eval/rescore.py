"""Offline re-scoring of saved eval.run outputs under every scorer version and variant (paper Revision 2).

    python -m eval.rescore runs/paper/results/P1/expert_v6/eval.json [more.json ...] [--nominal N.json]
    python -m eval.rescore --all          # every runs/paper/results/**/eval.json + runs/eval/sft_v1{1,4}_*_t7.json

Writes `rescore.json` next to a results-dir `eval.json` (runs/paper/results/<task>/<run>/rescore.json), else
`<stem>.rescore.json` next to the file; never touches eval.json or summary.json. Fields:
- score_v1: the stored row scores (tasks.scoring.score_answer, what every number so far used);
- score_v2: AUDIT_CODE C2 (a locate answer without a position scores 0 on the position half), from the saved
  answers (`row["answer"]`). Rows saved without answers (older outputs) cannot be re-scored under C2: then
  score_v2 is null and `score_v2_bounds` gives [lower, upper] (lower = every position-keyed df_ambiguity answer
  treated as position-less: 0.5 x v1);
- score_v2_verifier_strict: v2, and an answer the verifier rejected (end_reason verifier_failed, or a final
  verifier_code under the verifier-feedback-off ablation) scores 0 (C7); score_v1_verifier_strict likewise;
- headline: the same three on the C6 headline mask (tasks.scoring.is_headline: no collateral / siting areas, no
  task where the empty / "no" answer scores 1.0); non_headline the rest;
- per_area, near_object split (positive vs empty key, C5), the heldout_v1_disjoint subset when the run covers it,
  C2/C7 counts, a v1 recompute check, tool timeouts (C8) when logged;
- pairing (Npair / M* sets, when the nominal partner run is found or given): eval.pairing's twin failures under the
  tracker key and the true-emitter key (C3) and the act/abstain pair accuracy.
Every mean has a two-level cluster bootstrap 95 % CI (scenario seed, then rows; eval.run.cluster_bootstrap); with
--samples > 1 every sample row is a value in its seed's cluster.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DISJOINT = REPO / "eval" / "frozen" / "heldout_v1_disjoint" / "tasks.jsonl"


def _ci(clusters: dict) -> dict:
    from eval.run import cluster_bootstrap

    vals = [v for vs in clusters.values() for v in vs]
    if not vals:
        return {"n": 0, "mean": None, "ci95": None, "n_clusters": 0}
    cb = cluster_bootstrap(clusters)
    return {"n": cb["n"], "mean": round(cb["mean"], 4), "ci95": cb["ci95"], "n_clusters": cb["n_clusters"]}


def _verifier_rejected(row: dict) -> bool:
    """The final answer failed the verifier: ended verifier_failed, or (verifier feedback off) submitted with a
    failing verifier_code."""
    if row.get("end_reason") == "verifier_failed":
        return True
    return row.get("intent") is not None and row.get("verifier_code") is not None


def _has_position_key(task) -> bool:
    return task.area == "df_ambiguity" and (task.answer_key or {}).get("position") is not None


def rescore_rows(rows: list[dict], tasks: dict) -> list[dict]:
    """Per row: {task_id, seed, area, template, headline, v1, v2, v2_lo, v2_hi, strict flags...}. `tasks`: task_id
    -> Task. v2 is None when the row has no saved answer and C2 could apply (then v2_lo / v2_hi bound it)."""
    from eval.run import scenario_of
    from tasks.scoring import empty_scores_full, is_headline, score_answer, score_answer_v2
    from tools.control import SubmitArgs

    out = []
    for r in rows:
        t = tasks[r["task_id"]]
        v1 = float(r["score"])
        has_ans = "answer" in r
        ans = None
        if has_ans and r["answer"] is not None:
            ans = SubmitArgs.model_validate({"answer": r["answer"]}).answer
        c2 = False
        recompute_ok = None
        if has_ans:
            v2 = v1
            if ans is not None:
                c2 = (_has_position_key(t) and ans.intent == "df_ambiguity" and ans.position is None)
                if c2:
                    v2 = float(score_answer_v2(t, ans))
                if not (t.area == "siting" and (t.answer_key or {}).get("zones_wkt")):   # zones need the episode
                    recompute_ok = abs(float(score_answer(t, ans)) - v1) < 1e-6
            lo = hi = v2
        else:
            maybe = _has_position_key(t) and r.get("intent") == "df_ambiguity"
            v2 = None if maybe else v1
            lo, hi = (round(0.5 * v1, 3), v1) if maybe else (v1, v1)
        rej = _verifier_rejected(r)
        out.append({"task_id": r["task_id"], "sample": r.get("sample", 0), "seed": scenario_of(r), "area": t.area,
                    "template": t.template, "headline": is_headline(t), "empty_key": empty_scores_full(t),
                    "v1": v1, "v2": v2, "v2_lo": lo, "v2_hi": hi, "c2_applied": c2, "verifier_rejected": rej,
                    "v1_strict": 0.0 if rej else v1, "v2_strict": None if v2 is None else (0.0 if rej else v2),
                    "v2_strict_lo": 0.0 if rej else lo, "v2_strict_hi": 0.0 if rej else hi,
                    "recompute_ok": recompute_ok, "n_timeouts": r.get("n_timeouts")})
    return out


def _block(rs: list[dict]) -> dict:
    """The score fields of a row subset, each with its cluster CI."""
    def cl(field):
        c = defaultdict(list)
        for x in rs:
            c[x["seed"]].append(x[field])
        return c

    out = {"n": len(rs), "score_v1": _ci(cl("v1")), "score_v1_verifier_strict": _ci(cl("v1_strict"))}
    if rs and all(x["v2"] is not None for x in rs):
        out["score_v2"] = _ci(cl("v2"))
        out["score_v2_verifier_strict"] = _ci(cl("v2_strict"))
    else:
        out["score_v2"] = out["score_v2_verifier_strict"] = None
        if rs:
            out["score_v2_bounds"] = [round(statistics.mean(x["v2_lo"] for x in rs), 4),
                                      round(statistics.mean(x["v2_hi"] for x in rs), 4)]
            out["score_v2_verifier_strict_bounds"] = [round(statistics.mean(x["v2_strict_lo"] for x in rs), 4),
                                                      round(statistics.mean(x["v2_strict_hi"] for x in rs), 4)]
    return out


def _flat(b: dict) -> dict:
    """Compact per-area view: means only."""
    g = lambda k: (b.get(k) or {}).get("mean")  # noqa: E731
    return {"n": b["n"], "v1": g("score_v1"), "v2": g("score_v2"), "v2_strict": g("score_v2_verifier_strict"),
            **({"v2_bounds": b["score_v2_bounds"]} if "score_v2_bounds" in b else {})}


def load_tasks(set_path: str) -> tuple[dict, dict]:
    from tasks.base import Task

    p = REPO / set_path
    raw = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {r["task_id"]: Task.from_row(r) for r in raw}, {r["task_id"]: r for r in raw}


def rescore(res: dict, nominal: dict | None = None, world_key: bool = True) -> dict:
    """The rescore summary of one eval.run output (`nominal`: the paired N run for an M* set)."""
    from eval.run import set_sha256

    tasks, raw = load_tasks(res["set"])
    rows = res["rows"]
    rs = rescore_rows(rows, tasks)
    head = [x for x in rs if x["headline"]]
    out = {"set": res["set"], "set_sha256": set_sha256(REPO / res["set"]), "policy": res.get("policy"),
           "model_id": res.get("model_id"), "ablation": res.get("ablation"), "samples": res.get("samples", 1),
           "answers_available": all("answer" in r for r in rows),
           "scorer": {"v1": "tasks.scoring.score_answer", "v2": "tasks.scoring.score_answer_v2 (C2)",
                      "verifier_strict": "verifier-rejected answer = 0 (C7)",
                      "headline": "tasks.scoring.is_headline (C6)"},
           **_block(rs),
           "headline": _block(head), "non_headline": _block([x for x in rs if not x["headline"]]),
           "per_area": {a: _flat(_block([x for x in rs if x["area"] == a])) for a in sorted({x["area"] for x in rs})},
           "c2": {"n_position_keys": sum(_has_position_key(tasks[x["task_id"]]) for x in rs),
                  "n_applied": sum(x["c2_applied"] for x in rs),
                  "n_unresolved_no_answer": sum(x["v2"] is None for x in rs)},
           "c7": {"n_verifier_rejected": sum(x["verifier_rejected"] for x in rs),
                  "v1_mean_on_rejected": round(statistics.mean(x["v1"] for x in rs if x["verifier_rejected"]), 4)
                  if any(x["verifier_rejected"] for x in rs) else None},
           "c6": {"n_headline": len(head), "n_empty_key": sum(x["empty_key"] for x in rs),
                  "n_collateral_siting": sum(x["area"] in ("collateral", "siting") for x in rs)},
           "v1_recompute": {"checked": sum(x["recompute_ok"] is not None for x in rs),
                            "mismatch": sum(x["recompute_ok"] is False for x in rs)}}
    no = [x for x in rs if x["template"] == "near_object"]
    if no:
        out["near_object_split"] = {"positive_key": _block([x for x in no if not x["empty_key"]]),
                                    "empty_key": _block([x for x in no if x["empty_key"]])}
    if DISJOINT.exists():
        ids = {json.loads(x)["task_id"] for x in DISJOINT.read_text().splitlines() if x.strip()}
        sub = [x for x in rs if x["task_id"] in ids]
        if sub:
            out["heldout_disjoint"] = {**_block(sub), "per_family": {
                f: _flat(_block([x for x in sub if x["template"] == f])) for f in sorted({x["template"] for x in sub})}}
    if any(x["n_timeouts"] is not None for x in rs):
        from eval.run import timeout_summary

        out["timeouts"] = timeout_summary(rows)
    if nominal is not None and any(r.get("pair_id") for r in raw.values()):
        from eval.pairing import WorldKeys, paired_report

        _nt, nraw = load_tasks(nominal["set"])
        out["pairing"] = paired_report(nominal, nraw, res, raw, world=WorldKeys() if world_key else None)
        out["pairing"]["nominal_set"] = nominal["set"]
    return out


def out_path(p: Path) -> Path:
    return p.with_name("rescore.json") if p.name == "eval.json" else p.with_name(p.stem + ".rescore.json")


def nominal_partner(p: Path, res: dict) -> Path | None:
    """For <dir>/<model>_v6_M*/eval.json the paired nominal run <dir>/<model>_v6_Npair/eval.json, if it exists."""
    m = re.search(r"(v\d+)_M\w+$", p.parent.name)
    if p.name != "eval.json" or not m:
        return None
    cand = p.parent.with_name(p.parent.name[: m.start()] + f"{m.group(1)}_Npair") / "eval.json"
    return cand if cand.exists() else None


def all_inputs() -> list[Path]:
    ps = sorted((REPO / "runs" / "paper" / "results").glob("**/eval.json"))
    ps += sorted(p for v in ("11", "14") for p in (REPO / "runs" / "eval").glob(f"sft_v{v}_*_t7.json"))
    return ps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--nominal", default=None, help="paired nominal run for an M* input (default: found by name)")
    ap.add_argument("--no-world-key", action="store_true")
    ap.add_argument("--print", action="store_true", help="print a one-line table per input")
    a = ap.parse_args(argv)
    paths = [Path(p) if Path(p).is_absolute() else REPO / p for p in a.inputs] + (all_inputs() if a.all else [])
    table = []
    for p in paths:
        res = json.loads(p.read_text())
        if "rows" not in res or "set" not in res:
            continue
        npath = Path(a.nominal) if a.nominal else nominal_partner(p, res)
        nominal = json.loads(npath.read_text()) if npath else None
        rep = {"source": str(p.relative_to(REPO)).replace("\\", "/"),
               **({"nominal_source": str(npath.relative_to(REPO)).replace("\\", "/")} if npath else {}),
               **rescore(res, nominal, world_key=not a.no_world_key)}
        out_path(p).write_text(json.dumps(rep, indent=1))
        g = lambda b, k: (b.get(k) or {}).get("mean")  # noqa: E731
        h = rep["headline"]
        table.append([rep["source"], rep["n"], g(rep, "score_v1"), g(rep, "score_v2"), rep.get("score_v2_bounds"),
                      g(rep, "score_v2_verifier_strict"), h["n"], g(h, "score_v1"), g(h, "score_v2"),
                      h.get("score_v2_bounds"), g(h, "score_v2_verifier_strict")])
        print(json.dumps(table[-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
