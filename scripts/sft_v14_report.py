"""SFT v1.4 checkpoint selection, gate and results page (PLAN 2026-10-23 rf-convergence Task 7).

    python scripts/sft_v14_report.py select --evals step15=runs/eval/v14sel_step15.json ... --out runs/s79/v14/selected.json
    python scripts/sft_v14_report.py report --base sft_v11 --new sft_v14 --out docs/RESULTS/sft_v14.md \
        [--eval-dir runs/eval] [--suffix ""] [--stress-base X.json --stress-new Y.json]

select: the checkpoint with the best FAMILY-MACRO score on the held-out-like validation tasks (runs/s79/v14/
tasks_val.jsonl: unseen wordings on the val seeds + a seen-family sample), not on loss; within 0.005 of the best the
earliest step wins (least drift from v1.1).

report: eval files <eval-dir>/<model>_<set><suffix>.json for sets v3, v6, v6_W4, v6_W5, heldout. Per set: means, the
paired difference with a task-level and a two-level (scenario, task) cluster CI; held-out per family + macro, with a
cluster CI on the macro difference. Gate (Task 7):
  v3 no-regression   task-CI lower bound of (new - base) on frozen v3 > -0.02 (the M6b gate);
  beats v1.1 on v6   (new - base) on v6 > 0 with task-CI lower bound > 0;
  held-out >= v1.1   family-macro on the held-out set: new >= base (point estimate; CI reported).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.run import cluster_bootstrap, paired_bootstrap, paired_cluster_bootstrap, scenario_of  # noqa: E402

SETS = ("v3", "v6", "v6_W4", "v6_W5", "heldout")
TIE = 0.005
V3_FLOOR = -0.02


def family_macro(rows: list[dict]) -> float:
    fam = defaultdict(list)
    for r in rows:
        fam[r.get("template", "?")].append(float(r["score"]))
    return statistics.mean(statistics.mean(v) for v in fam.values()) if fam else 0.0


def select(results: dict[str, list[dict]]) -> dict:
    """results: {checkpoint label: eval rows}. Labels 'stepN' sort by N; 'final' sorts last."""
    def order(k: str) -> float:
        return float(k[4:]) if k.startswith("step") and k[4:].isdigit() else float("inf")

    sc = {k: round(family_macro(v), 4) for k, v in results.items()}
    best = max(sc.values())
    pick = min((k for k, v in sc.items() if v >= best - TIE), key=order)
    return {"selected": pick, "scores": sc, "best": best, "tie": TIE}


def paired_macro_ci(ra: list[dict], rb: list[dict], n: int = 2000, seed: int = 0) -> dict:
    """Family-macro of (b - a) on shared tasks, with a two-level (scenario, task) bootstrap CI."""
    ia = {r["task_id"]: r for r in ra}
    pairs = [(scenario_of(r), r.get("template", "?"), float(r["score"]) - float(ia[r["task_id"]]["score"]))
             for r in rb if r["task_id"] in ia]
    by_sc = defaultdict(list)
    for sc, fam, d in pairs:
        by_sc[sc].append((fam, d))

    def macro(items):
        fam = defaultdict(list)
        for f, d in items:
            fam[f].append(d)
        return statistics.mean(statistics.mean(v) for v in fam.values())

    keys = sorted(by_sc)
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        items = []
        for _k in keys:
            v = by_sc[keys[rng.randrange(len(keys))]]
            items += rng.choices(v, k=len(v))
        boots.append(macro(items))
    boots.sort()
    return {"n": len(pairs), "n_clusters": len(keys), "diff": round(macro([(f, d) for _, f, d in pairs]), 4),
            "ci95_cluster": [round(boots[int(0.025 * n)], 4), round(boots[min(n - 1, int(0.975 * n))], 4)]}


def gate(cmp: dict) -> dict:
    """cmp: {set: {"diff", "ci95_task", ...}, "heldout_macro": {"base", "new", ...}}."""
    out = {}
    v3 = cmp.get("v3")
    out["v3_no_regression"] = bool(v3 and v3["ci95_task"][0] > V3_FLOOR)
    v6 = cmp.get("v6")
    out["beats_v11_on_v6"] = bool(v6 and v6["diff"] > 0 and v6["ci95_task"][0] > 0)
    h = cmp.get("heldout_macro")
    out["heldout_ge_v11"] = bool(h and h["new"] >= h["base"])
    out["pass"] = all(out.values())
    return out


def _rows(p: Path) -> list[dict] | None:
    return json.loads(p.read_text())["rows"] if p.exists() else None


def cmd_select(a) -> int:
    res = {}
    for spec in a.evals:
        k, _, p = spec.partition("=")
        rows = _rows(REPO / p)
        if rows:
            res[k] = rows
    if not res:
        print("no eval files")
        return 1
    out = select(res)
    Path(REPO / a.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def cmd_report(a) -> int:
    ed = REPO / a.eval_dir
    cmp, lines = {}, []
    lines += [f"# SFT v1.4 vs v1.1 ({a.new} vs {a.base})", "",
              "PLAN 2026-10-23 rf-convergence Task 7. Same harness and tool outputs for both models (the weak_zones "
              "threshold guardrail is on for both). Paired over shared tasks. CIs: task-level bootstrap and two-level "
              "(scenario, task) cluster bootstrap; frozen v3 holds 2 test seeds, so its cluster CI mostly reflects "
              "the two scenarios.", "",
              "| Set | n | v1.1 | v1.4 | diff | CI task | CI cluster (seeds) |", "|---|---|---|---|---|---|---|"]
    held = None
    for s in SETS:
        rb, rn = _rows(ed / f"{a.base}_{s}{a.suffix}.json"), _rows(ed / f"{a.new}_{s}{a.suffix}.json")
        if not (rb and rn):
            lines.append(f"| {s} | - | missing | | | | |")
            continue
        t, c = paired_bootstrap(rb, rn)["_all"], paired_cluster_bootstrap(rb, rn)["_all"]
        mb = statistics.mean(r["score"] for r in rb)
        mn = statistics.mean(r["score"] for r in rn)
        cmp[s] = {"n": t["n"], "base": round(mb, 3), "new": round(mn, 3), "diff": t["diff"], "ci95_task": t["ci95"],
                  "ci95_cluster": c["ci95"], "n_clusters": c["n_clusters"]}
        lines.append(f"| {s} | {t['n']} | {mb:.3f} | {mn:.3f} | {t['diff']:+.3f} | {t['ci95']} | {c['ci95']} "
                     f"({c['n_clusters']}) |")
        if s == "heldout":
            held = (rb, rn)
    if held:
        rb, rn = held
        m = paired_macro_ci(rb, rn)
        cmp["heldout_macro"] = {"base": round(family_macro(rb), 3), "new": round(family_macro(rn), 3), **m}
        lines += ["", "## Held-out families (never trained)", "",
                  f"Family macro: v1.1 **{cmp['heldout_macro']['base']:.3f}**, v1.4 **{cmp['heldout_macro']['new']:.3f}**,"
                  f" diff {m['diff']:+.3f}, cluster CI {m['ci95_cluster']} ({m['n_clusters']} seeds).", "",
                  "| Family | n | v1.1 | v1.4 | diff | CI cluster |", "|---|---|---|---|---|---|"]
        fams = sorted({r["template"] for r in rn})
        for f in fams:
            fb = [r for r in rb if r["template"] == f]
            fn = [r for r in rn if r["template"] == f]
            pc = paired_cluster_bootstrap(fb, fn)["_all"]
            lines.append(f"| {f} | {len(fn)} | {statistics.mean(r['score'] for r in fb):.3f} | "
                         f"{statistics.mean(r['score'] for r in fn):.3f} | {pc['diff']:+.3f} | {pc['ci95']} |")
    for s in ("v6", "v3"):                       # per-family table on the big sets
        if s in cmp:
            rb, rn = _rows(ed / f"{a.base}_{s}{a.suffix}.json"), _rows(ed / f"{a.new}_{s}{a.suffix}.json")
            lines += ["", f"## Per family on {s}", "", "| Family | n | v1.1 | v1.4 | diff |", "|---|---|---|---|---|"]
            for f in sorted({r["template"] for r in rn}):
                fb = [r["score"] for r in rb if r["template"] == f]
                fn = [r["score"] for r in rn if r["template"] == f]
                if fb and fn:
                    lines.append(f"| {f} | {len(fn)} | {statistics.mean(fb):.3f} | {statistics.mean(fn):.3f} | "
                                 f"{statistics.mean(fn) - statistics.mean(fb):+.3f} |")
    for lab, p in (("v1.1", a.stress_base), ("v1.4", a.stress_new)):
        if p and (REPO / p).exists():
            s = json.loads((REPO / p).read_text())["summary"]
            cmp[f"stress_{lab}"] = {k: s.get(k) for k in ("runs", "correct", "as_written_correct", "answer_p50_s",
                                                           "answer_p95_s")}
    if any(k.startswith("stress_") for k in cmp):
        lines += ["", "## Demo v2b live stress (22 questions x 3 wordings, correct = score >= 0.95)", "",
                  "Note: the v1.4 training wordings target the round-2 misses (drone-search budgets, passive collateral,"
                  " LOS vs direct path), so the reworded half of this test is not independent for v1.4.", "",
                  "| Model | correct | as written | p50 s | p95 s |", "|---|---|---|---|---|"]
        for k, v in cmp.items():
            if k.startswith("stress_"):
                lines.append(f"| {k[7:]} | {v['correct']}/{v['runs']} | {v['as_written_correct']} | "
                             f"{v['answer_p50_s']} | {v['answer_p95_s']} |")
    g = gate(cmp)
    cmp["gate"] = g
    lines += ["", "## Gate", "",
              f"- v3 no-regression (task-CI lower > {V3_FLOOR}): **{'pass' if g['v3_no_regression'] else 'FAIL'}**",
              f"- beats v1.1 on v6 (diff > 0, task-CI lower > 0): **{'pass' if g['beats_v11_on_v6'] else 'FAIL'}**",
              f"- held-out family macro >= v1.1: **{'pass' if g['heldout_ge_v11'] else 'FAIL'}**",
              f"- **Verdict: {'PASS (demo-model candidate)' if g['pass'] else 'FAIL (v1.1 stays the demo model)'}**"]
    if a.selected and (REPO / a.selected).exists():
        sel = json.loads((REPO / a.selected).read_text())
        lines += ["", f"Checkpoint selected on the held-out-like val tasks (family macro): `{sel['selected']}`; "
                      f"scores {sel['scores']}."]
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(cmp, indent=1))
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--evals", nargs="+", required=True, help="label=path (label stepN or final)")
    s.add_argument("--out", default="runs/s79/v14/selected.json")
    r = sub.add_parser("report")
    r.add_argument("--base", default="sft_v11")
    r.add_argument("--new", default="sft_v14")
    r.add_argument("--eval-dir", default="runs/eval")
    r.add_argument("--suffix", default="")
    r.add_argument("--stress-base", default="runs/eval/demo_stress_live_v2b_r2.json")
    r.add_argument("--stress-new", default="runs/eval/demo_stress_live_v2b_v14.json")
    r.add_argument("--selected", default="runs/s79/v14/selected.json")
    r.add_argument("--out", default="docs/RESULTS/sft_v14.md")
    a = ap.parse_args()
    return cmd_select(a) if a.cmd == "select" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
