"""P9 (c): tool-use behaviour from trajectories (before vs after RL; any policy).

    python -m eval.paper.behaviour --traces runs/paper/results/P9/traces_expert_v6.jsonl --name expert_v6
    python -m eval.paper.behaviour --traces data/demo_replays_v2 --name demo_v2 --who sft
    python -m eval.paper.behaviour --eval runs/eval/sft_v11_v6_t7.json --name sft_v11_v6     # rows only

Inputs (any mix): normalised episode jsonl from eval/paper/collect_traces.py ({task_id, template, area, tools,
score, end_reason, ...}); harness Tracer jsonl files or dirs (events with `tool`, grouped by episode_id); gateway
replay json ({events, summary}, `tool_call` events). With only an eval/run.py output (no tool sequences) it reports
the step/tool-call statistics vs success.

Outputs (runs/paper/results/P9): <name>_ngrams.csv (tool 1-3-grams, overall and in successes vs failures),
<name>_behaviour.json (verify-before-submit rate, steps vs success, per family), figures when matplotlib exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# tools that cross-check evidence against the physics/world model rather than gather it
VERIFY_TOOLS = {"model_trust", "geometry_check", "predict_aoa", "sensor_consistency", "prop_query",
                "fit_candidates:rt_match"}
CONTROL = {"submit", "declare_intent", "ask_user"}


def _tool_name(name: str, args) -> str:
    if name == "fit_candidates" and isinstance(args, dict) and args.get("method") == "rt_match":
        return "fit_candidates:rt_match"
    return name


def load_episodes(paths: list[str], who: str | None = None) -> list[dict]:
    eps: list[dict] = []
    for p in paths:
        p = Path(REPO / p)
        files = sorted(p.rglob("*.json*")) if p.is_dir() else [p]
        tracer = defaultdict(list)
        for f in files:
            if f.suffix == ".json":
                d = json.loads(f.read_text())
                if "events" in d and "summary" in d:
                    s = d["summary"]
                    if who and s.get("who") != who:
                        continue
                    tools = [_tool_name(e["name"], e.get("arguments")) for e in d["events"] if e["type"] == "tool_call"]
                    eps.append({"task_id": s.get("task_id"), "template": s.get("template"), "area": None,
                                "tools": tools, "score": s.get("score"), "end_reason": s.get("end_reason"),
                                "turns": s.get("turns"), "policy": s.get("who")})
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if "tools" in e and "task_id" in e:
                    eps.append(e)
                elif "tool" in e and "episode_id" in e:
                    tracer[e["episode_id"]].append(e)
        for eid, evs in tracer.items():
            eps.append({"task_id": evs[0].get("task_id"), "template": None, "area": None,
                        "tools": [_tool_name(e["tool"], e.get("args")) for e in evs], "score": None,
                        "end_reason": None, "episode_id": eid})
    return eps


def ngrams(seq: list[str], n: int) -> list[tuple]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def verify_before_submit(tools: list[str]) -> bool:
    """A verification tool is called before the (last) submit."""
    last = max((i for i, t in enumerate(tools) if t == "submit"), default=len(tools))
    return any(t in VERIFY_TOOLS for t in tools[:last])


def analyse(eps: list[dict], thr: float = 0.8) -> tuple[dict, list[dict]]:
    ok = lambda e: e.get("score") is not None and e["score"] >= thr  # noqa: E731
    gram_rows = []
    for n in (1, 2, 3):
        allc, sc, fc = Counter(), Counter(), Counter()
        for e in eps:
            g = ngrams([t for t in e["tools"] if t != "declare_intent"], n)
            allc.update(g)
            (sc if ok(e) else fc).update(g)
        ns, nf = sum(map(ok, eps)), len(eps) - sum(map(ok, eps))
        for g, c in allc.most_common(40):
            gram_rows.append({"n": n, "gram": " > ".join(g), "count": c, "per_episode": round(c / max(len(eps), 1), 3),
                              "per_success_ep": round(sc[g] / max(ns, 1), 3),
                              "per_failure_ep": round(fc[g] / max(nf, 1), 3)})
    fam = defaultdict(list)
    for e in eps:
        fam[e.get("template") or "?"].append(e)

    def stats(es):
        if not es:
            return {}
        n_tools = [len([t for t in e["tools"] if t not in CONTROL]) for e in es]
        vbs = [verify_before_submit(e["tools"]) for e in es]
        sc = [e for e in es if e.get("score") is not None]
        return {"n": len(es), "mean_tool_calls": round(sum(n_tools) / len(es), 2),
                "verify_before_submit": round(sum(vbs) / len(es), 3),
                "success": round(sum(map(ok, sc)) / len(sc), 3) if sc else None,
                "success_if_verify": _rate([ok(e) for e, v in zip(es, vbs, strict=True) if v and e.get("score") is not None]),
                "success_if_no_verify": _rate([ok(e) for e, v in zip(es, vbs, strict=True)
                                               if not v and e.get("score") is not None]),
                "distinct_tools": round(sum(len(set(e["tools"]) - CONTROL) for e in es) / len(es), 2)}

    steps = defaultdict(list)
    for e in eps:
        if e.get("score") is not None:
            steps[min(len([t for t in e["tools"] if t not in CONTROL]), 10)].append(ok(e))
    res = {"overall": stats(eps), "per_family": {k: stats(v) for k, v in sorted(fam.items())},
           "success_by_tool_calls": {str(k): {"n": len(v), "success": _rate(v)} for k, v in sorted(steps.items())},
           "verify_tools": sorted(VERIFY_TOOLS)}
    return res, gram_rows


def _rate(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def rows_only(eval_json: str, thr: float = 0.8) -> dict:
    """Steps / tool calls vs success from an eval/run.py output (no tool sequences)."""
    rows = json.loads(Path(REPO / eval_json).read_text())["rows"]
    by_calls, by_turns, fam = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        s = r["score"] >= thr
        by_calls[min(r.get("tool_calls") or 0, 12)].append(s)
        by_turns[min(r.get("turns") or 0, 12)].append(s)
        fam[r["template"]].append(r)
    per_family = {}
    for k, v in sorted(fam.items()):
        ok_ = [r for r in v if r["score"] >= thr]
        bad = [r for r in v if r["score"] < thr]
        per_family[k] = {"n": len(v), "success": _rate([r["score"] >= thr for r in v]),
                         "mean_calls_success": round(sum(r["tool_calls"] for r in ok_) / len(ok_), 2) if ok_ else None,
                         "mean_calls_failure": round(sum(r["tool_calls"] for r in bad) / len(bad), 2) if bad else None,
                         "abstain_rate": _rate([bool(r.get("abstained")) for r in v])}
    calls = [r.get("tool_calls") or 0 for r in rows]
    succ = [float(r["score"] >= thr) for r in rows]
    n = len(rows)
    mc, ms = sum(calls) / n, sum(succ) / n
    cov = sum((c - mc) * (s - ms) for c, s in zip(calls, succ, strict=True)) / n
    sd_c = (sum((c - mc) ** 2 for c in calls) / n) ** 0.5
    sd_s = (sum((s - ms) ** 2 for s in succ) / n) ** 0.5
    return {"eval": eval_json, "n": n, "success": round(ms, 3), "mean_tool_calls": round(mc, 2),
            "corr_calls_success": round(cov / (sd_c * sd_s), 3) if sd_c and sd_s else None,
            "success_by_tool_calls": {str(k): {"n": len(v), "success": _rate(v)} for k, v in sorted(by_calls.items())},
            "success_by_turns": {str(k): {"n": len(v), "success": _rate(v)} for k, v in sorted(by_turns.items())},
            "end_reasons": dict(Counter(r["end_reason"] for r in rows)), "per_family": per_family}


def figure(res: dict, name: str, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sb = res.get("success_by_tool_calls") or {}
    if not sb:
        return
    ks = list(sb)
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.bar(range(len(ks)), [sb[k]["success"] or 0 for k in ks], color="#4a6fa5")
    for i, k in enumerate(ks):
        ax.text(i, (sb[k]["success"] or 0) + 0.02, f"n={sb[k]['n']}", ha="center", fontsize=6)
    ax.set_xticks(range(len(ks)), [k if k not in ("10", "12") else k + "+" for k in ks])
    ax.set_xlabel("tool calls per episode")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(name, fontsize=9)
    fig.savefig(out / f"{name}_steps.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="*", default=[])
    ap.add_argument("--eval", default=None, help="eval/run.py output: adds scores to traces, or rows-only analysis")
    ap.add_argument("--who", default=None, help="replay json: keep this summary.who (sft, scenescout)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", default="runs/paper/results/P9")
    a = ap.parse_args(argv)
    out = Path(REPO / a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"name": a.name}
    if a.eval:
        res["rows"] = rows_only(a.eval, a.thr)
    if a.traces:
        eps = load_episodes(a.traces, a.who)
        if a.eval:
            sc = {r["task_id"]: r for r in json.loads(Path(REPO / a.eval).read_text())["rows"]}
            for e in eps:
                if e.get("score") is None and e.get("task_id") in sc:
                    e["score"] = sc[e["task_id"]]["score"]
                    e["template"] = e.get("template") or sc[e["task_id"]]["template"]
        res["traces"], grams = analyse(eps, a.thr)
        with open(out / f"{a.name}_ngrams.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n", "gram", "count", "per_episode", "per_success_ep", "per_failure_ep"])
            w.writeheader()
            w.writerows(grams)
        figure(res["traces"], a.name, out)
    elif a.eval:
        figure(res["rows"], a.name, out)
    (out / f"{a.name}_behaviour.json").write_text(json.dumps(res, indent=1))
    head = {k: (v.get("overall") if k == "traces" else {x: v[x] for x in ("n", "success", "mean_tool_calls",
                                                                          "corr_calls_success")})
            for k, v in res.items() if k in ("traces", "rows")}
    print(json.dumps({"name": a.name, **head}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
