"""Run a policy on the frozen eval set and report SPEC 11 metrics.

    python -m eval.run --policy expert --out runs/eval/expert_v1.json
    python -m eval.run --policy model --model-id prompted:Qwen3-4B-Instruct-2507 --out runs/eval/prompted_v1.json
    python -m eval.run --compare runs/eval/a.json runs/eval/b.json      # paired bootstrap per area
    python -m eval.run --compare a.json b.json --cluster   # + two-level (scenario, task) cluster CI (S7.3)

The model policy talks to the OpenAI-compatible vLLM server (serve/llm.py) exactly as the UI does
(serve.agent.run_model). Episodes run concurrently, one AkshaEnv per worker thread.

Paper plan 2026-10-28 (T1, T2, T4):
    --samples N            each task N times, seed = --seed + sample (LLM request seed and env.reset seed)
    --provider P           local (default) | openrouter | openai | anthropic | gemini; keys from env vars only
                           (anthropic: native SDK, serve/anthropic_backend.py; claude-opus-5-5, claude-sonnet-5-5, ...)
    --api-max-completion-tokens, --reasoning-effort, --reasoning-models   API request settings (not EnvConfig)
    --no-anthropic-fallback  turn off the server-side refusal fallback (on by default where the model supports it)
Anthropic rows also carry `api`: {nudges, stop_reasons, fallback_ran, refusal, reasoning_effort, served_by}.
    --policy always_abstain | prior_centroid | nearest_sensor | random_track (eval/trivial.py),
             tracker_only | triangulate_only (eval/classical.py), parser_expert (eval/parser_expert.py, --parser)
    --truth-metrics        + Hit@1 on single-key track families (offline truth via tasks/truth_access)
Paper Revision 2 (AUDIT_CODE C8 + minors; defaults unchanged):
    --deterministic-tools  every tool timeout = DETERMINISTIC_TIMEOUT_MS (no effective wall-clock cutoff);
                           --tool-timeout-ms N sets another value. Rows always log `n_timeouts` + `timeouts`, the
                           summary `_timeouts`.
    --enforce-set-hash     refuse a set whose sha256 differs from its manifest's `tasks_sha256` (default: warn)
    --policy always_abstain_oracle_reason (fair abstain baseline), residual_check (wrong-twin flag, no LLM)
    --compare with --samples > 1 pairs rows on (task_id, sample) and reports dropped rows.
Rows carry `answer` (the submitted answer), `sample`, `seed`, `provider`, `tool_choice_used`, `usage`; the summary
gains `_grounding` (RFG score, Acc@d, pass@k) and `_end_reasons`. Scores are unchanged.
    python -m eval.run --policy model --provider openai --served o4-mini --model-id api:o4-mini \
        --api-max-completion-tokens 8192 --reasoning-effort low --set eval/frozen/v6/tasks.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from tasks.base import Task  # noqa: E402

_LOCAL = threading.local()


DETERMINISTIC_TIMEOUT_MS = 120_000      # C8 eval mode: high enough that no tool times out (wall-clock free)


def eval_config(ablation: str = "none", tool_timeout_ms: int | None = None):
    """The episode EnvConfig of an eval row: the ablation preset(s), and with `tool_timeout_ms` every tool's timeout
    set to that value (C8 deterministic mode; None = the default per-tool timeouts)."""
    from eval.ablations import apply_ablation
    from harness.config import EnvConfig

    cfg = apply_ablation(EnvConfig.load(), ablation)
    if tool_timeout_ms:
        cfg = EnvConfig(**{**cfg.model_dump(),
                           "tool_timeout_ms": {k: int(tool_timeout_ms) for k in cfg.tool_timeout_ms}})
    return cfg


def _env(site: str, model_id: str, ablation: str = "none", tool_timeout_ms: int | None = None) -> AkshaEnv:
    """One AkshaEnv per (site, ablation, timeout mode) and worker thread; `ablation` is an eval/ablations.py preset
    ('+' joins)."""
    envs = getattr(_LOCAL, "envs", None)
    if envs is None:
        envs = _LOCAL.envs = {}
    k = (site, ablation, tool_timeout_ms or None)
    if k not in envs:
        envs[k] = AkshaEnv(REPO / site, model_id=model_id, config=eval_config(ablation, tool_timeout_ms))
    return envs[k]


def set_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_set_hash(path: Path, man: dict, enforce: bool = False) -> dict:
    """Task-set sha256 vs the frozen manifest's `tasks_sha256` (audit minor 1). Returns {sha256, expected, status}
    with status ok | mismatch | no_manifest_hash; warns on stderr, raises SystemExit on a mismatch (or a missing
    hash) when `enforce`."""
    got = set_sha256(path)
    exp = (man or {}).get("tasks_sha256")
    status = "no_manifest_hash" if not exp else ("ok" if exp == got else "mismatch")
    if status != "ok":
        msg = f"set hash {status}: {path} sha256 {got[:16]} manifest {str(exp)[:16]}"
        if enforce:
            raise SystemExit(f"--enforce-set-hash: {msg}")
        print(f"WARNING {msg}", file=sys.stderr)
    return {"sha256": got, "expected": exp, "status": status}


TRIVIAL = ("always_abstain", "prior_centroid", "nearest_sensor", "random_track", "always_abstain_oracle_reason")
CLASSICAL = ("tracker_only", "triangulate_only", "residual_check")
POLICIES = ("expert", "expert_check", "model", *TRIVIAL, *CLASSICAL, "parser_expert")


def run_one(task: Task, policy: str, model_id: str, temperature: float, seed: int, served: str,
            wallclock_s: float = 0.0, sample: int = 0, opts: dict | None = None) -> dict:
    """wallclock_s > 0 enforces the serve-time wall-clock budget on the model loop (default off: concurrent eval
    latency is not single-user latency, and past numbers were measured without it).
    seed: this row's seed (the caller adds the sample index). opts: {client, provider, options (LLMOptions)} for the
    model policy; {parser_client, parser_model, parser_examples, parser_areas, parser_request} for parser_expert;
    {ablation} (eval/ablations.py preset, default "none") and {tool_timeout_ms} (C8 mode) for every policy."""
    from serve.agent import redact

    opts = opts or {}
    ablation = opts.get("ablation") or "none"
    if policy == "residual_check" and "with_consistency" not in ablation.split("+"):
        ablation = "with_consistency" if ablation == "none" else ablation + "+with_consistency"   # needs the tool
    env = _env(task.site or "data/site_kandla", model_id, ablation, opts.get("tool_timeout_ms"))
    t0 = time.time()
    gen_ms, toks, err = [], 0, None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0, "calls": 0}
    tool_choice, parsed = None, None
    api: dict | None = None          # anthropic backend extras (absent for local / chat-API rows)
    if policy == "model":
        from serve.agent import run_task

        for ev in run_task(env, task, client=opts.get("client"), temperature=temperature, model=served,
                           max_wallclock_s=wallclock_s or None, seed=seed, options=opts.get("options")):
            if ev["type"] == "assistant":
                gen_ms.append(ev["gen_ms"])
                toks += ev["tokens"]
                tool_choice = ev.get("tool_choice", tool_choice)
                for k, v in (ev.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + v
                usage["calls"] += 1
                if "stop_reasons" in ev:
                    api = api or {"nudges": 0, "stop_reasons": {}, "fallback_ran": False, "refusal": None,
                                  "reasoning_effort": ev.get("reasoning_effort"), "served_by": []}
                    api["nudges"] += ev.get("nudges", 0)
                    for s in ev["stop_reasons"]:
                        api["stop_reasons"][str(s)] = api["stop_reasons"].get(str(s), 0) + 1
                    api["fallback_ran"] = api["fallback_ran"] or bool(ev.get("fallback"))
                    api["refusal"] = ev.get("refusal") or api["refusal"]
                    if ev.get("served_by") and ev["served_by"] not in api["served_by"]:
                        api["served_by"].append(ev["served_by"])
            elif ev["type"] == "error" and "LLM server" in ev["message"]:
                err = ev["message"]
    else:
        env.reset(task, seed=seed)
        try:
            if policy == "expert":
                from tasks.expert_kandla import solve

                solve(env, task)
            elif policy == "expert_check":       # T-FIX T4 expert (needs ablation with_consistency+fix_line)
                from tasks.expert_check import solve

                solve(env, task)
            elif policy in TRIVIAL:
                from eval.trivial import solve

                solve(env, task, policy, seed)
            elif policy in CLASSICAL:
                from eval.classical import solve

                solve(env, task, policy)
            elif policy == "parser_expert":
                from eval.parser_expert import solve

                parsed = solve(env, task, opts["parser_client"], opts.get("parser_model", served),
                               opts.get("parser_examples"), opts.get("parser_areas"), opts.get("parser_request"))
                for k in ("prompt_tokens", "completion_tokens"):
                    usage[k] += parsed["usage"][k]
                usage["calls"] += 1
            else:
                raise ValueError(f"unknown policy {policy!r}")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    ans = env.answer
    vr = env.verifier_result or {}
    row = {"task_id": task.task_id, "area": task.area, "template": task.template, "split": task.split,
            "score": float(getattr(env, "answer_score", 0.0) or 0.0) if env.done else 0.0,
            "end_reason": getattr(env, "end_reason", "none"), "turns": env.turn, "gen_tokens": toks,
            "tool_calls": env.executor.calls, "tool_names": list(getattr(env.executor, "call_names", [])), "verifier_ok": bool(vr.get("ok")), "verifier_code": vr.get("code"),
            "intent": ans.intent if ans is not None else None,
            "abstained": bool(ans is not None and ans.intent == "abstain"),
            # demo-v2 S5.4: stated confidence (calibration: ECE, Brier, confident-wrong) and the verifier rules that
            # failed on the final answer (RF_AMBIGUOUS, INCOMPLETE_COLLATERAL, ...)
            "confidence": float(ans.confidence) if ans is not None else None,
            "flags": [r["code"] for r in (env.verifier_rules or []) if not r["ok"]],
            "latency_s": round(time.time() - t0, 3), "gen_ms": gen_ms, "error": redact(err), "model_id": model_id,
            # paper plan T1/T2: the submitted answer (offline grounding metrics), sample/seed, provider, usage
            "answer": ans.model_dump(mode="json") if ans is not None else None, "sample": sample, "seed": seed,
            "provider": (opts.get("provider") or "local") if policy == "model" else None,
            "tool_choice_used": tool_choice, "usage": usage, "ablation": ablation,
            # C8: every tool TIMEOUT of this episode (nondeterministic under load unless --deterministic-tools)
            "n_timeouts": len(getattr(getattr(env, "executor", None), "timeouts", None) or []),
            "timeouts": list(getattr(getattr(env, "executor", None), "timeouts", None) or [])}
    if parsed is not None:
        row["parsed"] = parsed
    if api is not None:
        row["api"] = api
    return row


def run_all(tasks: list[Task], policy: str, model_id: str, temperature: float, seed: int, served: str,
            wallclock_s: float = 0.0, samples: int = 1, concurrency: int = 1, opts: dict | None = None) -> list[dict]:
    """Every task `samples` times (sample s gets seed + s), in task order then sample order."""
    jobs = [(t, s) for t in tasks for s in range(max(1, samples))]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(lambda j: run_one(j[0], policy, model_id, temperature, seed + j[1], served, wallclock_s,
                                             sample=j[1], opts=opts), jobs))


def grounding_summary(rows: list[dict], tasks: list[Task], summ: dict, samples: int = 1,
                      truth_metrics: bool = False) -> dict:
    """Paper headline metrics next to the SPEC 11 summary (eval/grounding.py)."""
    from eval.grounding import pass_at_k, pass_hat_k, position_metrics, rfg_score, track_hit_metrics

    out = {"rfg": rfg_score(summ), "position": position_metrics(rows, tasks)}
    if samples > 1:
        ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= samples]
        out["pass_at_k"] = {k: round(pass_at_k(rows, k), 4) for k in ks}
        out["pass_hat_k"] = {k: round(pass_hat_k(rows, k), 4) for k in ks}
    if truth_metrics:
        out["track_hit"] = track_hit_metrics(rows, tasks, REPO)
    return out


def summarise(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["area"]].append(r)
    out = {}
    for area, rs in sorted(by.items()):
        out[area] = {"n": len(rs), "score": round(statistics.mean(r["score"] for r in rs), 3),
                     "verifier_reject_rate": round(sum(r["verifier_code"] is not None for r in rs) / len(rs), 3),
                     "abstain_rate": round(sum(r["abstained"] for r in rs) / len(rs), 3),
                     "turns_mean": round(statistics.mean(r["turns"] for r in rs), 2)}
    try:
        from scripts.make_tasks import HELD_OUT
    except Exception:  # noqa: BLE001
        HELD_OUT = set()
    seen = [r["score"] for r in rows if r.get("template") not in HELD_OUT and r["area"] != "unanswerable"]
    held = [r["score"] for r in rows if r.get("template") in HELD_OUT]
    fam = defaultdict(list)
    for r in rows:
        fam[r.get("template", "?")].append(r["score"])
    out["_generalisation"] = {"seen_families": round(statistics.mean(seen), 3) if seen else None, "n_seen": len(seen),
                              "held_out_families": round(statistics.mean(held), 3) if held else None,
                              "n_held_out": len(held)}
    out["_families"] = {k: [len(v), round(statistics.mean(v), 3)] for k, v in sorted(fam.items())}
    lat = sorted(r["latency_s"] for r in rows)
    out["_all"] = {"n": len(rows), "score": round(statistics.mean(r["score"] for r in rows), 3),
                   "macro_score": round(statistics.mean(v["score"] for k, v in out.items() if not k.startswith("_")), 3),
                   "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
                   "errors": sum(r["error"] is not None for r in rows),
                   "ungrounded": sum(r["verifier_code"] == "UNGROUNDED" for r in rows),
                   "no_answer": sum(r["intent"] is None for r in rows)}
    if any(r.get("confidence") is not None for r in rows):
        out["_calibration"] = {k: v for k, v in calibration(rows).items() if k != "bins"}
    return out


def _row_key(r: dict) -> tuple:
    """Pairing key of an eval row: (task_id, sample), so --samples > 1 keeps every sample (audit minor 2)."""
    return (r["task_id"], int(r.get("sample", 0) or 0))


def _paired_rows(a: list[dict], b: list[dict]) -> tuple[dict, dict, list, int]:
    ia = {_row_key(r): r for r in a}
    ib = {_row_key(r): r for r in b}
    return ia, ib, sorted(set(ia) & set(ib)), len(set(ia) ^ set(ib))


def paired_bootstrap(a: list[dict], b: list[dict], n: int = 2000, seed: int = 0) -> dict:
    """Per-area mean(b - a) with a 95% CI over the (task, sample) rows present in both runs (SPEC 11).
    `_all.n_dropped`: rows present in only one run (reported, not silently lost)."""
    ia, ib, both, dropped = _paired_rows(a, b)
    rng = random.Random(seed)
    by = defaultdict(list)
    for t in both:
        d = ib[t]["score"] - ia[t]["score"]
        by[ia[t]["area"]].append(d)
        by["_all"].append(d)
    res = {}
    for area, d in sorted(by.items()):
        means = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
        res[area] = {"n": len(d), "diff": round(statistics.mean(d), 3),
                     "ci95": [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]}
    if "_all" in res:
        res["_all"]["n_dropped"] = dropped
    return res


# --- S7.3: scenario-level statistics and confidence calibration (audit G Q48, F Q41) ----------------------------
CONFIDENT = 0.7          # confident-wrong: stated confidence >= 0.7 and score < 0.5


def scenario_of(row: dict) -> str:
    """Scenario seed of an eval row from its task id (k<seed>[X][W4]-<family>-<n>); the id itself otherwise."""
    import re

    m = re.match(r"k(\d+)", str(row.get("task_id", "")))
    return m.group(1) if m else str(row.get("task_id"))


def cluster_bootstrap(clusters: dict, n: int = 2000, seed: int = 0) -> dict:
    """Two-level bootstrap of a pooled mean: resample scenarios with replacement, then tasks within each drawn
    scenario with replacement. Percentile 95 % CI. clusters: {scenario: [values]}."""
    keys = sorted(k for k, v in clusters.items() if v)
    vals = [list(clusters[k]) for k in keys]
    allv = [x for v in vals for x in v]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        tot, cnt = 0.0, 0
        for _k in range(len(vals)):
            v = vals[rng.randrange(len(vals))]
            s = rng.choices(v, k=len(v))
            tot += sum(s)
            cnt += len(s)
        means.append(tot / cnt)
    means.sort()
    return {"n": len(allv), "n_clusters": len(keys), "mean": statistics.mean(allv),
            "ci95": [round(means[int(0.025 * n)], 4), round(means[min(n - 1, int(0.975 * n))], 4)]}


def paired_cluster_bootstrap(a: list[dict], b: list[dict], n: int = 2000, seed: int = 0) -> dict:
    """Per-area mean(b - a) over the (task, sample) rows present in both runs, with the two-level (scenario, task)
    CI."""
    ia, ib, both, dropped = _paired_rows(a, b)
    by: dict = defaultdict(lambda: defaultdict(list))
    for t in both:
        d = ib[t]["score"] - ia[t]["score"]
        sc = scenario_of(ia[t])
        by[ia[t]["area"]][sc].append(d)
        by["_all"][sc].append(d)
    res = {}
    for area, cl in sorted(by.items()):
        cb = cluster_bootstrap(cl, n, seed)
        res[area] = {"n": cb["n"], "n_clusters": cb["n_clusters"], "diff": round(cb["mean"], 3),
                     "ci95": [round(x, 3) for x in cb["ci95"]]}
    if "_all" in res:
        res["_all"]["n_dropped"] = dropped
    return res


def timeout_summary(rows: list[dict]) -> dict:
    """C8: tool TIMEOUTs over a run (rows without the field, i.e. older outputs, are not counted)."""
    known = [r for r in rows if "n_timeouts" in r]
    per_tool = Counter(t["tool"] for r in known for t in r.get("timeouts") or [])
    return {"n_rows_logged": len(known), "n_timeouts": sum(r["n_timeouts"] for r in known),
            "rows_with_timeout": sum(r["n_timeouts"] > 0 for r in known), "per_tool": dict(sorted(per_tool.items()))}


def calibration(rows: list[dict], n_bins: int = 10) -> dict:
    """Stated answer confidence vs the task score (0..1): ECE (equal-width bins), Brier, reliability bins and the
    confident-wrong rate (confidence >= 0.7 and score < 0.5), over rows that logged a confidence."""
    rs = [r for r in rows if r.get("confidence") is not None]
    out = {"n": len(rs), "n_missing": len(rows) - len(rs)}
    if not rs:
        return out
    bins = defaultdict(list)
    for r in rs:
        c = min(max(float(r["confidence"]), 0.0), 1.0)
        bins[min(int(c * n_bins), n_bins - 1)].append((c, float(r["score"])))
    rel, ece = [], 0.0
    for k in sorted(bins):
        v = bins[k]
        mc, ms = statistics.mean(c for c, _ in v), statistics.mean(s for _, s in v)
        ece += len(v) / len(rs) * abs(mc - ms)
        rel.append({"lo": k / n_bins, "hi": (k + 1) / n_bins, "n": len(v), "conf": round(mc, 3), "score": round(ms, 3)})
    conf_ = [r for r in rs if float(r["confidence"]) >= CONFIDENT]
    cw = [r for r in conf_ if float(r["score"]) < 0.5]
    out.update({"ece": round(ece, 4),
                "brier": round(statistics.mean((float(r["confidence"]) - float(r["score"])) ** 2 for r in rs), 4),
                "mean_conf": round(statistics.mean(float(r["confidence"]) for r in rs), 3),
                "mean_score": round(statistics.mean(float(r["score"]) for r in rs), 3),
                "confident_wrong_rate": round(len(cw) / len(rs), 4),
                "confident_wrong_of_confident": round(len(cw) / len(conf_), 4) if conf_ else None,
                "n_distinct_conf": len({round(float(r["confidence"]), 3) for r in rs}), "bins": rel})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES, default="expert")
    ap.add_argument("--model-id", default="prompted:Qwen3-4B-Instruct-2507")
    ap.add_argument("--served", default="scenescout", help="served model name (vLLM --served-model-name / LoRA)")
    ap.add_argument("--set", default="eval/frozen/v1/tasks.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wallclock-s", type=float, default=0.0, help="per-question wall-clock budget (0 = off)")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--cluster", action="store_true",
                    help="with --compare: also the two-level (scenario, then task) cluster bootstrap CI")
    ap.add_argument("--samples", type=int, default=1, help="episodes per task; sample s uses seed --seed + s")
    ap.add_argument("--families", default=None, help="comma list of task families to keep (before --limit)")
    ap.add_argument("--provider", default="local", choices=["local", "openrouter", "openai", "anthropic", "gemini"])
    ap.add_argument("--api-max-completion-tokens", type=int, default=None,
                    help="API completion cap per call incl. reasoning (separate from EnvConfig; default env budget, "
                         "anthropic 16000)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="API reasoning effort, e.g. low | medium | high (anthropic: output_config.effort; default "
                         "the model's API default, sent explicitly: opus-5-5 medium, sonnet-5-5 high)")
    ap.add_argument("--no-anthropic-fallback", action="store_true",
                    help="anthropic: no server-side refusal fallback (fallbacks='default' is on by default)")
    ap.add_argument("--reasoning-models", default=None,
                    help="comma list of model-name prefixes that get no temperature (default serve.agent's)")
    ap.add_argument("--parser", choices=["local", "provider"], default="local",
                    help="parser_expert: local vLLM server or --provider")
    ap.add_argument("--parser-model", default=None, help="parser_expert: model name (default --served)")
    ap.add_argument("--parser-examples", default="data/tasks/x3/train.jsonl",
                    help="parser_expert: train tasks for the few-shot prompt (held-out families excluded)")
    ap.add_argument("--ablation", default="none",
                    help="eval/ablations.py preset(s), '+' joins (e.g. with_consistency+no_rt); none = default env")
    ap.add_argument("--truth-metrics", action="store_true",
                    help="+ Hit@1 on single-key track families (offline truth read)")
    ap.add_argument("--deterministic-tools", action="store_true",
                    help=f"C8 eval mode: every tool timeout = {DETERMINISTIC_TIMEOUT_MS} ms (no wall-clock cutoff)")
    ap.add_argument("--tool-timeout-ms", type=int, default=None,
                    help="every tool timeout = this many ms (the deterministic mode with this value)")
    ap.add_argument("--enforce-set-hash", action="store_true",
                    help="refuse a set whose sha256 differs from (or is missing in) its manifest (default: warn)")
    a = ap.parse_args()
    if a.compare:
        ra, rb = (json.loads(Path(p).read_text())["rows"] for p in a.compare)
        if not a.cluster:
            print(json.dumps(paired_bootstrap(ra, rb), indent=1))
            return 0
        task, clus = paired_bootstrap(ra, rb), paired_cluster_bootstrap(ra, rb)
        print(json.dumps({k: {**task[k], "ci95_cluster": clus[k]["ci95"], "n_clusters": clus[k]["n_clusters"]}
                          for k in task}, indent=1))
        return 0
    path = REPO / a.set
    man0 = json.loads(path.with_name("manifest.json").read_text()) if path.with_name("manifest.json").exists() else {}
    set_hash = check_set_hash(path, man0, a.enforce_set_hash)
    tasks = [Task.from_row(json.loads(x)) for x in path.read_text().splitlines()]
    if a.families:
        keep = set(a.families.split(","))
        tasks = [t for t in tasks if t.template in keep]
    if a.limit:
        tasks = tasks[: a.limit]
    man = json.loads(path.with_name("manifest.json").read_text()) if path.with_name("manifest.json").exists() else {}
    opts: dict = {}
    if a.policy == "model":
        from serve.agent import LLMOptions, make_client

        prefixes = tuple(p for p in a.reasoning_models.split(",") if p) if a.reasoning_models else None
        opts = {"provider": a.provider, "client": make_client(a.provider),
                "options": LLMOptions(provider=a.provider, max_completion_tokens=a.api_max_completion_tokens,
                                      reasoning_effort=a.reasoning_effort, reasoning_prefixes=prefixes,
                                      anthropic_fallback=not a.no_anthropic_fallback)}
    elif a.policy == "parser_expert":
        from eval.parser_expert import load_examples
        from serve.agent import PROVIDERS, is_reasoning_model, make_client

        prov = a.provider if a.parser == "provider" else "local"
        if prov == "anthropic":
            ap.error("parser_expert --parser provider speaks Chat Completions; --provider anthropic is not supported")
        pmodel = a.parser_model or a.served
        if a.parser_examples == "none":     # fine-tuned parser (train/parser_data.py): zero-shot prompt it was trained on
            examples, areas = [], load_examples()[1]
        else:
            examples, areas = load_examples(REPO / a.parser_examples)
        req = {PROVIDERS[prov][2]: a.api_max_completion_tokens or 512}
        if not (prov != "local" and is_reasoning_model(pmodel)):
            req["temperature"] = 0.0
        if a.reasoning_effort:
            req["reasoning_effort"] = a.reasoning_effort
        opts = {"parser_client": make_client(prov), "parser_model": pmodel, "parser_examples": examples,
                "parser_areas": areas, "parser_request": req}
    from eval.ablations import ABLATIONS

    bad = [p for p in a.ablation.split("+") if p.strip() not in ABLATIONS]
    if bad:
        ap.error(f"unknown --ablation preset(s) {bad}; one of {sorted(ABLATIONS)}")
    opts["ablation"] = a.ablation
    tool_timeout_ms = a.tool_timeout_ms or (DETERMINISTIC_TIMEOUT_MS if a.deterministic_tools else None)
    if tool_timeout_ms:
        opts["tool_timeout_ms"] = tool_timeout_ms
    t0 = time.time()
    rows = run_all(tasks, a.policy, a.model_id, a.temperature, a.seed, a.served, a.wallclock_s, samples=a.samples,
                   concurrency=a.concurrency if a.policy in ("model", "parser_expert") else 1, opts=opts)
    summ = summarise(rows)
    summ["_all"]["wallclock_s"] = round(time.time() - t0, 1)
    summ["_end_reasons"] = dict(sorted(Counter(r["end_reason"] for r in rows).items()))
    summ["_grounding"] = grounding_summary(rows, tasks, summ, a.samples, a.truth_metrics)
    summ["_timeouts"] = timeout_summary(rows)
    print(json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"policy": a.policy, "model_id": a.model_id, "set": a.set,
                                           "provider": opts.get("provider"), "samples": a.samples, "seed": a.seed,
                                           "ablation": a.ablation, "tool_timeout_ms": tool_timeout_ms,
                                           "set_hash": set_hash,
                                           "manifest": man, "summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
