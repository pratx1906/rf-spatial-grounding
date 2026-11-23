"""Estimate the $ cost of an API eval from a smoke run's token usage (plan 2026-10-28 T2; audit M10).

    python scripts/api_cost.py runs/paper/smoke/opus55_20.json --preset claude-opus-5-5 --tasks 1500
    python scripts/api_cost.py runs/paper/smoke/api_o4mini_20.json --in-per-m 1.1 --out-per-m 4.4 \
        --cached-in-per-m 0.275 --tasks 1500

Prices are USD per million tokens. OpenAI (and other chat-API) prices are passed in; --preset fills the Anthropic
list prices from the claude-api skill (shared/model-migration.md, prompt-caching.md, 2026-10); explicit price flags
override a preset. Usage comes from the eval rows (`usage`): prompt_tokens is the whole prompt; cached_tokens (cache
reads) and cache_creation_tokens (Anthropic cache writes) are subsets of it, the rest bills at the input price.
Reasoning / thinking tokens are billed inside the completion count. The per-episode mean is extrapolated to --tasks
episodes (default: the rows in the file).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# USD per 1M tokens: in (uncached input), out, cached_in (cache read), cache_write (5-minute TTL write)
PRESETS = {
    "claude-opus-5-5": {"in": 4.0, "out": 20.0, "cached_in": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cached_in": 0.20, "cache_write": 2.5},
}


def estimate(rows: list[dict], in_per_m: float, out_per_m: float, cached_in_per_m: float | None = None,
             tasks: int | None = None, cache_write_per_m: float | None = None) -> dict:
    cached_in_per_m = in_per_m if cached_in_per_m is None else cached_in_per_m
    cache_write_per_m = in_per_m if cache_write_per_m is None else cache_write_per_m
    rs = [r for r in rows if r.get("usage")]
    if not rs:
        raise SystemExit("no rows with usage: run eval.run on the model policy first")
    per = []
    for r in rs:
        u = r["usage"]
        cached = int(u.get("cached_tokens", 0) or 0)
        written = int(u.get("cache_creation_tokens", 0) or 0)
        prompt = int(u.get("prompt_tokens", 0) or 0)
        per.append(((prompt - cached - written) * in_per_m + cached * cached_in_per_m + written * cache_write_per_m
                    + int(u.get("completion_tokens", 0) or 0) * out_per_m) / 1e6)
    n = tasks or len(rs)
    mean = statistics.mean(per)
    tot = lambda k: sum(int(r["usage"].get(k, 0) or 0) for r in rs)  # noqa: E731
    return {"n_rows": len(rs), "usd_per_task": round(mean, 6), "tasks": n, "usd_total": round(mean * n, 2),
            "tokens_per_task": {k: round(tot(k) / len(rs)) for k in ("prompt_tokens", "cached_tokens",
                                                                       "cache_creation_tokens", "completion_tokens",
                                                                       "reasoning_tokens")},
            "calls_per_task": round(tot("calls") / len(rs), 2),
            "prices_per_m": {"in": in_per_m, "cached_in": cached_in_per_m, "cache_write": cache_write_per_m,
                             "out": out_per_m}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("smoke", help="eval.run --out JSON of an API smoke run")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Anthropic list prices")
    ap.add_argument("--in-per-m", type=float, default=None, help="USD per 1M uncached input tokens")
    ap.add_argument("--out-per-m", type=float, default=None, help="USD per 1M output tokens (incl. reasoning)")
    ap.add_argument("--cached-in-per-m", type=float, default=None, help="USD per 1M cached (cache-read) input tokens")
    ap.add_argument("--cache-write-per-m", type=float, default=None,
                    help="USD per 1M cache-write tokens (Anthropic cache_creation; default: the input price)")
    ap.add_argument("--tasks", type=int, default=None, help="episodes to extrapolate to (tasks x samples)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pr = dict(PRESETS[a.preset]) if a.preset else {}
    for k, v in (("in", a.in_per_m), ("out", a.out_per_m), ("cached_in", a.cached_in_per_m),
                 ("cache_write", a.cache_write_per_m)):
        if v is not None:
            pr[k] = v
    if "in" not in pr or "out" not in pr:
        ap.error("give --preset or both --in-per-m and --out-per-m")
    r = estimate(json.loads(Path(a.smoke).read_text())["rows"], pr["in"], pr["out"], pr.get("cached_in"), a.tasks,
                 pr.get("cache_write"))
    if a.json:
        print(json.dumps(r))
    else:
        t = r["tokens_per_task"]
        print(f"{r['n_rows']} episodes measured: ${r['usd_per_task']:.4f} per episode, {t['prompt_tokens']} prompt "
              f"({t['cached_tokens']} cache read, {t['cache_creation_tokens']} cache write) / "
              f"{t['completion_tokens']} completion tokens per episode")
        print(f"estimate for {r['tasks']} episodes: ${r['usd_total']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
