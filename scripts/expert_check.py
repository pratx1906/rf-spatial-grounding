"""Run the scripted expert on generated tasks and report scores per area and family (SPEC 9.3 smoke test).

    python scripts/expert_check.py --seeds 13 --k 3 [--show-fail 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from scripts.build_scenarios import parse_seeds, seed_dir  # noqa: E402
from tasks.expert_kandla import solve  # noqa: E402
from tasks.kandla.gen_more import GenMore as Gen  # noqa: E402


def run(tasks, show_fail=0):
    envs = {}
    fam, area = defaultdict(list), defaultdict(list)
    fails = []
    for t in tasks:
        env = envs.setdefault(t.site, AkshaEnv(REPO / t.site))
        env.reset(t)
        try:
            solve(env, t)
        except Exception as e:  # noqa: BLE001
            fails.append((t.template, t.question, f"EXC {type(e).__name__}: {e}"))
            fam[t.template].append(0.0)
            area[t.area].append(0.0)
            continue
        sc = getattr(env, "answer_score", 0.0) if env.done else 0.0
        fam[t.template].append(sc)
        area[t.area].append(sc)
        if sc < 0.9:
            fails.append((t.template, t.question, json.dumps(env.answer.model_dump(mode="json") if env.answer else
                                                              {"end": getattr(env, "end_reason", "?"),
                                                               "verifier": env.verifier_result})[:300],
                          json.dumps(t.answer_key)[:300]))
    return fam, area, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show-fail", type=int, default=0)
    ap.add_argument("--world-variant", default=None, help="check on world-variant scenarios (D-031)")
    ap.add_argument("--obs", choices=["v1", "v2"], default="v1", help="v2: RF phase 2 sets (kandla_s<N>_X[_W])")
    a = ap.parse_args()
    tasks = []
    for s in parse_seeds(a.seeds):
        if a.obs == "v2":
            from scripts.build_scenarios import v2_dir

            g = Gen(v2_dir(s, a.world_variant), s, "check", tag="X" + (a.world_variant or ""))
        elif a.world_variant:
            from scripts.build_scenarios import variant_dir

            g = Gen(variant_dir(s, a.world_variant), s, "check", tag=a.world_variant, family_rng=True)
        else:
            g = Gen(seed_dir(s), s, "check")
        tasks += g.all(a.k)
        g.close()
    fam, area, fails = run(tasks)
    for k in sorted(area):
        v = area[k]
        print(f"AREA {k:16s} n={len(v):4d} mean={sum(v) / len(v):.3f}")
    for k in sorted(fam):
        v = fam[k]
        print(f"  {k:26s} n={len(v):4d} mean={sum(v) / len(v):.3f}")
    for f in fails[: a.show_fail]:
        print("FAIL", *f, sep="\n   ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
