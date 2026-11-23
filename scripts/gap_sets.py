"""Paired world-gap evaluation sets (D-031): pick up to --per-area tasks per area from the nominal twin (N), round
robin over families, then for every world variant keep exactly the tasks with the same (seed, template, question).
Truth (tracks, times) is identical across variants, so these are the same questions asked of different RF worlds;
each side is scored against its own world.

    python scripts/gap_sets.py [--per-area 50] [--variants W0 W1t W1 W2 W3]
    -> eval/frozen/v3w/{N,W0,...}.jsonl + manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.frozen_build import HASHED, sha  # noqa: E402


def key(r: dict) -> tuple:
    return (r["scenario_seed"], r["template"], r["question"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-area", type=int, default=50)
    ap.add_argument("--variants", nargs="+", default=["W0", "W1t", "W1", "W2", "W3"])
    ap.add_argument("--df-per-family", type=int, default=25)
    a = ap.parse_args()
    load = lambda v: [json.loads(x) for x in (REPO / "data" / "tasks" / f"test_{v}.jsonl").read_text().splitlines()]  # noqa: E731
    sets = {v: {key(r): r for r in load(v)} for v in a.variants}
    nominal = load("N")
    common = [r for r in nominal if all(key(r) in sets[v] for v in a.variants)]       # askable in every world
    rng = random.Random(1234)
    by = defaultdict(lambda: defaultdict(list))
    for r in common:
        by[r["area"]][r["template"]].append(r)
    pick = []
    for area in sorted(by):
        fam, chosen = by[area], []
        quota = min(a.per_area, sum(len(x) for x in fam.values()))
        while len(chosen) < quota:
            for f in sorted(fam):
                if fam[f] and len(chosen) < quota:
                    chosen.append(fam[f].pop(rng.randrange(len(fam[f]))))
        pick += chosen
    out = REPO / "eval" / "frozen" / "v3w"
    out.mkdir(parents=True, exist_ok=True)
    (out / "N.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pick))
    for v in a.variants:
        (out / f"{v}.jsonl").write_text("".join(json.dumps(sets[v][key(r)]) + "\n" for r in pick))
    cnt = defaultdict(int)
    for r in pick:
        cnt[r["area"]] += 1
    man = {"version": "v3w", "n_per_set": len(pick), "per_area": dict(cnt), "variants": ["N"] + a.variants,
           "pairing": "(scenario_seed, template, question) identical across all sets",
           "nominal_tasks": len(nominal), "askable_in_all_worlds": len(common),
           "code_sha": {p: sha(REPO / p) for p in HASHED},
           "tasks_sha": {f.name: sha(f) for f in sorted(out.glob("*.jsonl"))}}
    # DF tasks are selected from each world's own observations (e.g. direct_path needs exactly two bearings at that
    # sensor and time), so the same situation is almost never askable in every world (3 of 295). DF is therefore
    # compared UNPAIRED: per world, up to --df-per-family tasks of every DF family, same seeds (D-031).
    man["df_unpaired"] = {}
    for v in ["N"] + a.variants:
        rows = [r for r in load(v) if r["area"] == "df_ambiguity"]
        fam = defaultdict(list)
        for r in rows:
            fam[r["template"]].append(r)
        rr = random.Random(4321)
        pick_df = [x for f in sorted(fam) for x in rr.sample(fam[f], min(a.df_per_family, len(fam[f])))]
        (out / f"df_{v}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pick_df))
        man["df_unpaired"][v] = {f: min(a.df_per_family, len(fam[f])) for f in sorted(fam)}
    man["tasks_sha"] = {f.name: sha(f) for f in sorted(out.glob("*.jsonl"))}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("n_per_set", "per_area", "nominal_tasks", "askable_in_all_worlds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
