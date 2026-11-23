"""Freeze the evaluation set (SPEC 11): a stratified subset of the test split, version-hashed with the scoring code.

    python -m eval.frozen_build [--per-area 100] [--version v1] [--source data/tasks/test.jsonl]
    python -m eval.frozen_build --version rfg_core --source data/tasks/x3/test.jsonl \
        --areas operator_query,df_ambiguity,drone_search --target 1500
    python -m eval.frozen_build --version heldout_v1 --copy runs/s79/v14/heldout_tasks.jsonl

Default: --per-area tasks per area, round-robin over families. --target: about N tasks from --areas, stratified per
(family, seed cluster) (plan 2026-10-28 T5): every family gets min(available, Q) tasks with Q water-filled so the
total is N; each family's quota is split over the seed clusters (15-16, 51-60) in proportion to availability, and
round-robin over seeds inside a cluster. --copy freezes a task file verbatim. Writes
eval/frozen/<version>/{tasks.jsonl, manifest.json}. Changing tasks or scoring requires a new version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HASHED = ["tasks/scoring.py", "harness/verifier.py", "harness/answers.py", "harness/prompt.py", "harness/executor.py",
          "harness/env.py"] + sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in (REPO / "tools").glob("*.py")) +     sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in (REPO / "sim" / "belief").glob("*.py"))   # trust (D-031)
SEED_CLUSTERS = {"15-16": range(15, 17), "51-60": range(51, 61)}
RNG_SEED = 1234


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def seed_text(rows: list[dict]) -> str:
    """The scenario seeds of a set as compact ranges ("15-16,51-60")."""
    ss = sorted({int(r["scenario_seed"]) for r in rows})
    out, i = [], 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[j] + 1:
            j += 1
        out.append(f"{ss[i]}-{ss[j]}" if j > i else f"{ss[i]}")
        i = j + 1
    return ",".join(out)


def cluster_of(seed: int) -> str:
    return next((k for k, r in SEED_CLUSTERS.items() if seed in r), str(seed))


def pick_per_area(rows: list[dict], per_area: int, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["area"]].append(r)
    pick = []
    for area in sorted(by):
        fam = defaultdict(list)
        for r in by[area]:
            fam[r["template"]].append(r)
        quota, chosen = min(per_area, len(by[area])), []
        while len(chosen) < quota:                       # round-robin over families so every family is covered
            for f in sorted(fam):
                if fam[f] and len(chosen) < quota:
                    chosen.append(fam[f].pop(rng.randrange(len(fam[f]))))
        pick += chosen
    return pick


def family_quotas(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Water-filling: min(n_f, Q) per family with the largest Q whose total <= target; the remainder goes one each
    to the unsaturated families in name order. Total = min(target, sum n_f)."""
    if target >= sum(sizes.values()):
        return dict(sizes)
    q = 0
    while sum(min(n, q + 1) for n in sizes.values()) <= target:
        q += 1
    quota = {f: min(n, q) for f, n in sizes.items()}
    left = target - sum(quota.values())
    for f in sorted(sizes):
        if left and sizes[f] > quota[f]:
            quota[f] += 1
            left -= 1
    return quota


def split_quota(avail: dict[str, int], k: int) -> dict[str, int]:
    """k over clusters in proportion to availability (largest remainder; never above availability)."""
    tot = sum(avail.values())
    raw = {c: k * n / tot for c, n in avail.items()}
    out = {c: min(avail[c], int(v)) for c, v in raw.items()}
    for c in sorted(avail, key=lambda c: (-(raw[c] - int(raw[c])), c)):
        if sum(out.values()) >= k:
            break
        if out[c] < avail[c]:
            out[c] += 1
    while sum(out.values()) < k:                         # capacity fallback
        c = next(c for c in sorted(avail) if out[c] < avail[c])
        out[c] += 1
    return out


def pick_stratified(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """About `target` tasks stratified per (family, seed cluster), round-robin over seeds inside a cluster."""
    fam = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["task_id"]):
        fam[r["template"]].append(r)
    quota = family_quotas({f: len(v) for f, v in fam.items()}, target)
    pick = []
    for f in sorted(fam):
        cl = defaultdict(lambda: defaultdict(list))
        for r in fam[f]:
            cl[cluster_of(int(r["scenario_seed"]))][int(r["scenario_seed"])].append(r)
        kq = split_quota({c: sum(len(v) for v in s.values()) for c, s in cl.items()}, quota[f])
        for c in sorted(cl):
            seeds, chosen = cl[c], []
            while len(chosen) < kq[c]:
                for s in sorted(seeds):
                    if seeds[s] and len(chosen) < kq[c]:
                        chosen.append(seeds[s].pop(rng.randrange(len(seeds[s]))))
            pick += chosen
    return pick


def counts(rows: list[dict]) -> dict:
    return {"per_area": dict(sorted(Counter(r["area"] for r in rows).items())),
            "per_family": dict(sorted(Counter(r["template"] for r in rows).items())),
            "per_seed": {str(k): v for k, v in sorted(Counter(int(r["scenario_seed"]) for r in rows).items())},
            "per_family_cluster": {f: dict(sorted(Counter(cluster_of(int(r["scenario_seed"])) for r in rows
                                                          if r["template"] == f).items()))
                                   for f in sorted({r["template"] for r in rows})}}


def build(version: str, source: str, per_area: int = 100, areas: list[str] | None = None, target: int | None = None,
          copy: bool = False, root: Path = REPO / "eval" / "frozen") -> dict:
    out = root / version
    if out.exists():
        raise SystemExit(f"eval/frozen/{version} exists; frozen sets are immutable (pick a new version)")
    src = REPO / source
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    if any(r.get("split", "test") != "test" for r in rows):
        raise SystemExit(f"{source} contains non-test rows")
    rng = random.Random(RNG_SEED)
    if copy:
        pick, mode = rows, "copy"
    else:
        if areas:
            rows = [r for r in rows if r["area"] in set(areas)]
        if target:
            pick, mode = pick_stratified(rows, target, rng), f"stratified family x seed-cluster, target {target}"
        else:
            pick, mode = pick_per_area(rows, per_area, rng), f"per-area {per_area}, round-robin over families"
    if len({r["task_id"] for r in pick}) != len(pick):
        raise SystemExit("duplicate task ids")
    out.mkdir(parents=True)
    tf = out / "tasks.jsonl"
    tf.write_text("".join(json.dumps(r) + "\n" for r in pick))
    from scripts.make_tasks import COMPOSITIONAL, HELD_OUT

    man = {"version": version, "n": len(pick), **counts(pick), "tasks_sha": sha(tf),
           "tasks_sha256": hashlib.sha256(tf.read_bytes()).hexdigest(),
           "code_sha": {p: sha(REPO / p) for p in HASHED},
           "source": f"{source} (seeds {seed_text(pick)})",
           "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "selection": {"mode": mode, "areas": areas, "rng_seed": None if copy else RNG_SEED,
                         "seed_clusters": {k: [r.start, r.stop - 1] for k, r in SEED_CLUSTERS.items()}},
           "held_out_families": sorted({r["template"] for r in pick} & HELD_OUT),
           "compositional_families": sorted({r["template"] for r in pick} & COMPOSITIONAL),
           "data_sha": {s: sha(REPO / s / "observed.duckdb") for s in sorted({r["site"] for r in pick})
                        if (REPO / s / "observed.duckdb").exists()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-area", type=int, default=100)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--source", default="data/tasks/test.jsonl", help="test-split tasks file (repo-relative)")
    ap.add_argument("--areas", default=None, help="comma list of areas to keep")
    ap.add_argument("--target", type=int, default=None, help="stratify per (family, seed cluster) to ~N tasks")
    ap.add_argument("--copy", default=None, metavar="SRC", help="freeze SRC verbatim (repo-relative test tasks file)")
    a = ap.parse_args()
    man = build(a.version, a.copy or a.source, a.per_area, a.areas.split(",") if a.areas else None, a.target,
                copy=bool(a.copy))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
