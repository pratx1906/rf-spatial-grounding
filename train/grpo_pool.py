"""RL prompt pools and the fixed RL validation subset (paper plan P3 + Revision 1 B1/B3/M3).

    python -m train.grpo_pool val400  [--seed 0]                      # data/rl/val400.jsonl (selection set)
    python -m train.grpo_pool targets --model-val <eval.json> --expert-val <eval.json>   # data/rl/targets.json
    python -m train.grpo_pool build   [--seed 0] [--replay-frac 0.4] [--w-frac 0.0]      # data/rl/pool_{A,B,C}.jsonl
    python -m train.grpo_pool check   data/rl/pool_B.jsonl                # disjointness check against every eval set

Selection (Revision 1, B1): checkpoints are chosen only on `val400`, a fixed stratified 400-task subset of
data/tasks/x3/val_clean.jsonl (val seeds 13-14, held-out families already removed), greedy decoding, rule "best val
mean score, ties to the earlier iteration". No test set (v6, heldout, W4/W5, M*) is read here.

Targets (Revision 1, M3): re-derived from per-family scores on val_clean, NOT from v6. `targets` reads an eval/run.py
output of the init model (SFT v1.1) and of the scripted expert on val_clean and applies the pre-declared rule:
- A "model below expert": n >= MIN_N, expert >= 0.80 and expert - model >= 0.15;
- B "expert weak" (only RL scored against the key can beat the teacher): n >= MIN_N and expert < 0.80;
- replay: model >= 0.95 (not a target), mixed in at `--replay-frac` (30-50 %) so they do not regress.
The rule, both score tables and the resulting lists go into data/rl/targets.json (disclosed in the paper).
Behaviour buckets that do not depend on scores (docs/RESULTS/grpo_inputs.md section 3C):
- abstain calibration pairs: every *_ambiguous row (answer after ask_user) with clear zone_incursion rows of the same
  seeds (near_object itself is a held-out family and never enters a pool);
- drone-search budget wordings: ds_* rows reworded with tasks.kandla.diverse `train` wordings ("allotted",
  "allowed", "you get", terse imperative);
- entered_between rewordings that keep the verb of entry.
Held-out / compositional families (scripts.make_tasks.NOT_FOR_SELECTION) never enter a pool. Exploitable keys never
enter a pool either (AUDIT_CODE C6): the collateral and siting areas (their keys are the tools' own outputs, and the
verifier hint hands over missing collateral refs), and every row whose key an EMPTY answer already scores 1.0 (empty
track / event list, "no"), except the yes/no incursion families (train.rewards.episode.VIOLATION_FAMILIES) where the
"no" rows stay but are capped at the number of "yes" rows per family, so neither "always yes" nor "always no" pays.
noisy_band and first_into_zone have no rows in x3/train (their question texts collide with val/test and were dropped
by make_tasks); they are reported as `unavailable` for pools A/B and enter pool C only through W6-W8 rows.

Pools: A (RL-only from base), B (from SFT v1.1; also B14 and the RFT control) are the same rows, so the arms differ
only in the init / algorithm. C = B + hardware-mismatch train worlds M1-M4 on train seeds (data/tasks/x3_Mtrain/*.jsonl,
built by P6a; TODO until it exists, `build` then skips C and says so) + optional W6-W8 rows (`--w-frac`). Pool rows
carry `pool_bucket` and `world`; Task.from_row ignores them.
Disjointness (`check`, also run by `build`): no pool row shares a task_id, a scenario site or a scenario seed with any
eval set (eval/frozen/*/tasks.jsonl, x3 val/val_clean/test, x3_M0/M2/Npair, x3_W4/W5 test, val400).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "data" / "rl"
TRAIN = REPO / "data" / "tasks" / "x3" / "train.jsonl"
VAL_CLEAN = REPO / "data" / "tasks" / "x3" / "val_clean.jsonl"
W_TRAIN = {w: REPO / "data" / "tasks" / f"x3_{w}" / "train.jsonl" for w in ("W6", "W7", "W8")}
MTRAIN_DIR = REPO / "data" / "tasks" / "x3_Mtrain"          # TODO(P6a): written by runs/paper/p6a_after_queue.sh train
EVAL_SEEDS = {0, 13, 14, 15, 16, *range(51, 61)}            # demo, val, test seeds: never in a pool
MIN_N = 4
RULE = {"A": "n>=4 and expert>=0.80 and expert-model>=0.15", "B": "n>=4 and expert<0.80",
        "replay": "n>=4 and model>=0.95", "source": "x3/val_clean per-family scores (not v6; Revision 1 M3)"}
PAIR_FAMILIES = ("near_object_ambiguous", "zone_incursion_ambiguous")
MISSING_IN_TRAIN = ("noisy_band", "first_into_zone")


def rows_of(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_rows(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def not_for_selection() -> set[str]:
    from scripts.make_tasks import NOT_FOR_SELECTION

    return set(NOT_FOR_SELECTION)


# --- validation subset ------------------------------------------------------------------------------------------
def val_subset(rows: list[dict], n: int = 400, seed: int = 0) -> list[dict]:
    """Stratified by family: proportional allocation with at least min(2, n_f) per family; deterministic in seed."""
    banned = not_for_selection()
    rows = [r for r in rows if r["template"] not in banned]
    by = defaultdict(list)
    for r in rows:
        by[r["template"]].append(r)
    rng = random.Random(seed)
    fams = sorted(by)
    alloc = {f: min(len(by[f]), 2) for f in fams}
    rest = n - sum(alloc.values())
    if rest > 0:
        tot = sum(len(by[f]) - alloc[f] for f in fams)
        extra = {f: (len(by[f]) - alloc[f]) * rest / tot for f in fams}
        for f in fams:
            alloc[f] += int(extra[f])
        left = n - sum(alloc.values())
        for f in sorted(fams, key=lambda f: -(extra[f] - int(extra[f]))):
            if left <= 0:
                break
            if alloc[f] < len(by[f]):
                alloc[f] += 1
                left -= 1
    out = []
    for f in fams:
        out += rng.sample(sorted(by[f], key=lambda r: r["task_id"]), alloc[f])
    return sorted(out, key=lambda r: r["task_id"])


# --- targets ----------------------------------------------------------------------------------------------------
def family_scores(eval_json: dict) -> dict[str, tuple[int, float]]:
    fam = eval_json["summary"]["_families"]
    return {k: (int(v[0]), float(v[1])) for k, v in fam.items()}


def derive_targets(model: dict[str, tuple[int, float]], expert: dict[str, tuple[int, float]]) -> dict:
    banned = not_for_selection()
    a, b, replay, skipped = [], [], [], []
    for f in sorted(set(model) | set(expert)):
        if f in banned:
            continue
        if f not in model or f not in expert or min(model[f][0], expert[f][0]) < MIN_N:
            skipped.append(f)
            continue
        m, e = model[f][1], expert[f][1]
        if e < 0.80:
            b.append(f)
        elif e - m >= 0.15:
            a.append(f)
        elif m >= 0.95:
            replay.append(f)
    return {"rule": RULE, "A": a, "B": b, "replay": replay, "skipped_small_n": skipped,
            "model": {k: list(v) for k, v in sorted(model.items())},
            "expert": {k: list(v) for k, v in sorted(expert.items())}}


# --- pools ------------------------------------------------------------------------------------------------------
def _tag(r: dict, bucket: str, world: str | None = None) -> dict:
    r = dict(r)
    r["pool_bucket"], r["world"] = bucket, world
    return r


def _take(rng: random.Random, rows: list[dict], k: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["task_id"])
    return rng.sample(rows, min(k, len(rows)))


EXCLUDED_AREAS = ("collateral", "siting")


def empty_answer_full_credit(row: dict) -> bool:
    """True when an empty answer ("no", no tracks / events) scores >= 0.999 on this row's key (AUDIT_CODE C6)."""
    from types import SimpleNamespace

    from tasks.base import Task
    from tasks.scoring import score_answer

    area = row["area"]
    if area not in ("operator_query", "replay"):
        return False
    t = Task.from_row(row)
    if area == "replay":
        tries = [SimpleNamespace(intent="replay", evt_refs=[], time_s=None)]
    else:
        tries = [SimpleNamespace(intent="operator_query", track_refs=[], boolean=b, region=None, protocol=None,
                                 band=None) for b in (None, False)]
    for a in tries:
        try:
            if score_answer(t, a) >= 0.999:
                return True
        except (KeyError, AttributeError, TypeError):
            continue
    return False


def rl_eligible(row: dict) -> bool:
    from train.rewards.episode import VIOLATION_FAMILIES

    if row["area"] in EXCLUDED_AREAS:
        return False
    return row["template"] in VIOLATION_FAMILIES or not empty_answer_full_credit(row)


def balance_yes_no(rows: list[dict], rng: random.Random) -> list[dict]:
    """Yes/no incursion families: keep at most as many "no" rows as "yes" rows per family."""
    from train.rewards.episode import VIOLATION_FAMILIES

    out, by = [], defaultdict(lambda: ([], []))
    for r in rows:
        k = r.get("answer_key") or {}
        if r["template"] in VIOLATION_FAMILIES and k.get("boolean") is not None:
            by[r["template"]][0 if k["boolean"] else 1].append(r)
        else:
            out.append(r)
    for _f, (yes, no) in sorted(by.items()):
        out += yes + _take(rng, no, len(yes))
    return out


def build_core(train: list[dict], targets: dict, seed: int = 0, cap_a: int = 120, cap_b: int = 100,
               n_ds: int = 100, n_eb: int = 60, replay_frac: float = 0.4) -> tuple[list[dict], dict]:
    """Pool A/B rows from x3 train: targets + behaviour buckets + replay at `replay_frac` of the final pool."""
    from tasks.kandla.diverse import DS_FAMILIES, reword

    banned = not_for_selection()
    train = [r for r in train if r["template"] not in banned and r.get("scenario_seed") not in EVAL_SEEDS
             and rl_eligible(r)]
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in train:
        by[r["template"]].append(r)
    out: list[dict] = []
    for f in targets["A"]:
        out += [_tag(r, "target_A") for r in _take(rng, by.get(f, []), cap_a)]
    for f in targets["B"]:
        out += [_tag(r, "target_B") for r in _take(rng, by.get(f, []), cap_b)]
    have = {r["task_id"] for r in out}
    # abstain calibration pairs: ambiguous rows + as many clear zone_incursion rows from the same seeds
    amb = [r for f in PAIR_FAMILIES for r in by.get(f, [])]
    out += [_tag(r, "abstain_pair") for r in amb if r["task_id"] not in have]
    seeds = Counter(r["scenario_seed"] for r in amb)
    for s, k in sorted(seeds.items()):
        clear = [r for r in by.get("zone_incursion", []) if r["scenario_seed"] == s and r["task_id"] not in have]
        out += [_tag(r, "abstain_pair") for r in _take(rng, clear, k)]
    # wording robustness (keys unchanged): drone-search budgets and entered_between
    ds = [r for f in DS_FAMILIES for r in by.get(f, [])]
    for r in _take(rng, ds, n_ds):
        rw = reword(r, "train", rng)
        if rw is not None:
            out.append(_tag(rw, "ds_wording"))
    for r in _take(rng, by.get("entered_between", []), n_eb):
        rw = reword(r, "train", rng)
        if rw is not None:                        # the curated train wordings all keep an entry verb
            out.append(_tag(rw, "reword"))
    # replay of families already at >= 0.95 (not a target)
    n_rep = round(replay_frac / (1 - replay_frac) * len(out)) if replay_frac < 1 else 0
    fams = [f for f in targets["replay"] if by.get(f)]
    if fams and n_rep:
        per = max(1, n_rep // len(fams))
        rep = []
        for f in fams:
            rep += _take(rng, by[f], per)
        out += [_tag(r, "replay") for r in _take(rng, rep, n_rep)]
    out = balance_yes_no(out, rng)
    seen, uniq = set(), []
    for r in out:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            uniq.append(r)
    rng.shuffle(uniq)
    info = {"unavailable": [f for f in list(targets["A"]) + list(targets["B"]) + list(MISSING_IN_TRAIN)
                            if not by.get(f)]}
    return uniq, info


def build_world_rows(paths: dict[str, Path], families: set[str], n: int, seed: int, bucket: str) -> list[dict]:
    banned = not_for_selection()
    rng = random.Random(seed + 17)
    per = max(1, n // max(1, len(paths)))
    out = []
    for w, p in sorted(paths.items()):
        rows = [r for r in rows_of(p) if r["template"] in families and r["template"] not in banned
                and r.get("scenario_seed") not in EVAL_SEEDS and rl_eligible(r)]
        rows = balance_yes_no(rows, rng)
        out += [_tag(r, bucket, r.get("world_variant") or w) for r in _take(rng, rows, per)]
    return out


def mtrain_paths() -> dict[str, Path]:
    return {p.stem: p for p in sorted(MTRAIN_DIR.glob("*.jsonl"))} if MTRAIN_DIR.exists() else {}


# --- disjointness -----------------------------------------------------------------------------------------------
def eval_sets() -> list[Path]:
    t = REPO / "data" / "tasks"
    out = sorted((REPO / "eval" / "frozen").glob("*/tasks.jsonl"))
    out += [t / "x3" / n for n in ("val.jsonl", "val_clean.jsonl", "test.jsonl")]
    out += sorted(t.glob("x3_M*/test*.jsonl")) + sorted(t.glob("x3_Npair/test*.jsonl"))
    out += sorted(t.glob("x3_W4/*.jsonl")) + sorted(t.glob("x3_W5/*.jsonl"))
    out += [OUT / "val400.jsonl"]
    return [p for p in out if p.exists()]


def check_disjoint(pool: list[dict], sets: list[Path] | None = None) -> dict:
    """Raises ValueError on any shared task_id, scenario site or eval scenario seed; returns the counts (and the
    number of identical question texts, reported only: templated wordings recur across seeds by design)."""
    ids, sites, qs = set(), set(), set()
    for p in sets if sets is not None else eval_sets():
        for r in rows_of(p):
            ids.add(r["task_id"])
            sites.add(str(r.get("site", "")).rstrip("/"))
            qs.add(r["question"])
    sites.discard("")
    bad_id = [r["task_id"] for r in pool if r["task_id"] in ids]
    bad_site = sorted({r["site"] for r in pool if str(r.get("site", "")).rstrip("/") in sites})
    bad_seed = sorted({r.get("scenario_seed") for r in pool if r.get("scenario_seed") in EVAL_SEEDS})
    banned = sorted({r["template"] for r in pool} & not_for_selection())
    if bad_id or bad_site or bad_seed or banned:
        raise ValueError(f"pool overlaps eval data: ids {bad_id[:5]} ({len(bad_id)}), sites {bad_site[:5]}, "
                         f"seeds {bad_seed}, held-out families {banned}")
    return {"n": len(pool), "eval_ids": len(ids), "eval_sites": len(sites),
            "same_question_text": sum(r["question"] in qs for r in pool)}


# --- CLI --------------------------------------------------------------------------------------------------------
def _summary(rows: list[dict]) -> dict:
    return {"n": len(rows), "buckets": dict(Counter(r["pool_bucket"] for r in rows)),
            "worlds": dict(Counter(str(r.get("world")) for r in rows)),
            "families": dict(sorted(Counter(r["template"] for r in rows).items()))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("val400")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--n", type=int, default=400)
    t = sub.add_parser("targets")
    t.add_argument("--model-val", required=True)
    t.add_argument("--expert-val", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--replay-frac", type=float, default=0.4)
    b.add_argument("--w-frac", type=float, default=0.0, help="pool C: share of W6-W8 rows (optional mix)")
    b.add_argument("--m-frac", type=float, default=0.35, help="pool C: share of mismatch-world (M1-M4) rows")
    b.add_argument("--require-c", action="store_true", help="fail if the mismatch train worlds are missing")
    c = sub.add_parser("check")
    c.add_argument("pools", nargs="+")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "val400":
        rows = val_subset(rows_of(VAL_CLEAN), a.n, a.seed)
        sha = write_rows(OUT / "val400.jsonl", rows)
        man = {"source": "data/tasks/x3/val_clean.jsonl", "n": len(rows), "seed": a.seed, "sha256": sha,
               "rule": "best val mean score (greedy), ties to the earlier iteration",
               "families": dict(sorted(Counter(r["template"] for r in rows).items()))}
        (OUT / "val400.manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps(man))
        return 0
    if a.cmd == "targets":
        tg = derive_targets(family_scores(json.loads(Path(a.model_val).read_text())),
                            family_scores(json.loads(Path(a.expert_val).read_text())))
        tg["model_val"], tg["expert_val"] = a.model_val, a.expert_val
        (OUT / "targets.json").write_text(json.dumps(tg, indent=1))
        print(json.dumps({k: tg[k] for k in ("A", "B", "replay", "skipped_small_n")}))
        return 0
    if a.cmd == "check":
        for p in a.pools:
            print(p, json.dumps(check_disjoint(rows_of(Path(p)))))
        return 0
    targets = json.loads((OUT / "targets.json").read_text())
    core, info = build_core(rows_of(TRAIN), targets, seed=a.seed, replay_frac=a.replay_frac)
    man = {"seed": a.seed, "replay_frac": a.replay_frac, "targets": {k: targets[k] for k in ("A", "B", "replay")},
           "excluded_areas": list(EXCLUDED_AREAS), "excluded_keys": "empty answer scores 1.0 (yes/no incursion "
           "families kept, 'no' capped at 'yes' per family)",
           "rule": targets["rule"], **info, "pools": {}}
    for name in ("A", "B"):
        sha = write_rows(OUT / f"pool_{name}.jsonl", core)
        man["pools"][name] = {"sha256": sha, **_summary(core), "disjoint": check_disjoint(core),
                              "note": "A and B hold the same rows (arms differ only in init / algorithm)"}
    fams = {r["template"] for r in core}
    mp = mtrain_paths()
    if not mp:
        msg = f"pool C skipped: no mismatch train tasks under {MTRAIN_DIR.relative_to(REPO)} (TODO: P6a)"
        man["pools"]["C"] = {"skipped": msg}
        print(msg, file=sys.stderr)
        if a.require_c:
            raise SystemExit(msg)
    else:
        n_core = len(core)
        n_m = round(a.m_frac / (1 - a.m_frac - a.w_frac) * n_core)
        n_w = round(a.w_frac / (1 - a.m_frac - a.w_frac) * n_core) if a.w_frac else 0
        mrows = build_world_rows(mp, fams | {"zone_incursion", "swarm_in_zone", "count_in_zone"}, n_m, a.seed,
                                 "mtrain")
        wrows = build_world_rows({w: p for w, p in W_TRAIN.items() if p.exists()}, fams | set(MISSING_IN_TRAIN),
                                 n_w, a.seed, "world") if n_w else []
        pool_c = core + mrows + wrows
        random.Random(a.seed + 3).shuffle(pool_c)
        sha = write_rows(OUT / "pool_C.jsonl", pool_c)
        man["pools"]["C"] = {"sha256": sha, "m_frac": a.m_frac, "w_frac": a.w_frac, "mtrain_files": sorted(mp),
                             **_summary(pool_c), "disjoint": check_disjoint(pool_c)}
    (OUT / "pools.manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("n", "buckets", "skipped")}
                      for k, v in man["pools"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
