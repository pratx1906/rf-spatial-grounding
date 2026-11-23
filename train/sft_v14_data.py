"""SFT v1.4 data (PLAN 2026-10-23 rf-convergence Task 7): diverse questions + v3 base + ~50 % SFT v1.1 replay.

    python -m train.sft_v14_data gen      # diverse train tasks (train seeds) + held-out-LIKE val tasks (val seeds)
    python -m train.sft_v14_data record   # scripted expert on them + a balanced v3 base sample (CPU)
    python -m train.sft_v14_data mix      # noise + v1.1 replay + length check -> data/sft/train_v14.jsonl, val_v14.jsonl
    python -m train.sft_v14_data heldout  # held-out-family eval set from the test split (eval only)
    common: --tasks-dir data/tasks/x3 --work runs/s79/v14 [--tiny] (tiny: a few tasks per family, for a dry run)

Sources: `--tasks-dir` train.jsonl (seeds 1-12, 17-28; held-out families are already absent) and val.jsonl (seeds
13-14). Needs SCENESCOUT_DATA_VERSION=v3 (siting keys are computed on the v3 coverage model, as make_tasks does).

- Diverse train part (tasks.kandla.diverse, `train` wordings): rewordings of drone search, LOS, direct path,
  collateral (hypothetical / passive voice, incl. fratricide) and entered_between tasks (keys unchanged), plus
  weak_df / weak_detect questions with and without height and threshold (defaults 10 deg / 0.5 / 40 m when omitted;
  key from weak_zones with the effective values). weak_low_alt (and every HELD_OUT family) is never generated.
- v3 base: a balanced sample of the original train tasks across all seen families (the v3 data the gate tests).
- Replay (about half of the mix): SFT v1.1's own set (data/sft/train_v11.jsonl), topped up from the v1 balanced
  set (data/sft/train_bal_v2.jsonl) that v1.1 also learned.
- Validation, held-out-LIKE: the disjoint `val` wordings on the val seeds (rewordings + siting with omitted values)
  plus a seen-family sample in the original wording. `tasks_val.jsonl` is used for checkpoint selection by task
  score (scripts/sft_v14_report.py select); its expert records (val_v14.jsonl) give the val loss. No HELD_OUT family.
Only expert episodes the verifier accepted with score >= train.sft_data.MIN_SCORE are learned.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAIN_SEEDS = list(range(1, 13)) + list(range(17, 29))
VAL_SEEDS = [13, 14]
MAX_LEN = 15360
CAPS = {"ds_hover": 45, "ds_hover_budget2": 45, "ds_loiter": 45, "ds_swarm": 45, "los_check": 45, "direct_path": 45,
        "jam_all": 20, "jam_protected": 20, "jam_links": 20, "jam_aviation": 20, "jam_fratricide": 35,
        "entered_between": 40}                      # rewordings per family (train)
N_SITING = 420                                       # siting draws (train); ~65 % have no weak zone (dropped): ~150 kept
N_SITING_VAL = 80
VAL_PER_REWORD_FAMILY = 8
VAL_PER_SEEN_FAMILY = 2
N_BASE = 240                                         # v3 base sample (original wording, all seen families)
REPLAY_FRAC = 0.5
NOISE = 0.2                                          # operator-noise share on the new part (v1.1 used 0.35)
N_VAL_LOSS = 120                                     # train.sft reads at most 120 val records


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _siting(seeds: list[int], n: int, split: str, rng: random.Random, tiny: bool) -> tuple[list[dict], dict]:
    """weak_df / weak_detect tasks with varied or omitted height / threshold; keys from the generator's own
    weak_zones call with the EFFECTIVE values (defaults when omitted), as tasks/kandla/gen.py does."""
    from scripts.build_scenarios import v2_dir
    from tasks.kandla import diverse as dv
    from tasks.kandla.gen_more import GenMore

    g0 = GenMore(v2_dir(seeds[0], None), seeds[0], split, tag="X3d")
    regions = ["hero_zone"] + [z for z, v in g0.site.zones.items() if v.kind == "restricted"]
    g0.close()
    specs = []
    for _ in range(n):
        fam = rng.choice(dv.SITING_FAMILIES)
        h = None if rng.random() < 0.4 else rng.choice(dv.HEIGHTS)
        pool = dv.DF_THRESHOLDS if fam == "weak_df" else dv.DETECT_THRESHOLDS
        thr = None if rng.random() < 0.5 else rng.choice(pool)
        specs.append((rng.choice(seeds), fam, rng.choice(regions), rng.choice(list(dv.BAND_PHRASE)), h, thr))
    by_seed = defaultdict(list)
    for s in specs:
        by_seed[s[0]].append(s[1:])
    out, stats = [], Counter()
    for seed in sorted(by_seed):
        g = GenMore(v2_dir(seed, None), seed, split, tag="X3d")
        for fam, z, band, h, thr in by_seed[seed]:
            q, sp = dv.siting_question(fam, rng.choice(dv.BAND_PHRASE[band]), g.zone_name(z), h, thr, split, rng)
            assert not dv.is_low_alt_wording(q), q
            reg = {"type": "ref", "ref": f"zone:{z}"}
            d = g.call("weak_zones", metric=sp["metric"], band=band, threshold=sp["threshold"], region=reg,
                       height_m=sp["height_m"], limit=5)
            if not d["zones"]:
                stats["empty"] += 1
                continue
            geoms = [g.ctx.episode["ras"][zz["ref"]]["geom"].wkt for zz in d["zones"]]
            t = g.task("siting", fam, q, {"zones_wkt": geoms},
                       {"zone": f"zone:{z}", "band": band, **sp})
            r = t.to_row()
            r["categories"] = list(r.get("categories") or []) + ["diverse"]
            r["diverse_omitted"] = [k for k, v in (("height", h), ("threshold", thr)) if v is None]
            out.append(r)
            stats[f"{fam}|omit:{'+'.join(r['diverse_omitted']) or '-'}"] += 1
        g.close()
        if tiny and len(out) >= 4:
            break
    return out, dict(stats)


def cmd_gen(a) -> None:
    from scripts.make_tasks import HELD_OUT
    from tasks.kandla import diverse as dv

    rng = random.Random(1404)
    tdir, work = REPO / a.tasks_dir, REPO / a.work
    train, val = _rows(tdir / "train.jsonl"), _rows(tdir / "val.jsonl")
    eval_q = {r["question"] for r in val} | {r["question"] for r in _rows(tdir / "test.jsonl")}
    # --- train: rewordings + siting -----------------------------------------------------------------------------
    by = defaultdict(list)
    for r in train:
        by[r["template"]].append(r)
    caps = {f: (2 if a.tiny else c) for f, c in CAPS.items()}
    out = []
    for fam, cap in caps.items():
        pool = by.get(fam, [])[:]
        rng.shuffle(pool)
        for r in pool[:cap]:
            d = dv.reword(r, "train", rng)
            if d and d["question"] not in eval_q:
                out.append(d)
    sit, sit_stats = _siting(TRAIN_SEEDS, 6 if a.tiny else N_SITING, "train", rng, a.tiny)
    out += [r for r in sit if r["question"] not in eval_q]
    assert not {r["template"] for r in out} & HELD_OUT
    rng.shuffle(out)
    _write(work / "tasks_train.jsonl", out)
    # --- val (held-out-LIKE): unseen wordings on the val seeds + seen-family sample ---------------------------------
    vby = defaultdict(list)
    for r in val:
        vby[r["template"]].append(r)
    vout = []
    for fam in CAPS:
        pool = vby.get(fam, [])[:]
        rng.shuffle(pool)
        for r in pool[: (1 if a.tiny else VAL_PER_REWORD_FAMILY)]:
            d = dv.reword(r, "val", rng)
            if d:
                d["val_kind"] = "unseen_wording"
                vout.append(d)
    vs, vs_stats = _siting(VAL_SEEDS, 4 if a.tiny else N_SITING_VAL, "val", rng, a.tiny)
    for r in vs:
        r["val_kind"] = "unseen_wording"
    vout += vs
    seen = sorted(f for f in vby if f not in HELD_OUT)
    for fam in seen[: (6 if a.tiny else None)]:
        pool = vby[fam][:]
        rng.shuffle(pool)
        for r in pool[: (1 if a.tiny else VAL_PER_SEEN_FAMILY)]:
            vout.append({**r, "val_kind": "seen_family"})
    assert not {r["template"] for r in vout} & HELD_OUT
    trq = {r["question"] for r in out}
    vout = [r for r in vout if r["question"] not in trq]
    _write(work / "tasks_val.jsonl", vout)
    print(json.dumps({"train_tasks": len(out), "by_family": dict(Counter(r["template"] for r in out)),
                      "siting": sit_stats, "val_tasks": len(vout),
                      "val_by_kind": dict(Counter(r["val_kind"] for r in vout)), "val_siting": vs_stats}, indent=1))


def _base_sample(train: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Balanced v3 base sample: area shares of train.sft_subset, family round-robin inside an area."""
    from train.sft_subset import SHARE

    fam = defaultdict(lambda: defaultdict(list))
    for r in train:
        fam[r["area"]][r["template"]].append(r)
    out = []
    for area, share in SHARE.items():
        quota = max(1, round(n * share))
        pools = {f: rng.sample(v, len(v)) for f, v in fam[area].items()}
        got = []
        while len(got) < quota and any(pools.values()):
            for f in sorted(pools):
                if pools[f] and len(got) < quota:
                    got.append(pools[f].pop())
        out += got
    return out


def cmd_record(a) -> None:
    from train.sft_v12_data import _record

    rng = random.Random(1405)
    tdir, work = REPO / a.tasks_dir, REPO / a.work
    tasks = _rows(work / "tasks_train.jsonl")
    base = _base_sample(_rows(tdir / "train.jsonl"), 12 if a.tiny else N_BASE, rng)
    for r in base:
        r["categories"] = list(r.get("categories") or []) + ["v3_base"]
    _write(work / "tasks_base.jsonl", base)
    res = {"diverse": _record(tasks, work / "rec_diverse.jsonl"), "base": _record(base, work / "rec_base.jsonl"),
           "val": _record(_rows(work / "tasks_val.jsonl"), work / "rec_val.jsonl")}
    print(json.dumps(res, indent=1), flush=True)


def cmd_mix(a) -> None:
    from scripts import cuda_env  # noqa: F401  (HF_HOME)
    from scripts.augment_text import noisy
    from transformers import AutoTokenizer

    from harness.prompt import static_prefix
    from train.sft_data import tokenize_with_mask

    rng = random.Random(1406)
    work = REPO / a.work
    new = []
    for src, f in (("v14_diverse", "rec_diverse.jsonl"), ("v3_base", "rec_base.jsonl")):
        for rec in _rows(work / f):
            rec.pop("site", None)
            if rng.random() < NOISE:
                rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
                rec["noised"] = True
            rec["source"] = src
            new.append(rec)
    # replay = about half of the mix: all of SFT v1.1's own set, topped up from the v1 balanced set v1.1 also learned
    # (data/sft/train_bal_v2.jsonl; held-out families are absent from both). No example is repeated.
    from scripts.make_tasks import HELD_OUT

    v11 = _rows(REPO / "data/sft/train_v11.jsonl")
    n_rep = round(len(new) * REPLAY_FRAC / (1 - REPLAY_FRAC))
    rep = rng.sample(v11, min(n_rep, len(v11)))
    if len(rep) < n_rep and (REPO / "data/sft/train_bal_v2.jsonl").exists():
        have = {r["task_id"] for r in rep}
        pool = [r for r in _rows(REPO / "data/sft/train_bal_v2.jsonl")
                if r["task_id"] not in have and r["template"] not in HELD_OUT]
        rep += rng.sample(pool, min(n_rep - len(rep), len(pool)))
    for r in rep:
        r["source"] = "v11_replay"
    allrec = new + rep
    rng.shuffle(allrec)
    tools = static_prefix()[1]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    keep, too_long, lens = [], [], []
    for r in allrec:
        ids, _ = tokenize_with_mask(tok, r["messages"], tools, max_len=10 ** 9)
        if len(ids) > MAX_LEN:
            too_long.append((r["task_id"], len(ids)))
            continue
        lens.append(len(ids))
        keep.append(r)
    out = REPO / a.out
    _write(out, keep)
    vrec = _rows(work / "rec_val.jsonl")
    vby = defaultdict(list)
    for r in vrec:
        r.pop("site", None)
        vby[r["template"]].append(r)
    val = []                                           # family round-robin so 120 records cover every family
    while len(val) < N_VAL_LOSS and any(vby.values()):
        for f in sorted(vby):
            if vby[f] and len(val) < N_VAL_LOSS:
                val.append(vby[f].pop(0))
    vl = [len(tokenize_with_mask(tok, r["messages"], tools, max_len=10 ** 9)[0]) for r in val]
    val = [r for r, n in zip(val, vl, strict=True) if n <= MAX_LEN]
    _write(REPO / a.val_out, val)
    lens.sort()
    by = defaultdict(Counter)
    for r in keep:
        by[r["source"]][r["area"]] += 1
    src = Counter(r["source"] for r in keep)
    print(json.dumps({"train": len(keep), **dict(src), "replay_frac": round(src["v11_replay"] / max(1, len(keep)), 3),
                      "per_area": {s: dict(c) for s, c in by.items()}, "noised_new": sum(bool(r.get("noised")) for r in keep if r["source"] != "v11_replay"),
                      "too_long": too_long, "tools": len(tools), "len_max": lens[-1] if lens else 0,
                      "len_p50": lens[len(lens) // 2] if lens else 0, "val_n": len(val),
                      "val_len_max": max(vl) if vl else 0, "out": str(out)}, indent=1))


def cmd_heldout(a) -> None:
    """Held-out-family eval set from the test split (seeds 15-16, 51-60): per family up to --per-family tasks."""
    from scripts.make_tasks import HELD_OUT

    rng = random.Random(1407)
    by = defaultdict(list)
    for r in _rows(REPO / a.tasks_dir / "test.jsonl"):
        if r["template"] in HELD_OUT:
            by[r["template"]].append(r)
    out = []
    for f in sorted(by):
        pool = by[f][:]
        rng.shuffle(pool)
        out += pool[: (2 if a.tiny else a.per_family)]
    _write(REPO / a.work / "heldout_tasks.jsonl", out)
    print(json.dumps({"heldout_tasks": len(out), "by_family": dict(Counter(r["template"] for r in out))}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "record", "mix", "heldout"])
    ap.add_argument("--tasks-dir", default="data/tasks/x3")
    ap.add_argument("--work", default="runs/s79/v14")
    ap.add_argument("--out", default="data/sft/train_v14.jsonl")
    ap.add_argument("--val-out", default="data/sft/val_v14.jsonl")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--tiny", action="store_true", help="a few tasks per family (dry run)")
    a = ap.parse_args()
    {"gen": cmd_gen, "record": cmd_record, "mix": cmd_mix, "heldout": cmd_heldout}[a.stage](a)


if __name__ == "__main__":
    main()
