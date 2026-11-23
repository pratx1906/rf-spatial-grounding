"""SFT v1.3 data (demo-v2 S2.5b): height-varied siting examples on top of the v1.2 v2 base.

Diagnosis (S2.5): every v1.2 siting example asked about 40 m (weak_df, weak_detect) or used 15 m (suggest_*), so the
model passed height_m=40 whatever the question said and lost the held-out family weak_low_alt (15 m). v1.3 adds
weak_df / weak_detect questions at 5 m and 80 m (both are cache heights in rf_cache_v2). 15 m and the weak_low_alt
wording are never used, so the held-out test stays honest.

    python -m train.sft_v13_extra gen       # tasks on the v2 training seeds only (1-12, 17-28) -> data/tasks/v13_extra*.jsonl
    python -m train.sft_v13_extra record    # expert episodes (MIN_SCORE 0.9), CPU
    python -m train.sft_v13_extra mix       # v12_base + extra -> reword/noise/v1.1 replay -> data/sft/train_v13.jsonl

Siting keys come from the nominal coverage model (D-025), so a question's answer does not depend on the scenario:
each (family, zone, band, height) combination is used once, on a training seed picked at random. Validation items
are combinations held out of the training extra (disjoint keys), so val never repeats a training answer.
The key path is the generator's own (Gen.call("weak_zones") -> episode ras geom WKT -> {"zones_wkt": [...]});
tasks/kandla/gen.py is not changed.
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
HEIGHTS = (5.0, 80.0)                     # never 15 (held-out weak_low_alt) and not 40 (already in v1.2)
TASKS = REPO / "data/tasks/v13_extra.jsonl"
TASKS_VAL = REPO / "data/tasks/v13_extra_val.jsonl"
REC = REPO / "runs/dbg/s25b/v13_expert.jsonl"
REC_VAL = REPO / "runs/dbg/s25b/v13_expert_val.jsonl"
N_VAL_COMBOS = 8
REPLAY_FRAC = 0.30
MAX_LEN = 15360


def _template(fam: str, bp: str, zn: str, h: float, alt: int) -> tuple:
    """gen.py wording (alt 0) and one close variant (alt 1); the height is the question's own."""
    hs = f"{h:.0f}"
    if fam == "weak_df":
        return "df_err_p90_deg", 10.0, [
            f"Where is {bp} DF accuracy weak (2-node bearing error above 10 deg) inside the {zn} for drones at {hs} m?",
            f"Where inside the {zn} is {bp} DF accuracy weak (2-node bearing error above 10 deg) for drones at {hs} m?",
        ][alt]
    return "detect_prob", 0.5, [
        f"Where inside the {zn} would a drone at {hs} m on {bp} probably go undetected?",
        f"Where in the {zn} would a drone flying at {hs} m on {bp} probably go undetected?"][alt]


def cmd_gen() -> None:
    from scripts.build_scenarios import v2_dir
    from tasks.kandla.gen import BAND_PHRASE
    from tasks.kandla.gen_more import GenMore as Gen

    rng = random.Random(1303)
    g0 = Gen(v2_dir(TRAIN_SEEDS[0], None), TRAIN_SEEDS[0], "train", tag="Xh")
    regions = ["hero_zone"] + [z for z, v in g0.site.zones.items() if v.kind == "restricted"]
    # 80 m is rarely weak (weak_detect never), so each 80 m combination is asked in both wordings
    combos = [(fam, z, band, h, alt) for fam in ("weak_df", "weak_detect") for z in regions for band in BAND_PHRASE
              for h in HEIGHTS for alt in ((0, 1) if h == 80.0 else (rng.randrange(2),))]
    g0.close()
    rng.shuffle(combos)
    by_seed = defaultdict(list)
    for c in combos:
        by_seed[rng.choice(TRAIN_SEEDS)].append(c)
    eval_q = set()
    for s in ("val", "test"):
        p = REPO / "data/tasks/x" / f"{s}.jsonl"
        eval_q |= {json.loads(x)["question"] for x in p.read_text().splitlines()}
    rows, empty = [], 0
    for seed in sorted(by_seed):
        g = Gen(v2_dir(seed, None), seed, "train", tag="Xh")
        for fam, z, band, h, alt in by_seed[seed]:
            bp, zn = rng.choice(BAND_PHRASE[band]), g.zone_name(z)
            metric, thr, q = _template(fam, bp, zn, h, alt)
            assert "15 m" not in q and "low drones" not in q and q not in eval_q
            reg = {"type": "ref", "ref": f"zone:{z}"}
            d = g.call("weak_zones", metric=metric, band=band, threshold=thr, region=reg, height_m=h, limit=5)
            assert d["height_m"] == h, d["height_m"]
            if not d["zones"]:
                empty += 1
                continue
            geoms = [g.ctx.episode["ras"][zz["ref"]]["geom"].wkt for zz in d["zones"]]
            t = g.task("siting", fam, q, {"zones_wkt": geoms},
                       {"zone": f"zone:{z}", "band": band, "metric": metric, "threshold": thr, "height_m": h})
            r = t.to_row()
            r["v13_combo"] = [fam, z, band, h]
            rows.append(r)
        g.close()
    rng.shuffle(rows)
    # validation: N_VAL_COMBOS (family, zone, band, height) combinations held out, about 2 per family x height,
    # every row of a held-out combination goes to val (keys disjoint from train)
    keys = sorted({tuple(r["v13_combo"]) for r in rows})
    rng.shuffle(keys)
    vk, need = set(), Counter()
    for k in keys:
        if need[(k[0], k[3])] < 2 and len(vk) < N_VAL_COMBOS:
            need[(k[0], k[3])] += 1
            vk.add(k)
    val = [r for r in rows if tuple(r["v13_combo"]) in vk]
    train = [r for r in rows if tuple(r["v13_combo"]) not in vk]
    TASKS.write_text("".join(json.dumps(r) + "\n" for r in train))
    TASKS_VAL.write_text("".join(json.dumps(r) + "\n" for r in val))
    cnt = Counter((r["template"], r["v13_combo"][2], r["v13_combo"][3]) for r in train)
    print(json.dumps({"regions": len(regions), "combos": len(combos), "empty": empty, "train": len(train),
                      "val": len(val), "by_fam_band_h": {"|".join(map(str, k)): v for k, v in sorted(cnt.items())}},
                     indent=1))


def cmd_record() -> None:
    from train.sft_v12_data import _record

    rows = [json.loads(x) for x in TASKS.read_text().splitlines()]
    print(json.dumps({"train_record": _record(rows, REC)}, indent=1), flush=True)
    vrows = [json.loads(x) for x in TASKS_VAL.read_text().splitlines()]
    print(json.dumps({"val_record": _record(vrows, REC_VAL)}, indent=1), flush=True)


def cmd_mix(mix: float, noise: float) -> None:
    from scripts import cuda_env  # noqa: F401  (HF_HOME)
    from scripts.augment_text import noisy
    from transformers import AutoTokenizer

    from harness.prompt import static_prefix
    from train.sft_data import tokenize_with_mask

    rng = random.Random(3)
    para = defaultdict(list)
    for line in (REPO / "data/tasks/v12_para.jsonl").read_text().splitlines():
        r = json.loads(line)
        para[r["paraphrase_of"]].append(r["question"])
    base = [(x, "v2") for x in (REPO / "data/sft/v12_base.jsonl").read_text().splitlines()]
    base += [(x, "v13_siting_h") for x in REC.read_text().splitlines()]
    new = []
    for line, src in base:                                  # the v1.2 mix step (train.sft_v12_data.cmd_mix)
        rec = json.loads(line)
        rec.pop("site", None)
        alts = para.get(rec["task_id"], [])                 # none for the new siting questions
        if alts and rng.random() < mix:
            rec["messages"][1] = {"role": "user", "content": rng.choice(alts)}
            rec["paraphrased"] = True
        if noise and rng.random() < noise:
            rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
            rec["noised"] = True
        rec["source"] = src
        new.append(rec)
    v11 = [json.loads(x) for x in (REPO / "data/sft/train_v11.jsonl").read_text().splitlines()]
    n_rep = round(len(new) * REPLAY_FRAC / (1 - REPLAY_FRAC))
    rep = rng.sample(v11, min(n_rep, len(v11)))
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
    (REPO / "data/sft/train_v13.jsonl").write_text("\n".join(json.dumps(r) for r in keep) + "\n")
    val = (REPO / "data/sft/val_v12.jsonl").read_text().splitlines()
    val += [x for x in REC_VAL.read_text().splitlines()]
    val = [json.dumps({k: v for k, v in json.loads(x).items() if k != "site"}) for x in val]
    (REPO / "data/sft/val_v13.jsonl").write_text("\n".join(val) + "\n")
    vl = [len(tokenize_with_mask(tok, json.loads(x)["messages"], tools, max_len=10 ** 9)[0]) for x in val]
    lens.sort()
    by = defaultdict(Counter)
    for r in keep:
        by[r["source"]][r["area"]] += 1
    sit = Counter()
    for r in keep:
        if r["area"] == "siting":
            sit[r["template"]] += 1
    print(json.dumps({"train_v13": len(keep), **{s: sum(c.values()) for s, c in by.items()},
                      "per_area": {s: dict(c) for s, c in by.items()}, "siting_templates": dict(sit),
                      "too_long": too_long, "tools": len(tools), "len_max": lens[-1], "len_p50": lens[len(lens) // 2],
                      "len_mean": round(sum(lens) / len(lens)), "val_n": len(val), "val_len_max": max(vl)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "record", "mix"])
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.35)
    a = ap.parse_args()
    {"gen": cmd_gen, "record": cmd_record, "mix": lambda: cmd_mix(a.mix, a.noise)}[a.stage]()


if __name__ == "__main__":
    main()
