"""SFT v1.2 data (demo-v2 S2.5): v2 expert trajectories (rebuilt v2 observations, nominal x + world variants
W6/W7/W8) plus ~30% replay of the SFT v1.1 set, with the v1.1 augmentation recipe on the new part.

    python -m train.sft_v12_data record      # expert on a per-family capped sample of the v2 train tasks
    python -m train.sft_v12_data select      # balanced pick (train.sft_subset shares) -> v12_base + tasks to reword
    python scripts/paraphrase.py --src data/tasks/v12_tasks.jsonl --out data/tasks/v12_para.jsonl --k 1   (vLLM)
    python -m train.sft_v12_data mix         # reword (mix 0.6) + operator noise (0.35) + v1.1 replay; length check

Only expert episodes the verifier accepted with score >= MIN_SCORE are kept (train.sft_data), so wrong expert
answers on the weak v2 families are never imitated. Held-out families are absent from the train split (make_tasks).
Every example must fit max_len tokens with the CURRENT tool prompt prefix (harness.prompt.static_prefix).
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

SOURCES = ["data/tasks/x/train.jsonl", "data/tasks/x_W6/train.jsonl", "data/tasks/x_W7/train.jsonl",
           "data/tasks/x_W8/train.jsonl"]
VAL_SOURCE = "data/tasks/x/val.jsonl"
REC = REPO / "runs/dbg/s25/v12_expert.jsonl"          # every kept expert record (scratch, not committed)
REC_VAL = REPO / "runs/dbg/s25/v12_expert_val.jsonl"
CAP_PER_FAMILY = 220                                 # expert attempts per family (pass rates 0.3-1.0)
N_NEW = 560                                          # v2 examples; replay = 30% of the total
REPLAY_FRAC = 0.30
N_VAL = 70
MAX_LEN = 15360


def _record(rows: list[dict], out: Path) -> dict:
    from harness.env import AkshaEnv
    from tasks.base import Task
    from train.sft_data import MIN_SCORE, record

    rows = sorted(rows, key=lambda r: r["site"])        # one site in memory at a time
    seen, kept = Counter(), Counter()
    env, site = None, None
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            t = Task.from_row(r)
            if t.site != site:
                env, site = AkshaEnv(REPO / t.site), t.site
            try:
                rec = record(env, t)
            except Exception as e:  # noqa: BLE001
                print("skip", t.task_id, type(e).__name__, e, flush=True)
                continue
            seen[t.template] += 1
            if rec["score"] >= MIN_SCORE:
                kept[t.template] += 1
                rec["site"] = t.site
                f.write(json.dumps(rec) + "\n")
            if (i + 1) % 500 == 0:
                print(f"[record] {i + 1}/{len(rows)} kept {sum(kept.values())}", flush=True)
    return {"seen": sum(seen.values()), "kept": sum(kept.values()),
            "pass_by_family": {f: round(kept[f] / seen[f], 2) for f in sorted(seen)}}


def cmd_record() -> None:
    rng = random.Random(12)
    fam = defaultdict(list)
    for s in SOURCES:
        for line in (REPO / s).read_text().splitlines():
            r = json.loads(line)
            fam[r["template"]].append(r)
    rows = []
    for f in sorted(fam):
        pool = fam[f][:]
        rng.shuffle(pool)
        rows += pool[:CAP_PER_FAMILY]
    print(json.dumps({"train_record": _record(rows, REC)}, indent=1), flush=True)
    vrows = [json.loads(x) for x in (REPO / VAL_SOURCE).read_text().splitlines()]
    rng.shuffle(vrows)
    print(json.dumps({"val_record": _record(vrows[:400], REC_VAL)}, indent=1), flush=True)


def cmd_select() -> None:
    from train.sft_subset import SHARE

    rng = random.Random(7)
    fam = defaultdict(lambda: defaultdict(list))
    for line in REC.read_text().splitlines():
        r = json.loads(line)
        fam[r["area"]][r["template"]].append(line)
    out = []
    for area, share in SHARE.items():
        quota = int(round(N_NEW * share))
        pools = {f: rng.sample(v, len(v)) for f, v in fam[area].items()}
        got = []
        while len(got) < quota and any(pools.values()):
            for f in sorted(pools):                      # family round-robin (train.sft_subset)
                if pools[f] and len(got) < quota:
                    got.append(pools[f].pop())
        out += got
        print(area, len(got), "of", sum(len(v) for v in fam[area].values()))
    rng.shuffle(out)
    (REPO / "data/sft/v12_base.jsonl").write_text("\n".join(out) + "\n")
    tasks = {}
    for s in SOURCES:
        for line in (REPO / s).read_text().splitlines():
            tasks[json.loads(line)["task_id"]] = line
    ids = [json.loads(x)["task_id"] for x in out]
    (REPO / "data/tasks/v12_tasks.jsonl").write_text("\n".join(tasks[i] for i in ids) + "\n")
    # validation: balanced by area from the v2 val split (expert-accepted only)
    vby = defaultdict(list)
    for line in REC_VAL.read_text().splitlines():
        vby[json.loads(line)["area"]].append(line)
    val = []
    for area in sorted(vby):
        val += vby[area][: max(4, round(N_VAL * SHARE.get(area, 0.1)))]
    (REPO / "data/sft/val_v12.jsonl").write_text("\n".join(val) + "\n")
    print(json.dumps({"v12_base": len(out), "val_v12": len(val)}))


def cmd_mix(mix: float, noise: float) -> None:
    from scripts import cuda_env  # noqa: F401  (HF_HOME)
    from scripts.augment_text import noisy
    from transformers import AutoTokenizer

    from harness.prompt import static_prefix
    from train.sft_data import tokenize_with_mask

    rng = random.Random(3)
    para = defaultdict(list)
    pf = REPO / "data/tasks/v12_para.jsonl"
    if pf.exists():
        for line in pf.read_text().splitlines():
            r = json.loads(line)
            para[r["paraphrase_of"]].append(r["question"])
    new, swapped, noised = [], 0, 0
    for line in (REPO / "data/sft/v12_base.jsonl").read_text().splitlines():   # v1.1 recipe (train.sft_augment)
        rec = json.loads(line)
        alts = para.get(rec["task_id"], [])
        if alts and rng.random() < mix:
            rec["messages"][1] = {"role": "user", "content": rng.choice(alts)}
            rec["paraphrased"] = True
            swapped += 1
        if noise and rng.random() < noise:
            rec["messages"][1] = {"role": "user", "content": noisy(rec["messages"][1]["content"], rng)}
            rec["noised"] = True
            noised += 1
        rec["source"] = "v2"
        new.append(rec)
    v11 = [json.loads(x) for x in (REPO / "data/sft/train_v11.jsonl").read_text().splitlines()]
    n_rep = round(len(new) * REPLAY_FRAC / (1 - REPLAY_FRAC))
    rep = rng.sample(v11, min(n_rep, len(v11)))       # already carries the v1.1 rewording / noise
    for r in rep:
        r["source"] = "v11_replay"
    allrec = new + rep
    rng.shuffle(allrec)
    tools = static_prefix()[1]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    keep, too_long, lens = [], [], []
    for r in allrec:
        ids, lab = tokenize_with_mask(tok, r["messages"], tools, max_len=10 ** 9)
        if len(ids) > MAX_LEN:
            too_long.append((r["task_id"], len(ids)))
            continue
        lens.append(len(ids))
        keep.append(r)
    (REPO / "data/sft/train_v12.jsonl").write_text("\n".join(json.dumps(r) for r in keep) + "\n")
    vl = [len(tokenize_with_mask(tok, json.loads(x)["messages"], tools, max_len=10 ** 9)[0])
          for x in (REPO / "data/sft/val_v12.jsonl").read_text().splitlines()]
    lens.sort()
    by = defaultdict(Counter)
    for r in keep:
        by[r["source"]][r["area"]] += 1
    print(json.dumps({"train_v12": len(keep), "v2": sum(by["v2"].values()), "replay": sum(by["v11_replay"].values()),
                      "per_area": {s: dict(c) for s, c in by.items()}, "paraphrased": swapped, "noised": noised,
                      "too_long": too_long, "tools": len(tools), "len_max": lens[-1], "len_p50": lens[len(lens) // 2],
                      "len_mean": round(sum(lens) / len(lens)), "val_len_max": max(vl), "val_n": len(vl)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["record", "select", "mix"])
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.35)
    a = ap.parse_args()
    {"record": cmd_record, "select": cmd_select, "mix": lambda: cmd_mix(a.mix, a.noise)}[a.stage]()


if __name__ == "__main__":
    main()
