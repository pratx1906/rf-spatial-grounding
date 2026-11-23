"""T-FIX arm T4 SFT data (docs/paper/TWIN_FIX_PREREG.md): the check -> exclude -> answer expert under the T1 prompt.
    python -m train.tfix_data sample                    # task rows -> runs/paper/results/TFIX/t4/tasks.jsonl
    python -m train.tfix_data record --shard 0 --of 6   # CPU, one shard of the tasks (run the shards in parallel)
    python -m train.tfix_data mix                       # shards -> data/sft/train_tfix.jsonl (+ length check)
Sources (training seeds only; never seeds 51-60, never M1/M3/M4 worlds):
- M2 heading-bias worlds of training seeds 3, 6, 9, 12 (data/tasks/x3_Mtrain/train_M2d.jsonl): the check finds the
  suspect node and the expert re-fits without it.
- Nominal replay re-recorded under the same prompt and tool list (runs/s79/v14 train + base task rows): the check
  finds nothing and the normal investigation follows, so the policy keeps every other skill.
Only episodes the verifier accepted with score >= train.sft_data.MIN_SCORE are kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ABL = "with_consistency+fix_line"
W = REPO / "runs/paper/results/TFIX/t4"
N_M2, N_NOM = 420, 260
MAX_LEN = 15360


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def cmd_sample(a) -> None:
    rng = random.Random(20261101)
    m2 = [r for r in _rows(REPO / "data/tasks/x3_Mtrain/train_M2d.jsonl") if r["scenario_seed"] in (3, 6, 9, 12)]
    nom = _rows(REPO / "runs/s79/v14/tasks_train.jsonl") + _rows(REPO / "runs/s79/v14/tasks_base.jsonl")
    nom = [r for r in nom if r["scenario_seed"] not in range(51, 61)]
    by = {}
    for r in m2:
        by.setdefault(r["template"], []).append(r)
    pick = []
    while len(pick) < N_M2 and any(by.values()):          # family round-robin
        for f in sorted(by):
            if by[f] and len(pick) < N_M2:
                pick.append(by[f].pop(rng.randrange(len(by[f]))))
    for r in pick:
        r["tfix_source"] = "m2_train"
    nn = rng.sample(nom, min(N_NOM, len(nom)))
    for r in nn:
        r["tfix_source"] = "nominal_replay"
    W.mkdir(parents=True, exist_ok=True)
    out = pick + nn
    (W / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"m2": len(pick), "nominal": len(nn), "seeds": sorted({r["scenario_seed"] for r in out})}))


def cmd_record(a) -> None:
    from eval.ablations import apply_ablation
    from harness.config import EnvConfig
    from harness.env import AkshaEnv
    from tasks import expert_check
    from tasks.base import Task
    from train import sft_data

    cfg = apply_ablation(EnvConfig.load(), ABL)
    rows = sorted(_rows(W / "tasks.jsonl"), key=lambda r: r["site"])
    rows = rows[a.shard::a.of]
    import tasks.expert_kandla as ek
    orig = ek.solve
    out = W / f"rec_{a.shard}.jsonl"
    env, site, kept, seen = None, None, Counter(), Counter()
    with open(out, "w") as f:
        for r in rows:
            t = Task.from_row(r)
            if t.site != site:
                env, site = AkshaEnv(REPO / t.site, config=cfg), t.site
            ek.solve = expert_check.solve if t.area in ("operator_query", "df_ambiguity") else orig
            try:
                # sft_data.record imports solve from tasks.expert_kandla at call time -> the patched one
                rec = sft_data.record(env, t)
            except Exception as e:  # noqa: BLE001
                print("skip", t.task_id, type(e).__name__, e, flush=True)
                continue
            finally:
                ek.solve = orig
            seen[r["tfix_source"]] += 1
            if rec["score"] >= sft_data.MIN_SCORE:
                kept[r["tfix_source"]] += 1
                rec["source"] = r["tfix_source"]
                f.write(json.dumps(rec) + "\n")
    print(json.dumps({"shard": a.shard, "seen": seen, "kept": kept}), flush=True)


def cmd_mix(a) -> None:
    from scripts import cuda_env  # noqa: F401
    from transformers import AutoTokenizer

    from eval.ablations import apply_ablation
    from harness.config import EnvConfig
    from harness.prompt import static_prefix
    from train.sft_data import tokenize_with_mask

    tools = static_prefix(apply_ablation(EnvConfig.load(), ABL))[1]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    rng = random.Random(1101)
    recs = [r for p in sorted(W.glob("rec_*.jsonl")) for r in _rows(p)]
    keep, too_long = [], 0
    for r in recs:
        if len(tokenize_with_mask(tok, r["messages"], tools, max_len=10 ** 9)[0]) > MAX_LEN:
            too_long += 1
            continue
        keep.append(r)
    rng.shuffle(keep)
    calls = Counter(tc["function"]["name"] for r in keep for m in r["messages"] for tc in (m.get("tool_calls") or []))
    out = REPO / "data/sft/train_tfix.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(json.dumps({"train": len(keep), "by_source": Counter(r["source"] for r in keep), "too_long": too_long,
                      "sensor_consistency_calls": calls["sensor_consistency"], "tools_used": len(calls),
                      "out": str(out)}, indent=1, default=dict))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    r = sub.add_parser("record")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--of", type=int, default=1)
    sub.add_parser("mix")
    a = ap.parse_args()
    {"sample": cmd_sample, "record": cmd_record, "mix": cmd_mix}[a.cmd](a)


if __name__ == "__main__":
    main()
