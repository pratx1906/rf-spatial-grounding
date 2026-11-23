"""SFT of the SceneScout policy on expert trajectories (SPEC 9.3, M4). LoRA on Qwen3-4B-Instruct-2507.

    python -m train.sft --data data/sft/train.jsonl --val data/sft/val.jsonl --out checkpoints/sft_v1 \
        [--epochs 1] [--max-steps 0] [--lr 1e-4] [--accum 16]

Loss only on assistant tokens; the LM head is applied ONLY at those positions (10k-token episodes would otherwise
need ~6 GB of full-vocabulary logits). Rendering = the serving chat template + tool schemas (train.sft_data).
Holds the GPU lock (state sft). Logs runs/sft/<name>/log.jsonl; saves a PEFT adapter vLLM can serve with LoRA.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import cuda_env  # noqa: E402,F401
from scripts.gpu_lock import LOCK_PATH as LOCK, exit_if_busy, gpu_lock  # noqa: E402

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def load(tok, path: Path, tools, max_len: int, limit: int = 0):
    from train.sft_data import tokenize_with_mask

    out = []
    for line in path.read_text().splitlines()[: limit or None]:
        rec = json.loads(line)
        ids, lab = tokenize_with_mask(tok, rec["messages"], tools, max_len=max_len + 1)
        if len(ids) > max_len or not any(x != -100 for x in lab):
            continue
        out.append((ids, lab, rec["area"]))
    return out


def masked_loss(model, ids, labels):
    """Causal LM loss on labelled positions only, without materialising full-sequence logits."""
    import torch
    import torch.nn.functional as F

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids).last_hidden_state                  # [1, T, H]
    tgt = labels[:, 1:]
    sel = tgt != -100
    h = hidden[:, :-1][sel]                                               # [N, H]
    logits = base.lm_head(h).float()
    return F.cross_entropy(logits, tgt[sel]), int(sel.sum())


@exit_if_busy
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out", default="checkpoints/sft_v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=0, help="also save the adapter every N steps (checkpoint-<step>)")
    ap.add_argument("--init", default=None, help="continue training an existing LoRA adapter (e.g. checkpoints/sft_v1)")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after N val evaluations without a val-loss improvement (0 = off); the saved "
                         "checkpoint-<step> adapters are then ranked by task score (scripts/sft_v14_report.py select)")
    ap.add_argument("--ablation", default="none",
                    help="tool schemas of this eval.ablations preset (T-FIX T4: with_consistency+fix_line); none = default")
    ap.add_argument("--seed", type=int, default=0, help="data order + torch seed (paper: second training seed)")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="save and stop after this step with the full schedule kept (reproduce a selected checkpoint)")
    a = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation, fewer OOMs
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from harness.prompt import static_prefix

    if a.ablation == "notools":         # plain chat SFT without tool schemas (fine-tuned parser baseline)
        tools = None
    elif a.ablation != "none":
        from eval.ablations import apply_ablation
        from harness.config import EnvConfig

        tools = static_prefix(apply_ablation(EnvConfig.load(), a.ablation))[1]
    else:
        tools = static_prefix()[1]
    tok = AutoTokenizer.from_pretrained(BASE)
    t0 = time.time()
    train = load(tok, REPO / a.data, tools, a.max_len, a.limit)
    val = load(tok, REPO / a.val, tools, a.max_len, 120) if (REPO / a.val).exists() else []
    print(f"[sft] train {len(train)} val {len(val)} examples, tokenised in {time.time() - t0:.0f} s; "
          f"mean len {sum(len(x[0]) for x in train) / max(1, len(train)):.0f}, "
          f"learned tokens {sum(sum(l != -100 for l in x[1]) for x in train)}", flush=True)
    name = Path(a.out).name
    logdir = REPO / "runs" / "sft" / name
    logdir.mkdir(parents=True, exist_ok=True)
    log = open(logdir / "log.jsonl", "a")
    # The model server may still be releasing VRAM after it was stopped: wait (up to 3 min) for the lock to clear
    # and for 15 GB free before loading, instead of failing with an OOM half-way into the first step.
    for _ in range(36):
        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        if not LOCK.exists() and free_gb >= 15:          # SFT v1 peaked at 14.1 GB
            break
        print(f"[sft] waiting for the GPU: lock {'held' if LOCK.exists() else 'free'}, {free_gb:.1f} GB free", flush=True)
        time.sleep(5)
    free_b, total_b = torch.cuda.mem_get_info()
    ceiling = float(os.environ.get("SCENESCOUT_VRAM_CEILING_GB", "20")) * 2**30   # hard cap on total GPU use
    frac = max(0.2, min(0.95, (ceiling - (total_b - free_b)) / total_b))
    torch.cuda.set_per_process_memory_fraction(frac)          # beyond this: a clean CUDA OOM, never a spill to RAM
    print(f"[sft] VRAM cap {frac * total_b / 2**30:.1f} GB for this process (ceiling 20 GB total)", flush=True)
    with gpu_lock("sft"):
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map={"": 0})
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if a.init:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(REPO / a.init), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                                     "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
        steps = a.max_steps or math.ceil(len(train) * a.epochs / a.accum)
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        rng = random.Random(a.seed)
        torch.manual_seed(a.seed)
        order = []
        while len(order) < steps * a.accum:
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx

        def evaluate():
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for ids, lab, _ in val:
                    x = torch.tensor([ids], device="cuda")
                    y = torch.tensor([lab], device="cuda")
                    loss, k = masked_loss(model, x, y)
                    tot += float(loss) * k
                    n += k
            model.train()
            return tot / max(n, 1)

        model.train()
        t_start = time.time()
        best_val, since_best = float("inf"), 0
        if val:
            v0 = evaluate()
            best_val = v0
            log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
            log.flush()
            print(f"[sft] step 0 val_loss {v0:.4f}", flush=True)
        for step in range(steps):
            tl, tk = 0.0, 0
            for j in range(a.accum):
                ids, lab, _ = train[order[step * a.accum + j]]
                x = torch.tensor([ids], device="cuda")
                y = torch.tensor([lab], device="cuda")
                loss, k = masked_loss(model, x, y)
                (loss / a.accum).backward()
                tl += float(loss.detach()) * k
                tk += k
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            rec = {"step": step + 1, "loss": tl / max(tk, 1), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t_start, 1),
                   "vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            if val and ((step + 1) % a.val_every == 0 or step + 1 == steps):
                rec["val_loss"] = evaluate()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < steps:
                model.save_pretrained(REPO / a.out / f"checkpoint-{step + 1}")      # early-stopping candidates
            if (step + 1) % 5 == 0 or step + 1 == steps or "val_loss" in rec:
                eta = (time.time() - t_start) / (step + 1) * (steps - step - 1)
                print(f"[sft] step {step + 1}/{steps} loss {rec['loss']:.4f}"
                      + (f" val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
                      + f" vram {rec['vram_gb']} GB eta {eta / 60:.0f} min", flush=True)
            if a.stop_at and step + 1 >= a.stop_at:
                print(f"[sft] stop at step {step + 1} (--stop-at; schedule of {steps} steps kept)", flush=True)
                steps = step + 1
                break
            if a.early_stop_patience and "val_loss" in rec:
                if rec["val_loss"] < best_val - 1e-4:
                    best_val, since_best = rec["val_loss"], 0
                else:
                    since_best += 1
                if since_best >= a.early_stop_patience and step + 1 < steps:
                    print(f"[sft] early stop at step {step + 1}: val loss has not improved for {since_best} evals "
                          f"(best {best_val:.4f})", flush=True)
                    log.write(json.dumps({"step": step + 1, "early_stop": True, "best_val_loss": best_val}) + "\n")
                    log.flush()
                    steps = step + 1
                    break
        out = REPO / a.out
        model.save_pretrained(out)
        tok.save_pretrained(out)
        (out / "train_meta.json").write_text(json.dumps({"base": BASE, "data": a.data, "examples": len(train),
                                                         "steps": steps, "accum": a.accum, "lr": a.lr,
                                                         "lora_r": a.lora_r, "init": a.init, "best_val_loss": best_val,
                                                         "seed": a.seed, "stop_at": a.stop_at,
                                                         "wallclock_s": round(time.time() - t_start)},
                                                        indent=1))
        print(f"[sft] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
