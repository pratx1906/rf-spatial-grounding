"""RL framework dry run: TRL GRPO, multi-turn tool use, LoRA, colocated vLLM (SPEC 9.4, milestone M0).

    python scripts/rl_dryrun_trl.py [--steps 2] [--prompts-per-step 2] [--model Qwen/Qwen3-4B-Instruct-2507]

Two-turn toy tool env: the model must call `lookup(key)` to learn a code it cannot guess, then answer with it.
Records the SPEC 9.4 selection criteria into runs/m0/rl_dryrun_trl_<ts>.json and runs/m0/measurements.json:
works on sm_120, peak VRAM, seconds per step, episodes/hour, multi-turn masking correctness (checked on the actual
training tokens), and chat-template identity between the rollout engine and the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import cuda_env  # noqa: E402,F401  (HF_HOME, CUDA_HOME, FlashInfer workaround; before torch)
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402
from scripts.gpu_monitor import GpuMonitor  # noqa: E402
from scripts.provenance import provenance  # noqa: E402

OUT = REPO / "runs" / "m0"
SYSTEM = ("You are a lookup agent. Codes are secret: you cannot know them without the lookup tool. "
          "Call lookup exactly once with the key from the question, then reply with only the code.")


def make_table(n: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return {f"K{i:03d}": "".join(rng.choice(alphabet) for _ in range(6)) for i in range(n)}


class LookupEnv:
    """Toy two-turn tool environment. One instance per rollout (TRL creates them via environment_factory)."""

    def __init__(self, table: dict[str, str]):
        self.table = table
        self.key: str | None = None
        self.calls: list[str] = []

    def reset(self, key: str | None = None, **_) -> None:
        self.key = key
        self.calls = []
        return None

    def lookup(self, key: str) -> str:
        """Look up the secret code for a key.

        Args:
            key: The key from the question, for example "K007".

        Returns:
            The code for the key, or an error message if the key is unknown.
        """
        norm = key.strip().upper()
        self.calls.append(norm)
        code = self.table.get(norm)
        return f"code={code}" if code else f"error: unknown key {key!r}"

    def get_reward(self) -> float:
        """Process reward: exactly one lookup of the right key."""
        if not self.calls:
            return 0.0
        right = self.calls.count(self.key)
        return 0.2 if right and len(self.calls) == 1 else (0.1 if right else 0.0)


def final_text(completion) -> str:
    """Last assistant message content (conversational completions) or the raw string."""
    if isinstance(completion, str):
        return completion
    for msg in reversed(completion):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def answer_reward(completions, answer, **_) -> list[float]:
    """1.0 if the final reply contains the right code (outcome reward)."""
    return [1.0 if a in final_text(c) else 0.0 for c, a in zip(completions, answer, strict=True)]


_noise_rng = random.Random(1234)


def mechanics_noise_reward(completions, **_) -> list[float]:
    """MECHANICS CHECK ONLY: seeded noise so group advantages are non-zero and a real gradient reaches LoRA.

    The 4B model solves the toy task every time, so with only task rewards every group has zero variance, zero
    advantage, and zero gradient; the optimizer path would go untested. Never used outside the dry run.
    """
    return [_noise_rng.uniform(0.0, 0.5) for _ in completions]


def check_masking(batch, tokenizer, table_codes: set[str]) -> dict:
    """Verify on real training tokens: tool results are excluded from the loss, model tokens are included."""
    ids, cmask = batch["completion_ids"], batch["completion_mask"]
    tmask = batch.get("tool_mask")
    if tmask is None:
        return {"ok": False, "reason": "no tool_mask in training batch (tool tokens would be trained on)"}
    n_tool, results_masked, calls_trained, leaks = 0, 0, 0, []
    for row in range(ids.shape[0]):
        loss_ids = ids[row][(cmask[row] * tmask[row]).bool()].tolist()
        excl_ids = ids[row][(cmask[row].bool()) & (~tmask[row].bool())].tolist()
        trained, excluded = tokenizer.decode(loss_ids), tokenizer.decode(excl_ids)
        if not excl_ids:
            continue
        n_tool += 1
        results_masked += int("code=" in excluded or "error:" in excluded)
        calls_trained += int("lookup" in trained)  # the model's own tool-call tokens must be trained
        if "code=" in trained:  # tool output text must never be in the loss
            leaks.append(trained[:200])
    ok = n_tool > 0 and results_masked == n_tool and calls_trained == n_tool and not leaks
    return {"ok": ok, "rollouts_with_tool_result": n_tool, "tool_result_masked": results_masked,
            "model_tool_call_trained": calls_trained, "tool_output_leaked_into_loss": leaks[:2]}


def auto_vllm_util(headroom_gb: float) -> tuple[float, float]:
    import torch

    free, total = torch.cuda.mem_get_info()
    return round((free / 2**30 - headroom_gb) / (total / 2**30), 3), free / 2**30


@exit_if_busy
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--group", type=int, default=4, help="GRPO group size G (SPEC 9.5: 4 on the laptop)")
    p.add_argument("--prompts-per-step", type=int, default=8, help="SPEC 9.5: 8 to 16")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--max-completion", type=int, default=512)
    p.add_argument("--vllm-headroom-gb", type=float, default=None,
                   help="VRAM kept free when sizing the colocated vLLM engine; default 3 with sleep mode, 6 without "
                        "(without sleep, the engine stays resident while the trainer needs activation memory)")
    p.add_argument("--sleep", choices=["auto", "on", "off"], default="auto",
                   help="vLLM sleep mode (SPEC 9.5 option A). auto = off under WSL2, where cuMem* VMM calls fail")
    p.add_argument("--no-noise", action="store_true", help="disable the mechanics-only noise reward")
    a = p.parse_args()

    import platform as _pf

    in_wsl = "microsoft" in _pf.uname().release.lower()
    sleep = {"on": True, "off": False, "auto": not in_wsl}[a.sleep]
    headroom = a.vllm_headroom_gb if a.vllm_headroom_gb is not None else (3.0 if sleep else 6.0)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / f"rl_trl_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table = make_table(200, seed=0)
    rng = random.Random(1)
    keys = rng.sample(sorted(table), 64)
    ds = Dataset.from_list([{
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": f"What is the code for key {k}?"}],
        "key": k, "answer": table[k]} for k in keys])

    result: dict = {"framework": "trl", "model": a.model, "provenance": provenance(), "ok": False,
                    "vllm_sleep_mode": sleep, "vllm_headroom_gb": headroom, "group": a.group,
                    "prompts_per_step": a.prompts_per_step, "lora_r": a.lora_r,
                    "mechanics_noise_reward": not a.no_noise}
    captured: dict = {}

    class CaptureFirstBatch(GRPOTrainer):
        def compute_loss(self, model, inputs, *args, **kwargs):
            if not captured:
                captured.update({k: v.detach().cpu() for k, v in inputs.items()
                                 if k in ("completion_ids", "completion_mask", "tool_mask")})
            return super().compute_loss(model, inputs, *args, **kwargs)

    step_times: list[float] = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, *_, **__):
            self.t0 = time.time()

        def on_step_end(self, *_, **__):
            step_times.append(time.time() - self.t0)

    logs: list[dict] = []

    class LogCapture(TrainerCallback):
        def on_log(self, _args, _state, _control, logs=None, **__):  # noqa: A002 - HF callback signature
            if logs:
                captured_logs.append({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()})

    captured_logs = logs

    with gpu_lock("rl"):
        mon = GpuMonitor(run_dir, interval_s=2)
        with mon:
            try:
                tok = AutoTokenizer.from_pretrained(a.model)
                tok.padding_side = "left"
                model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
                model.to("cuda")
                util, free_gb = auto_vllm_util(headroom)
                result.update(vram_free_gb_after_trainer_load=round(free_gb, 1), vllm_gpu_util=util)
                print(f"[dryrun] free VRAM after trainer load {free_gb:.1f} GB -> vLLM util {util}")
                if util * 23.9 < 9.0:  # bf16 4B weights ~8 GB + a minimal KV cache
                    raise SystemExit(f"not enough VRAM for colocated vLLM (util {util}); close GPU apps")

                n_completions = a.prompts_per_step * a.group
                cfg = GRPOConfig(
                    output_dir=str(run_dir / "out"),
                    max_steps=a.steps,
                    per_device_train_batch_size=a.group,
                    gradient_accumulation_steps=a.prompts_per_step,
                    num_generations=a.group,
                    learning_rate=2e-5,
                    beta=0.005,                      # KL to the reference = adapter disabled (no 2nd weight copy)
                    epsilon=0.2,
                    max_completion_length=a.max_completion,
                    max_tool_calling_iterations=2,
                    temperature=0.9,
                    bf16=True,
                    gradient_checkpointing=True,
                    use_vllm=True,
                    vllm_mode="colocate",
                    vllm_gpu_memory_utilization=util,
                    vllm_enable_sleep_mode=sleep,    # SPEC 9.5 option A when available, else option B
                    vllm_max_model_length=2048,
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                    seed=0,
                )
                trainer = CaptureFirstBatch(
                    model=model,
                    processing_class=tok,
                    reward_funcs=[answer_reward] + ([] if a.no_noise else [mechanics_noise_reward]),
                    args=cfg,
                    train_dataset=ds,
                    peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                           target_modules="all-linear", task_type="CAUSAL_LM"),
                    environment_factory=lambda: LookupEnv(table),
                    callbacks=[StepTimer(), LogCapture()],
                )
                # SPEC 7.1: rollout engine and trainer must render prompts with the identical chat template.
                llm = getattr(getattr(trainer, "vllm_generation", None), "llm", None)
                eng_tok = llm.get_tokenizer() if llm is not None else None
                t_train = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16]
                t_roll = hashlib.sha256(((eng_tok.chat_template if eng_tok else "") or "").encode()).hexdigest()[:16]
                result["chat_template_identical"] = eng_tok is not None and t_train == t_roll
                result["chat_template_hash"] = {"trainer": t_train, "rollout_engine": t_roll}

                lora_before = {n: p.detach().float().cpu().clone() for n, p in trainer.model.named_parameters()
                               if p.requires_grad and "lora_B" in n}
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                trainer.train()
                changed = sum(int(not torch.equal(lora_before[n], p.detach().float().cpu()))
                              for n, p in trainer.model.named_parameters() if n in lora_before)
                result["lora_tensors_changed"] = f"{changed}/{len(lora_before)}"
                result["grad_norms"] = [lg.get("grad_norm") for lg in logs if "grad_norm" in lg]
                result["train_wall_s"] = round(time.time() - t0, 1)
                result["peak_torch_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                result["masking"] = check_masking(captured, tok, set(table.values()))
                result["logs"] = logs
                result["step_s"] = [round(s, 1) for s in step_times]
                per_step = sorted(step_times)[len(step_times) // 2] if step_times else None
                result["completions_per_step"] = n_completions
                result["episodes_per_hour"] = round(n_completions / per_step * 3600) if per_step else None
                rewards = [lg.get("reward") for lg in logs if "reward" in lg]
                result["mean_reward"] = rewards
                learned = changed > 0 or a.no_noise  # without noise the toy task can legitimately give 0 gradient
                result["ok"] = bool(result["masking"]["ok"] and result["chat_template_identical"]
                                    and step_times and learned)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001 - a failed dry run is a result, recorded in full
                import traceback

                result["error"] = f"{type(e).__name__}: {e}"
                (run_dir / "traceback.txt").write_text(traceback.format_exc())
                print(traceback.format_exc(), file=sys.stderr)
        s = GpuMonitor.summary(run_dir / "gpu_monitor.csv")
        result.update(peak_vram_used_mb_whole_gpu=s["peak_vram_used_mb"],
                      min_vram_free_mb=round(23.9 * 1024 - (s["peak_vram_used_mb"] or 0)),
                      median_temp_c=s["median_temp_c"], median_sm_clock_mhz=s["median_sm_clock_mhz"],
                      monitor_error=mon.error)

        (OUT / f"rl_dryrun_trl_{ts}.json").write_text(json.dumps(result, indent=2, default=str))
        meas = OUT / "measurements.json"
        merged = json.loads(meas.read_text()) if meas.exists() else {}
        merged.update({
            "rl_trl_ok": result["ok"],
            "rl_trl_step_s_median": (sorted(result.get("step_s", [])) or [None])[len(result.get("step_s", [])) // 2],
            "rl_trl_episodes_per_hour": result.get("episodes_per_hour"),
            "rl_trl_peak_vram_used_mb": result.get("peak_vram_used_mb_whole_gpu"),
            "rl_trl_masking_ok": result.get("masking", {}).get("ok"),
            "rl_trl_chat_template_identical": result.get("chat_template_identical"),
            "rl_trl_lora_tensors_changed": result.get("lora_tensors_changed"),
            "rl_trl_vllm_sleep_mode": sleep,
            "rl_trl_rollouts_per_step": a.prompts_per_step * a.group,
            "rl_trl_error": result.get("error"),
        })
        meas.write_text(json.dumps(merged, indent=2))

    print(json.dumps({k: v for k, v in result.items() if k not in ("logs", "provenance")}, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
