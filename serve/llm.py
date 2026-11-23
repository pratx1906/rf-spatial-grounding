"""Launch the vLLM OpenAI-compatible server for the agent model (SPEC 7.7).

    python -m serve.llm [--model Qwen/Qwen3-4B-Instruct-2507] [--port 8001]

Holds the GPU lock (state serve_eval) for the server's lifetime, sizes VRAM from free memory (D-016), enables
prefix caching and the Hermes-style tool parser used by Qwen chat templates.

Latency flags (demo-v2 S5.6, docs/design/audit/2026-10-21_latency.md): `--gpu-util` fixes the vLLM memory fraction
when the GPU is shared (skips the free-memory sizing); `--max-num-seqs` (a single operator needs few);
`--spec-ngram K` enables n-gram (prompt-lookup) speculative decoding with K draft tokens. Chunked prefill and CUDA
graphs are vLLM 0.31 defaults (V1 engine); the warm-up request is sent by the gateway (serve.gateway.warm_llm).

Other models (paper P2 size ladder): `--model <hf id or path>` picks a preset from the model name (`preset_for`):
Qwen3 2507 instruct models -> hermes tool parser (unchanged default); Qwen3 hybrid-thinking releases (e.g. Qwen3-8B)
-> hermes + chat template `enable_thinking=false`; gpt-oss -> `openai` tool parser + `openai_gptoss` reasoning parser
(harmony). `--tool-call-parser`, `--reasoning-parser`, `--chat-template-kwargs`, `--quantization`, `--dtype` override
the preset; `--gpu-memory-utilization` is an alias of `--gpu-util`. With no new flags and the default model the vLLM
command is exactly the old one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import cuda_env  # noqa: E402,F401
from scripts.bench_m0 import auto_gpu_util  # noqa: E402

# served models are in the project HF cache; vLLM still asks the Hub at start-up, and one failed lookup hung an RL
# run's server (2026-10-29 12:28). SCENESCOUT_HF_ONLINE=1 for a model not yet downloaded.
if os.environ.get("SCENESCOUT_HF_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Qwen3 hybrid-thinking releases (April 2025): thinking is on by default in their chat template
_QWEN3_HYBRID = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b",
                 "qwen3-235b-a22b")


def preset_for(model: str) -> dict:
    """Serving preset from the model name: tool parser, reasoning parser, default chat-template kwargs. Works on a hub
    id and on a pinned HF-cache snapshot path (.../models--Qwen--Qwen3-8B/snapshots/<sha>)."""
    parts = [x for x in model.replace("\\", "/").replace("--", "/").lower().split("/") if x]
    if any("gpt-oss" in x for x in parts):
        return {"tool_call_parser": "openai", "reasoning_parser": "openai_gptoss", "chat_template_kwargs": None}
    if any(x.split("-awq")[0].split("-fp8")[0] in _QWEN3_HYBRID for x in parts):
        return {"tool_call_parser": "hermes", "reasoning_parser": None,
                "chat_template_kwargs": {"enable_thinking": False}}
    return {"tool_call_parser": "hermes", "reasoning_parser": None, "chat_template_kwargs": None}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-model-len", type=int, default=15360)
    p.add_argument("--vram-ceiling-gb", type=float, default=20.0,
                   help="hard ceiling on TOTAL GPU memory in use (all processes) once vLLM is up; above ~22.5 GB on a "
                        "24 GB card Windows spills desktop graphics into shared system RAM and the laptop stutters")
    p.add_argument("--gpu-util", "--gpu-memory-utilization", dest="gpu_util", type=float, default=None,
                   help="explicit vLLM --gpu-memory-utilization (fraction of total VRAM); overrides the sizing from "
                        "--vram-ceiling-gb, e.g. 0.45 next to another GPU job")
    p.add_argument("--max-num-seqs", type=int, default=None, help="vLLM --max-num-seqs (default: vLLM's)")
    p.add_argument("--spec-ngram", type=int, default=0,
                   help="n-gram speculative decoding with this many draft tokens (0 = off)")
    p.add_argument("--lora", action="append", default=[], help="name=path of a PEFT adapter (repeatable); "
                                                                "requests with model=<name> use it")
    p.add_argument("--max-loras", type=int, default=0, help="vLLM --max-loras (adapters batched together; 0 = vLLM default)")
    p.add_argument("--max-lora-rank", type=int, default=64, help="vLLM --max-lora-rank (128 for rank-96 soup adapters)")
    p.add_argument("--quantization", default=None, help="vLLM --quantization (default: read from the model config)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tool-call-parser", default=None, help="override the preset tool parser (hermes / openai / ...)")
    p.add_argument("--reasoning-parser", default=None, help="override the preset reasoning parser")
    p.add_argument("--chat-template-kwargs", default=None,
                   help="JSON for vLLM --default-chat-template-kwargs; overrides the preset, e.g. "
                        "'{\"enable_thinking\": false}'")
    return p.parse_args(argv)


def build_cmd(a: argparse.Namespace, util: float, vllm: str = "vllm") -> list[str]:
    pre = preset_for(a.model)
    parser = getattr(a, "tool_call_parser", None) or pre["tool_call_parser"]
    cmd = [vllm, "serve", a.model, "--port", str(a.port), "--host", "127.0.0.1",
           "--served-model-name", "scenescout", "--dtype", getattr(a, "dtype", "bfloat16"),
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(util), "--enable-prefix-caching",
           "--enable-auto-tool-choice", "--tool-call-parser", parser, "--generation-config", "vllm"]
    if getattr(a, "quantization", None):
        cmd += ["--quantization", a.quantization]
    reasoning = getattr(a, "reasoning_parser", None) or pre["reasoning_parser"]
    if reasoning:
        cmd += ["--reasoning-parser", reasoning]
    ctk = getattr(a, "chat_template_kwargs", None)
    ctk = json.loads(ctk) if ctk else pre["chat_template_kwargs"]
    if ctk:
        cmd += ["--default-chat-template-kwargs", json.dumps(ctk)]
    if a.max_num_seqs:
        cmd += ["--max-num-seqs", str(a.max_num_seqs)]
    if a.spec_ngram > 0:
        cmd += ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": a.spec_ngram,
                                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2})]
    if a.lora:
        cmd += ["--enable-lora", "--max-lora-rank", str(getattr(a, "max_lora_rank", 64)), "--lora-modules", *a.lora]
        if getattr(a, "max_loras", 0):
            cmd += ["--max-loras", str(a.max_loras)]
    return cmd


@exit_if_busy
def main() -> int:
    a = parse_args()
    if a.gpu_util is not None:
        util, vram = a.gpu_util, {"vllm_gpu_util": a.gpu_util, "source": "--gpu-util"}
    else:
        import pynvml as nv

        nv.nvmlInit()
        total_gb = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0)).total / 2**30
        nv.nvmlShutdown()
        # vLLM takes (free - headroom); with headroom = total - ceiling, total usage ends at exactly the ceiling
        util, vram = auto_gpu_util(max(0.5, total_gb - a.vram_ceiling_gb))
    print(f"[serve.llm] VRAM {vram}", flush=True)
    cmd = build_cmd(a, util, str(Path(sys.executable).with_name("vllm")))
    with gpu_lock("serve_eval"):
        return subprocess.call(cmd, env=dict(os.environ))


if __name__ == "__main__":
    sys.exit(main())
