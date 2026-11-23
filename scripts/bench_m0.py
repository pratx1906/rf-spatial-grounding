"""M0 measurements (SPEC 2.5 "measure before planning"). Merges results into runs/m0/measurements.json.

    python scripts/bench_m0.py [--model Qwen/Qwen3-4B-Instruct-2507] [--warmup-s 60]

Measures, for the serving model in bf16 under vLLM: load time, single-stream decode tokens/s,
batched (16 concurrent) generation tokens/s, and prefix-cached prompt latency, plus median GPU clock and
temperature over the run. Run with the laptop plugged in and warm (SPEC 2.3).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import cuda_env  # noqa: E402,F401  (sets HF_HOME / CUDA_HOME before vLLM is imported)
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402
from scripts.gpu_monitor import GpuMonitor  # noqa: E402
from scripts.provenance import provenance  # noqa: E402

OUT = REPO / "runs" / "m0"

# Static-prefix stand-in (~1.8k tokens with the Qwen3 tokenizer; SPEC 7.5 budgets up to 2.5k).
PREFIX = ("You are the AKSHA-360 scene agent. Answer only with tool calls. " + "Tool: site_resolve(query). " * 300)


def auto_gpu_util(headroom_gb: float, cap: float = 0.92) -> tuple[float, dict]:
    """Largest vLLM gpu_memory_utilization that fits in the VRAM free *right now*, minus a safety margin.

    vLLM's fraction is of total VRAM and it refuses to start if that exceeds free memory. Windows apps hold VRAM
    that WSL processes cannot reclaim, so a fixed fraction OOMs on a busy desktop. Uses NVML, which does not
    create a CUDA context in this process (that would itself cost ~0.5 GB).
    """
    import pynvml as nv

    nv.nvmlInit()
    try:
        mem = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0))
    finally:
        nv.nvmlShutdown()
    total, free = mem.total / 2**30, mem.free / 2**30
    util = min(cap, (free - headroom_gb) / total)
    return round(util, 3), {"vram_total_gb": round(total, 1), "vram_free_gb_before_load": round(free, 1),
                            "vram_headroom_gb": headroom_gb, "vllm_gpu_util": round(util, 3)}


def save(m: dict) -> None:
    """Merge into measurements.json immediately, so a later failure never loses earlier numbers."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "measurements.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(m)
    path.write_text(json.dumps(merged, indent=2))


@exit_if_busy
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--warmup-s", type=float, default=60.0, help="sustained load before timing, so clocks settle")
    p.add_argument("--gpu-util", type=float, default=None, help="vLLM fraction of total VRAM; default: auto")
    # vLLM's CUDA graphs and the parent process use ~1 GB beyond its fraction (measured: 1.5 GB headroom left 0.6 GB
    # free at peak), so 2.5 GB here keeps >= 1.5 GB really free (SPEC 2.4 asks for 2+ GB of planned headroom).
    p.add_argument("--headroom-gb", type=float, default=2.5, help="VRAM left free in auto mode")
    a = p.parse_args()

    import psutil

    run_dir = OUT / f"bench_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    m: dict = {
        "free_ram_gb_idle": round(psutil.virtual_memory().available / 2**30, 1),
        "free_disk_gb_at_repo": round(shutil.disk_usage(REPO).free / 1e9, 1),
        "bench_run_dir": str(run_dir.relative_to(REPO)),
        "bench_measured_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance(), indent=2))

    with gpu_lock("bench"):
        if a.gpu_util is None:
            util, vram = auto_gpu_util(a.headroom_gb)
        else:
            util, vram = a.gpu_util, {"vllm_gpu_util": a.gpu_util}
        # weights (~8 GB for a 4B model in bf16) + CUDA graphs + KV cache for at least one 8192-token sequence
        need = 11.0 / vram.get("vram_total_gb", 24.0)
        if util < need:
            raise SystemExit(f"Not enough free VRAM: vLLM would get {util:.2f} of total, needs >= {need:.2f}. "
                             f"Free: {vram.get('vram_free_gb_before_load')} GB. Close GPU-heavy apps and retry.")
        m.update(vram)
        save(m)
        print(f"[bench] VRAM: {vram}")
        mon = GpuMonitor(run_dir, interval_s=2)
        with mon:
            from vllm import LLM, SamplingParams

            t0 = time.time()
            llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=8192,
                      enable_prefix_caching=True)
            m.update(bench_model=a.model, vllm_load_s=round(time.time() - t0, 1))
            save(m)

            sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.8, ignore_eos=True)
            t_loaded = time.time()  # clocks/temps are summarised from here on, not over download/load
            t_end = time.time() + a.warmup_s
            while time.time() < t_end:
                llm.generate(["Warm up the GPU."] * a.batch, sp, use_tqdm=False)

            t0 = time.time()
            out = llm.generate(["Describe the jetties at Kandla port."], sp, use_tqdm=False)
            m["decode_tok_s_single_stream"] = round(len(out[0].outputs[0].token_ids) / (time.time() - t0), 1)
            save(m)

            t0 = time.time()
            outs = llm.generate([f"Question {i}: list tank farms." for i in range(a.batch)], sp, use_tqdm=False)
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            m[f"gen_tok_s_batch{a.batch}"] = round(n / (time.time() - t0), 1)
            save(m)

            one = SamplingParams(max_tokens=1, temperature=0)
            llm.generate([PREFIX + "Q: warm cache"], one, use_tqdm=False)
            t0 = time.time()
            for i in range(10):
                llm.generate([PREFIX + f"Q: where is track {i}?"], one, use_tqdm=False)
            m["ttft_cached_prefix_ms"] = round((time.time() - t0) / 10 * 1000, 1)
            m["prefix_tokens"] = len(llm.get_tokenizer().encode(PREFIX))
            save(m)

        whole = GpuMonitor.summary(run_dir / "gpu_monitor.csv")
        s = GpuMonitor.summary(run_dir / "gpu_monitor.csv", since=t_loaded)
        m.update({
            "bench_under_load_samples": s["n_samples"],
            "bench_median_sm_clock_mhz": s["median_sm_clock_mhz"],
            "bench_median_temp_c": s["median_temp_c"],
            "bench_median_power_w": s["median_power_w"],
            "bench_peak_vram_used_mb (whole GPU, incl. Windows apps)": whole["peak_vram_used_mb"],
            "bench_min_vram_free_mb": round(vram.get("vram_total_gb", 0) * 1024 - (whole["peak_vram_used_mb"] or 0)),
            "bench_monitor_error": mon.error,
        })
        save(m)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
