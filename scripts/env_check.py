"""Environment smoke test (SPEC 2.2, milestone M0). Writes docs/ENV_REPORT.md and runs/m0/env_check.json.

    python scripts/env_check.py [--skip-vllm] [--vllm-model Qwen/Qwen3-0.6B]

Each check records ok / FAIL / skip plus detail. Exit code is 1 if any required check fails.
Measured numbers from scripts/bench_m0.py (runs/m0/measurements.json) are included when present.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import cuda_env  # noqa: E402,F401  (sets HF_HOME / CUDA_HOME before torch is imported)
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402
from scripts.provenance import provenance  # noqa: E402

OUT = REPO / "runs" / "m0"

RESULTS: list[dict] = []


def check(name: str, required: bool = True):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.time()
            try:
                detail = fn(*a, **kw)
                status = "ok"
            except SkipCheck as e:
                status, detail = "skip", str(e)
            except Exception as e:  # noqa: BLE001 - every failure is recorded, not raised
                status = "FAIL"
                detail = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc(limit=3)
                print(tb, file=sys.stderr)
            RESULTS.append({"check": name, "status": status, "required": required,
                            "detail": detail, "secs": round(time.time() - t0, 2)})
            print(f"[{status:>4}] {name}: {detail}")
        return run
    return deco


class SkipCheck(Exception):
    pass


@check("os")
def c_os():
    rel = platform.uname().release
    side = "WSL2" if "microsoft" in rel.lower() else platform.system()
    return f"{side} {platform.platform()} | python {platform.python_version()}"


@check("driver / nvidia-smi")
def c_driver():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,clocks.sm",
         "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


@check("torch + CUDA")
def c_torch():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}"


@check("compute capability sm_120")
def c_cap():
    import torch
    cap = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    assert cap == (12, 0), f"expected (12, 0), got {cap}"
    assert any(a in arches for a in ("sm_120", "compute_120")), f"torch build lacks sm_120 kernels: {arches}"
    return f"{cap}, torch arch list includes sm_120"


@check("bf16")
def c_bf16():
    import torch
    assert torch.cuda.is_bf16_supported()
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    assert y == y, "NaN in bf16 matmul"
    return "supported, matmul ok"


@check("SDPA attention (flash / mem-efficient)")
def c_sdpa():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
    ok = []
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q, is_causal=True)
            ok.append(backend.name)
        except RuntimeError:
            pass
    assert ok, "no fused SDPA backend works"
    return "working backends: " + ", ".join(ok)


@check("flash-attn package", required=False)
def c_flash_attn():
    try:
        import flash_attn
    except ImportError:
        raise SkipCheck("not installed; SDPA is the fallback (SPEC 14)") from None
    return f"flash_attn {flash_attn.__version__}"


@check("bitsandbytes 8-bit optimizer + 4-bit linear")
def c_bnb():
    import bitsandbytes as bnb
    import torch
    lin = torch.nn.Linear(512, 512, device="cuda")
    opt = bnb.optim.AdamW8bit(lin.parameters(), lr=1e-3)
    loss = lin(torch.randn(8, 512, device="cuda")).pow(2).mean()
    loss.backward()
    opt.step()
    q = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16, quant_type="nf4")
    q.load_state_dict(torch.nn.Linear(512, 512).state_dict())
    q = q.cuda()
    out = q(torch.randn(4, 512, device="cuda", dtype=torch.bfloat16))
    assert out.shape == (4, 512)
    return f"bitsandbytes {bnb.__version__}: AdamW8bit step ok, nf4 Linear4bit forward ok"


@check("peft + trl import")
def c_rl_libs():
    import peft
    import trl
    return f"peft {peft.__version__}, trl {trl.__version__} (RL framework dry runs are a separate M0 item)"


@check("nvcc (CUDA toolkit for vLLM/FlashInfer JIT)")
def c_nvcc():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found; run: wsl -u root -- bash scripts/wsl/install_cuda_toolkit.sh")
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    return f"{nvcc}: {out.strip().splitlines()[-1]}"


VLLM_SNIPPET = textwrap.dedent("""
    import json, time
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model={model!r}, dtype="bfloat16", gpu_memory_utilization=0.35, max_model_len=2048,
              enable_prefix_caching=True, enforce_eager=False)
    load_s = time.time() - t0
    out = llm.generate(["The port of Kandla is in"], SamplingParams(max_tokens=32, temperature=0))
    print("RESULT", json.dumps({{"load_s": round(load_s, 1), "text": out[0].outputs[0].text[:80]}}))
""")


@check("vLLM loads and generates (small model)")
def c_vllm(model: str):
    import vllm
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    p = subprocess.run([sys.executable, "-c", VLLM_SNIPPET.format(model=model)],
                       capture_output=True, text=True, timeout=1200, env=env)
    log = OUT / "vllm_check.log"
    OUT.mkdir(parents=True, exist_ok=True)
    log.write_text(f"exit {p.returncode}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}")
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        # The tail is usually a wrapper ("Engine core initialization failed"); report the first root-cause line.
        errs = [ln.strip() for ln in (p.stderr + p.stdout).splitlines()
                if re.search(r"\b\w*(Error|Exception)\b:", ln) and "Engine core initialization failed" not in ln]
        cause = errs[0] if errs else (p.stderr or p.stdout)[-300:]
        raise RuntimeError(f"exit {p.returncode}: {cause[:300]} (full log: {log.relative_to(REPO)})")
    r = json.loads(lines[-1][7:])
    return f"vllm {vllm.__version__}, {model} loaded in {r['load_s']} s, output: {r['text']!r}"


@check("Sionna RT", required=False)
def c_sionna():
    try:
        import sionna.rt  # noqa: F401
    except ImportError:
        raise SkipCheck("not in this venv by design (agent stack never imports Sionna); "
                        "checked separately by scripts/sionna_check.py") from None
    return "importable"


@check("host resources")
def c_host():
    import psutil
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    du = shutil.disk_usage(REPO)
    return (f"RAM total {vm.total / 2**30:.1f} GiB, available {vm.available / 2**30:.1f} GiB, "
            f"swap {sw.total / 2**30:.1f} GiB | disk at repo: free {du.free / 1e9:.0f} GB of {du.total / 1e9:.0f} GB "
            f"({du.free / du.total:.0%}) | cpu threads {os.cpu_count()}")


def write_report(path: Path, measurements: dict | None) -> None:
    prov = provenance()
    lines = [
        "# Environment report (M0)",
        "",
        f"Generated by `scripts/env_check.py` on {time.strftime('%Y-%m-%d %H:%M')} at git "
        f"`{(prov['git_hash'] or 'unknown')[:10]}`{' **with uncommitted changes**' if prov['git_dirty'] else ''}. "
        "Re-run to refresh; do not hand-edit the tables. "
        "Full results with library versions: `runs/m0/env_check.json`.",
        "",
        "## Automated checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for r in RESULTS:
        req = "" if r["required"] else " (optional)"
        detail = str(r["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['check']}{req} | {r['status']} | {detail} |")
    lines += ["", "## Measured numbers (SPEC 2.5)", ""]
    if measurements:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in measurements.items()]
    else:
        lines.append("Not measured yet. Run `scripts/bench_m0.py`.")
    manual = REPO / "docs" / "ENV_MANUAL.md"
    if manual.exists():
        lines += ["", manual.read_text().strip()]
    path.write_text("\n".join(lines) + "\n")


@exit_if_busy
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-vllm", action="store_true", help="debug only: the run is then INCOMPLETE (exit 1)")
    p.add_argument("--vllm-model", default="Qwen/Qwen3-0.6B")
    a = p.parse_args()

    c_os()
    c_driver()
    c_host()
    c_nvcc()
    with gpu_lock("env_check"):
        c_torch()
        c_cap()
        c_bf16()
        c_sdpa()
        c_flash_attn()
        c_bnb()
        c_rl_libs()
        if a.skip_vllm:
            RESULTS.append({"check": "vLLM loads and generates (small model)", "status": "skip",
                            "required": True, "detail": "--skip-vllm", "secs": 0})
        else:
            c_vllm(a.vllm_model)
    c_sionna()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env_check.json").write_text(json.dumps({"provenance": provenance(), "results": RESULTS}, indent=2))
    meas_path = OUT / "measurements.json"
    measurements = json.loads(meas_path.read_text()) if meas_path.exists() else None
    write_report(REPO / "docs" / "ENV_REPORT.md", measurements)

    failed = [r["check"] for r in RESULTS if r["required"] and r["status"] == "FAIL"]
    skipped = [r["check"] for r in RESULTS if r["required"] and r["status"] == "skip"]
    if failed:
        print("\nFAILED: " + ", ".join(failed))
    elif skipped:
        print("\nINCOMPLETE (required checks skipped): " + ", ".join(skipped))
    else:
        print("\nAll required checks passed.")
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    sys.exit(main())
