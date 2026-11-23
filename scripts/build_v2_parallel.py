"""Build RF phase 2 (v2) scenario sets with several GPU workers under a hard VRAM guard (WSL).

    python scripts/build_v2_parallel.py --workers 5 --jobs 1-12,17-28:X 13-16:X 15-16:W4 15-16:W5 1-3:W6 4-6:W7 7-9:W8
    SCENESCOUT_DATA_VERSION=v21 python scripts/build_v2_parallel.py --workers 5 --jobs 0:X 13-16:X ...

The data version (SCENESCOUT_DATA_VERSION) passes through to every worker; all output dirs are checked against the
demo-root guard before any worker starts (--allow-demo-roots: v2 only, sets SCENESCOUT_ALLOW_DEMO_ROOTS=1 for the
workers). Logs: runs/v2_build/ (v2) or runs/v2_build_<version>/.

Each job is one `scripts/build_scenarios.py --obs v2 --seeds N [--world-variant W]` process. The parent holds the
global GPU lock; workers lock private files. Total GPU memory is polled every second; above --ceiling-gb every
worker is killed (no spill, no OOM) and the remaining jobs are reported as not run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import ENV_ALLOW, add_allow_flag, data_version, guard_write  # noqa: E402
from scripts.build_scenarios import parse_seeds, v2_dir  # noqa: E402
from scripts.gpu_lock import gpu_lock  # noqa: E402

LOG = REPO / "runs" / "v2_build" / "build.log"


def log_path(version: str | None = None) -> Path:
    v = data_version(version)
    return LOG if v == "v2" else REPO / "runs" / f"v2_build_{v}" / "build.log"


def vram_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ceiling-gb", type=float, default=22.0)
    ap.add_argument("--jobs", nargs="+", required=True, help="SEEDS:X or SEEDS:W<n>")
    ap.add_argument("--cast", choices=["default", "varied"], default="default", help="passed to build_scenarios")
    add_allow_flag(ap)
    a = ap.parse_args()
    jobs = []
    for spec in a.jobs:
        seeds, _, var = spec.partition(":")
        for s in parse_seeds(seeds):
            jobs.append((s, None if var in ("", "X") else var))
    for s, var in jobs:                          # refuse demo roots before any worker starts
        guard_write(v2_dir(s, var), a.allow_demo_roots)
    LOG = log_path()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    procs, failed, done, peak, tripped = [], [], [], [0], []
    lock_dir = LOG.parent / "locks"
    lock_dir.mkdir(exist_ok=True)
    for f in lock_dir.glob("*.lock"):        # private worker locks of a previous (killed) run
        f.unlink()

    def guard():
        while not tripped and (jobs or procs):
            mb = vram_mb()
            peak[0] = max(peak[0], mb)
            if mb > a.ceiling_gb * 1024:
                tripped.append(mb)
                for p in list(procs):
                    p.kill()
            time.sleep(1)

    def worker(k: int):
        thr = str(max(1, (os.cpu_count() or 8) // a.workers))     # avoid numpy/torch thread oversubscription
        env = {**os.environ, "AKSHA_GPU_LOCK": str(lock_dir / f"w{k}.lock"), "OMP_NUM_THREADS": thr,
               "MKL_NUM_THREADS": thr, "OPENBLAS_NUM_THREADS": thr, "TORCH_NUM_THREADS": thr}
        if a.allow_demo_roots:
            env[ENV_ALLOW] = "1"
        while jobs and not tripped:
            seed, var = jobs.pop(0)
            cmd = [sys.executable, "scripts/build_scenarios.py", "--obs", "v2", "--seeds", str(seed)]
            if var:
                cmd += ["--world-variant", var]
            if a.cast != "default":
                cmd += ["--cast", a.cast]
            t0 = time.time()
            with (LOG.parent / f"s{seed}_{var or 'X'}.log").open("w") as fh:
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
                procs.append(p)
                rc = p.wait()
                procs.remove(p)
            rec = {"seed": seed, "variant": var or "X", "rc": rc, "s": round(time.time() - t0), "peak_mb": peak[0]}
            (done if rc == 0 else failed).append(rec)
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    with gpu_lock("df_sim"):
        g = threading.Thread(target=guard, daemon=True)
        g.start()
        ws = [threading.Thread(target=worker, args=(k,)) for k in range(a.workers)]
        for w in ws:
            w.start()
            time.sleep(20)                        # stagger start-up (CUDA context + cache loads)
        for w in ws:
            w.join()
    summary = {"done": len(done), "failed": len(failed), "not_run": len(jobs), "peak_mb": peak[0], "guard": tripped}
    print(json.dumps(summary), flush=True)
    with LOG.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    return 0 if not failed and not tripped else 1


if __name__ == "__main__":
    sys.exit(main())
