"""Trace living-world variant caches (D-031) on the Windows GPU: all sensors x bands, spec_diff, one process per
variant. World variants -> data/site_kandla/world_cache/rf/<id>; belief variants -> rf_cache/ensemble/<id>.

    .venv-sionna-win\\Scripts\\python scripts\\precompute_variants.py --only W0          (control first)
    .venv-sionna-win\\Scripts\\python scripts\\precompute_variants.py --only W1t W1 W2 W3 B1 B2 B3

Logs time and peak GPU memory per variant to runs/rf_variants/precompute.log.
--stack v2 scene and cache roots follow the data version (SCENESCOUT_DATA_VERSION, envbuild.variant_roots); the demo
roots (rf_cache_v2, world_cache_v2) are written only with --allow-demo-roots, never under v21/v3.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SITE = REPO / "data" / "site_kandla"
LOG = REPO / "runs" / "rf_variants" / "precompute.log"


CEILING_MB = 22 * 1024        # user allows ~23 GB (2026-10-20); hard guard 1 GB below
PROCS: list = []


def peak_vram(stop: threading.Event, box: list) -> None:
    """Track peak GPU memory; above CEILING_MB kill the workers at once (a parallel batch that outgrows the ceiling
    must fail loudly, not spill)."""
    while not stop.is_set():
        try:
            mb = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5).stdout.split()[0])
            box[0] = max(box[0], mb)
            if mb > CEILING_MB:
                box.append(f"ceiling exceeded: {mb} MiB")
                for pr in list(PROCS):
                    with contextlib.suppress(Exception):
                        pr.kill()
        except Exception:  # noqa: BLE001
            pass
        stop.wait(1)


def check_stack(stack: str, version: str) -> None:
    """The v1 stack writes the legacy unversioned roots (rf_cache/, world_cache/): refuse it under v21/v3 data."""
    if stack == "v1" and version != "v2":
        raise SystemExit(f"--stack v1 writes the legacy roots; refused under data version {version} (use --stack v2)")


def failed_label(vid: str, shift: int | None) -> str:
    """Failed-run label: the variant, plus the lattice shift when the run traced a shifted lattice."""
    return vid if shift is None else f"{vid}@p{int(shift)}"


def command(vid: str, spec: dict, stack: str, chunk: int, solver: dict | None = None) -> list[str]:
    """The air7311_precompute command of one variant. v2: dual-pol grid, heights 2-80 m, UTD fix, the variant's
    sea state (Ament roughness, S7.4); solver: optional samples_per_src / max_depth / top_k overrides, and
    lattice_shift (a seeded shifted ray lattice, RF convergence Task 4/5: written to <cache>/lattice/p<shift>)."""
    from envbuild.variant_roots import cache_root, lattice_root, scene_dir

    side = spec[vid]["side"]
    shift = (solver or {}).get("lattice_shift")
    cmd = [sys.executable, "-m", "envbuild.air7311_precompute", "--sensors", "S1", "S2", "S3", "S4", "S5",
           "--setups", spec[vid].get("setup", "spec_diff"), "--radius-m", "1500", "--step-m", "20",
           "--chunk", str(chunk), "--scene-dir", str(scene_dir(vid, side, stack)),
           "--out-root", str(lattice_root(cache_root(vid, side, stack), shift))]
    if stack == "v2":
        cmd += ["--heights", "2", "5", "15", "40", "80", "--dual-pol", "--fix-diffraction"]
        if int(spec[vid].get("sea_state", 0)) > 0:
            cmd += ["--sea-state", str(int(spec[vid]["sea_state"]))]
    for k, flag in (("samples_per_src", "--samples-per-src"), ("max_depth", "--max-depth"), ("top_k", "--top-k"),
                    ("lattice_shift", "--lattice-shift")):
        if (solver or {}).get(k) is not None:
            cmd += [flag, str(int(solver[k]))]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--stack", choices=["v1", "v2"], default="v1",
                    help="v2: rf_cache_v2 / world_cache_v2, dual-pol grid, heights 2-80 m, fixed diffraction")
    ap.add_argument("--parallel", type=int, default=1,
                    help="variants traced side by side (each Sionna run alternates GPU tracing with CPU path "
                         "extraction; two workers at half chunk keep the GPU busy at about the same peak VRAM)")
    ap.add_argument("--samples-per-src", type=int, default=None, help="solver override (convergence, S7.4)")
    ap.add_argument("--max-depth", type=int, default=None, help="solver override (convergence, S7.4)")
    ap.add_argument("--top-k", type=int, default=None, help="paths kept per cell (default air7311_precompute.TOP_K)")
    ap.add_argument("--lattice-shifts", nargs="*", type=int, default=None,
                    help="trace every variant once per seeded shifted ray lattice into <cache>/lattice/p<seed> "
                         "(RF convergence Task 5 fallback: per-cell spread over patterns)")
    from envbuild.variant_roots import add_allow_flag, cache_root, data_version, guard_write

    add_allow_flag(ap)
    a = ap.parse_args()
    check_stack(a.stack, data_version())
    from scripts.gpu_lock import gpu_lock

    spec = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())["variants" if a.stack == "v1" else "variants_v2"]
    if a.stack == "v2":
        for vid in a.only:                     # refuse demo roots before any trace
            guard_write(cache_root(vid, spec[vid]["side"], "v2"), a.allow_demo_roots)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    solver = {k: v for k, v in (("samples_per_src", a.samples_per_src), ("max_depth", a.max_depth),
                                ("top_k", a.top_k)) if v is not None}
    shifts = a.lattice_shifts or [None]

    stop, box = threading.Event(), [0]
    threading.Thread(target=peak_vram, args=(stop, box), daemon=True).start()
    queue, failed = [(vid, sh) for vid in a.only for sh in shifts], []
    lock_dir = LOG.parent / "worker_locks"
    lock_dir.mkdir(exist_ok=True)

    def worker(k: int) -> None:
        # the parent holds the global GPU lock for the whole batch; each worker locks a private file
        env = {**os.environ, "AKSHA_GPU_LOCK": str(lock_dir / f"w{k}.lock")}
        while True:
            try:
                vid, sh = queue.pop(0)
            except IndexError:
                return
            t0 = time.time()
            if len(box) > 1:
                failed.append(failed_label(vid, sh))
                continue
            sol = solver if sh is None else {**solver, "lattice_shift": sh}
            pr = subprocess.Popen(command(vid, spec, a.stack, a.chunk, sol), cwd=REPO, env=env)
            PROCS.append(pr)
            rc = pr.wait()
            PROCS.remove(pr)
            rec = {"variant": vid, "stack": a.stack, "side": spec[vid]["side"], "rc": rc, "lattice_shift": sh,
                   "seconds": round(time.time() - t0, 1), "peak_vram_mb_so_far": box[0], "workers": a.parallel}
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if rc:
                failed.append(failed_label(vid, sh))

    with gpu_lock("sionna_precompute"):
        ths = [threading.Thread(target=worker, args=(k,)) for k in range(max(1, a.parallel))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    stop.set()
    print(json.dumps({"done": a.only, "failed": failed, "peak_vram_mb": box[0], "guard": box[1:]}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
