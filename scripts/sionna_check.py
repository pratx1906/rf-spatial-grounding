"""Sionna RT smoke test and timing (SPEC 2.1, 2.2, 2.5). Runs in its own venv, never in the agent venv.

    python scripts/sionna_check.py [--variant cuda_ad_mono_polarized|llvm_ad_mono_polarized] [--n-rx 1000]

Loads a built-in scene, places a grid of receivers, computes paths, and merges into runs/m0/measurements.json:
which Mitsuba variant worked on this OS side (CUDA/OptiX vs CPU LLVM) and seconds per 1k receivers.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402
from scripts.provenance import provenance  # noqa: E402


def run(variant: str, n_rx: int) -> dict:
    import mitsuba as mi

    mi.set_variant(variant)
    import sionna.rt as rt

    scene = rt.load_scene(rt.scene.simple_street_canyon)
    arr = dict(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.tx_array = rt.PlanarArray(**arr)
    scene.rx_array = rt.PlanarArray(**arr)
    scene.add(rt.Transmitter(name="tx", position=[-32.0, 10.0, 32.0]))
    side = int(n_rx**0.5)
    for i in range(side):
        for j in range(side):
            scene.add(rt.Receiver(name=f"rx{i}_{j}", position=[-40 + 80 * i / side, -40 + 80 * j / side, 1.5]))

    solver = rt.PathSolver()
    kw = dict(max_depth=3, los=True, specular_reflection=True, diffuse_reflection=False, refraction=True)

    def once() -> tuple[float, list]:
        t0 = time.time()
        a, _tau = solver(scene, **kw).cir(out_type="numpy")
        return time.time() - t0, list(a.shape)

    cold, shape = once()  # includes Dr.Jit kernel compilation; same arguments as the timed runs
    warm = sorted(once()[0] for _ in range(3))
    secs = warm[1]  # median of 3
    n = side * side
    return {
        "variant": variant,
        "n_rx": n,
        "secs_cold": round(cold, 3),
        "secs": round(secs, 3),
        "secs_per_1k_rx": round(secs / n * 1000, 3),
        "cir_shape": shape,
    }


@exit_if_busy
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None, help="default: try CUDA then LLVM")
    p.add_argument("--n-rx", type=int, default=1024)
    a = p.parse_args()
    side = "wsl" if "microsoft" in platform.uname().release.lower() else platform.system().lower()

    results = {}
    variants = [a.variant] if a.variant else ["cuda_ad_mono_polarized", "llvm_ad_mono_polarized"]
    out = REPO / "runs" / "m0"
    out.mkdir(parents=True, exist_ok=True)
    with gpu_lock("sionna_precompute"):
        for v in variants:
            try:
                results[v] = run(v, a.n_rx)
                print(f"[ok] {v}: {results[v]}")
            except Exception as e:  # noqa: BLE001
                results[v] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"[FAIL] {v}: {results[v]['error']}")

        import sionna.rt

        record = {"provenance": provenance(), "results": results}
        (out / f"sionna_{side}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(record, indent=2))
        path = out / "measurements.json"
        merged = json.loads(path.read_text()) if path.exists() else {}
        for v, r in results.items():
            key = f"sionna_{side}_{v.split('_')[0]}"
            merged[key] = (f"{r['secs_per_1k_rx']} s per 1k rx (depth 3, warm median of 3; cold {r['secs_cold']} s)"
                           if "secs" in r else "FAIL: " + r["error"][:120])
        merged[f"sionna_rt_version_{side}"] = getattr(sionna.rt, "__version__", "unknown")
        path.write_text(json.dumps(merged, indent=2))
    return 0 if any("secs" in r for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
