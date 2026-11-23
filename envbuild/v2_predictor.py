"""Agent-side propagation predictor caches for the v2 belief (rf_cache_v2/sionna_rt/<band>/aoa_<S>.npz).

The agent's tools (predict_aoa, prop_query, effect_footprint, fit_candidates) read node-centred path caches in the
layout of rf_cache/sionna_rt. For v2 they are the Blender-geometry nominal AIR7311 caches (fixed diffraction,
V-pol), converted: gain_db = 10 log10 |h|^2 per path; same grid (20 m, 2-80 m heights). No new tracing.

    python -m envbuild.v2_predictor [--allow-demo-roots]     (WSL, CPU)

Reads and writes the rf_cache of the data version (SCENESCOUT_DATA_VERSION, envbuild.variant_roots.data_roots); the
demo rf_cache_v2 is written only with --allow-demo-roots (v2), never under v21/v3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "data" / "site_kandla" / "rf_cache_v2"          # demo; main() uses data_roots()["rf_cache"]
BANDS = ("sub_ghz_868_915", "ism_2400", "ism_5800")
KEEP = ("xs", "ys", "heights", "names", "az_deg", "el_deg", "tau_ns", "ptype", "via", "h_re", "h_im", "kept_energy")


def main() -> None:
    from envbuild.variant_roots import add_allow_flag, data_roots, guard_write

    ap = argparse.ArgumentParser()
    add_allow_flag(ap)
    a = ap.parse_args()
    V2 = data_roots()["rf_cache"]
    out = guard_write(V2 / "sionna_rt", a.allow_demo_roots)
    n = 0
    for band in BANDS:
        (out / band).mkdir(parents=True, exist_ok=True)
        for f in sorted(V2.glob(f"air7311_*/spec_diff/{band}.npz")):
            sid = f.parts[-3].split("_")[1]
            z = np.load(f)
            d = {k: z[k] for k in KEEP}
            p = d["h_re"].astype("f8") ** 2 + d["h_im"].astype("f8") ** 2
            d["gain_db"] = np.where(p > 0, 10 * np.log10(np.maximum(p, 1e-30)), -300.0).astype("f4")
            np.savez_compressed(out / band / f"aoa_{sid}.npz", **d)
            n += 1
    (out / "manifest.json").write_text(json.dumps({"source": f"{V2.name}/air7311_*/spec_diff (Blender geometry, "
                                                             "sionna_fix)", "files": n}, indent=1))
    print("wrote", n, "predictor caches")


if __name__ == "__main__":
    main()
