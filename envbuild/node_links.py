"""Node-to-node path gains for fratricide (RF phase 2, Task 5). Windows Sionna venv.

    .venv-sionna-win\\Scripts\\python -m envbuild.node_links     -> data/site_kandla/rf_cache_v2/node_links.json

Scene and output follow the data version (SCENESCOUT_DATA_VERSION: <rf_cache>/scene, <rf_cache>/node_links.json);
the demo rf_cache_v2 is written only with --allow-demo-roots (v2), never under v21/v3 (checked before tracing).

A jammer at one DF node desensitises the others through the propagation path between their masts. The node path
caches cover 1.5 km at drone heights (2-80 m), but the masts are up to 2.6 km apart at 12 m and 25 m, so the 20
directed links are traced explicitly (v2 nominal scene, fixed diffraction, iso V-pol at both ends, all bands).
Belief-side product (nominal model): collateral_check may read it. A jammer on the SAME mast couples through
antenna isolation, not propagation: SAME_MAST_ISOLATION_DB (estimated, [HUMAN]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.rf_precompute import BAND_HZ  # noqa: E402

SITE = REPO / "data" / "site_kandla"
SAME_MAST_ISOLATION_DB = 30.0


def main() -> int:
    import argparse

    from envbuild.variant_roots import add_allow_flag, data_roots, guard_write

    ap = argparse.ArgumentParser()
    add_allow_flag(ap)
    a_ = ap.parse_args()
    rf = data_roots()["rf_cache"]
    f_out = guard_write(rf / "node_links.json", a_.allow_demo_roots)
    from envbuild import sionna_fix
    from envbuild.air7311_precompute import load, place, solve
    from scripts.gpu_lock import gpu_lock

    patch = sionna_fix.apply()
    sens = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
            if s["type"] == "fixed_df"}
    ids = sorted(sens)
    out = {"patch": patch, "same_mast_isolation_db": SAME_MAST_ISOLATION_DB, "fidelity": "nominal model (belief)",
           "links": {}}
    with gpu_lock("sionna_precompute"):
        for band, f in BAND_HZ.items():
            scene, _ = load(f, rf / "scene", band)
            for a in ids:
                tx = np.array([sens[a]["x_m"], sens[a]["y_m"], sens[a]["z_m"]])
                rx = np.array([[sens[b]["x_m"], sens[b]["y_m"], sens[b]["z_m"]] for b in ids if b != a])
                place(scene, tx, rx)
                p = solve(scene, "spec_diff")
                ar, ai = (np.array(x) for x in p.a)
                tau = np.array(p.tau)
                n = len(rx)
                h = ((ar + 1j * ai).reshape(n, -1) * np.exp(-2j * np.pi * f * np.nan_to_num(tau.reshape(n, -1))))
                g = (np.abs(h) ** 2).sum(1)
                inter = np.array(p.interactions)
                for k, b in enumerate(b_ for b_ in ids if b_ != a):
                    d = float(np.linalg.norm(rx[k] - tx))
                    out["links"].setdefault(band, {})[f"{a}->{b}"] = {
                        "gain_db": round(float(10 * np.log10(g[k])) if g[k] > 0 else -300.0, 2),
                        "distance_m": round(d, 1),
                        "fspl_db": round(float(20 * np.log10(4 * np.pi * d * f / 299792458.0)), 2)}
                print(band, a, "done", flush=True)
    f_out.write_text(json.dumps(out, indent=1))
    print(json.dumps({b: {k: v["gain_db"] for k, v in d.items()} for b, d in out["links"].items()}, indent=0)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
