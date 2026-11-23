"""S7.3 protocol classifier confusion vs SNR (sim.rf.classify on detections of the v2 chain), single path and
Sionna multipath.

Single path: the classifier labels recorded by `characterise_detection.py pd` (runs/s73/det_pd.json, 6 azimuths).
Multipath: the same waveforms through real cached Sionna channels: cells drawn at random (fixed seed) from the v2
belief cache data/site_kandla/rf_cache_v2/sionna_rt/ism_2400/aoa_S*.npz (read only; all valid paths of the cell,
gains, delays and arrival directions as traced), normalised so SNR is the total per-element in-band SNR.

    python scripts/characterise_classifier.py run --cells 12 --workers 8
    python scripts/characterise_classifier.py report
    .venv-sionna-win/Scripts/python scripts/characterise_classifier.py plot     # figure (needs matplotlib)
Outputs: runs/s73/cls_mp.json, runs/s73/cls_confusion.json, docs/RESULTS/figures/s73_confusion.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import characterise_detection as D  # noqa: E402

CACHE = REPO / "data" / "site_kandla" / "rf_cache_v2" / "sionna_rt" / "ism_2400"
CLASSES = ["ocusync_like", "wifi", "lora", "gfsk", "fm_video", "unknown", "missed"]
SNR_BINS = {"low (-6..2 dB)": (-6, 2), "mid (4..10 dB)": (4, 10), "high (12..20 dB)": (12, 20)}


def sample_cells(n: int, seed: int = 7) -> list[dict]:
    """n multipath channels from the cache: one random cell (sensor, height, x, y) with >= 2 valid paths each."""
    rng = np.random.default_rng(seed)
    files = sorted(CACHE.glob("aoa_S*.npz"))
    out = []
    while len(out) < n:
        f = files[int(rng.integers(len(files)))]
        z = np.load(f)
        pt = z["ptype"]
        for _ in range(200):
            ih, iy, ix = (int(rng.integers(s)) for s in pt.shape[:3])
            ok = pt[ih, iy, ix] != 255
            if ok.sum() >= 2:
                h = (z["h_re"][ih, iy, ix] + 1j * z["h_im"][ih, iy, ix])[ok]
                tau = np.nan_to_num(z["tau_ns"][ih, iy, ix][ok]) * 1e-9
                p = np.abs(h) ** 2
                if not np.isfinite(p).all() or p.sum() <= 0:
                    continue
                h = h / np.sqrt(p.sum())
                tau = tau - tau.min() + 2e-6
                w = p / p.sum()
                mean = (w * tau).sum()
                out.append({"file": f.name, "cell": [ih, iy, ix], "h": h, "tau": tau,
                            "az": np.nan_to_num(z["az_deg"][ih, iy, ix][ok]), "el": np.nan_to_num(z["el_deg"][ih, iy, ix][ok]),
                            "n_paths": int(ok.sum()), "k_factor_db": float(10 * np.log10(p.max() / max(p.sum() - p.max(), 1e-30))),
                            "rms_delay_ns": float(np.sqrt((w * (tau - mean) ** 2).sum()) * 1e9)})
                break
    return out


def confusion(rows: list[dict]) -> dict:
    """{waveform: {snr bin: {class: fraction}}} plus accuracy among detections and the unknown rate."""
    out = {}
    for name, truth in D.WAVEFORMS.items():
        rs = [r for r in rows if r["name"] == name and "labels" in r]
        if not rs:
            continue
        out[name] = {"truth": truth}
        for b, (lo, hi) in SNR_BINS.items():
            labs = [x for r in rs if lo <= r["snr"] <= hi for x in r["labels"]]
            if not labs:
                continue
            det = [x for x in labs if x != "missed"]
            out[name][b] = {"n": len(labs), **{c: round(labs.count(c) / len(labs), 3) for c in CLASSES},
                            "acc_detected": round(det.count(truth) / len(det), 3) if det else None,
                            "unknown_detected": round(det.count("unknown") / len(det), 3) if det else None}
        per = {}
        for s in sorted({r["snr"] for r in rs}):
            labs = [x for r in rs if r["snr"] == s for x in r["labels"] if x != "missed"]
            per[str(s)] = round(labs.count(truth) / len(labs), 3) if labs else None
        out[name]["acc_detected_vs_snr"] = per
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "plot"])
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--out", default=str(REPO / "runs" / "s73"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "plot":                      # figures only (the WSL venv has no matplotlib: run on Windows)
        plot(json.loads((out / "cls_confusion.json").read_text()))
        return 0
    if a.what == "run":
        cells = sample_cells(a.cells)
        jobs = [(n, s, -1.0, 100 + k, True, (c["h"], c["tau"], c["az"], c["el"]))
                for n in D.WAVEFORMS for s in D.SNRS for k, c in enumerate(cells)]
        print(f"multipath classifier: {len(jobs)} dwells over {len(cells)} cached channels", flush=True)
        rows = D.run_pool(D.pd_job, jobs, a.workers)
        meta = [{k: v for k, v in c.items() if k not in ("h", "tau", "az", "el")} for c in cells]
        (out / "cls_mp.json").write_text(json.dumps({"cells": meta, "rows": rows}, indent=1))
    sp = json.loads((out / "det_pd.json").read_text())["rows"]
    mp = json.loads((out / "cls_mp.json").read_text())
    res = {"single_path": confusion(sp), "multipath": confusion(mp["rows"]),
           "multipath_cells": {"n": len(mp["cells"]),
                               "n_paths_median": float(np.median([c["n_paths"] for c in mp["cells"]])),
                               "rms_delay_ns_median": float(np.median([c["rms_delay_ns"] for c in mp["cells"]])),
                               "k_factor_db_median": float(np.median([c["k_factor_db"] for c in mp["cells"]]))},
           "pd_multipath": {n: {"snr50_db": v["snr50_db"], "snr90_db": v["snr90_db"], "pd": v["pd"]}
                            for n, v in D.summarise_pd(mp["rows"]).items()}}
    (out / "cls_confusion.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("multipath_cells", "pd_multipath")}, indent=1))
    for kind in ("single_path", "multipath"):
        print(kind)
        for name, v in res[kind].items():
            for b in SNR_BINS:
                if b in v:
                    q = v[b]
                    top = sorted(((q[c], c) for c in CLASSES), reverse=True)[:3]
                    print(f"  {name:10s} {b:16s} acc_det={q['acc_detected']} unk={q['unknown_detected']} "
                          f"{', '.join(f'{c} {x:.2f}' for x, c in top)}")
    plot(res)
    return 0


def plot(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = REPO / "docs" / "RESULTS" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for i, kind in enumerate(("single_path", "multipath")):
        for j, b in enumerate(SNR_BINS):
            ax = axs[i, j]
            names = [n for n in res[kind] if b in res[kind][n]]
            m = np.array([[res[kind][n][b][c] for c in CLASSES] for n in names])
            ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
            for (r, c), v in np.ndenumerate(m):
                if v >= 0.005:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.6 else "black")
            ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set_title(f"{kind.replace('_', ' ')}, {b}", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "s73_confusion.png", dpi=130)
    print("figure:", figs / "s73_confusion.png")


if __name__ == "__main__":
    sys.exit(main())
