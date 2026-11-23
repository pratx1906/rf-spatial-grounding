"""World check and map for the paired unseen-area arms (offline; truth read for checking only).
    python -m eval.paper.na_paired_verify --out runs/paper/results/NA/verify_paired
Per arm (U seeds 101-120, F 201-220): share of drone truth points and of answer-key locations in 250 m cells that no
training scenario (seeds 1-12, 17-28) ever used; new-zone ids in training tasks; observed tracks entering the anchor
zones. Writes verify_paired.json and map_paired.png (training traffic grey, arm U red, arm F blue).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.paper.newarea_verify import C, SC, TRAIN, _obs_tracks, _truth_pts, _zones
from sim.world.kandla_scenario import PAIRED_ANCHORS

ARMS = {"U": (range(101, 121), "na_u"), "F": (range(201, 221), "na_f")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/NA/verify_paired")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import Point, shape

    tr = np.concatenate([_truth_pts(s) for s in TRAIN])
    cells = {(int(x // C), int(y // C)) for x, y, _ in tr}
    rep = {}
    pts = {}
    for arm, (seeds, cast) in ARMS.items():
        seeds = [s for s in seeds if (SC / f"kandla_s{s}_X/truth/truth.duckdb").exists()]
        P = np.concatenate([_truth_pts(s) for s in seeds]); pts[arm] = P
        z = _zones(seeds[0]); geo = {k: shape(z[k]["geometry"]) for k in PAIRED_ANCHORS[cast]}
        T = [json.loads(x) for x in Path(f"eval/frozen/na{arm}_v1/tasks.jsonl").read_text().splitlines() if x.strip()] \
            if Path(f"eval/frozen/na{arm}_v1/tasks.jsonl").exists() else []
        keys = [(t["answer_key"]["position"]["x_m"], t["answer_key"]["position"]["y_m"]) for t in T
                if isinstance(t["answer_key"].get("position"), dict)]
        obs = {s: _obs_tracks(s) for s in seeds}
        rep[arm] = {"seeds": len(seeds), "anchors": list(PAIRED_ANCHORS[cast]),
                    "drone_points_in_never_trained_cells": round(float(np.mean(
                        [(int(x // C), int(y // C)) not in cells for x, y, _ in P])), 3),
                    "answer_positions": len(keys),
                    "answer_positions_in_never_trained_cells": round(float(np.mean(
                        [(int(x // C), int(y // C)) not in cells for x, y in keys])), 3) if keys else None,
                    "observed_tracks_entering_anchor_zones_per_scenario": round(float(np.mean(
                        [sum(any(g.contains(Point(x, y)) for g in geo.values() for _, x, y in v)
                             for v in obs[s].values()) for s in seeds])), 2),
                    "train_points_inside_anchor_zones": int(sum(any(g.contains(Point(x, y)) for g in geo.values())
                                                                for x, y, _ in tr[::5]) * 5)}
    train_txt = Path("data/tasks/x3/train.jsonl").read_text()
    rep["unseen_zone_ids_in_train_tasks"] = {z: train_txt.count(z) for z in PAIRED_ANCHORS["na_u"]}
    rep["familiar_zone_ids_in_train_tasks"] = {z: train_txt.count(z) for z in PAIRED_ANCHORS["na_f"]}
    (out / "verify_paired.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(tr[::6, 0], tr[::6, 1], s=1, c="#9aa5b1", alpha=0.3, label="training drone traffic (24 scenarios)")
    for arm, col in (("F", "#1f77b4"), ("U", "#d62728")):
        p = pts[arm][::3]
        ax.scatter(p[:, 0], p[:, 1], s=1.5, c=col, alpha=0.45, label=f"arm {arm} drones ({'unseen' if arm == 'U' else 'familiar'})")
    z = _zones(101)
    for zid, f in z.items():
        if f["properties"]["kind"] == "restricted":
            g = shape(f["geometry"])
            col = "#d62728" if zid in PAIRED_ANCHORS["na_u"] else "#1f77b4" if zid in PAIRED_ANCHORS["na_f"] else "#555"
            ax.plot(*g.exterior.xy, color=col, lw=2.2 if col != "#555" else 0.8)
    import yaml
    for n in yaml.safe_load((SC / "kandla_s101_X/sensors.yaml").read_text())["sensors"]:
        ax.plot(n["pose"]["x_m"], n["pose"]["y_m"], "^", ms=9, color="#2ca02c", mec="k")
    ax.set_xlim(-3500, 3500); ax.set_ylim(-4000, 3000); ax.set_aspect("equal")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.set_title("Paired arms: unseen zones (red) vs familiar zones TF-8, TF-3 (blue)")
    ax.legend(loc="upper left", fontsize=7, markerscale=5)
    fig.tight_layout(); fig.savefig(out / "map_paired.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
