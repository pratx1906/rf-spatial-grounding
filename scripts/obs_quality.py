"""Observation quality vs truth across scenario seeds (offline; reads truth via tasks.truth_access only).

    python scripts/obs_quality.py --seeds 0-16 [--out docs/RESULTS/air7311_observations.md]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_scenarios import parse_seeds, seed_dir  # noqa: E402
from tasks.truth_access import TruthStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    per_sensor = defaultdict(lambda: [0, 0, 0, 0])        # groups, strongest is LOS, any LOS, has ghost
    errs, n_tracks, kinds = [], 0, defaultdict(int)
    for seed in parse_seeds(a.seeds):
        d = seed_dir(seed)
        truth = TruthStore(d)
        lab = truth.hyp_labels()
        tof = truth.track_of()
        con = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
        for tid, sid, t, hs, pw in con.execute(
                "SELECT track_id, sensor_id, t_s, list(hyp_id), list(power_db) FROM bearings GROUP BY 1,2,3").fetchall():
            g = per_sensor[sid]
            g[0] += 1
            top = hs[int(np.argmax(pw))]
            g[1] += lab[top] == "LOS"
            g[2] += any(lab[h] == "LOS" for h in hs)
            g[3] += any(lab[h] != "LOS" for h in hs)
        for tid, t, x, y in con.execute("SELECT track_id, t_s, x_m, y_m FROM track_points WHERE t_s % 10 = 0").fetchall():
            eid, kind = tof[tid]
            p = truth.position(eid, t)
            if p:
                errs.append(math.hypot(x - p[0], y - p[1]))
        n_tracks += len(tof)
        for _, k in tof.values():
            kinds[k] += 1
        con.close()
        truth.close()
    e = np.array(errs)
    lines = ["| Node | bearing groups | strongest = direct | direct present | has ghost peak |", "|---|---|---|---|---|"]
    for sid in sorted(per_sensor):
        g = per_sensor[sid]
        lines.append(f"| {sid} | {g[0]} | {100 * g[1] / g[0]:.1f}% | {100 * g[2] / g[0]:.1f}% | {100 * g[3] / g[0]:.1f}% |")
    lines += ["", f"Tracks: {n_tracks} ({dict(kinds)}); fixes compared: {len(e)}",
              f"Fix error vs truth: median {np.median(e):.1f} m, p90 {np.percentile(e, 90):.1f} m, "
              f">100 m {100 * np.mean(e > 100):.1f}%"]
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
