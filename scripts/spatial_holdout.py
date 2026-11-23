"""Within-site spatial hold-out (paper plan Revision 1, item 11). Offline (reads truth/ for emitter positions).

The held-out cluster is the WEST TANK-FARM area: tank farm TF-3 and TF-8 restricted zones, their convex hull, buffered
by BUFFER_M. A task falls in the cluster when any of these holds:
  - it names a cluster zone (params.zone / zone_phrase, answer_key.zone / region_ref) or a cluster tank farm object;
  - its answer-key position lies in the region;
  - its track (params.track or the first key track) is in the region at the task time (observed track point nearest
    to t, or the track's median position for time-free tasks), or the true emitter behind it is (truth, offline);
  - a siting key zone (zones_wkt) intersects the region.

    SCENESCOUT_DATA_VERSION=v3 python scripts/spatial_holdout.py

Writes eval/frozen/spatial_v1 (via eval.frozen_build --copy of the cluster tasks of the test seeds, 15-16 and 51-60;
refuses to overwrite) and data/rl/spatial_exclude.txt (task ids of TRAIN pools in the cluster, one per line, with a
'# source' header per file), plus data/rl/spatial_exclude.manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
from shapely import wkt as swkt
from shapely.geometry import Point, shape
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CLUSTER_ZONES = ("zone:tank_farm_tf3_restricted", "zone:tank_farm_tf8_restricted")
CLUSTER_WORDS = ("tf-3", "tf-8", "tf3", "tf8", "tank_farm_tf3", "tank_farm_tf8", "tank farm 3", "tank farm 8")
BUFFER_M = 100.0
TEST_SRC = REPO / "data/tasks/x3/test.jsonl"
TRAIN_SRC = ["data/tasks/x3/train.jsonl", "data/tasks/x3_W6/train.jsonl", "data/tasks/x3_W7/train.jsonl",
             "data/tasks/x3_W8/train.jsonl", "data/tasks/x3/val_clean.jsonl"]
TRAIN_GLOB = "data/tasks/x3_Mtrain/train_*.jsonl"


def region(site_dir: Path):
    zs = {("zone:" + f["properties"]["id"]): shape(f["geometry"])
          for f in json.loads((site_dir / "zones.geojson").read_text())["features"]}
    return unary_union([zs[z] for z in CLUSTER_ZONES]).convex_hull.buffer(BUFFER_M)


@lru_cache(maxsize=64)
def site(site: str) -> dict:
    d = REPO / site
    o = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    pts = o.execute("SELECT track_id, t_s, x_m, y_m FROM track_points").fetchdf()
    o.close()
    from tasks.truth_access import TruthStore

    ts = TruthStore(d)
    tl = {t: e for t, (e, _) in ts.track_of().items()}
    ep = ts.con.execute("SELECT emitter_id, t_s, x_m, y_m FROM emitter_points").fetchdf()
    ts.close()
    return {"reg": region(d), "pts": {k: g[["t_s", "x_m", "y_m"]].to_numpy() for k, g in pts.groupby("track_id")},
            "tl": tl, "ep": {k: g[["t_s", "x_m", "y_m"]].to_numpy() for k, g in ep.groupby("emitter_id")}}


def _at(a: np.ndarray, t: float | None):
    if a is None or not len(a):
        return None
    if t is None:
        return np.median(a[:, 1:], axis=0)
    i = int(np.argmin(np.abs(a[:, 0] - t)))
    return a[i, 1:] if abs(a[i, 0] - t) <= 10 else None


def in_cluster(task: dict) -> tuple[bool, str]:
    p, k = task.get("params") or {}, task.get("answer_key") or {}
    refs = [p.get("zone"), k.get("zone"), k.get("region_ref")]
    if any(r in CLUSTER_ZONES for r in refs if isinstance(r, str)):
        return True, "zone_ref"
    words = " ".join(str(p.get(x) or "") for x in ("zone_phrase", "obj_phrase", "phrase")) + " " + task["question"]
    if any(w in words.lower() for w in CLUSTER_WORDS):
        return True, "named"
    if not task.get("site") or not (REPO / task["site"]).exists():
        return False, ""
    s = site(task["site"])
    reg = s["reg"]
    pos = k.get("position")
    if isinstance(pos, dict) and reg.contains(Point(pos["x_m"], pos["y_m"])):
        return True, "key_position"
    for w in k.get("zones_wkt") or []:
        if swkt.loads(w).intersects(reg):
            return True, "siting_zone"
    tr = p.get("track") or next(iter(k.get("track_refs") or []), None)
    if isinstance(tr, str):
        tid = tr.split("@")[0].removeprefix("trk:")
        t = p.get("t", k.get("t_s"))
        t = float(t) if isinstance(t, (int, float)) else None
        xy = _at(s["pts"].get(tid), t)
        if xy is not None and reg.contains(Point(*xy)):
            return True, "track_observed"
        e = s["tl"].get(tid)
        xy = _at(s["ep"].get(e), t) if e else None
        if xy is not None and reg.contains(Point(*xy)):
            return True, "track_true"
    return False, ""


def main() -> int:
    """--exclude-only: only rewrite data/rl/spatial_exclude.txt (e.g. after new train pools such as x3_Mtrain)."""
    only = "--exclude-only" in sys.argv[1:]
    out_set = REPO / "eval/frozen/spatial_v1"
    if out_set.exists() and not only:
        raise SystemExit(f"{out_set} exists; refusing to overwrite (use --exclude-only)")
    test = [json.loads(x) for x in TEST_SRC.read_text().splitlines() if x.strip()]
    keep, why = [], Counter()
    for t in test:
        ok, w = in_cluster(t)
        if ok:
            keep.append(t)
            why[w] += 1
    if not only:
        tmp = REPO / "data/tasks/spatial_v1_test.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
        subprocess.run([sys.executable, "-m", "eval.frozen_build", "--version", "spatial_v1", "--copy",
                        str(tmp.relative_to(REPO))], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    srcs = TRAIN_SRC + sorted(str(p.relative_to(REPO)) for p in REPO.glob(TRAIN_GLOB))
    lines, per_src = [], {}
    for src in srcs:
        f = REPO / src
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        ids = [r["task_id"] for r in rows if in_cluster(r)[0]]
        per_src[src] = {"rows": len(rows), "excluded": len(ids)}
        lines += [f"# source {src}"] + ids
    ex = REPO / "data/rl/spatial_exclude.txt"
    ex.parent.mkdir(parents=True, exist_ok=True)
    ex.write_text("\n".join(lines) + "\n")
    man = {"cluster": "west tank-farm area: convex hull of " + ", ".join(CLUSTER_ZONES) + f", buffered {BUFFER_M:.0f} m",
           "rule": "scripts/spatial_holdout.py in_cluster (zone/object names, key position, siting zones, track "
                   "position observed or true at task time)",
           "test_source": str(TEST_SRC.relative_to(REPO)), "test_tasks": len(test), "held_out_tasks": len(keep),
           "held_out_by_reason": dict(why), "held_out_by_area": dict(Counter(r["area"] for r in keep)),
           "train_exclusions": per_src, "exclude_sha256": hashlib.sha256(ex.read_bytes()).hexdigest()}
    (REPO / "data/rl/spatial_exclude.manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))
    return 0


if __name__ == "__main__":
    sys.exit(main())
