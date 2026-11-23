"""P10: the human-question form (docs/paper/human_questions/form.csv + FORM.md slot table). 100 slots over the
seed-15/16 test scenarios (v3 data). Each slot: scenario, time window, what an operator sees in the console (zones,
DF sensor sites, named site objects, and the OBSERVED tracks in the window with their band; observed tables only,
never truth/), the answer type to aim at, and an empty question column.

    SCENESCOUT_DATA_VERSION=v3 python scripts/human_form.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from shapely import wkt as swkt
from shapely.geometry import Point, shape

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "paper" / "human_questions"
SCEN = {15: "data/scenarios_v3/kandla_s15_X", 16: "data/scenarios_v3/kandla_s16_X"}
BAND = {"ism_2400": "2.4 GHz", "ism_5800": "5.8 GHz", "sub_ghz_868_915": "868/915 MHz"}
# answer type -> slots per seed, and the one-line aim shown to the writer
TYPES = {
    "operator_query": (15, "a question about drones/tracks now: where, how many, in which zone, near what, real or not"),
    "df_ambiguity": (8, "a question about bearings at a sensor: which is the direct path, where the emitter really is"),
    "replay": (7, "a question about what happened earlier: when something entered/left/was first seen"),
    "drone_search": (5, "ask the system to send the DF drone (D1) to pin down a track's position"),
    "siting": (5, "where coverage is weak or where one more DF sensor would help"),
    "collateral": (5, "what a jammer at a given sensor/band/power would also hit (own links, assets, aviation)"),
    "unanswerable": (5, "a question the system should refuse or say it cannot answer (actions, unknown things, future)"),
}
WINDOWS = [(t, t + 60) for t in range(0, 600, 60)]


def scene(seed: int) -> dict:
    d = REPO / SCEN[seed]
    zones = [(f["properties"]["id"], f["properties"]["name"], shape(f["geometry"]))
             for f in json.loads((d / "zones.geojson").read_text())["features"]]
    sens = [s for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"] if s["type"] == "fixed_df"]
    obj = pd.read_parquet(d / "objects.parquet")
    keep = obj[obj.kind.isin(["tank_farm", "jetty", "substation", "silo"]) | (obj.name_source == "osm")]
    objs = [(r.name, swkt.loads(r.wkt).centroid) for r in keep.itertuples()]
    con = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    tracks = con.execute("SELECT track_id, band, first_seen, last_seen FROM tracks").fetchdf()
    pts = con.execute("SELECT track_id, t_s, x_m, y_m FROM track_points").fetchdf()
    con.close()
    return {"zones": zones, "sensors": sens, "objs": objs, "tracks": tracks, "pts": pts}


def visible(sc: dict, w: tuple[int, int], rng: random.Random) -> str:
    t0, t1 = w
    tr = sc["tracks"][(sc["tracks"].first_seen <= t1) & (sc["tracks"].last_seen >= t0)]
    mid = sc["pts"][(sc["pts"].t_s >= t0) & (sc["pts"].t_s <= t1)]
    zones_hit, near = set(), set()
    for _, p in mid.groupby("track_id").median().iterrows():
        pt = Point(p.x_m, p.y_m)
        zones_hit.update(n for _, n, g in sc["zones"] if g.contains(pt) and "Hero" not in n and "perimeter" not in n)
        near.update(n for n, c in sorted(sc["objs"], key=lambda o: o[1].distance(pt))[:1])
    items = [f"trk:{r.track_id} ({BAND.get(r.band, r.band)})" for r in tr.itertuples()]
    rng.shuffle(items)
    trk = ", ".join(items[:6]) + (f" and {len(items) - 6} more" if len(items) > 6 else "")
    z = "; ".join(sorted(zones_hit)[:4]) or "none of the restricted zones"
    o = "; ".join(sorted(near)[:3]) or "-"
    return (f"{len(tr)} tracks on screen ({trk or 'none'}). Track positions fall in: {z}. "
            f"Site objects near them: {o}.")


def main() -> int:
    rng = random.Random(10)
    rows = []
    for seed in (15, 16):
        sc = scene(seed)
        slots = [ty for ty, (n, _) in TYPES.items() for _ in range(n)]
        rng.shuffle(slots)
        for i, ty in enumerate(slots):
            w = WINDOWS[i % len(WINDOWS)]
            rows.append({"slot": 0, "scenario": f"kandla_s{seed}_X", "t_window_s": f"{w[0]}-{w[1]}",
                         "answer_type": ty, "aim": TYPES[ty][1], "what_the_operator_sees": visible(sc, w, rng),
                         "question": "", "writer": "", "key_checked_by": ""})
    for i, r in enumerate(rows, 1):
        r["slot"] = i
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "form.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sens = scene(15)["sensors"]
    (OUT / "_context.json").write_text(json.dumps({
        "zones": [n for _, n, _ in scene(15)["zones"]],
        "sensors": {s["id"]: s["mount"] for s in sens}, "n_slots": len(rows),
        "per_type": {k: 2 * v[0] for k, v in TYPES.items()}}, indent=1))
    print(json.dumps({"slots": len(rows), "out": str(OUT / "form.csv")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
