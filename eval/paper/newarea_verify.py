"""Verify the unseen-area set (newarea_v1) in the world and visually (offline; reads truth/ for checking only).
    python -m eval.paper.newarea_verify --out runs/paper/results/NA/verify
Checks:
  1. zones: the two new zones exist in every new-area scenario and in no other scenario / the site build;
  2. training traffic: truth drone points of train seeds (1-12, 17-28) inside the new zones (expect 0) and the share of
     new-area drone points in 250 m cells training never used;
  3. observed side: agent-visible tracks of the new seeds inside the new zones (the tracker formed tracks there);
  4. answer keys: every key position / track of the frozen tasks, located on the map (share inside the pocket and in
     never-trained cells); keyed tracks resolved to their true emitter, distance key -> true position at the key time;
  5. text: new-zone names never appear in the training tasks; question texts never equal a training question.
Writes verify.json, map.png (whole port) and zoom.png (the pocket), and samples.md (20 questions with keys).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

SC = Path("data/scenarios_v3")
TRAIN = list(range(1, 13)) + list(range(17, 29))
NEW = list(range(61, 67))
NZ = ("south_warehouse_restricted", "jetty_j6_restricted")
POCKET = (-1000, 1250, -2250, -1250)
C = 250.0


def _zones(seed: int) -> dict:
    return {f["properties"]["id"]: f for f in json.loads((SC / f"kandla_s{seed}_X/zones.geojson").read_text())["features"]}


def _truth_pts(seed: int):
    c = duckdb.connect(str(SC / f"kandla_s{seed}_X/truth/truth.duckdb"), read_only=True)
    return np.array(c.execute("select p.x_m, p.y_m, p.t_s from emitter_points p join emitters e using(emitter_id) "
                              "where e.kind not in ('friendly_fixed')").fetchall())


def _obs_tracks(seed: int) -> dict:
    c = duckdb.connect(str(SC / f"kandla_s{seed}_X/observed.duckdb"), read_only=True)
    tabs = [t for (t,) in c.execute("show tables").fetchall()]
    tp = next(t for t in tabs if "track_point" in t)
    cols = [x[0] for x in c.execute(f"describe {tp}").fetchall()]
    tid = next(x for x in cols if "track" in x)
    rows = c.execute(f"select {tid}, x_m, y_m, t_s from {tp}").fetchall()
    out = {}
    for k, x, y, t in rows:
        out.setdefault(k, []).append((t, x, y))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/NA/verify")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import Point, shape

    rep = {}
    # 1. zones
    zs = {s: _zones(s) for s in NEW}
    rep["zones_in_new_seeds"] = {s: [z for z in NZ if z in zs[s]] for s in NEW}
    other = [p.name for p in SC.iterdir() if p.is_dir() and not any(p.name.startswith(f"kandla_s{s}_") for s in NEW)
             and (p / "zones.geojson").exists() and any(z in (p / "zones.geojson").read_text() for z in NZ)]
    rep["new_zones_in_other_scenarios"] = other
    rep["new_zones_in_site_build"] = any(z in Path("data/site_kandla/site_build.json").read_text() for z in NZ)
    geo = {z: shape(zs[NEW[0]][z]["geometry"]) for z in NZ}

    # 2. training traffic
    tr = np.concatenate([_truth_pts(s) for s in TRAIN])
    cells = {(int(x // C), int(y // C)) for x, y, _ in tr}
    rep["train_points_inside_new_zones"] = {z: int(sum(g.contains(Point(x, y)) for x, y, _ in tr[::5]) * 5) for z, g in geo.items()}
    nw = {s: _truth_pts(s) for s in NEW}
    allnew = np.concatenate(list(nw.values()))
    rep["new_points_in_never_trained_cells"] = round(float(np.mean([(int(x // C), int(y // C)) not in cells for x, y, _ in allnew])), 3)
    x0, x1, y0, y1 = POCKET
    rep["new_points_in_pocket"] = round(float(np.mean((allnew[:, 0] > x0) & (allnew[:, 0] < x1) & (allnew[:, 1] > y0) & (allnew[:, 1] < y1))), 3)

    # 3. observed tracks in the new zones
    obs = {s: _obs_tracks(s) for s in NEW}
    rep["observed_tracks_entering_new_zones"] = {
        s: {z: sum(any(g.contains(Point(x, y)) for _, x, y in pts) for pts in obs[s].values()) for z, g in geo.items()}
        for s in NEW}

    # 4. answer keys on the map
    T = [json.loads(x) for x in Path("eval/frozen/newarea_v1/tasks.jsonl").read_text().splitlines() if x.strip()]
    kp, in_pocket = [], set()
    for t in T:
        k = t["answer_key"]; s = t["scenario_seed"]
        if isinstance(k.get("position"), dict):
            kp.append((k["position"]["x_m"], k["position"]["y_m"], t["template"], t["task_id"]))
        refs = list(k.get("track_refs") or []) + ([t["params"]["track"]] if t["params"].get("track") else [])
        tt = k.get("t_s") or t["params"].get("t")
        for r in refs:
            pts = obs[s].get(r.split(":")[-1]) or obs[s].get(r)
            if pts:
                p = min(pts, key=lambda q: abs(q[0] - tt)) if tt is not None else pts[len(pts) // 2]
                kp.append((p[1], p[2], t["template"], t["task_id"]))
    kp_a = np.array([(x, y) for x, y, _, _ in kp]) if kp else np.zeros((0, 2))
    x0, x1, y0, y1 = POCKET
    in_pocket = sorted({i for x, y, _, i in kp if x0 < x < x1 and y0 < y < y1})
    (out / "pocket_task_ids.json").write_text(json.dumps(in_pocket))
    rep["tasks_with_key_in_pocket"] = len(in_pocket)
    rep["answer_key_locations"] = len(kp)
    if len(kp):
        rep["key_locations_in_pocket"] = round(float(np.mean((kp_a[:, 0] > x0) & (kp_a[:, 0] < x1) & (kp_a[:, 1] > y0) & (kp_a[:, 1] < y1))), 3)
        rep["key_locations_in_never_trained_cells"] = round(float(np.mean([(int(x // C), int(y // C)) not in cells for x, y in kp_a])), 3)
    rep["tasks"] = len(T); rep["per_area"] = dict(Counter(t["area"] for t in T))
    rep["tasks_naming_new_zones"] = sum(any(z in json.dumps(t) for z in NZ) for t in T)

    # 5. text leakage
    train_txt = Path("data/tasks/x3/train.jsonl").read_text()
    rep["new_zone_ids_in_train_tasks"] = {z: train_txt.count(z) for z in NZ}
    tq = {json.loads(x)["question"] for x in train_txt.splitlines() if x.strip()}
    rep["questions_identical_to_a_train_question"] = sum(t["question"] in tq for t in T)
    (out / "verify.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))

    # samples
    random.seed(1)
    smp = random.sample(T, 20)
    (out / "samples.md").write_text("\n".join(f"- **{t['template']}** (seed {t['scenario_seed']}): {t['question']}\n  "
                                              f"key `{json.dumps(t['answer_key'])[:220]}`" for t in smp) + "\n")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from shapely import wkt

    objs = pd.read_parquet(SC / f"kandla_s{NEW[0]}_X/objects.parquet")
    import yaml
    sens = yaml.safe_load((SC / f"kandla_s{NEW[0]}_X/sensors.yaml").read_text())["sensors"]
    for name, lim in [("map", (-3500, 3500, -4000, 3500)), ("zoom", (-1600, 1800, -3000, -800))]:
        fig, ax = plt.subplots(figsize=(9, 9))
        tri = tr[::7]
        ax.scatter(tri[:, 0], tri[:, 1], s=1, c="#9aa5b1", alpha=0.25, label="training drone traffic (24 seeds)")
        for _, o in objs.iterrows():
            g = wkt.loads(o.wkt)
            if g.geom_type == "Polygon" and o.kind in ("building", "tank", "jetty"):
                ax.fill(*g.exterior.xy, color="#444" if o.kind != "jetty" else "#6b4f2a", lw=0, alpha=0.8)
        for zid, f in zs[NEW[0]].items():
            g = shape(f["geometry"])
            if f["properties"]["kind"] == "restricted" and g.geom_type == "Polygon":
                new = zid in NZ
                ax.plot(*g.exterior.xy, color="#d62728" if new else "#1f77b4", lw=2.2 if new else 1)
                if new:
                    ax.text(g.centroid.x, g.centroid.y + 260, f["properties"]["name"], color="#d62728", ha="center", fontsize=9, weight="bold")
        cols = plt.cm.tab10.colors
        for i, s in enumerate(NEW):
            p = nw[s][::2]
            ax.scatter(p[:, 0], p[:, 1], s=2, color=cols[i], alpha=0.6, label=f"new-area truth, seed {s}")
        if len(kp):
            ax.scatter(kp_a[:, 0], kp_a[:, 1], marker="x", s=18, c="k", lw=0.8, label="answer-key locations")
        for s_ in sens:
            ax.plot(s_["pose"]["x_m"], s_["pose"]["y_m"], "^", ms=10, color="#2ca02c", mec="k")
            ax.text(s_["pose"]["x_m"] + 40, s_["pose"]["y_m"] + 40, s_["id"], fontsize=8, clip_on=True)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        ax.set_title("Unseen-area test: new zones (red), training traffic (grey), new-area drones (colour)")
        ax.legend(loc="upper left", fontsize=7, markerscale=4)
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
