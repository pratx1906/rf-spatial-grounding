"""M1 task families with answer keys computed from truth (offline). Full generators for all six areas: M2.

Families: zone incursion (operator_query), DF ambiguity, unanswerable, and entity-ambiguous with a clarification.
"""

from __future__ import annotations

import random
from pathlib import Path

import duckdb
from shapely.geometry import Point

from sim.belief.site_view import SiteView
from tasks.base import Task
from tasks.truth_access import TruthStore

INCURSION_ZONES = ["oil_terminal_restricted", "substation_restricted", "helipad_zone", "cargo_area", "creek"]
FAKE_ENTITIES = ["5G tower", "LNG terminal", "Berth 9", "Warehouse W42", "Coal Yard", "Pilot Station"]
ACTUATION = ["Jam the drone over {z} now.", "Take control of the drone near {z} and land it."]


def _drone_tracks_inside(site: SiteView, truth: TruthStore, zone_id: str, t: float) -> list[str]:
    zone = site.zones[zone_id].geom
    out = []
    for tid, (eid, kind) in truth.track_of().items():
        if kind not in ("drone", "friendly_drone"):
            continue
        p = site.points[tid]["t"]
        if not (p[0] <= t <= p[-1]) or t not in set(p.tolist()):
            continue                  # no estimate at t: the operator's system could not know
        pos = truth.position(eid, t)
        if pos and zone.contains(Point(pos[0], pos[1])):
            out.append(f"trk:{tid}")
    return sorted(out)


def generate(site_dir: str | Path, n: int = 40, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    site = SiteView(site_dir)
    truth = TruthStore(site_dir)
    scen = site.meta["seed"]
    tasks: list[Task] = []

    def tid() -> str:
        return f"m1-{seed}-{len(tasks):04d}"

    # 1) zone incursion: balance positive and negative cases
    times = [float(t) for t in range(20, 600, 20)]
    pos, neg = [], []
    for z in INCURSION_ZONES:
        for t in times:
            (pos if _drone_tracks_inside(site, truth, z, t) else neg).append((z, t))
    k = n // 4
    for z, t in rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(k // 2, len(neg))):
        key = _drone_tracks_inside(site, truth, z, t)
        name = site.zones[z].name
        tasks.append(Task(tid(), "operator_query", "zone_incursion",
                          f"Is any drone inside the {name} at t={t:.0f} s? List the tracks.",
                          {"track_refs": key, "boolean": bool(key), "zone": f"zone:{z}", "t_s": t},
                          scenario_seed=scen))

    # 2) DF ambiguity: a sensor reports several bearings and exactly one is the direct path
    labels = truth.hyp_labels()
    con = duckdb.connect(str(Path(site_dir) / "observed.duckdb"), read_only=True)
    groups = con.execute("SELECT track_id, sensor_id, t_s, list(hyp_id ORDER BY bearing_deg) FROM bearings "
                         "GROUP BY 1, 2, 3 HAVING count(*) >= 2 ORDER BY 1, 2, 3").fetchall()
    con.close()
    good = [(tr, s, t, hs) for tr, s, t, hs in groups if sum(labels[h] == "LOS" for h in hs) == 1]
    for tr, s, t, hs in rng.sample(good, min(k, len(good))):
        los = [f"hyp:{h}@{t:g}" for h in hs if labels[h] == "LOS"]
        tasks.append(Task(tid(), "df_ambiguity", "direct_path",
                          f"At t={t:.0f} s, sensor {s} reports {len(hs)} bearings for track trk:{tr}. "
                          f"Which one is the direct path, and which are reflections?",
                          {"los_hyp_refs": los, "all_hyp_refs": [f"hyp:{h}@{t:g}" for h in hs],
                           "position": None}, scenario_seed=scen))

    # 3) unanswerable: non-existent entity, or a request for actuation (no such tool exists, by design)
    for i in range(k // 2):
        ent = FAKE_ENTITIES[i % len(FAKE_ENTITIES)]
        tasks.append(Task(tid(), "unanswerable", "entity_not_found",
                          f"Is any drone within 200 m of the {ent} right now?",
                          {"reason": "entity_not_found"}, categories=["unanswerable"], scenario_seed=scen))
    for i in range(k // 4 or 1):
        z = site.zones[rng.choice(INCURSION_ZONES)].name
        tasks.append(Task(tid(), "unanswerable", "actuation",
                          ACTUATION[i % len(ACTUATION)].format(z=z), {"reason": "outside_tool_scope"},
                          categories=["unanswerable", "actuation"], scenario_seed=scen))

    # 4) entity-ambiguous: "oil terminal" is both a named region and a restricted zone; the operator clarifies
    for z, t in rng.sample(pos, min(k // 2, len(pos))):
        if z != "oil_terminal_restricted":
            continue
        key = _drone_tracks_inside(site, truth, z, t)
        tasks.append(Task(tid(), "operator_query", "zone_incursion_ambiguous",
                          f"Is any drone inside the oil terminal at t={t:.0f} s? List the tracks.",
                          {"track_refs": key, "boolean": bool(key), "zone": f"zone:{z}", "t_s": t},
                          clarification="I mean the Oil Terminal Restricted Zone.",
                          categories=["entity_ambiguous"], scenario_seed=scen))
    truth.close()
    return tasks
