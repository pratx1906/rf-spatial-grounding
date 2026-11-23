"""Kandla scenario (WORLD side, never agent-visible): single drones, a decoy, a friendly patrol, and swarms.

All emitters are synthetic (ENV spec 14.2); positions are anchored to real site geometry (tank farms, jetties) by id.

cast="default" (every built seed so far): the scripted cast below, bit-identical across versions.
cast="varied" (S7.3, audit G Q48; test seeds only, built by S7.5): a randomised cast, `make_varied_emitters`:
  drone count 1-20 (some as one swarm), altitude 5-150 m, EIRP 10-30 dBm, same-band density 0.2-1.0 (the share of
  drones on one dominant band), mixed transit / loiter / hover tracks, 0-2 friendly drones and 0-1 decoy.
"""

from __future__ import annotations

import math
import random

from sim.world.scenario import DT_S, DURATION_S, Emitter, _fly, _freq

PROTO = {"ism_2400": "ocusync-like", "ism_5800": "ocusync-like", "sub_ghz_868_915": "lora-like"}
BW = {"sub_ghz_868_915": 500.0, "ism_2400": 10000.0, "ism_5800": 20000.0}


def _drone(rng, eid, wps, band, t0, t1, kind="drone", hop=None, swarm_id=None, speed=None) -> Emitter:
    sched = [(0.0, band)] + ([((t0 + t1) / 2, hop)] if hop else [])
    return Emitter(eid, kind, sched, {b: _freq(rng, b) for _, b in sched}, rng.uniform(17, 23), PROTO[band],
                   BW[band], t0, t1, wps, speed or rng.uniform(7, 13), swarm_id=swarm_id)


def _formation(shape: str, n: int, spacing: float) -> list[tuple[float, float]]:
    """(along, across) offsets relative to the leader, in metres."""
    if shape == "v":
        offs = [(0.0, 0.0)]
        for i in range(1, n):
            rank = (i + 1) // 2
            side = 1 if i % 2 else -1
            offs.append((-rank * spacing, side * rank * spacing))
        return offs
    return [(0.0, (i - (n - 1) / 2) * spacing) for i in range(n)]   # line abreast


def _swarm(rng, sid, n, shape, spacing, wps, band, t0, t1, speed) -> list[Emitter]:
    leader = _fly(wps, speed, round(t0 / DT_S) * DT_S, min(t1, DURATION_S))
    offs = _formation(shape, n, spacing)
    members = []
    for k, (al, ac) in enumerate(offs):
        e = _drone(rng, f"{sid}_m{k + 1}", wps, band, t0, t1, swarm_id=sid, speed=speed)
        pts, heading = [], 0.0
        jx, jy = rng.uniform(-4, 4), rng.uniform(-4, 4)
        for i, (t, x, y, z) in enumerate(leader):
            if i + 1 < len(leader):
                nx, ny = leader[i + 1][1], leader[i + 1][2]
                if math.hypot(nx - x, ny - y) > 0.5:
                    heading = math.atan2(ny - y, nx - x)
            ox = al * math.cos(heading) - ac * math.sin(heading)
            oy = al * math.sin(heading) + ac * math.cos(heading)
            pts.append((t, x + ox + jx + rng.gauss(0, 1.5), y + oy + jy + rng.gauss(0, 1.5),
                        max(10.0, z + rng.gauss(0, 2))))
        e.points = pts
        members.append(e)
    return members


CASTS = ("default", "varied", "newarea", "na_u", "na_f")
# Paired unseen-area experiment (paper 2026-10-31): arm U (seeds 101-120) anchors on the unseen zones, arm F (seeds
# 201-220) on the two trained zones with the weakest sensor geometry (TF-8, TF-3). Both arms draw the cast from the
# same random stream (seed % 100), so drone count, bands, EIRP, modes and timing are identical per pair; only the place
# differs. Entry/exit points: the half of a 1.5 km ring around the anchors' centroid that faces away from the site.
PAIRED_ANCHORS = {"na_u": ("south_warehouse_restricted", "jetty_j6_restricted"),
                  "na_f": ("tank_farm_tf8_restricted", "tank_farm_tf3_restricted")}
# Unseen-area test (paper, 2026-10-31): the zones added by envbuild.site_kandla.add_newarea_zones, in a pocket south of
# the tank farms (x -1000..1000 m, y -2200..-1250 m) where no training scenario (seeds 1-12, 17-28) ever put a drone.
NEWAREA_ZONES = ("south_warehouse_restricted", "jetty_j6_restricted")
ALT_RANGE_M = (5.0, 150.0)
EIRP_RANGE_DBM = (10.0, 30.0)


def make_emitters(seed: int, zones: dict, objs: list, cast: str = "default") -> list[Emitter]:
    if cast not in CASTS:
        raise ValueError(f"unknown cast {cast!r}; expected one of {CASTS}")
    if cast == "varied":
        return make_varied_emitters(seed, zones, objs)
    if cast == "newarea":
        return make_varied_emitters(seed, zones, objs, newarea=True)
    if cast in PAIRED_ANCHORS:
        return make_varied_emitters(seed % 100, zones, objs, paired=PAIRED_ANCHORS[cast])
    rng = random.Random(seed * 7919 + 11)
    farms = sorted([o for o in objs if o.kind == "tank_farm"],
                   key=lambda o: math.hypot(o.geom.centroid.x, o.geom.centroid.y))
    # the target farm and the loiter farm vary by seed among the four farms nearest the hero-zone centre
    near = farms[:4] if len(farms) >= 4 else farms
    order = rng.sample(range(len(near)), len(near)) if seed else list(range(len(near)))
    target = near[order[0]].geom.centroid
    second = near[order[1]].geom.centroid if len(near) > 1 else target
    jet = zones["jetty_restricted"].geom.centroid
    jetties = sorted([o for o in objs if o.kind == "jetty"], key=lambda o: math.hypot(o.geom.centroid.x,
                                                                                      o.geom.centroid.y))

    def alt():
        return rng.uniform(40, 110)

    ems = [
        # intruder from the creek (east) into the nearest tank farm's restricted zone, then out to the north-west
        _drone(rng, "emt_a", [(2600, rng.uniform(-1800, -800), alt()), (target.x, target.y, alt()),
                              (target.x - 1200, target.y + 1500, alt())],
               rng.choice(["ism_2400", "ism_5800"]), rng.uniform(0, 60), rng.uniform(420, 560)),
        # loiterer over the second tank farm
        _drone(rng, "emt_b", [(-2600, rng.uniform(-500, 500), 70.0), (second.x, second.y, 70.0),
                              (second.x + 5, second.y + 5, 70.0)], "sub_ghz_868_915", rng.uniform(30, 100), 600.0),
        # band-hopper transiting north to south over the jetties
        _drone(rng, "emt_c", [(jet.x - 300, 2600, alt()), (jet.x, jet.y, alt()), (jet.x + 200, -2600, alt())],
               "ism_2400", rng.uniform(80, 160), rng.uniform(500, 600), hop="ism_5800"),
    ]
    # friendly port inspection drone patrolling the jetty line
    jx = sorted([o for o in objs if o.kind == "jetty"], key=lambda o: -o.geom.centroid.y)
    patrol = [(o.geom.centroid.x + 60, o.geom.centroid.y, 45.0) for o in jx[:4]]
    patrol += list(reversed(patrol))[1:]
    f = _drone(rng, "emt_f", patrol, "ism_5800", 0.0, DURATION_S, kind="friendly_drone", speed=8.0)
    ems.append(f)
    # decoy: static ground transmitter beside the target tank farm (Wi-Fi-like, drone-confusable)
    ems.append(Emitter("emt_x", "decoy", [(0.0, "ism_2400")], {"ism_2400": _freq(rng, "ism_2400")}, 15.0,
                       "wifi-like", 20000.0, rng.uniform(0, 60), DURATION_S,
                       [(target.x + rng.uniform(150, 250), target.y - rng.uniform(150, 250), 3.0)], 0.0))
    for e in ems:
        e.points = _fly(e.waypoints, e.speed_mps, round(e.t_start / DT_S) * DT_S, min(e.t_end, DURATION_S))
    # hovering intruder near a jetty head (photographing a berth): slow approach, then a long hover
    jh = jetties[rng.randrange(min(4, len(jetties)))].geom.centroid if jetties else jet
    hx, hy = jh.x + rng.uniform(-250, 250), jh.y + rng.uniform(-250, 250)
    ems.append(_drone(rng, "emt_h", [(hx + 900, hy - 700, 55.0), (hx, hy, 45.0), (hx + 3, hy + 3, 45.0)],
                      rng.choice(["ism_2400", "ism_5800"]), rng.uniform(40, 200), DURATION_S, speed=6.0))
    ems[-1].points = _fly(ems[-1].waypoints, ems[-1].speed_mps, round(ems[-1].t_start / DT_S) * DT_S, DURATION_S)
    # swarm 1: 8 drones in a V from a corner (seed 0: north-east) towards the jetties, then holding over them
    cx, cy = [(3200, 3200), (3200, -3200), (-3200, 3200), (-3200, -3200)][rng.randrange(4) if seed else 0]
    ems += _swarm(rng, "swarm_1", 8, "v", 25.0,
                  [(cx, cy, 90.0), (jet.x + 150, jet.y + 150, 70.0), (jet.x + 160, jet.y + 140, 70.0)],
                  "ism_5800", rng.uniform(100, 160), DURATION_S, 12.0)
    # swarm 2: 5 drones line abreast across the tank farms, from the west (seed 0) or the east
    sx = -3200 if (not seed or rng.random() < 0.5) else 3200
    ems += _swarm(rng, "swarm_2", 5, "line", 35.0,
                  [(sx, target.y - 300, 60.0), (target.x, target.y, 60.0), (-sx * 0.8, target.y + 400, 60.0)],
                  "sub_ghz_868_915", rng.uniform(220, 300), DURATION_S, 10.0)
    return ems


def make_varied_emitters(seed: int, zones: dict, objs: list, newarea: bool = False,
                         paired: tuple | None = None) -> list[Emitter]:
    """Randomised cast for extra test seeds (audit G Q48). Own RNG stream; the default cast is untouched.
    newarea: the same generator anchored on the unseen-area zones (NEWAREA_ZONES), with entry/exit points on the
    southern half of the 3 km ring so that transits approach from the side away from the trained area."""
    rng = random.Random(seed * 104729 + 17 + (7 if newarea else 13 if paired else 0))
    bands = list(PROTO)
    if paired:
        anchors = [zones[z].geom.centroid for z in paired]
        acx, acy = sum(p.x for p in anchors) / 2, sum(p.y for p in anchors) / 2
        away = math.atan2(acy, acx)
    elif newarea:
        anchors = [zones[z].geom.centroid for z in NEWAREA_ZONES]
    else:
        anchors = [o.geom.centroid for o in objs if o.kind in ("tank_farm", "jetty")]
    if not anchors:
        anchors = [zones["jetty_restricted"].geom.centroid]
    n = rng.randint(1, 20)
    dominant = rng.choice(bands)
    density = rng.uniform(0.2, 1.0)            # probability that a drone uses the dominant band
    lo, hi = ALT_RANGE_M

    def band():
        return dominant if rng.random() < density else rng.choice([b for b in bands if b != dominant])

    def alt(base):
        return min(hi, max(lo, base * rng.uniform(0.85, 1.15)))

    def edge():
        if paired:                  # half ring (1.5 km) around the anchors, facing away from the site centre
            a = away + rng.uniform(-math.pi / 2, math.pi / 2)
            return acx + 1500 * math.cos(a), acy + 1500 * math.sin(a)
        if newarea:                 # southern half of a 1.5 km ring around the pocket, away from the trained area
            a = rng.uniform(math.pi, 2 * math.pi)
            return 1500 * math.cos(a), -1900 + 1500 * math.sin(a)
        a = rng.uniform(0, 2 * math.pi)
        return 3000 * math.cos(a), 3000 * math.sin(a)

    ems: list[Emitter] = []
    n_swarm = rng.randint(3, min(10, n)) if n >= 6 and rng.random() < 0.4 else 0
    k = 0
    for i in range(n - n_swarm):
        k += 1
        base = rng.uniform(lo, hi)
        tgt = rng.choice(anchors)
        mode = rng.choice(["transit", "loiter", "hover"])
        ex, ey = edge()
        if mode == "transit":
            ox, oy = edge()
            wps = [(ex, ey, alt(base)), (tgt.x, tgt.y, alt(base)), (ox, oy, alt(base))]
        elif mode == "loiter":
            r = rng.uniform(80, 400)
            wps = [(ex, ey, alt(base))] + [(tgt.x + r * math.cos(a), tgt.y + r * math.sin(a), alt(base))
                                           for a in (0.0, 1.6, 3.1, 4.7, 0.0)]
        else:
            hx, hy = tgt.x + rng.uniform(-300, 300), tgt.y + rng.uniform(-300, 300)
            wps = [(hx + rng.uniform(-900, 900), hy + rng.uniform(-900, 900), alt(base)), (hx, hy, alt(base)),
                   (hx + 3, hy + 3, alt(base))]
        t0 = rng.uniform(0, 300)
        t1 = DURATION_S if mode != "transit" else min(DURATION_S, t0 + rng.uniform(200, 450))
        e = _drone(rng, f"emt_v{k:02d}", wps, band(), t0, t1, speed=rng.uniform(3, 18))
        e.eirp_dbm = rng.uniform(*EIRP_RANGE_DBM)
        e.points = _fly(e.waypoints, e.speed_mps, round(e.t_start / DT_S) * DT_S, min(e.t_end, DURATION_S))
        ems.append(e)
    if n_swarm:
        sx, sy = edge()
        tgt = rng.choice(anchors)
        base = rng.uniform(max(lo, 15.0), hi)
        members = _swarm(rng, "swarm_v", n_swarm, rng.choice(["v", "line"]), rng.uniform(15, 50),
                         [(sx, sy, base), (tgt.x, tgt.y, base), (tgt.x + 10, tgt.y + 10, base)], band(),
                         rng.uniform(60, 300), DURATION_S, rng.uniform(6, 15))
        for e in members:
            e.eirp_dbm = rng.uniform(*EIRP_RANGE_DBM)
            e.points = [(t, x, y, min(hi, max(lo, z))) for t, x, y, z in e.points]
        ems += members
    for j in range(rng.randint(0, 2)):
        a, b = rng.sample(anchors, 2) if len(anchors) > 1 else (anchors[0], anchors[0])
        wps = [(a.x + 60, a.y, 45.0), (b.x + 60, b.y, 45.0), (a.x + 60, a.y, 45.0)]
        f = _drone(rng, f"emt_fv{j + 1}", wps, rng.choice(bands), 0.0, DURATION_S, kind="friendly_drone", speed=8.0)
        f.eirp_dbm = rng.uniform(*EIRP_RANGE_DBM)
        f.points = _fly(f.waypoints, f.speed_mps, 0.0, DURATION_S)
        ems.append(f)
    if rng.random() < 0.5:
        t = rng.choice(anchors)
        d = Emitter("emt_xv", "decoy", [(0.0, "ism_2400")], {"ism_2400": _freq(rng, "ism_2400")},
                    rng.uniform(*EIRP_RANGE_DBM), "wifi-like", 20000.0, rng.uniform(0, 60), DURATION_S,
                    [(t.x + rng.uniform(-250, 250), t.y + rng.uniform(-250, 250), 3.0)], 0.0)
        d.points = _fly(d.waypoints, 0.0, round(d.t_start / DT_S) * DT_S, DURATION_S)
        ems.append(d)
    return ems
