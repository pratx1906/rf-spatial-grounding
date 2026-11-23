"""Emitters and their true trajectories for one scenario (WORLD side, never agent-visible)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import yaml

from harness.config import REPO

BANDS = yaml.safe_load((REPO / "configs" / "tools" / "bands.yaml").read_text())["bands"]
DT_S = 2.0
DURATION_S = 600.0


@dataclass
class Emitter:
    id: str
    kind: str                 # drone | decoy | friendly_drone
    band_schedule: list[tuple[float, str]]   # (from_t, band)
    freq_mhz: dict[str, float]
    eirp_dbm: float
    protocol: str
    bandwidth_khz: float
    t_start: float
    t_end: float
    waypoints: list[tuple[float, float, float]]
    speed_mps: float
    points: list[tuple[float, float, float, float]] = field(default_factory=list)   # t, x, y, z
    swarm_id: str | None = None

    def band_at(self, t: float) -> str:
        band = self.band_schedule[0][1]
        for t0, b in self.band_schedule:
            if t >= t0:
                band = b
        return band


def _fly(wps: list[tuple[float, float, float]], speed: float, t0: float, t1: float) -> list[tuple]:
    """Constant-speed flight along waypoints, hovering at the last one until t1."""
    pts, seg, pos = [], 0, list(wps[0])
    t = t0
    while t <= t1 + 1e-9:
        pts.append((round(t, 1), pos[0], pos[1], pos[2]))
        step = speed * DT_S
        while step > 0 and seg < len(wps) - 1:
            tx, ty, tz = wps[seg + 1]
            d = math.dist(pos, (tx, ty, tz))
            if d <= step:
                pos, step, seg = [tx, ty, tz], step - d, seg + 1
            else:
                f = step / d
                pos = [pos[0] + (tx - pos[0]) * f, pos[1] + (ty - pos[1]) * f, pos[2] + (tz - pos[2]) * f]
                step = 0
        t += DT_S
    return pts


def _freq(rng: random.Random, band: str) -> float:
    lo, hi = BANDS[band]
    return round(rng.uniform(lo + 1, hi - 1), 1)


def make_emitters(seed: int, zones: dict) -> list[Emitter]:
    rng = random.Random(seed * 7919 + 1)
    term = zones["oil_terminal_restricted"].geom
    sub = zones["substation_restricted"].geom
    tx, ty = term.centroid.x, term.centroid.y

    def alt():
        return rng.uniform(40, 120)

    def drone(eid: str, wps, band: str, t0: float, t1: float, hop: str | None = None, kind: str = "drone"):
        sched = [(0.0, band)] + ([((t0 + t1) / 2, hop)] if hop else [])
        freqs = {b: _freq(rng, b) for _, b in sched}
        proto = {"ism_2400": "ocusync-like", "ism_5800": "ocusync-like", "sub_ghz_868_915": "lora-like"}[band]
        return Emitter(eid, kind, sched, freqs, rng.uniform(17, 23), proto,
                       {"sub_ghz_868_915": 500.0}.get(band, 10000.0), t0, t1, wps, rng.uniform(6, 14))

    ems = [
        # 1: incursion from the creek into the oil terminal restricted zone
        drone("emt_a", [(rng.uniform(-900, -600), -1100, alt()), (tx + rng.uniform(-80, 80), ty, alt()),
                        (tx + rng.uniform(150, 300), ty + rng.uniform(200, 400), alt())],
              rng.choice(["ism_2400", "ism_5800"]), rng.uniform(0, 60), rng.uniform(420, 560)),
        # 2: loiterer over the cargo area
        drone("emt_b", [(1600, rng.uniform(200, 600), alt()), (rng.uniform(300, 900), rng.uniform(0, 300), 70.0),
                        (rng.uniform(300, 900), rng.uniform(0, 300), 70.0)],
              "sub_ghz_868_915", rng.uniform(30, 120), rng.uniform(450, 600)),
        # 3: band-hopper transiting the north approach corridor to the substation zone
        drone("emt_c", [(rng.uniform(-150, 150), 1900, alt()), (sub.centroid.x, sub.centroid.y, alt()),
                        (1500, rng.uniform(-200, 200), alt())],
              "ism_2400", rng.uniform(100, 200), rng.uniform(500, 600), hop="ism_5800"),
        # 4: friendly port inspection drone patrolling the jetties
        drone("emt_f", [(-1300, -350, 50.0), (-500, -350, 50.0), (-500, -700, 50.0), (-1300, -700, 50.0)],
              "ism_5800", 0.0, DURATION_S, kind="friendly_drone"),
    ]
    if rng.random() < 0.7:  # 5: optional second intruder crossing the site
        ems.append(drone("emt_d", [(-1600, rng.uniform(400, 900), alt()), (1600, rng.uniform(-200, 300), alt())],
                         rng.choice(["ism_2400", "ism_5800", "sub_ghz_868_915"]),
                         rng.uniform(200, 300), DURATION_S))
    # 6: decoy: a static ground emitter that looks drone-like on the 2.4 GHz band
    dx, dy = rng.uniform(-200, 1200), rng.uniform(-100, 500)
    ems.append(Emitter("emt_x", "decoy", [(0.0, "ism_2400")], {"ism_2400": _freq(rng, "ism_2400")}, 15.0,
                       "wifi-like", 20000.0, rng.uniform(0, 100), DURATION_S, [(dx, dy, 2.0)], 0.0))
    for e in ems:
        e.points = _fly(e.waypoints, e.speed_mps, round(e.t_start / DT_S) * DT_S, min(e.t_end, DURATION_S))
    return ems
