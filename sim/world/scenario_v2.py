"""v2 scenario layer (RF phase 2, Task 7; WORLD side): protocols drawn per band and paired controllers.

Applied on top of sim.world.kandla_scenario.make_emitters for the v2 sets only (v1 scenarios are unchanged).
  protocols  per band, weighted: 2.4 GHz OcuSync-like / Wi-Fi / ELRS-2.4; 5.8 GHz OcuSync-like / analogue FM / Wi-Fi;
             sub-GHz ELRS (LoRa) / SiK telemetry (GFSK). Swarm members share their leader's protocol.
  controllers one ground controller (kind `controller`, 1.5 m, stationary, synthetic placement) per non-swarm drone
             and per swarm, 300-2500 m from the launch point, transmitting the matching uplink: OcuSync -> RC FHSS
             GFSK bursts; ELRS -> LoRa uplink; Wi-Fi -> Wi-Fi; analogue FM -> ELRS-2.4 RC link; SiK -> SiK.
             Truth links controller and drone by `pair_id`; observations never see it.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

from sim.world.scenario import Emitter

PROTOCOLS = {"ism_2400": [("ocusync_like", 0.5), ("wifi", 0.25), ("elrs24", 0.25)],
             "ism_5800": [("ocusync_like", 0.5), ("fm_video", 0.3), ("wifi", 0.2)],
             "sub_ghz_868_915": [("elrs", 0.7), ("sik", 0.3)]}
BW_KHZ = {"ocusync_like": 10000.0, "wifi": 20000.0, "elrs24": 812.5, "fm_video": 18000.0, "elrs": 500.0,
          "sik": 128.0, "rc_fhss": 1500.0}
UPLINK = {"ocusync_like": "rc_fhss", "elrs": "elrs", "elrs24": "elrs24", "wifi": "wifi", "fm_video": "elrs24",
          "sik": "sik"}
UPLINK_BAND = {"rc_fhss": "ism_2400", "elrs24": "ism_2400", "wifi": None, "elrs": "sub_ghz_868_915",
               "sik": "sub_ghz_868_915"}


def _pick(rng: random.Random, band: str) -> str:
    r, acc = rng.random(), 0.0
    for p, w in PROTOCOLS[band]:
        acc += w
        if r <= acc:
            return p
    return PROTOCOLS[band][-1][0]


def apply_v2(emitters: list[Emitter], seed: int, land_ok) -> list[Emitter]:
    """Returns new emitters (drones with v2 protocols + controllers). land_ok(x, y) -> True on open ground."""
    rng = random.Random(seed * 6151 + 29)
    out, swarm_proto, ctrl_for = [], {}, []
    for e in emitters:
        if e.kind not in ("drone", "friendly_drone", "decoy"):
            out.append(e)
            continue
        band = e.band_schedule[0][1]
        if e.swarm_id:
            p = swarm_proto.setdefault(e.swarm_id, _pick(rng, band))
        else:
            p = _pick(rng, band) if e.kind != "decoy" else "wifi"
        e2 = replace(e, protocol=p, bandwidth_khz=BW_KHZ[p])
        out.append(e2)
        if e.kind == "drone" and (not e.swarm_id or e.id.endswith("_m1")):
            ctrl_for.append(e2)
    for d in ctrl_for:
        x0, y0 = d.points[0][1], d.points[0][2]
        for _ in range(200):
            r, a = rng.uniform(300, 2500), rng.uniform(0, 2 * math.pi)
            x, y = x0 + r * math.cos(a), y0 + r * math.sin(a)
            if land_ok(x, y):
                break
        up = UPLINK[d.protocol]
        band = UPLINK_BAND[up] or d.band_schedule[0][1]
        pts = [(t, x, y, 1.5) for t, *_ in d.points]
        c = Emitter(f"ctrl_{d.id}", "controller", [(0.0, band)], {band: d.freq_mhz.get(band, d.freq_mhz[d.band_schedule[0][1]])},
                    rng.uniform(17, 27), up, BW_KHZ[up], d.t_start, d.t_end, [(x, y, 1.5)], 0.0)
        c.points = pts
        c.pair_id = d.id
        c.placement = "synthetic"
        out.append(c)
    return out


def jammer_events(seed: int, sensor_ids: list[str], max_eirp_dbm: float = 47.0) -> list[dict]:
    """0-2 mitigation-TX windows per scenario (synthetic): a DF node transmits band-limited noise through its PA.
    Truth AND belief: the system knows when it transmitted (effector log)."""
    rng = random.Random(seed * 9973 + 5)
    out = []
    for k in range(rng.choice([0, 1, 1, 2])):
        band = rng.choice(["ism_2400", "ism_5800", "sub_ghz_868_915"])
        f = {"ism_2400": 2441.75, "ism_5800": rng.choice([5775.0, 5825.0]), "sub_ghz_868_915": 895.0}[band]
        t0 = float(rng.randrange(60, 480, 10))
        out.append({"id": f"jam{k + 1}", "sensor": rng.choice(sensor_ids), "band": band, "f_mhz": f,
                    "bw_mhz": {"sub_ghz_868_915": 20.0, "ism_2400": 80.0, "ism_5800": 50.0}[band],
                    "eirp_dbm": round(rng.uniform(max_eirp_dbm - 7, max_eirp_dbm), 1), "t0_s": t0,
                    "t1_s": t0 + float(rng.randrange(60, 130, 10)), "placement": "synthetic"})
    return out
