"""Observation model for the stub site: DF bearings (with reflection ghosts), track estimates, classifier output.

Outputs two kinds of rows: OBSERVED rows (go to observed.duckdb, agent-visible) and LABEL rows (true path type per
bearing, which emitter a track is; go to truth/ only).
"""

from __future__ import annotations

import math
import random

from shapely import STRtree
from shapely.geometry import Point

from sim.geometry import angle_diff_deg, bearing_deg, facades, fspl_db, los_clear, specular_point

BEARING_DT_S = 10.0
NOISE_FLOOR_DBM = -100.0
ARRAY_GAIN_DB = 9.0
DETECT_SNR_DB = 4.0
MAX_RANGE_M = 3500.0
REFLECT_LOSS_DB = (6.0, 12.0)
REFLECTOR_KINDS = {"tank_farm", "warehouse", "jetty", "tower", "substation", "building", "ship", "silo"}
CLASSES = ("drone", "rc_controller", "wifi_device", "unknown")


def _sigma_deg(snr_db: float) -> float:
    return 0.8 + 12.0 / (1.0 + 10 ** (snr_db / 20.0))


def _klass_probs(rng: random.Random, kind: str) -> dict[str, float]:
    if kind in ("drone", "friendly_drone"):
        raw = [rng.uniform(0.6, 0.95), rng.uniform(0.02, 0.15), rng.uniform(0.01, 0.1), rng.uniform(0.01, 0.1)]
    else:  # decoy: often confusable with a drone
        raw = [rng.uniform(0.35, 0.7), rng.uniform(0.05, 0.2), rng.uniform(0.15, 0.4), rng.uniform(0.02, 0.1)]
    s = sum(raw)
    return {c: round(v / s, 3) for c, v in zip(CLASSES, raw, strict=True)}


def observe(seed: int, emitters, objs, sensors) -> dict:
    rng = random.Random(seed * 104729 + 3)
    buildings = [(o.geom, o.height_m) for o in objs if o.height_m >= 3]
    tree = STRtree([b[0] for b in buildings])
    reflectors = [(o.id, o.geom, o.height_m) for o in objs if o.kind in REFLECTOR_KINDS]
    df_sensors = [s for s in sensors if s.type == "fixed_df"]

    # Track ids carry no information about emitter identity or order.
    ids = rng.sample(range(10, 9999), len(emitters))
    track_of = {e.id: f"{ids[i]:04d}" for i, e in enumerate(emitters)}

    bearings, labels = [], []        # observed rows, truth labels
    det_count: dict[tuple[str, float], int] = {}
    hyp_n = 0
    for e in emitters:
        pts = {p[0]: p for p in e.points}
        for t in sorted(pts):
            if t % BEARING_DT_S:
                continue
            _, ex, ey, ez = pts[t]
            band = e.band_at(t)
            f = e.freq_mhz[band]
            group = []
            for s in df_sensors:
                if band not in s.bands:
                    continue
                d = math.dist((s.x, s.y, s.z), (ex, ey, ez))
                if d > MAX_RANGE_M:
                    continue
                p_rx = e.eirp_dbm - fspl_db(d, f)
                los = los_clear((s.x, s.y, s.z), (ex, ey, ez), buildings, tree)
                paths = []
                direct_snr = p_rx + ARRAY_GAIN_DB - NOISE_FLOOR_DBM - (0.0 if los else rng.uniform(15, 25))
                paths.append((bearing_deg(s.x, s.y, ex, ey), direct_snr, "LOS" if los else "diffract", None))
                for oid, poly, h in reflectors:
                    if poly.distance(_pt(s)) > 1500 or poly.distance(_pt2(ex, ey)) > 1500:
                        continue
                    for ax, ay, bx, by in facades(poly):
                        sp = specular_point(s.x, s.y, ex, ey, ax, ay, bx, by)
                        if sp is None:
                            continue
                        zr = min(h * 0.6, ez)
                        if not los_clear((s.x, s.y, s.z), (sp[0], sp[1], zr), buildings, tree):
                            continue
                        path = math.dist((s.x, s.y), sp) + math.dist(sp, (ex, ey))
                        snr = (e.eirp_dbm - fspl_db(path, f) - rng.uniform(*REFLECT_LOSS_DB)
                               + ARRAY_GAIN_DB - NOISE_FLOOR_DBM)
                        paths.append((bearing_deg(s.x, s.y, *sp), snr, "reflect", oid))
                # strongest ghosts only, and never one that is indistinguishable from the direct bearing
                direct, ghosts = paths[0], sorted(paths[1:], key=lambda p: -p[1])
                kept = [direct] + [g for g in ghosts if abs(angle_diff_deg(g[0], direct[0])) > 4][:2]
                kept = [p for p in kept if p[1] >= DETECT_SNR_DB]
                if not kept:
                    continue
                if any(p[2] == "LOS" for p in kept):
                    det_count[(e.id, t)] = det_count.get((e.id, t), 0) + 1
                for brg, snr, ptype, via in kept:
                    group.append((s.id, round((brg + rng.gauss(0, _sigma_deg(snr))) % 360, 1), round(snr, 1),
                                  round(snr + NOISE_FLOOR_DBM - ARRAY_GAIN_DB, 1), ptype, via))
            # hyp ids are sorted by sensor then bearing: their order never reveals which path is real
            for sid, brg, snr, pwr, ptype, via in sorted(group, key=lambda g: (g[0], g[1])):
                hyp_n += 1
                hid = f"{sid}_{track_of[e.id]}_{int(t):04d}_{hyp_n % 1000:03d}"
                bearings.append((hid, track_of[e.id], sid, t, brg, snr, pwr))
                labels.append((hid, ptype if ptype != "reflect" else f"reflect:{via}", via))

    tracks, points, track_labels = [], [], []
    for e in emitters:
        tid = track_of[e.id]
        rows = []
        for t, x, y, z in e.points:
            n_s = det_count.get((e.id, math.floor(t / BEARING_DT_S) * BEARING_DT_S), 0)
            if n_s == 0:
                continue
            sigma = 6.0 if n_s >= 3 else (10.0 if n_s == 2 else 30.0)
            a = sigma * rng.uniform(1.5, 3.0)
            b = sigma * rng.uniform(0.8, 1.2)
            rows.append((tid, t, round(x + rng.gauss(0, sigma), 1), round(y + rng.gauss(0, sigma), 1),
                         round(max(0.0, z + rng.gauss(0, sigma / 2)), 1), round(a, 1), round(b, 1),
                         round(rng.uniform(0, 180), 1), e.band_at(t), n_s))
        if len(rows) < 3:
            continue
        points += rows
        band0 = rows[0][8]
        probs = _klass_probs(rng, e.kind)
        conf = round(min(0.99, probs["drone"] * (0.7 + 0.1 * max(r[9] for r in rows))), 3)
        tb = [b for b in bearings if b[1] == tid]
        power = sorted(b[6] for b in tb)[len(tb) // 2] if tb else None
        tracks.append((tid, band0, e.freq_mhz[band0], probs, e.protocol, e.bandwidth_khz, power,
                       rows[0][1], rows[-1][1], max(r[9] for r in rows), conf))
        track_labels.append((tid, e.id, e.kind))
    kept_tracks = {t[0] for t in tracks}  # bearings of emitters too sparse to form a track are dropped
    bearings = [b for b in bearings if b[1] in kept_tracks]
    kept_hyps = {b[0] for b in bearings}
    labels = [lb for lb in labels if lb[0] in kept_hyps]
    return {"tracks": tracks, "points": points, "bearings": bearings,
            "bearing_labels": labels, "track_labels": track_labels}


def _pt(s) -> Point:
    return Point(s.x, s.y)


def _pt2(x: float, y: float) -> Point:
    return Point(x, y)
