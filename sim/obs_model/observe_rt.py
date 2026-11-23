"""Observation model driven by Sionna RT caches + a DF triangulation tracker (main spec 4A.1, ENV spec 11).

WORLD side of the belief/world split: it reads the nominal Sionna cache and applies the episode's world
perturbation (main spec S3: per-object ghost power scaling, ghost dropout, SNR-dependent angle noise), then emits:
- observed bearings (all resolvable peaks per sensor; no path labels), and
- track estimates from least-squares triangulation of each sensor's STRONGEST peak (what a simple real tracker
  does), so a ghost that is stronger than the direct path really does bias the fix.
Path labels (LOS / reflect:<obj> / diffract:<obj>) go to truth only.
"""

from __future__ import annotations

import math
import random

import numpy as np

from sim.belief.rf_cache import DIFFRACTION, LOS, RFCache, ellipse, triangulate
from sim.geometry import angle_diff_deg, bearing_deg
from sim.obs_model.observe import ARRAY_GAIN_DB, NOISE_FLOOR_DBM, _klass_probs, _sigma_deg

BEARING_DT_S = 10.0
DETECT_SNR_DB = 6.0
RESOLUTION_DEG = 5.0          # 8-element UCA: peaks closer than this merge (e.g. ground bounce at the direct azimuth)
MAX_PEAKS = 3
GHOST_SCALE_DB = 3.0          # world perturbation per (sensor, reflecting object), gap level G1
GHOST_DROPOUT = 0.10


def _label(p: dict) -> tuple[str, str | None]:
    if p["ptype"] == LOS:
        return "LOS", None
    if p["ptype"] == DIFFRACTION:
        return f"diffract:{p['via']}", p["via"]
    return f"reflect:{p['via']}", p["via"]


def observe_rt(seed: int, emitters, site_build: dict, cache: RFCache) -> dict:
    rng = random.Random(seed * 104729 + 7)
    sensors = [s for s in site_build["sensors"] if s.type == "fixed_df"]
    ghost_scale: dict[tuple[str, str], float] = {}
    ids = rng.sample(range(10, 9999), len(emitters))
    track_of = {e.id: f"{ids[i]:04d}" for i, e in enumerate(emitters)}

    bearings, labels, fixes = [], [], {}
    hyp_n = 0
    for e in emitters:
        pts = {p[0]: p for p in e.points}
        for t in sorted(pts):
            if t % BEARING_DT_S:
                continue
            _, ex, ey, ez = pts[t]
            band = e.band_at(t)
            per_sensor = []
            for s in sensors:
                if band not in s.bands:
                    continue
                paths = cache.paths(band, s.id, ex, ey, ez)
                if not paths:
                    continue
                d_true = max(math.dist((s.x, s.y, s.z), (ex, ey, ez)), 1.0)
                peaks = []
                for p in paths:
                    d_cell = max(math.dist((s.x, s.y, s.z), p["cell"]), 1.0)
                    pw = e.eirp_dbm + p["gain_db"] - 20 * math.log10(d_true / d_cell)
                    lab, via = _label(p)
                    if p["ptype"] == LOS:          # exact geometry for the direct path; cached angles for ghosts
                        brg = bearing_deg(s.x, s.y, ex, ey)
                        el = math.degrees(math.atan2(ez - s.z, math.hypot(ex - s.x, ey - s.y)))
                    else:
                        brg, el = p["az"], p["el"]
                        key = (s.id, via or "?")
                        if key not in ghost_scale:
                            ghost_scale[key] = rng.uniform(-GHOST_SCALE_DB, GHOST_SCALE_DB)
                        pw += ghost_scale[key]
                        if rng.random() < GHOST_DROPOUT:
                            continue
                    peaks.append([brg, el, 10 ** (pw / 10), lab, via])
                peaks.sort(key=lambda q: -q[2])
                merged = []
                for q in peaks:
                    for m in merged:
                        if abs(angle_diff_deg(q[0], m[0])) < RESOLUTION_DEG:
                            m[2] += q[2]
                            break
                    else:
                        merged.append(list(q))
                kept = []
                for brg, el, pw_lin, lab, via in merged[:MAX_PEAKS]:
                    pw = 10 * math.log10(pw_lin)
                    snr = pw + ARRAY_GAIN_DB - NOISE_FLOOR_DBM
                    if snr < DETECT_SNR_DB:
                        continue
                    sig = _sigma_deg(snr)
                    kept.append((round((brg + rng.gauss(0, sig)) % 360, 1), round(el + rng.gauss(0, sig), 1),
                                 round(snr, 1), round(pw, 1), lab, via, sig))
                if kept:
                    per_sensor.append((s, kept))
            for s, kept in per_sensor:
                for brg, el, snr, pw, lab, via, _ in sorted(kept, key=lambda k: k[0]):
                    hyp_n += 1
                    hid = f"{s.id}_{track_of[e.id]}_{int(t):04d}_{hyp_n % 1000:03d}"
                    bearings.append((hid, track_of[e.id], s.id, t, brg, snr, pw))
                    labels.append((hid, lab, via))
            # tracker: each sensor's strongest peak -> weighted LS fix; leave-one-out gating with >= 3 sensors
            obs = []
            for s, kept in per_sensor:
                top = max(kept, key=lambda q: q[2])
                obs.append((s.x, s.y, top[0], top[6]))
            fix = triangulate(obs)
            if fix is not None and len(obs) >= 3:
                best = fix
                for i in range(len(obs)):
                    alt = triangulate(obs[:i] + obs[i + 1:])
                    if alt is not None and alt[2] < 0.5 * best[2]:
                        best = alt
                fix = best
            if fix is None:
                continue
            xy, cov, _res = fix
            near_s, near_k = min(per_sensor, key=lambda sk: math.hypot(xy[0] - sk[0].x, xy[1] - sk[0].y))
            el = max(near_k, key=lambda q: q[2])[1]
            z = near_s.z + math.hypot(xy[0] - near_s.x, xy[1] - near_s.y) * math.tan(math.radians(el))
            a, b_, th = ellipse(cov)
            fixes.setdefault(e.id, []).append((t, float(xy[0]), float(xy[1]), max(0.0, z), a, b_, th,
                                               len(per_sensor), band))

    tracks, points, track_labels = [], [], []
    for e in emitters:
        fx = fixes.get(e.id, [])
        if len(fx) < 2:
            continue
        tid = track_of[e.id]
        rows = []
        for (t0, x0, y0, z0, a0, b0, th0, n0, bd0), (t1, x1, y1, z1, a1, b1, _th1, _n1, _bd1) in zip(fx, fx[1:]):
            if t1 - t0 > 3 * BEARING_DT_S:
                continue                                         # coverage gap: never bridge
            for t in np.arange(t0, t1, 2.0):
                f = (t - t0) / (t1 - t0)
                rows.append((tid, float(t), round(x0 + f * (x1 - x0), 1), round(y0 + f * (y1 - y0), 1),
                             round(z0 + f * (z1 - z0), 1), round(min(a0 + f * (a1 - a0), 2000.0), 1),
                             round(min(b0 + f * (b1 - b0), 2000.0), 1), round(th0, 1), bd0, n0))
        t, x, y, z, a, b_, th, n, bd = fx[-1]
        rows.append((tid, float(t), round(x, 1), round(y, 1), round(z, 1), round(min(a, 2000.0), 1),
                     round(min(b_, 2000.0), 1), round(th, 1), bd, n))
        rows.sort(key=lambda r: r[1])
        if len(rows) < 3:
            continue
        points += rows
        probs = _klass_probs(rng, e.kind)
        conf = round(min(0.99, probs["drone"] * (0.7 + 0.1 * max(r[9] for r in rows))), 3)
        tb = [bb for bb in bearings if bb[1] == tid]
        power = sorted(bb[6] for bb in tb)[len(tb) // 2] if tb else None
        tracks.append((tid, rows[0][8], e.freq_mhz[rows[0][8]], probs, e.protocol, e.bandwidth_khz, power,
                       rows[0][1], rows[-1][1], max(r[9] for r in rows), conf))
        track_labels.append((tid, e.id, e.kind))
    kept_tracks = {t[0] for t in tracks}
    bearings = [b for b in bearings if b[1] in kept_tracks]
    kept_h = {b[0] for b in bearings}
    labels = [lb for lb in labels if lb[0] in kept_h]
    return {"tracks": tracks, "points": points, "bearings": bearings, "bearing_labels": labels,
            "track_labels": track_labels}
