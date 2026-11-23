"""Typed event extraction from OBSERVED track estimates (what the pipeline knows; SPEC 5 "Time")."""

from __future__ import annotations

import math
from collections import defaultdict

from shapely.geometry import Point
from shapely.prepared import prep

EVENT_ZONE_KINDS = {"perimeter", "restricted", "corridor", "aviation", "named_region"}
LOITER_WINDOW_S = 60.0
LOITER_RADIUS_M = 40.0
DEBOUNCE = 3   # consecutive 2 s estimates (6 s) on the new side of a zone boundary


def extract_events(points: list[tuple], zones) -> list[tuple]:
    """points rows: (track_id, t, x, y, z, a, b, theta, band, n_sensors). Returns event rows
    (evt_id, type, track_id, zone_id, t_s, x_m, y_m, detail)."""
    by_track = defaultdict(list)
    for r in points:
        by_track[r[0]].append(r)
    zp = [(z.id, prep(z.geom)) for z in zones if z.kind in EVENT_ZONE_KINDS]
    events = []
    for tid in sorted(by_track):
        rows = sorted(by_track[tid], key=lambda r: r[1])
        first = rows[0]
        events.append(("detection_start", tid, None, first[1], first[2], first[3], first[8]))
        inside_prev = {zid: p.contains(Point(first[2], first[3])) for zid, p in zp}
        for zid, inside in inside_prev.items():
            if inside:  # already inside at first detection
                events.append(("zone_enter", tid, zid, first[1], first[2], first[3], "at_detection"))
        # Debounce: a zone state change needs DEBOUNCE consecutive estimates on the new side (noisy estimates near a
        # boundary would otherwise flicker enter/exit). The event time is the first estimate of the new state.
        pending: dict[str, list] = {}
        for prev, r in zip(rows, rows[1:]):
            pt = Point(r[2], r[3])
            for zid, p in zp:
                inside = p.contains(pt)
                if inside == inside_prev[zid]:
                    pending.pop(zid, None)
                    continue
                cand = pending.setdefault(zid, [r, 0])
                cand[1] += 1
                if cand[1] >= DEBOUNCE:
                    f = cand[0]
                    events.append(("zone_enter" if inside else "zone_exit", tid, zid, f[1], f[2], f[3], None))
                    inside_prev[zid] = inside
                    pending.pop(zid)
            if r[8] != prev[8]:
                events.append(("band_change", tid, None, r[1], r[2], r[3], f"{prev[8]}->{r[8]}"))
        # loiter: a window of >= LOITER_WINDOW_S where every estimate stays within LOITER_RADIUS_M of the mean
        i, in_loiter = 0, False
        for k in range(len(rows)):
            while rows[k][1] - rows[i][1] > LOITER_WINDOW_S:
                i += 1
            win = rows[i:k + 1]
            if rows[k][1] - rows[i][1] < LOITER_WINDOW_S - 1e-6:
                continue
            mx = sum(r[2] for r in win) / len(win)
            my = sum(r[3] for r in win) / len(win)
            tight = all(math.hypot(r[2] - mx, r[3] - my) <= LOITER_RADIUS_M for r in win)
            if tight and not in_loiter:
                events.append(("loiter", tid, None, win[0][1], round(mx, 1), round(my, 1), f"{LOITER_WINDOW_S:.0f}s"))
            in_loiter = tight
    events.sort(key=lambda e: (e[3], e[1], e[0]))
    return [(f"{n + 1:05d}", *e) for n, e in enumerate(events)]
