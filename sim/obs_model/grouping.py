"""Pipeline-side swarm grouping from OBSERVED track estimates (belief side; never uses truth).

Two tracks are linked when their estimates stay within LINK_M of each other for at least MIN_COMMON_S of common
time. Connected components with >= MIN_GROUP tracks become groups G1, G2, ... ordered by first detection.
"""

from __future__ import annotations

import math
from collections import defaultdict

LINK_M = 150.0
MIN_COMMON_S = 40.0
MIN_GROUP = 3
DT_S = 2.0


def group_tracks(points: list[tuple]) -> dict[str, str]:
    """points rows: (track_id, t, x, y, ...). Returns {track_id: group_id} for grouped tracks only."""
    pos: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    first: dict[str, float] = {}
    for r in points:
        pos[r[0]][r[1]] = (r[2], r[3])
        first[r[0]] = min(first.get(r[0], r[1]), r[1])
    tids = sorted(pos)
    parent = {t: t for t in tids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            common = pos[a].keys() & pos[b].keys()
            close = sum(1 for t in common if math.dist(pos[a][t], pos[b][t]) <= LINK_M)
            if close * DT_S >= MIN_COMMON_S:
                parent[find(a)] = find(b)
    comps = defaultdict(list)
    for t in tids:
        comps[find(t)].append(t)
    groups = sorted((c for c in comps.values() if len(c) >= MIN_GROUP), key=lambda c: min(first[t] for t in c))
    return {t: f"G{k}" for k, c in enumerate(groups, start=1) for t in c}
