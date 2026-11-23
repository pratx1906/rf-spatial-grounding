"""Bulk-stockpile mound mesh over any simple footprint polygon, concave ones included (demo-v2 S6 roof/stockpile fix).

The old stepped mound scaled the footprint ring toward its vertex mean. On a concave footprint (the 97-vertex coal
yard near ENU (786, -2775)) those scaled rings cross the footprint and each other: a starburst in Blender and dark
wedges in the baked web tile. Here the footprint is triangulated properly (constrained Delaunay over the boundary,
densified, plus an interior grid) and each vertex is lifted by its distance to the boundary, i.e. a stack with
angle-of-repose flanks and a flat top where the yard is wide enough:

    z = z0 + min(h, d * tan(repose))

Pure numpy (runs in Blender and in the tests). In Blender `mathutils.geometry.delaunay_2d_cdt` triangulates; outside
Blender a Bowyer-Watson Delaunay over the same points is used and triangles whose centroid falls outside the
footprint are dropped (the boundary is densified to the grid step, so no triangle bridges a concavity).
"""

from __future__ import annotations

import math

import numpy as np

REPOSE_DEG = 35.0          # coal / coke angle of repose (typ. 32-38 deg)


def signed_area(ring) -> float:
    P = np.asarray(ring, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_convex(ring, tol: float = 0.98) -> bool:
    """True when the footprint is (nearly) convex: area / convex-hull area >= tol."""
    hull = convex_hull(ring)
    ha = abs(signed_area(hull))
    return ha <= 0 or abs(signed_area(ring)) / ha >= tol


def convex_hull(ring) -> list[tuple[float, float]]:
    pts = sorted(set(map(tuple, np.asarray(ring, float).tolist())))
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def inside(pts: np.ndarray, ring) -> np.ndarray:
    """Even-odd point-in-polygon for many points."""
    P = np.asarray(ring, float)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = P[:, 0][None, :], P[:, 1][None, :]
    x1, y1 = np.roll(P[:, 0], -1)[None, :], np.roll(P[:, 1], -1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return (np.count_nonzero(cross, axis=1) % 2) == 1


def dist_to_boundary(pts: np.ndarray, ring) -> np.ndarray:
    P = np.asarray(ring, float)
    a, b = P, np.roll(P, -1, axis=0)
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(2) / L2[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - q) ** 2).sum(2)).min(1)


def densify(ring, step: float) -> np.ndarray:
    P = np.asarray(ring, float)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        out += [a + (b - a) * k / n for k in range(n)]
    return np.asarray(out)


def _bowyer_watson(pts: np.ndarray) -> list[tuple[int, int, int]]:
    """Plain Delaunay triangulation (pure python; small inputs and tests)."""
    lo, hi = pts.min(0), pts.max(0)
    c, s = (lo + hi) / 2, float(max(hi - lo)) * 20 + 1.0
    sup = np.array([[c[0] - s, c[1] - s], [c[0] + s, c[1] - s], [c[0], c[1] + s]])
    V = np.vstack([pts, sup])
    n = len(pts)
    tris = [(n, n + 1, n + 2)]

    def circ(t):
        (ax, ay), (bx, by), (cx, cy) = V[t[0]], V[t[1]], V[t[2]]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2

    cc = {tris[0]: circ(tris[0])}
    for i in range(n):
        px, py = V[i]
        bad = [t for t in tris if (px - cc[t][0]) ** 2 + (py - cc[t][1]) ** 2 < cc[t][2] * (1 - 1e-12)]
        edges: dict[tuple[int, int], int] = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                k = tuple(sorted(e))
                edges[k] = edges.get(k, 0) + 1
        bs = set(bad)
        tris = [t for t in tris if t not in bs]
        for t in bad:
            cc.pop(t, None)
        for (a, b), m in edges.items():
            if m == 1:
                t = (a, b, i)
                tris.append(t)
                cc[t] = circ(t)
    return [t for t in tris if max(t) < n]


def _cdt(pts: np.ndarray, n_boundary: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    try:
        from mathutils import Vector
        from mathutils.geometry import delaunay_2d_cdt
    except ImportError:
        return pts, _bowyer_watson(pts)
    loop = list(range(n_boundary))
    edges = [(i, (i + 1) % n_boundary) for i in loop]
    vs, _e, fs, *_ = delaunay_2d_cdt([Vector((float(x), float(y))) for x, y in pts], edges, [loop], 1, 1e-6)
    out = np.array([[v[0], v[1]] for v in vs], float)
    tris = []
    for f in fs:                                           # output faces are triangles; fan anything larger
        tris += [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
    return out, tris


def mound(ring, z0: float, h: float, step: float = 6.0, repose_deg: float = REPOSE_DEG):
    """Vertices (x, y, z) and CCW (upward) triangles of a stockpile over `ring` (simple polygon, site ENU, m)."""
    R = np.asarray(ring, float)
    if signed_area(R) < 0:
        R = R[::-1]
    B = densify(R, step)
    lo, hi = R.min(0), R.max(0)
    gx = np.arange(lo[0] + step / 2, hi[0], step)
    gy = np.arange(lo[1] + step / 2, hi[1], step)
    G = np.array([(x, y) for y in gy for x in gx], float).reshape(-1, 2)
    if len(G):
        G = G[inside(G, R)]
        G = G[dist_to_boundary(G, R) > 0.35 * step]       # no slivers against the boundary
    P2, tris = _cdt(np.vstack([B, G]) if len(G) else B, len(B))
    keep = []
    for t in tris:
        a, b, c = P2[t[0]], P2[t[1]], P2[t[2]]
        ar = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if abs(ar) < 1e-6 or not inside(((a + b + c) / 3)[None], R)[0]:
            continue
        keep.append(tuple(t) if ar > 0 else (t[0], t[2], t[1]))
    d = dist_to_boundary(P2, R)
    z = z0 + np.minimum(h, d * math.tan(math.radians(repose_deg)))
    return [(float(x), float(y), float(zz)) for (x, y), zz in zip(P2, z)], keep
