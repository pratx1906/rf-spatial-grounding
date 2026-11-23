"""RF relevance of mesh components (RF convergence plan, Task 3b). numpy only: imported inside Blender by
envbuild/blender/rf_simplify.py (export) and outside Blender by scripts/rf_geom_check.py (independent measurement on
the full PLY export), so the export and the check share ONE definition of "cross-section".

At 0.9 / 2.44 / 5.8 GHz (lambda = 33 / 12 / 5 cm) a member whose cross-section is below ~lambda/2 of the highest
band scatters weakly, but it is cut into many small facets, and every facet is a chance for a ray to hit or miss
(ray-luck noise). Wires, handrails, ladder rungs and thin rails go; every shell, wall, roof, deck, pipe and member
whose cross-section reaches the threshold stays.

Cross-section of one connected component (`cross_section`):
1. face-width prefilter: every face of a member with cross-section c lies on its surface, so its width
   (2 area / longest edge) is <= sqrt(2) c; one wider face means a surface, not a thin member (kind "sheet");
2. global oriented box: PCA major axis, then the points projected on the plane normal to it; the cross-section is
   the side of the smallest square enclosing that section (min over orientations of the larger of the two widths;
   `section_size`): the diameter of a round or polygonal tube, the side of a square bar at any rotation (a PCA box
   reads a rotated square up to sqrt(2) too large), min(b, (a + b) / sqrt 2) for an a x b bar. It never grows when
   points are removed, so a partial section (a ball through a tube built from single long triangles, as the jetty
   handrails are) never reads larger than the full one, which the larger side of a minimum-area rectangle can (a
   needle along a diamond's diagonal). Below the threshold: kind "straight";
3. otherwise local: the global value is exact for a straight member but too large for a bent or curved one (a
   handrail ring around a tank has a 15 m global box; a 6 m handrail piece bent at its stations reads 0.06-0.10 m).
   Sample the surface and edges densely; in each of up to 64 balls (radius r = 2 x threshold) take the section of
   the ball's points across the local axis, the smaller of two axis estimates: the PCA major axis of the ball itself
   and of a 5 r ball (a 2 r ball of ~70 points tilts its own axis by ~2 deg; a 5 r ball averages the two directions
   at a station bend, 22.5 deg on a 2.4 m tank-roof handrail arc, and both inflate a 4 cm section by up to 50 %).
   Balls with fewer than 8 points are skipped. Bends and junctions only ever INFLATE a ball's section, so the
   member's section is the low end of the distribution: local = the 25th percentile over the balls (a component is
   dropped only when a quarter of its sampled area already reads at least that thin, so a 0.15 m pipe carrying thin
   brackets stays). cross = min(global, local): a sampled ball is a subset of the section, so for a straight member
   the local value is <= the exact global one (within ~1e-4 m: section corners are mesh edges and are sampled).
   Kind "curved" when local < 0.9 x global, else "prism".

`slab_top_faces` recognises a closed horizontal slab (a 0.12 m ML-building roof box) and returns its top faces: the
top keeps its exact area; bottom and sides add nothing to RF beyond the top and its edges.
"""

from __future__ import annotations

import numpy as np

SQRT2 = 2 ** 0.5
_ANG = np.radians(np.arange(360.0))
_DIRS = np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1)


def _hull(p: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point."""
    p = np.unique(np.round(p, 7), axis=0)
    if len(p) < 3:
        return p
    pts = [tuple(x) for x in p]

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out

    lo, up = half(pts), half(reversed(pts))
    return np.array(lo[:-1] + up[:-1])


def min_rect(p2: np.ndarray) -> tuple[float, float]:
    """(short, long) sides of the minimum-area rectangle enclosing 2D points (rotating calipers on the hull)."""
    p2 = np.asarray(p2, float)
    if len(p2) > 64:            # hull vertices with a normal cone > 1 deg are extreme along one of 360 directions
        p2 = p2[np.unique(np.argmax(p2 @ _DIRS.T, axis=0))]
    h = _hull(p2)
    if len(h) == 0:
        return 0.0, 0.0
    if len(h) == 1:
        return 0.0, 0.0
    if len(h) == 2:
        return 0.0, float(np.linalg.norm(h[1] - h[0]))
    e = np.roll(h, -1, axis=0) - h
    L = np.linalg.norm(e, axis=1)
    e, L = e[L > 1e-12], L[L > 1e-12]
    u = e / L[:, None]
    w = np.stack([-u[:, 1], u[:, 0]], axis=1)
    eu, ew = np.ptp(h @ u.T, axis=0), np.ptp(h @ w.T, axis=0)
    k = int(np.argmin(eu * ew))
    a, b = float(eu[k]), float(ew[k])
    return min(a, b), max(a, b)


def section_size(p2: np.ndarray) -> tuple[float, float]:
    """(cross, other): cross = side of the smallest enclosing square of 2D points = min over orientations of
    max(width_u, width_w); other = the smaller width at that orientation. Candidates: the edge-flush orientations
    (exact for squares and rectangles) and a 0.25 deg grid, zoomed four times (orientation error < 1e-7 deg)."""
    p2 = np.asarray(p2, float)
    if len(p2) > 64:
        p2 = p2[np.unique(np.argmax(p2 @ _DIRS.T, axis=0))]
    h = _hull(p2)
    if len(h) < 2:
        return 0.0, 0.0

    def ext(ang):
        u = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        w = np.stack([-u[:, 1], u[:, 0]], axis=1)
        eu, ew = np.ptp(h @ u.T, axis=0), np.ptp(h @ w.T, axis=0)
        return np.maximum(eu, ew), np.minimum(eu, ew)

    e = np.roll(h, -1, axis=0) - h if len(h) > 2 else h[1:] - h[:1]
    flush = np.mod(np.arctan2(e[:, 1], e[:, 0]), np.pi / 2)          # edge-flush orientations (width minima)
    ang = np.concatenate([np.radians(np.arange(0.0, 90.0, 0.25)), flush])
    mx, _ = ext(ang)
    a0, step = ang[int(np.argmin(mx))], np.radians(0.25)
    for _ in range(4):                                               # zoom on the optimum: 0.25 deg / 50^4
        ang = np.concatenate([[a0], a0 + np.linspace(-step, step, 101)])
        mx, _ = ext(ang)
        a0, step = ang[int(np.argmin(mx))], step / 50
    mx, mn = ext(np.array([a0]))
    return float(mx[0]), float(mn[0])


def _major_axis(p: np.ndarray) -> np.ndarray:
    c = p - p.mean(0)
    return np.linalg.eigh(c.T @ c)[1][:, 2]


def section_across(p: np.ndarray, axis: np.ndarray) -> float:
    """Cross-section (section_size) of points projected on the plane normal to `axis`."""
    axis = axis / np.linalg.norm(axis)
    b1 = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    c = p - p.mean(0)
    return section_size(np.stack([c @ b1, c @ b2], axis=1))[0]


def obb_cross(p: np.ndarray) -> tuple[float, float, float]:
    """(length along the PCA major axis, cross-section = smallest enclosing square side of the section (see
    section_size), the smaller section width at that orientation)."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return 0.0, 0.0, 0.0
    c = p - p.mean(0)
    _, vec = np.linalg.eigh(c.T @ c)
    ax, b1, b2 = vec[:, 2], vec[:, 1], vec[:, 0]
    length = float(np.ptp(c @ ax))
    cross, other = section_size(np.stack([c @ b1, c @ b2], axis=1))
    return length, cross, other


def tri_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def face_widths(v: np.ndarray, f: np.ndarray, areas: np.ndarray | None = None) -> np.ndarray:
    """2 area / longest edge per triangle (the triangle's height over its longest side)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    lmax = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
                      np.linalg.norm(a - c, axis=1))
    ar = tri_areas(v, f) if areas is None else areas
    return np.where(lmax > 0, 2 * ar / np.maximum(lmax, 1e-30), 0.0)


def _samples(v, f, areas, rng, per_m2: float, per_m: float, cap: int) -> np.ndarray:
    """Area-weighted surface points plus points along every triangle edge (the section corners of a faceted tube are
    edges: without them the sampled hull underestimates the section)."""
    tot = float(areas.sum())
    n = int(min(cap, max(400, tot * per_m2)))
    t = rng.choice(len(f), size=n, p=areas / tot)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = v[f[t, 0]], v[f[t, 1]], v[f[t, 2]]
    surf = (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    p0, p1 = v[e[:, 0]], v[e[:, 1]]
    el = np.linalg.norm(p1 - p0, axis=1)
    m = int(min(cap, max(200, el.sum() * per_m)))
    k = rng.choice(len(e), size=m, p=el / el.sum()) if el.sum() > 0 else np.zeros(0, int)
    tt = rng.random(len(k))[:, None]
    return np.concatenate([surf, p0[k] + tt * (p1[k] - p0[k]), v])


def cross_section(v: np.ndarray, f: np.ndarray, min_cross_m: float, rng=None, n_balls: int = 64,
                  min_pts: int = 8) -> dict:
    """Cross-section of ONE connected component (vertices v [n,3], triangles f [m,3]); see the module docstring.
    Returns {cross, kind, length, area}; `cross` is exact for kinds straight/prism/curved, a lower bound for sheet."""
    v, f = np.asarray(v, float), np.asarray(f, np.int64)
    rng = rng if rng is not None else np.random.default_rng(0)
    areas = tri_areas(v, f) if len(f) else np.zeros(0)
    area = float(areas.sum())
    if len(f) == 0 or area <= 0:
        return {"cross": float("nan"), "kind": "empty", "length": 0.0, "area": area}
    used = np.unique(f)
    length, gcross, _ = obb_cross(v[used])
    wmax = float(face_widths(v, f, areas)[areas > 1e-12].max(initial=0.0))
    if wmax > SQRT2 * min_cross_m * 1.02:
        return {"cross": max(gcross, wmax / SQRT2), "kind": "sheet", "length": length, "area": area}
    if gcross < min_cross_m:
        return {"cross": gcross, "kind": "straight", "length": length, "area": area}
    r = 2 * min_cross_m
    p = _samples(v, f, areas, rng, per_m2=1500.0, per_m=300.0, cap=150_000)
    cen = p[rng.choice(len(p), size=min(n_balls, len(p)), replace=False)]
    loc = []
    for c in cen:
        d2 = np.einsum("ij,ij->i", p - c, p - c)
        q, big = p[d2 <= r * r], p[d2 <= (5 * r) ** 2]
        if len(q) >= min_pts:
            loc.append(min(section_across(q, _major_axis(big)), section_across(q, _major_axis(q))))
    if not loc:
        return {"cross": gcross, "kind": "prism", "length": length, "area": area}
    med = float(np.percentile(loc, 25))
    return {"cross": min(gcross, med), "kind": "curved" if med < 0.9 * gcross else "prism", "length": length,
            "area": area}


def removal_class(kind: str, length: float) -> str:
    """Report class of a removed thin component."""
    if kind == "curved":
        return "ring/loop/curved rail"
    return "wire/rod" if length >= 2.0 else "post/rung/short bar"


def slab_top_faces(v: np.ndarray, f: np.ndarray, max_t: float = 0.15, min_top_m2: float = 0.25,
                   tol: float = 1e-4) -> np.ndarray | None:
    """Boolean mask of the top faces if (v, f) is a closed horizontal slab: z range in (0, max_t], every face
    horizontal (at z max or z min) or vertical, top area == bottom area (within 0.1 %) >= min_top_m2. Else None."""
    v, f = np.asarray(v, float), np.asarray(f, np.int64)
    if len(f) < 4:
        return None
    z = v[f][:, :, 2]
    z0, z1 = float(z.min()), float(z.max())
    if not (0 < z1 - z0 <= max_t):
        return None
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 0
    nz = np.zeros(len(f))
    nz[ok] = n[ok, 2] / ln[ok]
    horiz, vert = np.abs(nz) > 0.999, np.abs(nz) < 0.001
    if not np.all(horiz | vert | ~ok):
        return None
    top = horiz & (np.abs(z - z1) < tol).all(1)
    bot = horiz & (np.abs(z - z0) < tol).all(1)
    if not np.all(top | bot | ~horiz):
        return None
    ar = ln / 2
    at, ab = float(ar[top].sum()), float(ar[bot].sum())
    if at < min_top_m2 or abs(at - ab) > 1e-3 * at:
        return None
    return top
