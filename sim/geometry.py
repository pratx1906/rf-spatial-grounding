"""2D geometry in the local ENU frame (metres). Bearings are degrees clockwise from true north (SPEC 4)."""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon


def bearing_deg(x0: float, y0: float, x1: float, y1: float) -> float:
    """Bearing from (x0, y0) to (x1, y1): 0 = north (+y), 90 = east (+x)."""
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference a - b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def ellipse_polygon(x: float, y: float, a_m: float, b_m: float, theta_deg: float, n: int = 32) -> Polygon:
    """Uncertainty ellipse; theta_deg is the bearing of the semi-major axis (clockwise from north)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ex, ey = a_m * np.cos(t), b_m * np.sin(t)          # major axis along local +x' before rotation
    th = math.radians(90.0 - theta_deg)                # bearing -> math angle
    xs = x + ex * math.cos(th) - ey * math.sin(th)
    ys = y + ex * math.sin(th) + ey * math.cos(th)
    return Polygon(np.column_stack([xs, ys]))


def ray_height_at(z0: float, z1: float, frac: float) -> float:
    return z0 + (z1 - z0) * frac


def los_clear(p0: tuple[float, float, float], p1: tuple[float, float, float],
              buildings: list[tuple[Polygon, float]], tree=None) -> bool:
    """True if the 3D segment p0->p1 clears every building (footprint, height) it crosses in 2D.

    The ray's height is interpolated linearly; a building blocks if the ray is below its roof where it enters it.
    """
    seg = LineString([p0[:2], p1[:2]])
    length = seg.length
    if length == 0:
        return True
    candidates = tree.query(seg) if tree is not None else range(len(buildings))
    for i in candidates:
        poly, h = buildings[int(i)]
        if not seg.intersects(poly):
            continue
        if poly.contains(Point(p0[:2])) or poly.contains(Point(p1[:2])):
            continue  # sensor mounted on, or emitter above, this building
        inter = seg.intersection(poly)
        for cx, cy in shapely.get_coordinates(inter):   # works for lines, points, multi-part and collections
            frac = math.hypot(cx - p0[0], cy - p0[1]) / length
            if ray_height_at(p0[2], p1[2], frac) < h:
                return False
    return True


def mirror_point(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    """Mirror of P across the infinite line through A and B."""
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    fx, fy = ax + t * dx, ay + t * dy
    return 2 * fx - px, 2 * fy - py


def specular_point(sx: float, sy: float, ex: float, ey: float,
                   ax: float, ay: float, bx: float, by: float) -> tuple[float, float] | None:
    """Single-bounce reflection point on wall segment AB for sensor S and emitter E, if it exists.

    Uses the image method: reflect E across AB; the specular point is where S->E' crosses AB. Both S and E must be
    on the same side of the wall.
    """
    def side(px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    s_side, e_side = side(sx, sy), side(ex, ey)
    if s_side == 0 or e_side == 0 or (s_side > 0) != (e_side > 0):
        return None
    mx, my = mirror_point(ex, ey, ax, ay, bx, by)
    # intersection of S->M with segment AB
    d1x, d1y = mx - sx, my - sy
    d2x, d2y = bx - ax, by - ay
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-9:
        return None
    t = ((ax - sx) * d2y - (ay - sy) * d2x) / den
    u = ((ax - sx) * d1y - (ay - sy) * d1x) / den
    if 0.0 < t < 1.0 and 0.02 < u < 0.98:  # strictly inside the facade, away from corners
        return sx + t * d1x, sy + t * d1y
    return None


def facades(poly: Polygon) -> list[tuple[float, float, float, float]]:
    """Exterior wall segments of a footprint polygon."""
    c = list(poly.exterior.coords)
    return [(c[i][0], c[i][1], c[i + 1][0], c[i + 1][1]) for i in range(len(c) - 1)]


def fspl_db(dist_m: float, freq_mhz: float) -> float:
    """Free-space path loss (Friis), dB."""
    d_km = max(dist_m, 1.0) / 1000.0
    return 20 * math.log10(d_km) + 20 * math.log10(freq_mhz) + 32.44


def stable_seed(*parts) -> int:
    """Process-independent seed from any parts (builtin hash() of str is randomised per process)."""
    import zlib

    return zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF
