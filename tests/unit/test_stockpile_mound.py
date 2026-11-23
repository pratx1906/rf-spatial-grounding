"""Stockpile mounds over concave footprints (demo-v2 S6): proper triangulation, no starburst."""

import json
from pathlib import Path

import numpy as np
import pytest

from envbuild.blender import mound as M

REPO = Path(__file__).resolve().parents[2]
L_SHAPE = [(0, 0), (60, 0), (60, 20), (20, 20), (20, 60), (0, 60)]


def _check(ring, z0, h, step):
    V, T = M.mound(ring, z0, h, step)
    V = np.asarray(V)
    area = 0.0
    for a, b, c in T:
        p, q, r = V[a, :2], V[b, :2], V[c, :2]
        ar = 0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))
        assert ar > 0, "every triangle faces up (CCW)"
        area += ar
        for u, v in ((p, q), (q, r), (r, p)):          # no edge leaves the footprint (bridging a concavity)
            mid = (u + v) / 2
            assert M.inside(mid[None], ring)[0] or M.dist_to_boundary(mid[None], ring)[0] < 1e-6
    assert area == pytest.approx(abs(M.signed_area(ring)), rel=1e-6)
    d = M.dist_to_boundary(V[:, :2], ring)
    assert np.all(V[d < 1e-6, 2] == pytest.approx(z0))          # the toe sits on the ground
    assert V[:, 2].max() <= z0 + h + 1e-9 and V[:, 2].min() >= z0 - 1e-9
    return V, T


def test_l_shape_concave():
    assert not M.is_convex(L_SHAPE)
    V, T = _check(L_SHAPE, 2.0, 8.0, 4.0)
    # the inner corner (20, 20) is on the boundary: no vertex is lifted there, and the old starburst (a ring scaled
    # toward the vertex mean, which crosses the footprint) cannot occur: nothing lies outside it
    assert V[:, 2].max() == pytest.approx(2.0 + min(8.0, 10.0 * np.tan(np.radians(M.REPOSE_DEG))), abs=0.6)


def test_kandla_coal_yard_footprint():
    d = json.loads((REPO / "data" / "site_kandla" / "blender" / "s2_structures.json").read_text())
    ring = max((p["outer"] for p in d["stockpiles"]), key=len)       # the 97-vertex yard near ENU (786, -2775)
    assert len(ring) > 50 and not M.is_convex(ring)
    _check(ring, 1.5, 12.0, 14.0)


def test_convex_box_flat_top():
    ring = [(0, 0), (100, 0), (100, 50), (0, 50)]
    assert M.is_convex(ring)
    V, _ = _check(ring, 0.0, 6.0, 5.0)
    assert V[:, 2].max() == pytest.approx(6.0)                     # wide enough for the full height
