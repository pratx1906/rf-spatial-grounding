"""numpy-side tests of envbuild/blender/rf_relevance (RF convergence plan, Task 3b): the cross-section measurement
shared by the Blender export and scripts/rf_geom_check.py --removed-ok."""

import math

import numpy as np

from envbuild.blender.rf_relevance import cross_section, min_rect, obb_cross, section_size, slab_top_faces


def _box(size, center=(0, 0, 0), rot_z=0.0):
    sx, sy, sz = (s / 2 for s in size)
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], float)
    c, s = math.cos(rot_z), math.sin(rot_z)
    v = v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + np.asarray(center, float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))], np.int64)
    return v, f


def _torus(R, r, nu=64, nv=8):
    v = np.array([[(R + r * math.cos(2 * math.pi * j / nv)) * math.cos(2 * math.pi * i / nu),
                   (R + r * math.cos(2 * math.pi * j / nv)) * math.sin(2 * math.pi * i / nu),
                   r * math.sin(2 * math.pi * j / nv)] for i in range(nu) for j in range(nv)])
    f = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, ((i + 1) % nu) * nv + j
            f += [(a, b, c), (a, c, d)]
    return v, np.array(f, np.int64)


def test_min_rect_rotated_square_is_exact():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) * 0.04
    rot = np.array([[math.cos(0.6), -math.sin(0.6)], [math.sin(0.6), math.cos(0.6)]])
    a, b = min_rect(sq @ rot.T)
    assert abs(a - 0.04) < 1e-6 and abs(b - 0.04) < 1e-6          # hull points rounded to 0.1 um


def test_section_size_square_circle_and_subsets():
    rot = np.array([[math.cos(0.6), -math.sin(0.6)], [math.sin(0.6), math.cos(0.6)]])
    sq = (np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) * 0.04) @ rot.T
    assert abs(section_size(sq)[0] - 0.04) < 1e-7
    circ = 0.03 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200))], 1)
    assert abs(section_size(circ)[0] - 0.06) < 1e-4
    diamond = np.array([[0.03, 0], [0, 0.03], [-0.03, 0], [0, -0.03]])
    needle = diamond[[0, 2]]                                            # a partial section along the diagonal
    assert section_size(needle)[0] <= section_size(diamond)[0] + 1e-9 < min_rect(needle)[1]


def test_obb_cross_rotated_post():
    v, _ = _box((0.045, 0.045, 1.1), rot_z=0.7)          # a PCA box would read up to 0.064
    length, cross, small = obb_cross(v)
    assert abs(length - 1.1) < 1e-6 and abs(cross - 0.045) < 1e-6 and abs(small - 0.045) < 1e-6


def test_cross_section_kinds():
    rng = np.random.default_rng(0)
    v, f = _box((30, 0.02, 0.02))
    assert cross_section(v, f, 0.05, rng)["kind"] == "straight"
    v, f = _torus(10.3, 0.02)
    cs = cross_section(v, f, 0.05, rng)
    assert cs["kind"] == "curved" and 0.03 < cs["cross"] < 0.05, cs
    v, f = _box((1.0, 0.3, 0.05))
    cs = cross_section(v, f, 0.05, rng)
    assert abs(cs["cross"] - 0.35 / 2 ** 0.5) < 1e-4, cs                # 0.3 x 0.05 bar: min(b, (a + b) / sqrt 2)
    v, f = _box((40, 10, 8))
    assert cross_section(v, f, 0.05, rng)["kind"] == "sheet"


def test_slab_top_faces():
    v, f = _box((12.4, 8.4, 0.12), center=(0, 0, 6.06))
    top = slab_top_faces(v, f)
    assert top is not None and top.sum() == 2
    vv, ff = _box((12, 8, 6))
    assert slab_top_faces(vv, ff) is None
