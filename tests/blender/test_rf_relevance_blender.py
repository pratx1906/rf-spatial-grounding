"""Blender-side tests of the RF-relevance pass (RF convergence plan, Task 3b): simplify_bmesh(..., rf_relevance=...)
drops components whose cross-section is below min_cross_m and collapses closed roof slabs to their top face.

    blender -b --factory-startup --python tests/blender/test_rf_relevance_blender.py

Run by tests/unit/test_rf_simplify.py (skipped without Blender). Never saves preferences or files. Prints one
`PASS <name>` / `FAIL <name>: <why>` line per case and `RESULT <passed>/<total>`; exit code 0 only if all pass.
"""

import math
import sys
import traceback
from pathlib import Path

import bmesh
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

REL = {"min_cross_m": 0.05}


def _simplify(bm, rel=REL, **kw):
    from envbuild.blender.rf_simplify import simplify_bmesh
    return simplify_bmesh(bm, angle_deg=1.0, min_extent_m=0.10, rf_relevance=rel, **kw)


def _bbox(bm):
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _area(bm):
    return sum(f.calc_area() for f in bm.faces)


def _box(bm, size, center=(0, 0, 0), rot_z=0.0):
    r = bmesh.ops.create_cube(bm, size=1.0)
    vs = r["verts"]
    bmesh.ops.scale(bm, vec=Vector(size), verts=vs)
    if rot_z:
        bmesh.ops.rotate(bm, verts=vs, cent=Vector(), matrix=Matrix.Rotation(rot_z, 3, "Z"))
    bmesh.ops.translate(bm, vec=Vector(center), verts=vs)
    return vs


def _tube(bm, radius, length, segments=8, center=(0, 0, 0), axis="X"):
    r = bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=length)
    if axis == "X":
        bmesh.ops.rotate(bm, verts=r["verts"], cent=Vector(), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"))
    bmesh.ops.translate(bm, vec=Vector(center), verts=r["verts"])
    return r["verts"]


def _ring(bm, major_r, minor_r, z, major_seg=64, minor_seg=8):
    """A torus (a handrail ring) built from quads: major_seg stations of a minor_seg-gon section."""
    rows = []
    for i in range(major_seg):
        a = 2 * math.pi * i / major_seg
        ca, sa = math.cos(a), math.sin(a)
        row = []
        for j in range(minor_seg):
            b = 2 * math.pi * j / minor_seg
            rr = major_r + minor_r * math.cos(b)
            row.append(bm.verts.new((rr * ca, rr * sa, z + minor_r * math.sin(b))))
        rows.append(row)
    for i in range(major_seg):
        r0, r1 = rows[i], rows[(i + 1) % major_seg]
        for j in range(minor_seg):
            bm.faces.new((r0[j], r0[(j + 1) % minor_seg], r1[(j + 1) % minor_seg], r1[j]))


def test_wire_2cm_30m_dropped():
    bm = bmesh.new()
    _box(bm, (20, 20, 6))                                             # a building the wire passes over
    _tube(bm, 0.01, 30.0, segments=6, center=(0, 0, 12))              # 2 cm x 30 m wire
    a0 = _area(bm)
    st = _simplify(bm)
    bb = _bbox(bm)
    assert bb[5] < 3.5 and abs((bb[3] - bb[0]) - 20) < 1e-4, f"wire kept: {bb}"
    assert st["thin_removed"] == 1 and st["thin_max_cross_m"] < 0.05, st
    assert abs(_area(bm) - (a0 - st["thin_removed_area_m2"])) < 1e-3, (a0, _area(bm), st)


def test_handrail_ring_on_tank_dropped_shell_kept():
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=10.0, radius2=10.0, depth=12.0)   # tank shell
    shell_area = _area(bm)
    _ring(bm, 10.3, 0.02, z=6.5)                                      # 4 cm handrail ring just outside the top
    st = _simplify(bm)
    assert st["thin_removed"] == 1, st
    assert 0.03 < st["thin_max_cross_m"] < 0.05, st
    bb = _bbox(bm)
    assert max(bb[3], -bb[0]) <= 10.0 + 1e-4 and bb[5] <= 6.0 + 1e-4, f"ring kept: {bb}"
    assert abs(_area(bm) - shell_area) < 1e-3 * shell_area, (_area(bm), shell_area)


def test_pipe_15cm_kept():
    bm = bmesh.new()
    _tube(bm, 0.075, 30.0, segments=12)
    a0 = _area(bm)
    st = _simplify(bm)
    assert st["thin_removed"] == 0, st
    bb = _bbox(bm)
    assert abs((bb[3] - bb[0]) - 30.0) <= 0.01 and abs(_area(bm) - a0) < 1e-3 * a0, (bb, _area(bm), a0)


def test_ladder_3cm_rungs_dropped():
    bm = bmesh.new()
    _box(bm, (4, 4, 10), center=(0, 0, 5))                            # the wall the ladder is fixed to
    wall_area = _area(bm)
    for x in (-0.225, 0.225):                                         # two 3 cm stiles, 8 m tall
        _box(bm, (0.03, 0.03, 8.0), center=(x, 2.2, 4.5))
    for k in range(25):                                               # 25 rungs of 3 cm, every 0.3 m
        _box(bm, (0.45, 0.03, 0.03), center=(0, 2.2, 1.0 + 0.3 * k), rot_z=0.0)
    st = _simplify(bm)
    assert st["thin_removed"] == 27, st
    assert st["thin_max_cross_m"] < 0.05, st
    assert abs(_area(bm) - wall_area) < 1e-3 * wall_area, (_area(bm), wall_area)
    assert _bbox(bm)[4] <= 2.0 + 1e-4, _bbox(bm)


def test_roof_slab_keeps_exact_top_area():
    bm = bmesh.new()
    _box(bm, (12, 8, 6), center=(0, 0, 3))                            # the building
    _box(bm, (12.4, 8.4, 0.12), center=(0, 0, 6.06))                  # 0.12 m roof slab on top of it
    top = 12.4 * 8.4
    st = _simplify(bm, rel={"min_cross_m": 0.05, "slab_top": True})
    assert st["slabs_collapsed"] == 1, st
    slab = [f for f in bm.faces if min(v.co.z for v in f.verts) > 6.0 + 1e-4]
    assert slab and all(abs(f.normal.z) > 0.999 for f in slab), "slab sides/bottom kept"
    a = sum(f.calc_area() for f in slab)
    assert abs(a - top) < 1e-6 * top, (a, top)
    assert all(abs(v.co.z - 6.12) < 1e-5 for f in slab for v in f.verts), "top moved"
    # without slab_top the slab stays a closed box
    bm2 = bmesh.new()
    _box(bm2, (12.4, 8.4, 0.12), center=(0, 0, 6.06))
    st2 = _simplify(bm2)
    assert st2["slabs_collapsed"] == 0 and abs(_area(bm2) - (2 * top + 2 * 0.12 * (12.4 + 8.4))) < 1e-3, st2


def test_slab_thick_member_kept():
    bm = bmesh.new()
    _box(bm, (6, 0.12, 0.12), center=(0, 0, 2))                       # a 12 cm beam: above the threshold
    _box(bm, (1.0, 0.3, 0.05), center=(0, 3, 0))                      # a 1.0 x 0.3 x 0.05 m tread: section 0.3 m
    st = _simplify(bm)
    assert st["thin_removed"] == 0, st
    assert len(bm.faces) >= 12, len(bm.faces)


def main() -> int:
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for c in cases:
        try:
            c()
            ok += 1
            print(f"PASS {c.__name__}", flush=True)
        except Exception as e:       # report every case, not only the first failure
            print(f"FAIL {c.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"RESULT {ok}/{len(cases)}", flush=True)
    return 0 if ok == len(cases) else 1


sys.exit(main())
