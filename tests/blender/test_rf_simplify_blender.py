"""Blender-side tests of envbuild/blender/rf_simplify.simplify_bmesh (RF convergence plan, Task 3).

    blender -b --factory-startup --python tests/blender/test_rf_simplify_blender.py

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


def _simplify(bm, **kw):
    from envbuild.blender.rf_simplify import simplify_bmesh
    return simplify_bmesh(bm, **kw)


def _bbox(bm):
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _tris(bm):
    return sum(len(f.verts) - 2 for f in bm.faces)


def _box(bm, size, center=(0, 0, 0), cuts=0):
    r = bmesh.ops.create_cube(bm, size=1.0)
    vs = r["verts"]
    bmesh.ops.scale(bm, vec=Vector(size), verts=vs)
    bmesh.ops.translate(bm, vec=Vector(center), verts=vs)
    if cuts:
        edges = list({e for v in vs for e in v.link_edges})
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts, use_grid_fill=True)
    return vs


def test_box_2000_faces():
    bm = bmesh.new()
    _box(bm, (40, 10, 8), cuts=18)
    assert len(bm.faces) >= 2000, len(bm.faces)
    bb0 = _bbox(bm)
    st = _simplify(bm, angle_deg=1.0, min_extent_m=0.10)
    assert _tris(bm) <= 12, f"{_tris(bm)} tris"
    assert all(len(f.verts) == 3 for f in bm.faces), "not triangulated"
    assert max(abs(a - b) for a, b in zip(bb0, _bbox(bm))) < 1e-4, (bb0, _bbox(bm))
    assert st["tris_before"] >= 4000 and st["tris_after"] == _tris(bm), st


def test_thin_long_pipe_survives():
    bm = bmesh.new()
    r = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.075, radius2=0.075, depth=30.0)
    bmesh.ops.rotate(bm, verts=r["verts"], cent=Vector(), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"))
    st = _simplify(bm, angle_deg=1.0, min_extent_m=0.10)
    assert len(bm.faces) > 0, "pipe removed"
    bb = _bbox(bm)
    assert abs((bb[3] - bb[0]) - 30.0) <= 0.01, bb
    assert abs((bb[4] - bb[1]) - 0.15) <= 0.01 and st["removed_components"] == 0, (bb, st)


def test_cylinder_radius_kept():
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=20.0, radius2=20.0, depth=15.0)
    _simplify(bm, angle_deg=1.0, min_extent_m=0.10)
    rs = [math.hypot(v.co.x, v.co.y) for v in bm.verts if abs(v.co.z) > 1.0]   # rim vertices
    assert len(rs) >= 96, len(rs)
    assert all(abs(r - 20.0) <= 0.1 for r in rs), (min(rs), max(rs))
    # side faces (normal horizontal) must still span the full circle
    side = [f for f in bm.faces if abs(f.normal.z) < 0.5]
    assert len(side) >= 48, len(side)


def test_bolt_removed():
    bm = bmesh.new()
    _box(bm, (10, 10, 1))
    _box(bm, (0.05, 0.05, 0.05), center=(20, 0, 0))                 # a 5 cm bolt, separate component
    st = _simplify(bm, angle_deg=1.0, min_extent_m=0.10)
    bb = _bbox(bm)
    assert bb[3] < 6.0, f"bolt kept: {bb}"
    assert st["removed_components"] == 1, st
    assert abs((bb[3] - bb[0]) - 10.0) < 1e-4, bb


def test_material_boundary_kept():
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=20, y_segments=10, size=10.0)
    for f in bm.faces:
        f.material_index = 0 if f.calc_center_median().x < 0 else 1
    a0 = {m: sum(f.calc_area() for f in bm.faces if f.material_index == m) for m in (0, 1)}
    _simplify(bm, angle_deg=1.0, min_extent_m=0.10)
    a1 = {m: sum(f.calc_area() for f in bm.faces if f.material_index == m) for m in (0, 1)}
    assert set(f.material_index for f in bm.faces) == {0, 1}, "material groups merged"
    assert all(abs(a0[m] - a1[m]) < 1e-3 for m in (0, 1)), (a0, a1)
    for f in bm.faces:                                               # no face straddles the material boundary
        xs = [v.co.x for v in f.verts]
        assert (max(xs) <= 1e-4) or (min(xs) >= -1e-4), xs
    assert len(bm.faces) <= 8, len(bm.faces)


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
