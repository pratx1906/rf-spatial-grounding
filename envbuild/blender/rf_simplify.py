"""Simplified RF copy of the Blender geometry (RF convergence plan, Task 3; bmesh only, importable inside Blender).

simplify_bmesh runs on a mesh already in the site frame (metres, after the world transform), in this order:
1. weld vertices closer than `weld_m` (< 1 mm by default: a larger weld fuses the ~1 cm double skins of thin walls),
   within one material only: coincident faces of two materials (an ML-building wall carries a steel cladding face on
   its concrete face) would otherwise fuse into one face of either material;
2. dissolve_degenerate (zero-area faces and zero-length edges left by the weld);
3. drop connected components whose bounding-box MAX extent is below `min_extent_m` (bolts and rivets go; a 30 m x
   0.15 m pipe has a 30 m max extent and stays);
4. dissolve_limit on EDGES only, planar (`angle_deg` between face normals), delimited by material so RF material
   classes never merge and per-face material indices survive. Vertices are not passed: dissolve_limit's vertex pass
   removes boundary vertices whose turn is below the same angle and, applied repeatedly, straightens gently curved
   strips (the 8 km rail at z = 0.5 m lost up to 1.6 m of its curve at 1 deg);
5. delete vertices left loose, then dissolve degree-2 vertices within `collinear_m` (1 mm) of the chord between their
   neighbours, iterated (a chord-distance bound, so a gentle curve is never straightened);
6. triangulate (beauty; handles the concave n-gons a planar dissolve can create);
7. area guard: every connected component's surface area must survive within `area_tol` (planar merges conserve area;
   the 1 deg tilt costs < 0.02 %). A component that changed more (a merged n-gon whose triangulation over-fills its
   outline, or a wall lost in a dissolve; both seen on ML-building tiles) is restored from its unsimplified copy.

With `rf_relevance={"min_cross_m": 0.05, "slab_top": bool}` (Task 3b) an RF-relevance pass runs between steps 3 and 4
(rf_relevance_pass): components whose measured cross-section (envbuild/blender/rf_relevance.cross_section) is below
min_cross_m are dropped (wires, handrails, ladder rungs, thin rails, rings), and with slab_top closed horizontal slabs
<= 0.15 m thick keep only their top faces (exact top area).

Curved surfaces keep their facets as long as the facet angle exceeds `angle_deg` (a 48-gon tank shell has 7.5 deg
between facets, so it is untouched at 1 deg). Larger angles on curved collections are only allowed when
scripts/rf_geom_check.py passes for them (radius error < 0.1 m).
"""

import math

import bmesh
import numpy as np


def _tri_count(bm) -> int:
    return sum(len(f.verts) - 2 for f in bm.faces)


def _components(bm) -> list[list]:
    """Connected components over edges (union-find on vertex indices); loose vertices are their own component."""
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    parent = list(range(len(bm.verts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in bm.edges:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        if a != b:
            parent[a] = b
    comps: dict[int, list] = {}
    for v in bm.verts:
        comps.setdefault(find(v.index), []).append(v)
    return list(comps.values())


def drop_small_components(bm, min_extent_m: float) -> int:
    """Delete components whose bbox max extent < min_extent_m; returns how many were removed."""
    if min_extent_m <= 0 or not bm.verts:
        return 0
    doomed, n = [], 0
    for comp in _components(bm):
        xs = [v.co.x for v in comp]
        ys = [v.co.y for v in comp]
        zs = [v.co.z for v in comp]
        ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if ext < min_extent_m:
            doomed.extend(comp)
            n += 1
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    return n


def dissolve_collinear(bm, tol_m: float, max_iter: int = 50) -> int:
    """Dissolve degree-2 vertices lying within tol_m of the chord between their two neighbours, repeated until
    nothing changes (one simultaneous pass leaves some members of a collinear run behind). Distance, not angle: an
    angle threshold lets long chords drift from gentle curves; a chord tolerance bounds the deviation to a few tol_m."""
    total = 0
    for _ in range(max_iter):
        doomed, taken = [], set()
        for v in bm.verts:
            if len(v.link_edges) != 2 or v in taken:
                continue
            p, q = (e.other_vert(v) for e in v.link_edges)
            if p in taken or q in taken:          # judge each vertex against neighbours that stay this round
                continue
            d = q.co - p.co
            L2 = d.length_squared
            if L2 == 0:
                continue
            t = max(0.0, min(1.0, (v.co - p.co).dot(d) / L2))
            if (p.co + t * d - v.co).length <= tol_m:
                doomed.append(v)
                taken.add(v)
        if not doomed:
            break
        bmesh.ops.dissolve_verts(bm, verts=doomed, use_face_split=False, use_boundary_tear=False)
        total += len(doomed)
    return total


def _comp_layer(bm):
    """Tag every face with its connected component id (an int face layer that survives dissolves)."""
    lay = bm.faces.layers.int.get("rf_comp") or bm.faces.layers.int.new("rf_comp")
    vid = {}
    for k, comp in enumerate(_components(bm)):
        for v in comp:
            vid[v] = k
    for f in bm.faces:
        f[lay] = vid[f.verts[0]]
    return lay


def _area_by(bm, lay) -> dict:
    out: dict = {}
    for f in bm.faces:
        out[f[lay]] = out.get(f[lay], 0.0) + f.calc_area()
    return out


def _restore(bm, orig, lay, bad: set) -> None:
    """Replace the components `bad` in bm by their copies from `orig` (same layers), triangulated."""
    gone = [f for f in bm.faces if f[lay] in bad]
    if gone:
        bmesh.ops.delete(bm, geom=gone, context="FACES")
    olay = orig.faces.layers.int.get("rf_comp")
    vmap, new = {}, []
    for f in orig.faces:
        if f[olay] not in bad:
            continue
        vs = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = bm.verts.new(v.co)
            vs.append(vmap[v])
        try:
            nf = bm.faces.new(vs)
        except ValueError:                     # face already exists (duplicate in the source): keep one
            continue
        nf.material_index = f.material_index
        nf[lay] = f[olay]
        new.append(nf)
    if new:
        bmesh.ops.triangulate(bm, faces=new, quad_method="BEAUTY", ngon_method="BEAUTY")


def rf_relevance_pass(bm, min_cross_m: float = 0.05, slab_top: bool = False, seed: int = 0) -> dict:
    """Drop components with cross-section < min_cross_m; with slab_top, collapse closed horizontal slabs to their top
    faces. Returns stats (thin_* and slab_*, thin_by_class)."""
    from envbuild.blender.rf_relevance import (SQRT2, cross_section, face_widths, removal_class, slab_top_faces,
                                               tri_areas)

    st = {"thin_removed": 0, "thin_removed_tris": 0, "thin_removed_area_m2": 0.0, "thin_max_cross_m": 0.0,
          "thin_by_class": {}, "slabs_collapsed": 0, "slab_removed_tris": 0, "slab_removed_area_m2": 0.0,
          "slab_top_area_m2": 0.0}
    if not bm.faces:
        return st
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()
    label = np.empty(len(bm.verts), np.int64)
    for k, comp in enumerate(_components(bm)):
        for v in comp:
            label[v.index] = k
    lt = bm.calc_loop_triangles()
    T = np.array([[lp.vert.index for lp in t] for t in lt], np.int64)
    tface = np.array([t[0].face.index for t in lt], np.int64)
    V = np.array([v.co[:] for v in bm.verts], float)
    tcomp = label[T[:, 0]]
    ar = tri_areas(V, T)
    w = face_widths(V, T, ar)
    nc = int(label.max()) + 1
    wmax = np.zeros(nc)
    np.maximum.at(wmax, tcomp, np.where(ar > 1e-12, w, 0.0))
    z = V[T][:, :, 2]
    zlo, zhi = np.full(nc, np.inf), np.full(nc, -np.inf)
    np.minimum.at(zlo, tcomp, z.min(1))
    np.maximum.at(zhi, tcomp, z.max(1))
    order = np.argsort(tcomp, kind="stable")
    starts = np.searchsorted(tcomp[order], np.arange(nc + 1))
    rng = np.random.default_rng(seed)
    doomed_faces: set = set()
    thin_cand = wmax <= SQRT2 * min_cross_m * 1.02
    slab_cand = (zhi - zlo <= 0.15) & (zhi > zlo) if slab_top else np.zeros(nc, bool)
    for k in np.flatnonzero(thin_cand | slab_cand):
        sel = order[starts[k]:starts[k + 1]]
        if len(sel) == 0:
            continue
        used, f_loc = np.unique(T[sel], return_inverse=True)
        f_loc = f_loc.reshape(-1, 3)
        v_loc = V[used]
        if thin_cand[k]:
            cs = cross_section(v_loc, f_loc, min_cross_m, rng)
            if cs["cross"] == cs["cross"] and cs["cross"] < min_cross_m:      # not NaN, below the threshold
                doomed_faces.update(tface[sel].tolist())
                a = float(ar[sel].sum())
                st["thin_removed"] += 1
                st["thin_removed_tris"] += len(sel)
                st["thin_removed_area_m2"] += a
                st["thin_max_cross_m"] = max(st["thin_max_cross_m"], cs["cross"])
                c = st["thin_by_class"].setdefault(removal_class(cs["kind"], cs["length"]),
                                                   {"n": 0, "tris": 0, "area_m2": 0.0, "max_cross_m": 0.0})
                c["n"] += 1
                c["tris"] += len(sel)
                c["area_m2"] += a
                c["max_cross_m"] = max(c["max_cross_m"], cs["cross"])
                continue
        if slab_cand[k]:
            top = slab_top_faces(v_loc, f_loc)
            if top is None:
                continue
            topf = set(tface[sel][top].tolist())
            gone = [fi for fi in set(tface[sel].tolist()) if fi not in topf]
            if not gone:
                continue
            doomed_faces.update(gone)
            st["slabs_collapsed"] += 1
            st["slab_removed_tris"] += int((~top).sum())
            st["slab_removed_area_m2"] += float(ar[sel][~top].sum())
            st["slab_top_area_m2"] += float(ar[sel][top].sum())
    if doomed_faces:
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in sorted(doomed_faces)], context="FACES")
    return st


def simplify_bmesh(bm, angle_deg: float = 1.0, min_extent_m: float = 0.10, weld_m: float = 0.0005,
                   collinear_m: float = 0.001, area_tol: float = 0.002, rf_relevance: dict | None = None) -> dict:
    """Simplify `bm` in place (site frame, metres) and triangulate it. Returns stats: tris_before, tris_after,
    removed_components (plus faces_before/after; with rf_relevance also the rf_relevance_pass stats)."""
    tris_before, faces_before = _tri_count(bm), len(bm.faces)
    if weld_m > 0 and bm.verts:
        by_mat: dict = {}
        for v in bm.verts:
            mats = {f.material_index for f in v.link_faces}
            if len(mats) == 1:
                by_mat.setdefault(mats.pop(), []).append(v)
        for vs in by_mat.values():
            bmesh.ops.remove_doubles(bm, verts=vs, dist=weld_m)
        bmesh.ops.dissolve_degenerate(bm, dist=weld_m, edges=bm.edges[:])
    removed = drop_small_components(bm, min_extent_m)
    rel = rf_relevance_pass(bm, **rf_relevance) if rf_relevance else {}
    lay = _comp_layer(bm)
    orig = bm.copy()
    area0 = _area_by(bm, lay)
    if angle_deg > 0 and bm.faces:
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle_deg), use_dissolve_boundaries=False,
                                 verts=[], edges=bm.edges[:], delimit={"MATERIAL"})
        loose = [v for v in bm.verts if not v.link_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        dissolve_collinear(bm, collinear_m)
    if bm.faces:
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    # zero-area slivers from triangulating collinear n-gon runs carry no RF surface: drop them
    slivers = [f for f in bm.faces if f.calc_area() < 1e-9]
    if slivers:
        bmesh.ops.delete(bm, geom=slivers, context="FACES_ONLY")
    area1 = _area_by(bm, lay)
    bad = {k for k, a0 in area0.items() if abs(area1.get(k, 0.0) - a0) > area_tol * a0 + 1e-6}
    if bad:
        _restore(bm, orig, lay, bad)
    orig.free()
    bm.faces.layers.int.remove(lay)
    return {"tris_before": tris_before, "tris_after": _tri_count(bm), "removed_components": removed,
            "restored_components": len(bad),
            "faces_before": faces_before, "faces_after": len(bm.faces), **rel}
