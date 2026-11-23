"""Geometric fidelity of the simplified RF copy (RF convergence plan, Task 3, Step 6).

    python scripts/rf_geom_check.py --full data/site_kandla/rf_scene_blender --simpl data/site_kandla/rf_scene_blender_rf

Per shape (one PLY per Blender object x RF material, matched by shape id), full export vs simplified copy:
- bbox delta (max over the 6 coordinates) <= 0.5 m;
- top height delta (z max) <= 0.1 m;
- surface area delta <= 5 %;
- one-sided Hausdorff p99 <= 0.25 m: area-weighted sample points on the FULL mesh, exact distance to the nearest
  triangle of the SIMPLIFIED mesh (numba BVH; distances above the 2 m search cap are reported as the cap).
Collections given with --strict COLLECTION (curved collections simplified with a larger angle) must also keep the
max sample distance (the radius error) below 0.1 m.

Gate: >= 99 % of shapes pass, and every failing shape is a component < 0.5 m (bbox max extent) by design.
Writes runs/s76/geom_check.json (per shape) and runs/s76/geom_check_summary.json.

--removed-ok (Task 3b, the RF-relevance copy rf_scene_blender_rfr): parts of the full mesh may be missing on purpose.

    python scripts/rf_geom_check.py --removed-ok --simpl data/site_kandla/rf_scene_blender_rfr --out runs/s76/geom_check_t3b.json

Per shape, the full mesh is split into connected components (vertices welded at 0.5 mm). Every full triangle gets 4
points (centroid + 3 inner points); a triangle is FAR when all 4 lie > --far-tol (5 mm) from the simplified mesh.
- removed component: >= 80 % of its area is far. Its cross-section is MEASURED here on the full PLY
  (envbuild/blender/rf_relevance.cross_section, the same definition as the export) and it must be < --min-cross-m;
- slab underside: a closed horizontal slab <= 0.15 m thick (rf_relevance.slab_top_faces) whose top faces are all
  near and whose side faces are all far (its bottom usually coincides with the roof face of the building box under
  it, so the bottom is near either way): bottom and sides count as removed, the top must survive (the area check of
  the kept surfaces then proves its area);
- everything else is KEPT, and the per-shape checks above (bbox, top, area, Hausdorff p99) run on the kept surfaces
  only (a shape whose every component was removed has no kept surface and is listed as removed).
Removed components are reported by class (rf_relevance.removal_class + slab underside): count, triangles, area, max
cross-section. Gate: kept-surface checks pass >= 99 % of the shapes with kept surfaces, every failure is by design
(< 0.5 m), every removed component measures < min_cross_m, every slab underside is <= 0.15 m thick, no extra shapes.
Triangles per collection are reported for full / --ref (Task 3 flat merge) / simplified.
"""

import argparse
import json
from pathlib import Path

import sys

import numba as nb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CAP = 2.0            # metres: search radius of the nearest-triangle query
LIM = {"bbox_m": 0.5, "top_m": 0.1, "area_rel": 0.05, "haus_p99_m": 0.25, "strict_max_m": 0.1, "small_m": 0.5}


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The binary PLY written by envbuild/blender/export_rf.write_ply (float xyz, uchar-count int faces, all tris)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode().splitlines()
    nv = int(next(ln for ln in head if ln.startswith("element vertex")).split()[-1])
    nf = int(next(ln for ln in head if ln.startswith("element face")).split()[-1])
    v = np.frombuffer(raw, dtype="<f4", count=3 * nv, offset=end).reshape(nv, 3).astype(np.float64)
    rec = np.frombuffer(raw, dtype=np.dtype([("n", "u1"), ("i", "<i4", 3)]), count=nf, offset=end + 12 * nv)
    assert (rec["n"] == 3).all(), path
    return v, rec["i"].astype(np.int64)


def tri_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def sample_points(v, f, n, rng) -> np.ndarray:
    ar = tri_areas(v, f)
    tot = ar.sum()
    if tot <= 0:
        return v[f[:, 0]][:n]
    t = rng.choice(len(f), size=n, p=ar / tot)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = v[f[t, 0]], v[f[t, 1]], v[f[t, 2]]
    return (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c


# ---------------------------------------------------------------- numba BVH nearest-triangle distance
@nb.njit(cache=True)
def _build(lo, hi, cen, leaf):
    n = len(cen)
    order = np.arange(n)
    cap = 2 * n + 1
    bmin = np.empty((cap, 3))
    bmax = np.empty((cap, 3))
    left = -np.ones(cap, np.int64)
    start = np.zeros(cap, np.int64)
    count = np.zeros(cap, np.int64)
    stack = np.empty((cap, 3), np.int64)
    sp = 0
    stack[0, 0], stack[0, 1], stack[0, 2] = 0, 0, n
    sp = 1
    nn = 1
    while sp > 0:
        sp -= 1
        node, s, e = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        for k in range(3):
            mn, mx = 1e30, -1e30
            for i in range(s, e):
                t = order[i]
                if lo[t, k] < mn:
                    mn = lo[t, k]
                if hi[t, k] > mx:
                    mx = hi[t, k]
            bmin[node, k], bmax[node, k] = mn, mx
        if e - s <= leaf:
            start[node], count[node] = s, e - s
            continue
        ext = np.empty(3)
        for k in range(3):
            cmn, cmx = 1e30, -1e30
            for i in range(s, e):
                c = cen[order[i], k]
                cmn = min(cmn, c)
                cmx = max(cmx, c)
            ext[k] = cmx - cmn
        ax = np.argmax(ext)
        seg = order[s:e].copy()
        idx = np.argsort(cen[seg, ax])
        order[s:e] = seg[idx]
        m = (s + e) // 2
        left[node] = nn
        stack[sp, 0], stack[sp, 1], stack[sp, 2] = nn, s, m
        sp += 1
        stack[sp, 0], stack[sp, 1], stack[sp, 2] = nn + 1, m, e
        sp += 1
        nn += 2
    return order, bmin[:nn], bmax[:nn], left[:nn], start[:nn], count[:nn]


@nb.njit(cache=True, inline="always")
def _dot(u0, u1, u2, w0, w1, w2):
    return u0 * w0 + u1 * w1 + u2 * w2


@nb.njit(cache=True)
def _pt_tri_d2(p, a, b, c):
    """Squared distance point-triangle (Ericson, Real-Time Collision Detection 5.1.5); scalar code (no BLAS)."""
    ab0, ab1, ab2 = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    ac0, ac1, ac2 = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    ap0, ap1, ap2 = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    d1, d2 = _dot(ab0, ab1, ab2, ap0, ap1, ap2), _dot(ac0, ac1, ac2, ap0, ap1, ap2)
    u, w = 0.0, 0.0                       # q = a + u*ab + w*ac
    if d1 <= 0 and d2 <= 0:
        pass
    else:
        bp0, bp1, bp2 = p[0] - b[0], p[1] - b[1], p[2] - b[2]
        d3, d4 = _dot(ab0, ab1, ab2, bp0, bp1, bp2), _dot(ac0, ac1, ac2, bp0, bp1, bp2)
        vc = d1 * d4 - d3 * d2
        cp0, cp1, cp2 = p[0] - c[0], p[1] - c[1], p[2] - c[2]
        d5, d6 = _dot(ab0, ab1, ab2, cp0, cp1, cp2), _dot(ac0, ac1, ac2, cp0, cp1, cp2)
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        if d3 >= 0 and d4 <= d3:
            u = 1.0
        elif vc <= 0 and d1 >= 0 and d3 <= 0:
            u = d1 / (d1 - d3)
        elif d6 >= 0 and d5 <= d6:
            w = 1.0
        elif vb <= 0 and d2 >= 0 and d6 <= 0:
            w = d2 / (d2 - d6)
        elif va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            u, w = 1.0 - t, t
        else:
            den = va + vb + vc
            if den <= 0:                  # degenerate triangle: fall back to the nearest vertex
                da = ap0 * ap0 + ap1 * ap1 + ap2 * ap2
                db = bp0 * bp0 + bp1 * bp1 + bp2 * bp2
                dc = cp0 * cp0 + cp1 * cp1 + cp2 * cp2
                return min(da, db, dc)
            u, w = vb / den, vc / den
    r0 = ap0 - u * ab0 - w * ac0
    r1 = ap1 - u * ab1 - w * ac1
    r2 = ap2 - u * ab2 - w * ac2
    return r0 * r0 + r1 * r1 + r2 * r2


@nb.njit(parallel=True, cache=True)
def _query(pts, A, B, C, order, bmin, bmax, left, start, count, cap):
    out = np.empty(len(pts))
    for i in nb.prange(len(pts)):
        p = pts[i]
        best = cap * cap
        stack = np.empty(128, np.int64)
        stack[0] = 0
        sp = 1
        while sp > 0:
            sp -= 1
            nd = stack[sp]
            d2 = 0.0
            for k in range(3):
                if p[k] < bmin[nd, k]:
                    d2 += (bmin[nd, k] - p[k]) ** 2
                elif p[k] > bmax[nd, k]:
                    d2 += (p[k] - bmax[nd, k]) ** 2
            if d2 >= best:
                continue
            if left[nd] < 0:
                for j in range(start[nd], start[nd] + count[nd]):
                    t = order[j]
                    d = _pt_tri_d2(p, A[t], B[t], C[t])
                    if d < best:
                        best = d
            else:
                stack[sp] = left[nd]
                stack[sp + 1] = left[nd] + 1
                sp += 2
        out[i] = np.sqrt(best)
    return out


def nearest_dist(pts: np.ndarray, v: np.ndarray, f: np.ndarray, cap: float = CAP) -> np.ndarray:
    if len(f) == 0:
        return np.full(len(pts), cap)
    A, B, C = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    lo, hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
    order, bmin, bmax, left, start, count = _build(lo, hi, (lo + hi) / 2, 4)
    return _query(pts, A, B, C, order, bmin, bmax, left, start, count, cap)


# ---------------------------------------------------------------- --removed-ok (Task 3b)
@nb.njit(cache=True)
def _union_find(ff, n):
    parent = np.arange(n)
    for i in range(len(ff)):
        for j in (1, 2):
            a, b = ff[i, 0], ff[i, j]
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[b] = a
    for x in range(n):
        r = x
        while parent[r] != r:
            r = parent[r]
        parent[x] = r
    return parent


def components(v: np.ndarray, f: np.ndarray, weld_m: float = 0.0005) -> np.ndarray:
    """Component label per triangle (vertices welded on a weld_m grid, as the export welds at 0.5 mm)."""
    if len(f) == 0:
        return np.zeros(0, np.int64)
    _, inv = np.unique(np.round(v / weld_m).astype(np.int64), axis=0, return_inverse=True)
    ff = inv.ravel()[f]
    root = _union_find(ff, int(ff.max()) + 1)
    _, lab = np.unique(root[ff[:, 0]], return_inverse=True)
    return lab.ravel()


def face_points(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """[nf, 4, 3]: centroid and three inner points (barycentric 0.6/0.2/0.2) of every triangle."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.stack([(a + b + c) / 3, 0.6 * a + 0.2 * b + 0.2 * c, 0.2 * a + 0.6 * b + 0.2 * c,
                     0.2 * a + 0.2 * b + 0.6 * c], axis=1)


def check_shape_removed_ok(full: Path, simpl: Path | None, rng, per_m2: float, nmin: int, nmax: int,
                           far_tol: float, min_cross: float, max_slab_t: float = 0.15) -> dict:
    """Task 3b: classify the full mesh into removed components, slab undersides and kept surfaces, measure every
    removed component, and run the shape checks on the kept surfaces only."""
    from envbuild.blender.rf_relevance import cross_section, removal_class, slab_top_faces

    vf, ff = read_ply(full)
    have = simpl is not None and simpl.exists()
    vs, fs = read_ply(simpl) if have else (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    ar = tri_areas(vf, ff)
    d = nearest_dist(face_points(vf, ff).reshape(-1, 3), vs, fs).reshape(-1, 4)
    far = (d > far_tol).all(1)
    lab = components(vf, ff)
    nc = int(lab.max()) + 1 if len(lab) else 0
    a_c = np.bincount(lab, ar, nc)
    farfrac = np.bincount(lab, ar * far, nc) / np.maximum(a_c, 1e-30)
    order = np.argsort(lab, kind="stable")
    starts = np.searchsorted(lab[order], np.arange(nc + 1))
    kept = np.ones(len(ff), bool)
    removed, slabs = [], []
    for k in range(nc):
        if farfrac[k] <= 0:
            continue
        sel = order[starts[k]:starts[k + 1]]
        used, fl = np.unique(ff[sel], return_inverse=True)
        fl = fl.reshape(-1, 3)
        if farfrac[k] >= 0.8:
            cs = cross_section(vf[used], fl, min_cross, rng)
            removed.append({"class": removal_class(cs["kind"], cs["length"]), "kind": cs["kind"],
                            "cross_m": cs["cross"], "length_m": cs["length"], "tris": int(len(sel)),
                            "area_m2": float(ar[sel].sum())})
            kept[sel] = False
            continue
        if np.ptp(vf[used][:, 2]) <= max_slab_t:
            top = slab_top_faces(vf[used], fl, max_t=max_slab_t)
            if top is None:
                continue
            zl = vf[used][fl][:, :, 2]
            sides = ~top & ~(np.abs(zl - zl.min()) < 1e-4).all(1)
            # the top survives, the sides are gone; the bottom is gone or coincides with the building's own roof
            # face under the slab (a double surface), so only the sides can prove the collapse
            if sides.any() and not far[sel][top].any() and far[sel][sides].all():
                slabs.append({"class": "slab underside", "thick_m": float(np.ptp(vf[used][:, 2])),
                              "tris": int((~top).sum()), "area_m2": float(ar[sel][~top].sum()),
                              "top_area_m2": float(ar[sel][top].sum())})
                kept[sel[~top]] = False
    row = {"tris_full": int(len(ff)), "tris_kept": int(kept.sum()), "area_full": float(ar.sum()),
           "removed": removed, "slabs": slabs, "tris_simpl": int(len(fs))}
    if not kept.any():
        row.update(removed_all=True, ok=True, why=[], extent_full=float(np.ptp(vf, axis=0).max()))
        if have and len(fs):
            row.update(ok=False, why=["simplified surface left where everything was removed"])
        return row
    fk = ff[kept]
    vk = vf[np.unique(fk)]
    ak = float(ar[kept].sum())
    bbf = np.r_[vk.min(0), vk.max(0)]
    row["extent_full"] = float((bbf[3:] - bbf[:3]).max())
    row["area_kept"] = ak
    if not have:
        row.update(missing=True, ok=False, why=["missing"])
        return row
    as_ = float(tri_areas(vs, fs).sum())
    bbs = np.r_[vs.min(0), vs.max(0)]
    n = int(np.clip(ak * per_m2, nmin, nmax))
    dd = nearest_dist(sample_points(vf, fk, n, rng), vs, fs)
    row.update(area_simpl=as_, bbox_delta=float(np.abs(bbf - bbs).max()), top_delta=float(abs(bbf[5] - bbs[5])),
               area_rel=float(abs(as_ - ak) / ak) if ak > 0 else 0.0, haus_p99=float(np.percentile(dd, 99)),
               haus_max=float(dd.max()), n_samples=n)
    why = [w for w, bad in (("bbox", row["bbox_delta"] > LIM["bbox_m"]), ("top", row["top_delta"] > LIM["top_m"]),
                            ("area", row["area_rel"] > LIM["area_rel"]),
                            ("hausdorff", row["haus_p99"] > LIM["haus_p99_m"])) if bad]
    row["ok"], row["why"] = not why, why
    return row


def tris_per_collection(root: Path) -> dict:
    m = json.loads((root / "objects.json").read_text())
    out: dict = {}
    for r in m["shapes"]:
        out[r["collection"]] = out.get(r["collection"], 0) + int(r["tris"])
    return out


def main_removed_ok(a) -> None:
    full, simpl = (REPO / a.full), (REPO / a.simpl)
    mf = json.loads((full / "objects.json").read_text())
    ms = json.loads((simpl / "objects.json").read_text())
    simpl_shapes = {r["shape"] for r in ms["shapes"]}
    rng = np.random.default_rng(a.seed)
    rows = {}
    for i, r in enumerate(mf["shapes"]):
        s = r["shape"]
        row = check_shape_removed_ok(full / "meshes" / f"{s}.ply",
                                     (simpl / "meshes" / f"{s}.ply") if s in simpl_shapes else None,
                                     rng, a.per_m2, a.nmin, a.nmax, a.far_tol, a.min_cross_m)
        row["collection"] = r["collection"]
        row["by_design"] = (not row["ok"]) and row.get("extent_full", 0.0) < LIM["small_m"]
        rows[s] = row
        if (i + 1) % 100 == 0:
            print(f"[geom_check --removed-ok] {i + 1}/{len(mf['shapes'])}", flush=True)
    extra = sorted(simpl_shapes - set(rows))
    checked = {s: r for s, r in rows.items() if not r.get("removed_all")}
    n_ok = sum(r["ok"] for r in checked.values())
    pass_pct = 100 * n_ok / max(len(checked), 1)
    fails = {s: {k: row.get(k) for k in ("collection", "why", "extent_full", "bbox_delta", "top_delta", "area_rel",
                                          "haus_p99", "haus_max", "by_design")}
             for s, row in rows.items() if not row["ok"]}
    classes: dict = {}
    for row in rows.values():
        for c in row["removed"] + row["slabs"]:
            key = f"{row['collection']} | {c['class']}"
            e = classes.setdefault(key, {"n": 0, "tris": 0, "area_m2": 0.0, "max_cross_m": 0.0, "max_thick_m": 0.0})
            e["n"] += 1
            e["tris"] += c["tris"]
            e["area_m2"] += c["area_m2"]
            if "cross_m" in c:
                e["max_cross_m"] = max(e["max_cross_m"], c["cross_m"])
            else:
                e["max_thick_m"] = max(e["max_thick_m"], c["thick_m"])
    rem = [c for row in rows.values() for c in row["removed"]]
    slb = [c for row in rows.values() for c in row["slabs"]]
    max_cross = max((c["cross_m"] for c in rem), default=0.0)
    bad_removed = [(s, c) for s, row in rows.items() for c in row["removed"]
                   if not (c["cross_m"] < a.min_cross_m)]                      # NaN (empty) counts as bad
    max_thick = max((c["thick_m"] for c in slb), default=0.0)
    tf, tr = tris_per_collection(full), tris_per_collection(simpl)
    tref = tris_per_collection(REPO / a.ref) if a.ref and (REPO / a.ref / "objects.json").exists() else {}
    tri_table = {c: {"full": tf[c], "ref": tref.get(c), "simpl": tr.get(c, 0)}
                 for c in sorted(tf, key=lambda c: -tf[c])}
    gate = (pass_pct >= 99.0 and all(f["by_design"] for f in fails.values()) and not extra and not bad_removed
            and max_thick <= 0.15)
    summary = {"mode": "removed_ok", "full": a.full, "simpl": a.simpl, "ref": a.ref,
               "params": ms.get("simplify", {}).get("params"),
               "limits": {**LIM, "far_tol_m": a.far_tol, "min_cross_m": a.min_cross_m},
               "shapes": len(rows), "shapes_removed_entirely": len(rows) - len(checked), "shapes_checked": len(checked),
               "pass": n_ok, "pass_pct": round(pass_pct, 2), "gate": bool(gate),
               "removed_components": len(rem), "removed_tris": sum(c["tris"] for c in rem),
               "removed_area_m2": round(sum(c["area_m2"] for c in rem), 1), "removed_max_cross_m": max_cross,
               "removed_not_thin": len(bad_removed), "slab_undersides": len(slb),
               "slab_removed_area_m2": round(sum(c["area_m2"] for c in slb), 1),
               "slab_top_area_m2": round(sum(c["top_area_m2"] for c in slb), 1), "slab_max_thick_m": max_thick,
               "removed_by_class": dict(sorted(classes.items(), key=lambda kv: -kv[1]["tris"])),
               "tris": {"full": sum(tf.values()), "ref": sum(tref.values()) if tref else None,
                        "simpl": sum(tr.values())},
               "tris_per_collection": tri_table, "failures": fails, "extra_shapes": extra,
               "removed_not_thin_list": [{"shape": s, **c} for s, c in bad_removed][:50]}
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "shapes": rows}, indent=1))
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'collection':18s} {'full':>9s} {'ref':>9s} {'simpl':>9s}")
    for c, t in tri_table.items():
        print(f"{c:18s} {t['full']:9d} {t['ref'] if t['ref'] is not None else -1:9d} {t['simpl']:9d}")
    print(f"{'class':45s} {'n':>6s} {'tris':>8s} {'area_m2':>9s} {'max_cross':>9s}")
    for k, e in summary["removed_by_class"].items():
        print(f"{k:45s} {e['n']:6d} {e['tris']:8d} {e['area_m2']:9.1f} "
              f"{e['max_cross_m'] if e['max_cross_m'] else e['max_thick_m']:9.4f}")
    print(f"removed {len(rem)} components (max cross {max_cross:.4f} m, not thin {len(bad_removed)}); slab undersides "
          f"{len(slb)} (max thick {max_thick:.3f} m); kept-surface shapes pass {n_ok}/{len(checked)} = {pass_pct:.2f} %;"
          f" failures {len(fails)} (by design {sum(f['by_design'] for f in fails.values())}); gate "
          f"{'PASS' if gate else 'FAIL'}")
    for s, f in list(fails.items())[:30]:
        print("FAIL", s, f)
    for s, c in bad_removed[:20]:
        print("NOT THIN", s, c)


# ---------------------------------------------------------------- per shape
def check_shape(full: Path, simpl: Path | None, rng, strict: bool, per_m2: float, nmin: int, nmax: int) -> dict:
    vf, ff = read_ply(full)
    af = float(tri_areas(vf, ff).sum())
    bbf = np.r_[vf.min(0), vf.max(0)]
    row = {"tris_full": int(len(ff)), "area_full": af, "extent_full": float((bbf[3:] - bbf[:3]).max())}
    if simpl is None or not simpl.exists():
        row.update(tris_simpl=0, missing=True, ok=False, why=["missing"])
        return row
    vs, fs = read_ply(simpl)
    as_ = float(tri_areas(vs, fs).sum())
    bbs = np.r_[vs.min(0), vs.max(0)]
    n = int(np.clip(af * per_m2, nmin, nmax))
    d = nearest_dist(sample_points(vf, ff, n, rng), vs, fs)
    row.update(tris_simpl=int(len(fs)), area_simpl=as_, bbox_delta=float(np.abs(bbf - bbs).max()),
               top_delta=float(abs(bbf[5] - bbs[5])), area_rel=float(abs(as_ - af) / af) if af > 0 else 0.0,
               haus_p99=float(np.percentile(d, 99)), haus_max=float(d.max()), n_samples=n)
    why = []
    if row["bbox_delta"] > LIM["bbox_m"]:
        why.append("bbox")
    if row["top_delta"] > LIM["top_m"]:
        why.append("top")
    if row["area_rel"] > LIM["area_rel"]:
        why.append("area")
    if row["haus_p99"] > LIM["haus_p99_m"]:
        why.append("hausdorff")
    if strict and row["haus_max"] > LIM["strict_max_m"]:
        why.append("radius")
    row["ok"], row["why"] = not why, why
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", default="data/site_kandla/rf_scene_blender")
    ap.add_argument("--simpl", default="data/site_kandla/rf_scene_blender_rf")
    ap.add_argument("--out", default="runs/s76/geom_check.json")
    ap.add_argument("--strict", action="append", default=None, metavar="COLLECTION",
                    help="collections that must keep max distance < 0.1 m (default: the simplify angle overrides)")
    ap.add_argument("--per-m2", type=float, default=2.0, help="sample points per m^2 of full surface")
    ap.add_argument("--nmin", type=int, default=2000)
    ap.add_argument("--nmax", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--removed-ok", action="store_true",
                    help="Task 3b: thin components and slab undersides may be removed (measured and reported)")
    ap.add_argument("--min-cross-m", type=float, default=0.05,
                    help="--removed-ok: every removed component must measure below this")
    ap.add_argument("--far-tol", type=float, default=0.005, help="--removed-ok: a triangle farther is missing")
    ap.add_argument("--ref", default="data/site_kandla/rf_scene_blender_rf",
                    help="--removed-ok: a third export whose triangles per collection are reported (Task 3)")
    a = ap.parse_args()
    if a.removed_ok:
        return main_removed_ok(a)
    full, simpl = (REPO / a.full), (REPO / a.simpl)
    mf = json.loads((full / "objects.json").read_text())
    ms = json.loads((simpl / "objects.json").read_text())
    sp = ms.get("simplify", {})
    strict = set(a.strict if a.strict is not None else (sp.get("params", {}).get("angle_override") or {}))
    simpl_shapes = {r["shape"] for r in ms["shapes"]}
    rng = np.random.default_rng(a.seed)
    rows = {}
    for i, r in enumerate(mf["shapes"]):
        s = r["shape"]
        row = check_shape(full / "meshes" / f"{s}.ply",
                          (simpl / "meshes" / f"{s}.ply") if s in simpl_shapes else None,
                          rng, r["collection"] in strict, a.per_m2, a.nmin, a.nmax)
        row["collection"] = r["collection"]
        row["by_design"] = (not row["ok"]) and row["extent_full"] < LIM["small_m"]
        rows[s] = row
        if (i + 1) % 100 == 0:
            print(f"[geom_check] {i + 1}/{len(mf['shapes'])}", flush=True)
    extra = sorted(simpl_shapes - set(rows))
    cols: dict = {}
    for s, row in rows.items():
        c = cols.setdefault(row["collection"], {"shapes": 0, "pass": 0, "tris_full": 0, "tris_simpl": 0,
                                                "haus_p99_max": 0.0, "haus_max": 0.0})
        c["shapes"] += 1
        c["pass"] += int(row["ok"])
        c["tris_full"] += row["tris_full"]
        c["tris_simpl"] += row["tris_simpl"]
        c["haus_p99_max"] = max(c["haus_p99_max"], row.get("haus_p99", 0.0))
        c["haus_max"] = max(c["haus_max"], row.get("haus_max", 0.0))
    for c in cols.values():
        c["tris_pct"] = round(100 * c["tris_simpl"] / max(c["tris_full"], 1), 1)
    fails = {s: {k: row.get(k) for k in ("collection", "why", "extent_full", "bbox_delta", "top_delta", "area_rel",
                                          "haus_p99", "haus_max", "by_design")}
             for s, row in rows.items() if not row["ok"]}
    n_ok = sum(r["ok"] for r in rows.values())
    pass_pct = 100 * n_ok / len(rows)
    tf, ts = sum(r["tris_full"] for r in rows.values()), sum(r["tris_simpl"] for r in rows.values())
    gate = pass_pct >= 99.0 and all(f["by_design"] for f in fails.values()) and not extra
    summary = {"full": a.full, "simpl": a.simpl, "params": sp.get("params"), "limits": LIM, "strict": sorted(strict),
               "shapes": len(rows), "pass": n_ok, "pass_pct": round(pass_pct, 2), "gate": bool(gate),
               "tris_full": tf, "tris_simpl": ts, "tris_pct": round(100 * ts / tf, 1),
               "per_collection": dict(sorted(cols.items(), key=lambda kv: -kv[1]["tris_full"])),
               "failures": fails, "extra_shapes": extra}
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "shapes": rows}, indent=1))
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'collection':18s} {'tris_full':>9s} {'tris_simpl':>10s} {'%':>6s} {'pass':>9s} {'p99max':>7s} {'max':>6s}")
    for k, c in summary["per_collection"].items():
        print(f"{k:18s} {c['tris_full']:9d} {c['tris_simpl']:10d} {c['tris_pct']:6.1f} {c['pass']:4d}/{c['shapes']:<4d} "
              f"{c['haus_p99_max']:7.3f} {c['haus_max']:6.3f}")
    print(f"TOTAL tris {tf} -> {ts} ({summary['tris_pct']} %); shapes pass {n_ok}/{len(rows)} = {pass_pct:.2f} %; "
          f"failures {len(fails)} (by design {sum(f['by_design'] for f in fails.values())}); gate "
          f"{'PASS' if gate else 'FAIL'}")
    for s, f in list(fails.items())[:30]:
        print("FAIL", s, f)


if __name__ == "__main__":
    main()
