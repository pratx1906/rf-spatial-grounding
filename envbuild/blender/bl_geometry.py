"""Mesh building blocks (pure python + numpy, bpy only at the end)."""

from __future__ import annotations

import math

import bpy
import numpy as np


def signed_area(ring) -> float:
    a = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        a += x0 * y1 - x1 * y0
    return a / 2


def ccw(ring):
    ring = [tuple(p) for p in ring]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring if signed_area(ring) > 0 else ring[::-1]


def point_in_poly(x, y, ring) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def offset_polyline(pts, half_w, closed=False):
    """Left/right offsets of a 2D polyline with mitred joins (miter limited)."""
    P = np.asarray(pts, float)
    n = len(P)
    L = np.zeros_like(P)
    R = np.zeros_like(P)
    for i in range(n):
        if closed:
            d0 = P[i] - P[i - 1]
            d1 = P[(i + 1) % n] - P[i]
        else:
            d0 = P[i] - P[i - 1] if i > 0 else P[1] - P[0]
            d1 = P[i + 1] - P[i] if i < n - 1 else P[i] - P[i - 1]
        d0 = d0 / (np.linalg.norm(d0) + 1e-9)
        d1 = d1 / (np.linalg.norm(d1) + 1e-9)
        t = d0 + d1
        t = t / (np.linalg.norm(t) + 1e-9)
        nrm = np.array([-t[1], t[0]])
        miter = half_w / max(0.35, float(np.dot(nrm, [-d1[1], d1[0]])))
        L[i] = P[i] + nrm * miter
        R[i] = P[i] - nrm * miter
    return L, R


class MB:
    """Accumulates verts/faces/material slots/uvs, then builds one Blender mesh object."""

    def __init__(self):
        self.v: list[tuple] = []
        self.f: list[tuple] = []
        self.m: list[int] = []
        self.uv: list[list | None] = []
        self.slots: list[str] = []
        self.fc: list = []            # optional per-face RGBA (alpha = weight of the imagery colour)
        self.cur_fc = (0.0, 0.0, 0.0, 0.0)

    def slot(self, name: str) -> int:
        if name not in self.slots:
            self.slots.append(name)
        return self.slots.index(name)

    def add_v(self, pts) -> int:
        b = len(self.v)
        self.v.extend(tuple(map(float, p)) for p in pts)
        return b

    def face(self, idx, mat, uv=None):
        self.f.append(tuple(idx))
        self.m.append(self.slot(mat))
        self.uv.append(uv)
        self.fc.append(self.cur_fc)

    # ---- primitives ----------------------------------------------------------------------------------------
    def prism(self, ring, z0, z1, mat, top=True, bottom=False, side_mat=None, top_mat=None):
        ring = ccw(ring)
        n = len(ring)
        b = self.add_v([(x, y, z0) for x, y in ring])
        t = self.add_v([(x, y, z1) for x, y in ring])
        s = 0.0
        for i in range(n):
            j = (i + 1) % n
            seg = math.dist(ring[i], ring[j])
            self.face((b + i, b + j, t + j, t + i), side_mat or mat,
                      [(s, z0), (s + seg, z0), (s + seg, z1), (s, z1)])
            s += seg
        if top:
            self.face([t + i for i in range(n)], top_mat or mat, [(x, y) for x, y in ring])
        if bottom:
            self.face([b + i for i in reversed(range(n))], mat, [(x, y) for x, y in reversed(ring)])

    def box(self, c, size, mat, rot=0.0):
        cx, cy, cz = c
        sx, sy, sz = (s / 2 for s in size)
        cr, sr = math.cos(rot), math.sin(rot)
        ring = [(cx + dx * cr - dy * sr, cy + dx * sr + dy * cr) for dx, dy in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))]
        self.prism(ring, cz - sz, cz + sz, mat, top=True, bottom=True)

    def cylinder(self, cx, cy, z0, z1, r, seg, mat, top=True, bottom=False, inner=False, top_mat=None):
        ring = [(cx + r * math.cos(2 * math.pi * k / seg), cy + r * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        if inner:
            ring = ring[::-1]
            n = seg
            b = self.add_v([(x, y, z0) for x, y in ring])
            t = self.add_v([(x, y, z1) for x, y in ring])
            for i in range(n):
                j = (i + 1) % n
                self.face((b + i, b + j, t + j, t + i), mat)
            return
        self.prism(ring, z0, z1, mat, top=top, bottom=bottom, top_mat=top_mat)

    def cone(self, cx, cy, z0, r, apex_h, seg, mat):
        b = self.add_v([(cx + r * math.cos(2 * math.pi * k / seg), cy + r * math.sin(2 * math.pi * k / seg), z0)
                        for k in range(seg)])
        a = self.add_v([(cx, cy, z0 + apex_h)])
        for k in range(seg):
            self.face((b + k, b + (k + 1) % seg, a), mat)

    def annulus(self, cx, cy, z, r0, r1, seg, mat, thick=0.0):
        inn = [(cx + r0 * math.cos(2 * math.pi * k / seg), cy + r0 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        out = [(cx + r1 * math.cos(2 * math.pi * k / seg), cy + r1 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        zs = [z] if thick == 0 else [z - thick, z]
        lay = []
        for zz in zs:
            lay.append((self.add_v([(x, y, zz) for x, y in inn]), self.add_v([(x, y, zz) for x, y in out])))
        i_t, o_t = lay[-1]
        for k in range(seg):
            j = (k + 1) % seg
            self.face((i_t + k, o_t + k, o_t + j, i_t + j), mat)
        if thick:
            i_b, o_b = lay[0]
            for k in range(seg):
                j = (k + 1) % seg
                self.face((i_b + j, o_b + j, o_b + k, i_b + k), mat)
                self.face((o_b + k, o_b + j, o_t + j, o_t + k), mat)
                self.face((i_b + j, i_b + k, i_t + k, i_t + j), mat)

    def tube(self, p0, p1, r, seg, mat, caps=False):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        d = p1 - p0
        ln = np.linalg.norm(d)
        if ln < 1e-6:
            return
        d /= ln
        a = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, a)
        u /= np.linalg.norm(u)
        w = np.cross(d, u)
        ring = [(math.cos(2 * math.pi * k / seg), math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        b = self.add_v([p0 + r * (c * u + s * w) for c, s in ring])
        t = self.add_v([p1 + r * (c * u + s * w) for c, s in ring])
        for k in range(seg):
            j = (k + 1) % seg
            self.face((b + k, b + j, t + j, t + k), mat)
        if caps:
            self.face([t + k for k in range(seg)], mat)
            self.face([b + k for k in reversed(range(seg))], mat)

    def ribbon(self, pts, half_w, z_of, mat, z_off=0.0, closed=False, u_scale=1.0):
        """Flat strip following a 2D polyline; UV u = metres along, v = 0..1 across."""
        if len(pts) < 2:
            return
        L, R = offset_polyline(pts, half_w, closed)
        zl = z_of(L) + z_off
        zr = z_of(R) + z_off
        bl = self.add_v([(x, y, z) for (x, y), z in zip(L, zl)])
        br = self.add_v([(x, y, z) for (x, y), z in zip(R, zr)])
        n = len(pts)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1))])
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            j = (i + 1) % n
            sj = s[j] if j > i else s[i] + math.dist(pts[i], pts[j])
            self.face((br + i, br + j, bl + j, bl + i), mat,
                      [(s[i] * u_scale, 0.0), (sj * u_scale, 0.0), (sj * u_scale, 1.0), (s[i] * u_scale, 1.0)])

    def wall(self, pts, thick, z0_of, h, mat, closed=False):
        """Vertical wall of thickness `thick` along a polyline, base following the ground."""
        if len(pts) < 2:
            return
        L, R = offset_polyline(pts, thick / 2, closed)
        n = len(pts)
        zl = z0_of(L)
        zr = z0_of(R)
        lb = self.add_v([(x, y, z) for (x, y), z in zip(L, zl)])
        lt = self.add_v([(x, y, z + h) for (x, y), z in zip(L, zl)])
        rb = self.add_v([(x, y, z) for (x, y), z in zip(R, zr)])
        rt = self.add_v([(x, y, z + h) for (x, y), z in zip(R, zr)])
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            j = (i + 1) % n
            self.face((lb + j, lb + i, lt + i, lt + j), mat)
            self.face((rb + i, rb + j, rt + j, rt + i), mat)
            self.face((lt + i, rt + i, rt + j, lt + j), mat)

    # ---- build ---------------------------------------------------------------------------------------------
    def build(self, name, M, collection, smooth=False, props=None):
        if not self.f:
            return None
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        for s in self.slots:
            me.materials.append(M[s])
        me.polygons.foreach_set("material_index", self.m)
        uvl = me.uv_layers.new(name="UVMap")
        co = np.array(self.v)
        uvs = []
        for fi, f in enumerate(self.f):
            uv = self.uv[fi]
            if uv is None:
                p = co[list(f)]
                nrm = np.cross(p[1] - p[0], p[2] - p[0])
                if abs(nrm[2]) > 0.7 * (np.linalg.norm(nrm) + 1e-9):
                    uv = [(q[0], q[1]) for q in p]
                else:
                    uv = [(q[0] + q[1], q[2]) for q in p]
            uvs.extend(uv)
        uvl.data.foreach_set("uv", np.asarray(uvs, np.float32).ravel())
        # always present (zeros = no tint): a missing attribute reads as black with full weight on the GPU
        ca = me.color_attributes.new("roof_col", "FLOAT_COLOR", "FACE")
        ca.data.foreach_set("color", np.asarray(self.fc, np.float32).ravel())
        me.validate(clean_customdata=False)
        me.update()
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(name, me)
        collection.objects.link(ob)
        for k, v in (props or {}).items():
            ob[k] = v
        return ob
