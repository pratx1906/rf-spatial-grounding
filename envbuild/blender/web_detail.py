"""Bake-time surface detail and markings for the console world (used by bake_web.py; the .blend is never saved).

Why: bake_web.py joins a collection's objects into one before baking (one atlas, one draw call). Joining collapses
two per-object shader inputs of the Blender materials:
- Object Info `Random`: per-tank paint (white / light grey / aluminium / cream), per-shed cladding palette, per-ship
  hull colour all became ONE value per collection;
- Texture Coordinate `Generated`: the 0..1 box of the object; run-off streaks and dust skirts of a tank were spread
  over the whole collection's height range.
stamp() writes both per object as mesh attributes before the join (web_rand, web_gen) and patch() rewires the
materials to read them, so the bake matches the per-object look of the Blender build.

On top of that, markings and surface detail the Blender materials do not carry (all ILLUSTRATIVE, not surveyed):
- tank shells: plate courses (2.44 m) and staggered vertical seams; the tank's console id (T001...) stencilled on two
  sides (synthetic ids, the console's labels, not real tank numbers); cone roofs darker with radial plate seams and
  a weathered rim; floating-roof decks grey plate with seams;
- jetty decks and the cargo wharf: concrete panel joints and per-panel tone in the deck's own axis, a yellow safety
  line 0.45 m inside every deck edge (flat strips 2 cm above the deck, no new 3D structure);
- metal roofs: sheet laps every 1 m and translucent skylight strips every 7.5 m along the ridge, dark gutters;
- flat roofs: membrane patches and parapet dirt; shed walls: vertical sheet seams and a concrete dado;
- every baked material: light edge definition from a Cycles Bevel normal (sun-caught arrises), so boxes read as
  solids from an oblique view.
Per-face frames (web_fax = long-edge axis cos/sin and half width across, web_fc = face centre) give roofs and decks
their own orientation after the join.
"""

from __future__ import annotations

import math
import re
import zlib

import bpy
import bmesh
import numpy as np

from envbuild.blender.bl_materials import NT

MARKINGS = ("illustrative: plate seams, panel joints, deck safety lines, roof sheet laps and skylights, edge wear; "
            "tank side stencils are the console's synthetic tank ids, not real tank numbers")
LABEL_COLS, LABEL_CELL = 16, (32, 9)          # label atlas: 16 columns of 32 x 9 px cells (5x7 font, 1 px margin)
_FONT = {                                       # 5x7 stencil font, rows top -> bottom
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
LABEL_BEARINGS = (-113.0, 67.0)                # math angle (deg from east): faces the default SW view and its opposite


class T(NT):
    """NT over an existing tree (does not clear it)."""

    def __init__(self, mat):                    # noqa: D401
        self.nt = mat.node_tree
        self.x = -4000

    def vattr(self, name, typ="GEOMETRY"):
        return self.n("ShaderNodeAttribute", attribute_name=name, attribute_type=typ)

    def mul(self, col, f):
        """colour x scalar factor."""
        m = self.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        self._in(m, "Factor", 1.0, "VALUE")
        self._in(m, "A", col, "RGBA")
        cc = self.n("ShaderNodeCombineColor")
        for k in range(3):
            if isinstance(f, bpy.types.NodeSocket):
                self.link(f, cc.inputs[k])
            else:
                cc.inputs[k].default_value = f
        self._in(m, "B", cc.outputs[0], "RGBA")
        return self.sock(m.outputs, "Result", "RGBA")

    def line(self, coord, period, half_w, soft=0.03, offset=0.0):
        """1 on lines every `period` m (half width half_w m), 0 between."""
        f = self.math("FRACT", self.math("DIVIDE", self.math("ADD", coord, offset), period))
        d = self.math("MULTIPLY", self.math("MINIMUM", f, self.math("SUBTRACT", 1.0, f)), period)
        return self.maprange(d, half_w + soft, max(half_w - soft, 0.0))

    def cellnoise(self, a, b, pa, pb, seed=0.0):
        """white noise per (a // pa, b // pb) cell, 0..1."""
        c = self.n("ShaderNodeCombineXYZ")
        self.link(self.math("FLOOR", self.math("DIVIDE", a, pa)), c.inputs[0])
        self.link(self.math("FLOOR", self.math("DIVIDE", b, pb)), c.inputs[1])
        c.inputs[2].default_value = seed
        w = self.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
        self.link(c.outputs[0], w.inputs["Vector"])
        return self.sock(w.outputs, "Value")


def _hash01(s: str) -> float:
    return (zlib.crc32(s.encode()) & 0xFFFFFF) / float(0x1000000)


# ------------------------------------------------------------------------------------------------ geometry stamps
def _set_attr(me, name, domain, typ, values):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, typ, domain)
    key = "value" if typ == "FLOAT" else "vector"
    a.data.foreach_set(key, np.ascontiguousarray(values, np.float32).ravel())


def _face_frames(me):
    """Per face: long-edge axis (cos, sin), half width across it, centre (x, y). Vectorised over loops."""
    nf, nl = len(me.polygons), len(me.loops)
    if not nf:
        return np.zeros((0, 3)), np.zeros((0, 3))
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[:, :2].astype(np.float64)
    ls = np.empty(nf, np.int32)
    lt = np.empty(nf, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    vi = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", vi)
    face_of = np.repeat(np.arange(nf), lt)
    pos_in = np.arange(nl) - ls[face_of]
    nxt = ls[face_of] + (pos_in + 1) % lt[face_of]
    p0, p1 = co[vi], co[vi[nxt]]
    e = p1 - p0
    el = np.hypot(e[:, 0], e[:, 1])
    # longest edge per face
    order = np.lexsort((el, face_of))
    last = np.r_[np.nonzero(np.diff(face_of[order]))[0], nl - 1]
    best = order[last]
    ax = e[best] / np.maximum(el[best], 1e-9)[:, None]
    cen = np.zeros((nf, 2))
    np.add.at(cen, face_of, p0)
    cen /= lt[:, None]
    rel = p0 - cen[face_of]
    across = np.abs(-rel[:, 0] * ax[face_of, 1] + rel[:, 1] * ax[face_of, 0])
    hw = np.zeros(nf)
    np.maximum.at(hw, face_of, across)
    fax = np.c_[ax, hw]
    fc = np.c_[cen, np.zeros(nf)]
    return fax, fc


def stamp(o, group: str, tank_index: dict) -> None:
    """Per-object attributes that survive bake_web.join(): web_rand, web_gen; tanks: web_tank, web_tank2; faces:
    web_fax, web_fc."""
    me = o.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0)
    _set_attr(me, "web_rand", "POINT", "FLOAT", np.full(n, _hash01(o.name), np.float32))
    _set_attr(me, "web_gen", "POINT", "FLOAT_VECTOR", (co - lo) / np.maximum(hi - lo, 1e-6))
    fax, fc = _face_frames(me)
    _set_attr(me, "web_fax", "FACE", "FLOAT_VECTOR", fax)
    _set_attr(me, "web_fc", "FACE", "FLOAT_VECTOR", fc)
    if group == "tanks":
        shell = _shell_verts(o)
        sc = co[shell] if len(shell) else co
        mw = np.array(o.matrix_world)
        w = sc @ mw[:3, :3].T + mw[:3, 3]
        cx, cy = (w[:, 0].min() + w[:, 0].max()) / 2, (w[:, 1].min() + w[:, 1].max()) / 2
        r = float(o.get("diameter_m") or (w[:, 0].max() - w[:, 0].min())) / 2
        z0, z1 = float(w[:, 2].min()), float(w[:, 2].max())
        idx = tank_index.get(o.name, 0)
        _set_attr(me, "web_tank", "POINT", "FLOAT_VECTOR", np.tile([cx, cy, idx], (n, 1)))
        _set_attr(me, "web_tank2", "POINT", "FLOAT_VECTOR", np.tile([r, z0, z1 - z0], (n, 1)))


def _shell_verts(o) -> np.ndarray:
    me = o.data
    slot = [i for i, s in enumerate(o.material_slots) if s.material and s.material.name == "tank_paint"]
    if not slot:
        return np.zeros(0, int)
    vs = set()
    for p in me.polygons:
        if p.material_index in slot:
            vs.update(p.vertices)
    return np.fromiter(vs, int)


def tank_ids(objs) -> tuple[dict, list]:
    """object name -> label cell index; labels in cell order ("T001" from 'Storage tank T001')."""
    idx, labels = {}, []
    for o in sorted(objs, key=lambda x: x.name):
        m = re.search(r"(T\d{3})", str(o.get("name", "")))
        labels.append(m.group(1) if m else "")
        idx[o.name] = len(labels) - 1
    return idx, labels


def label_image(labels: list) -> bpy.types.Image:
    """Atlas of stencil labels (RGB = paint), Blender image rows bottom -> top."""
    cw, ch = LABEL_CELL
    rows = max(1, math.ceil(len(labels) / LABEL_COLS))
    W, H = LABEL_COLS * cw, rows * ch
    a = np.zeros((H, W), np.float32)              # row 0 = top here, flipped at the end
    for i, s in enumerate(labels):
        cx, cy = (i % LABEL_COLS) * cw, (i // LABEL_COLS) * ch
        x = cx + (cw - 6 * len(s) + 1) // 2
        for chr_ in s:
            g = _FONT.get(chr_)
            if g:
                for r, row in enumerate(g):
                    for c, bit in enumerate(row):
                        if bit == "1":
                            a[cy + 1 + r, x + c] = 1.0
            x += 6
    img = bpy.data.images.get("web_tank_labels") or bpy.data.images.new("web_tank_labels", W, H, alpha=True)
    img.colorspace_settings.name = "Non-Color"     # before the pixels: a colour-space change regenerates the image
    px = np.ones((H, W, 4), np.float32)
    px[..., :3] = a[::-1, :, None]                 # paint mask in RGB (Blender: first row is the bottom)
    img.pixels.foreach_set(px.ravel())
    # as a file image: Cycles then reads exactly these pixels (pack() would regenerate a blank generated image)
    import os
    import tempfile
    img.filepath_raw = os.path.join(tempfile.gettempdir(), "web_tank_labels.png")
    img.file_format = "PNG"
    img.save()
    img.source = "FILE"
    img.reload()
    return img


def deck_lines(o, inset=0.45, width=0.16, lift=0.02) -> int:
    """Yellow safety lines: a flat strip `inset` m inside every edge of the deck / wharf top faces (illustrative)."""
    names = [s.material.name if s.material else "" for s in o.material_slots]
    me = o.data
    tops = [p for p in me.polygons if p.normal.z > 0.9 and p.area > 100 and p.center.z > 2.0 and
            names[p.material_index] in ("concrete", "web_deck", "wharf_yard")]
    if not tops:
        return 0
    mat = paint_material()
    if mat.name not in names:
        o.data.materials.append(mat)
    mi = list(o.data.materials).index(mat)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    nstrips = 0
    for p in tops:
        vs = [me.vertices[i].co.copy() for i in p.vertices]
        z = p.center.z + lift
        k = len(vs)
        for i in range(k):
            a, b = vs[i], vs[(i + 1) % k]
            dx, dy = b.x - a.x, b.y - a.y
            L = math.hypot(dx, dy)
            if L < 2.0:
                continue
            ux, uy = dx / L, dy / L
            nx, ny = -uy, ux                          # left of the edge = inside for a CCW (normal up) polygon
            t0, t1 = min(inset + width, L / 2), max(L - inset - width, L / 2)
            if t1 - t0 < 1.0:
                continue
            q = []
            for t, d in ((t0, inset), (t1, inset), (t1, inset + width), (t0, inset + width)):
                q.append(bm.verts.new((a.x + ux * t + nx * d, a.y + uy * t + ny * d, z)))
            f = bm.faces.new(q)
            f.material_index = mi
            nstrips += 1
    bm.to_mesh(me)
    bm.free()
    return nstrips


def deck_material(o) -> int:
    """Jetty deck tops get their own copy of `concrete` (panel joints only there, not on every concrete face)."""
    names = [s.material.name if s.material else "" for s in o.material_slots]
    if "concrete" not in names:
        return 0
    src = bpy.data.materials["concrete"]
    deck = bpy.data.materials.get("web_deck") or src.copy()
    deck.name = "web_deck"
    if "web_deck" not in names:
        o.data.materials.append(deck)
    di = list(o.data.materials).index(deck)
    ci = names.index("concrete")
    n = 0
    for p in o.data.polygons:
        if p.material_index == ci and p.normal.z > 0.9 and p.center.z > 2.0:
            p.material_index = di
            n += 1
    return n


def paint_material():
    m = bpy.data.materials.get("web_paint_yellow")
    if m:
        return m
    m = bpy.data.materials.new("web_paint_yellow")
    m.use_nodes = True
    t = NT(m)
    p = t.bsdf((0.60, 0.42, 0.04), 0.6)
    worn = t.maprange(t.noise(t.coord("Object"), 0.9, 4).outputs["Fac"], 0.35, 0.7, 0.0, 0.55)
    t.link(t.mixc(worn, (0.60, 0.42, 0.04), (0.36, 0.34, 0.30)), p.inputs["Base Color"])
    return m


# ------------------------------------------------------------------------------------------------ material patches
_DONE: set = set()


def _base_socket(t: T):
    p = next((n for n in t.nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if p is None:
        return None, None
    inp = p.inputs["Base Color"]
    if inp.is_linked:
        return p, inp.links[0].from_socket
    rgb = tuple(inp.default_value)[:3]
    c = t.n("ShaderNodeRGB")
    c.outputs[0].default_value = (*rgb, 1.0)
    return p, c.outputs[0]


def _rewire_object_inputs(t: T) -> None:
    """Object Info Random -> web_rand; Generated -> web_gen (per object, survive the join)."""
    for node in list(t.nt.nodes):
        if node.type == "OBJECT_INFO":
            out = node.outputs.get("Random")
            if out and out.is_linked:
                a = t.vattr("web_rand")
                for l in list(out.links):
                    t.link(a.outputs["Fac"], l.to_socket)
        elif node.type == "TEX_COORD":
            out = node.outputs.get("Generated")
            if out and out.is_linked:
                a = t.vattr("web_gen")
                for l in list(out.links):
                    t.link(a.outputs["Vector"], l.to_socket)


def _edges(t: T, col, amount=0.22, radius=0.22):
    """Lighten arrises (Bevel normal vs true normal): sun-caught edges, paint wear."""
    bv = t.n("ShaderNodeBevel")
    bv.samples = 6
    t.set(bv, "Radius", radius)
    geo = t.n("ShaderNodeNewGeometry")
    d = t.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    t.link(bv.outputs["Normal"], d.inputs[0])
    t.link(geo.outputs["Normal"], d.inputs[1])
    e = t.maprange(t.sock(d.outputs, "Value"), 0.985, 0.80)
    return t.mul(col, t.math("ADD", 1.0, t.math("MULTIPLY", e, amount)))


def _frame(t: T):
    """(along, across, edge distance) in the face's own long-edge frame (world metres)."""
    co = t.xyz(t.coord("Object"))
    fax = t.xyz(t.vattr("web_fax").outputs["Vector"])
    fc = t.xyz(t.vattr("web_fc").outputs["Vector"])
    rx, ry = t.math("SUBTRACT", co[0], fc[0]), t.math("SUBTRACT", co[1], fc[1])
    along = t.math("ADD", t.math("MULTIPLY", rx, fax[0]), t.math("MULTIPLY", ry, fax[1]))
    across = t.math("SUBTRACT", t.math("MULTIPLY", ry, fax[0]), t.math("MULTIPLY", rx, fax[1]))
    edge = t.math("SUBTRACT", fax[2], t.math("ABSOLUTE", across))
    # world-anchored along coordinate (stable joints between neighbouring faces of the same deck)
    wa = t.math("ADD", t.math("MULTIPLY", co[0], fax[0]), t.math("MULTIPLY", co[1], fax[1]))
    wc = t.math("SUBTRACT", t.math("MULTIPLY", co[1], fax[0]), t.math("MULTIPLY", co[0], fax[1]))
    return along, across, edge, wa, wc, co


def _tank_shell(t: T, col):
    co = t.xyz(t.coord("Object"))
    tk = t.xyz(t.vattr("web_tank").outputs["Vector"])
    tk2 = t.xyz(t.vattr("web_tank2").outputs["Vector"])
    dx, dy = t.math("SUBTRACT", co[0], tk[0]), t.math("SUBTRACT", co[1], tk[1])
    ang = t.math("ARCTAN2", dy, dx)
    r, zr, H = tk2[0], t.math("SUBTRACT", co[2], tk2[1]), tk2[2]
    course = t.math("FLOOR", t.math("DIVIDE", zr, 2.44))
    hseam = t.line(zr, 2.44, 0.05)
    arc = t.math("ADD", t.math("MULTIPLY", ang, r), t.math("MULTIPLY", course, 4.8))
    vseam = t.line(arc, 9.6, 0.04)
    seam = t.math("MAXIMUM", hseam, t.math("MULTIPLY", vseam, 0.8))
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", seam, 0.16)))
    # top course: weathered band under the roof edge (rain run-off line)
    top = t.math("MULTIPLY", t.maprange(zr, t.math("SUBTRACT", H, 1.2), t.math("SUBTRACT", H, 0.4)), 0.12)
    col = t.mul(col, t.math("SUBTRACT", 1.0, top))
    # stencilled id on two sides
    img = t.n("ShaderNodeTexImage", interpolation="Closest", extension="CLIP")
    img.image = bpy.data.images["web_tank_labels"]
    lh = t.math("MINIMUM", t.math("MAXIMUM", t.math("MULTIPLY", H, 0.15), 1.6), 3.6)
    cw, ch = LABEL_CELL
    rows = max(1, math.ceil(img.image.size[1] / ch))
    lw = t.math("MULTIPLY", lh, cw / (ch - 2) * 1.0)
    cxi = t.math("MODULO", tk[2], LABEL_COLS)
    cyi = t.math("FLOOR", t.math("DIVIDE", tk[2], LABEL_COLS))
    mask = None
    for b in LABEL_BEARINGS:
        da = t.math("SUBTRACT", ang, math.radians(b))
        da = t.math("SUBTRACT", da, t.math("MULTIPLY", t.math("FLOOR", t.math("DIVIDE", t.math("ADD", da, math.pi), 2 * math.pi)), 2 * math.pi))
        u = t.math("ADD", t.math("DIVIDE", t.math("MULTIPLY", da, r), lw), 0.5)
        v = t.math("ADD", t.math("DIVIDE", t.math("SUBTRACT", zr, t.math("MULTIPLY", H, 0.62)), t.math("MULTIPLY", lh, ch / (ch - 2))), 0.5)
        inside = t.math("MULTIPLY", t.math("MULTIPLY", t.math("GREATER_THAN", u, 0.0), t.math("LESS_THAN", u, 1.0)),
                        t.math("MULTIPLY", t.math("GREATER_THAN", v, 0.0), t.math("LESS_THAN", v, 1.0)))
        uu = t.math("DIVIDE", t.math("ADD", cxi, t.math("MINIMUM", t.math("MAXIMUM", u, 0.0), 0.999)), LABEL_COLS)
        # atlas rows: cell 0 at the top of the image (bottom-up v)
        vv = t.math("DIVIDE", t.math("SUBTRACT", rows, t.math("ADD", cyi, t.math("SUBTRACT", 1.0, t.math("MINIMUM", t.math("MAXIMUM", v, 0.0), 0.999)))), rows)
        uv = t.n("ShaderNodeCombineXYZ")
        t.link(uu, uv.inputs[0])
        t.link(vv, uv.inputs[1])
        im = img if mask is None else t.n("ShaderNodeTexImage", interpolation="Closest", extension="CLIP")
        im.image = img.image
        t.link(uv.outputs[0], im.inputs["Vector"])
        m = t.math("MULTIPLY", t.xyz(t.sock(im.outputs, "Color"))[0], inside)
        mask = m if mask is None else t.math("MAXIMUM", mask, m)
    mask = t.math("MULTIPLY", mask, t.math("GREATER_THAN", H, 7.0))       # no stencil on small day tanks
    return t.mixc(t.math("MULTIPLY", mask, 0.92), col, (0.035, 0.035, 0.04))


def _tank_roof(t: T, col, floating=False):
    co = t.xyz(t.coord("Object"))
    tk = t.xyz(t.vattr("web_tank").outputs["Vector"])
    tk2 = t.xyz(t.vattr("web_tank2").outputs["Vector"])
    dx, dy = t.math("SUBTRACT", co[0], tk[0]), t.math("SUBTRACT", co[1], tk[1])
    rr = t.math("SQRT", t.math("ADD", t.math("MULTIPLY", dx, dx), t.math("MULTIPLY", dy, dy)))
    ang = t.math("ARCTAN2", dy, dx)
    r = tk2[0]
    rnd = t.vattr("web_rand").outputs["Fac"]
    if floating:
        base = t.mixc(t.maprange(rnd, 0.2, 0.8), (0.27, 0.27, 0.26), (0.36, 0.355, 0.34))
        seam = t.math("MAXIMUM", t.line(co[0], 2.4, 0.04), t.line(co[1], 7.2, 0.04))
    else:
        # cone roofs a shade darker than the shell (weathered paint, no fresh coat on the roof)
        base = t.mul(col, t.maprange(rnd, 0.0, 1.0, 0.66, 0.80))
        radial = t.line(t.math("MULTIPLY", ang, 18.0 / math.pi), 1.0, 0.035)       # 36 roof plates
        seam = t.math("MAXIMUM", t.math("MULTIPLY", radial, t.maprange(rr, 1.5, 3.0)), t.line(rr, 3.0, 0.04))
    base = t.mul(base, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", seam, 0.22)))
    rim = t.maprange(t.math("SUBTRACT", r, rr), 2.2, 0.3)
    dirt = t.maprange(t.noise(t.coord("Object"), 0.25, 4).outputs["Fac"], 0.4, 0.7, 0.25, 0.75)
    return t.mixc(t.math("MULTIPLY", rim, dirt), base, (0.30, 0.25, 0.20))


def _deck(t: T, col, px=6.0, py=4.0, joint=0.08):
    along, across, edge, wa, wc, co = _frame(t)
    j = t.math("MAXIMUM", t.line(wa, px, joint), t.line(wc, py, joint))
    exp_ = t.line(wa, px * 5, 0.10)                               # expansion joints
    tone = t.maprange(t.cellnoise(wa, wc, px, py), 0.0, 1.0, 0.84, 1.10)
    col = t.mul(col, tone)
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", t.math("MAXIMUM", j, exp_), 0.40)))
    stain = t.maprange(t.noise(t.coord("Object"), 0.18, 5).outputs["Fac"], 0.45, 0.75)
    return t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", stain, 0.12)))


def _metal_roof(t: T, col):
    along, across, edge, wa, wc, co = _frame(t)
    laps = t.line(wa, 1.05, 0.03)
    sky = t.math("MULTIPLY", t.line(wa, 7.5, 0.5, 0.05, 3.75), t.maprange(edge, 1.0, 1.6))   # skylight sheets
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", laps, 0.10)))
    col = t.mixc(t.math("MULTIPLY", sky, 0.55), col, (0.72, 0.74, 0.72))
    gutter = t.maprange(edge, 0.35, 0.12)
    streak = t.maprange(t.noise(t.vscale(t.coord("Object"), 0.3, 0.3, 1.0), 0.6, 4).outputs["Fac"], 0.5, 0.75)
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", streak, 0.12)))
    return t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", gutter, 0.40)))


def _flat_roof(t: T, col):
    along, across, edge, wa, wc, co = _frame(t)
    tone = t.maprange(t.cellnoise(wa, wc, 5.0, 3.5, 7.0), 0.0, 1.0, 0.90, 1.07)
    col = t.mul(col, tone)
    seams = t.math("MAXIMUM", t.line(wa, 5.0, 0.04), t.line(wc, 3.5, 0.03))
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", seams, 0.10)))
    dirt = t.maprange(edge, 0.9, 0.15)
    pond = t.maprange(t.noise(t.coord("Object"), 0.08, 3).outputs["Fac"], 0.58, 0.72)
    return t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", t.math("MAXIMUM", dirt, t.math("MULTIPLY", pond, 0.6)), 0.25)))


def _shed_wall(t: T, col):
    along, across, edge, wa, wc, co = _frame(t)
    seams = t.line(wa, 1.0, 0.03)
    col = t.mul(col, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", seams, 0.12)))
    dado = t.maprange(co[2], 1.3, 1.15)
    return t.mixc(dado, col, (0.36, 0.345, 0.32))


def _steel(t: T, col):
    """Painted steel: some members rust red-brown (per-object share, illustrative)."""
    rs = t.maprange(t.noise(t.coord("Object"), 0.05, 2).outputs["Fac"], 0.55, 0.68)
    rnd = t.maprange(t.vattr("web_rand").outputs["Fac"], 0.55, 0.75)
    return t.mixc(t.math("MULTIPLY", t.math("MAXIMUM", rs, rnd), 0.55), col, (0.30, 0.15, 0.085))


PATCH = {"tank_paint": _tank_shell, "roof_cone": _tank_roof, "deck_plate": lambda t, c: _tank_roof(t, c, True),
         "web_deck": _deck, "wharf_yard": lambda t, c: _deck(t, c, 7.5, 7.5, 0.07), "roof_metal": _metal_roof,
         "roof_flat": _flat_roof, "wall_cladding": _shed_wall, "steel": _steel}
NO_EDGES = {"web_paint_yellow", "ground", "far_ground", "asphalt", "asphalt_plain", "ballast", "coal", "bulk_white",
            "web_deck", "wharf_yard", "bund_floor"}


def patch(mats: list) -> list:
    done = []
    for m in mats:
        if m.name in _DONE or not m.use_nodes:
            continue
        _DONE.add(m.name)
        t = T(m)
        _rewire_object_inputs(t)
        p, col = _base_socket(t)
        if p is None:
            continue
        f = PATCH.get(m.name)
        if f:
            col = f(t, col)
        if m.name not in NO_EDGES:
            col = _edges(t, col)
        t.link(col, p.inputs["Base Color"])
        done.append(m.name)
    return done


def prepare(objs: list, group: str) -> dict:
    """Before bake_web.join(): single-user meshes, deck materials + safety lines (jetties), per-object stamps."""
    info = {"strips": 0, "deck_faces": 0}
    tank_index = {}
    if group == "tanks":
        tank_index, labels = tank_ids(objs)
        label_image(labels)
        info["labels"] = len([x for x in labels if x])
    for o in objs:
        if o.data.users > 1:
            o.data = o.data.copy()
    if group == "jetties":
        for o in objs:
            info["deck_faces"] += deck_material(o)
            info["strips"] += deck_lines(o)
    for o in objs:
        stamp(o, group, tank_index)
    return info


def strip_attrs(objs: list) -> None:
    """Drop the bake-only attributes before the glTF export (they must not ship as custom vertex data)."""
    for o in objs:
        for name in ("web_rand", "web_gen", "web_fax", "web_fc", "web_tank", "web_tank2"):
            if name in o.data.attributes:
                o.data.attributes.remove(o.data.attributes[name])


def cut_long(objs: list, step: float = 40.0) -> int:
    """Long faces (a 1 km jetty deck n-gon, deck lines) and large connected flat areas (road networks) become one UV
    island each, and the packer then
    shrinks every island of the collection to fit the longest: most of the atlas stayed empty (jetties: ~80%).
    Cut faces longer than `step` on a world grid and split the cut edges, so islands are at most `step` long."""
    n_cut = 0
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        cut_edges = []
        for axis in (0, 1):
            def ext(f):
                c = [v.co[axis] for v in f.verts]
                return min(c), max(c)
            # n-gons and quads only: bisecting long fan triangles (hip roofs on concave footprints) left slivers that
            # baked black; the grid-cell split below still breaks those fans into separate islands
            cand = [f for f in bm.faces if len(f.verts) > 3 and (lambda e: e[1] - e[0] > step)(ext(f))]
            if not cand:
                continue
            lo = min(ext(f)[0] for f in cand)
            hi = max(ext(f)[1] for f in cand)
            for k in range(math.floor(lo / step) + 1, math.floor(hi / step) + 1):
                c = k * step
                faces = [f for f in cand if f.is_valid and (lambda e: e[0] < c - 0.01 and e[1] > c + 0.01)(ext(f))]
                if not faces:
                    continue
                edges = list({e for f in faces for e in f.edges})
                verts = list({v for f in faces for v in f.verts})
                no = (1.0, 0.0, 0.0) if axis == 0 else (0.0, 1.0, 0.0)
                co = (c, 0.0, 0.0) if axis == 0 else (0.0, c, 0.0)
                res = bmesh.ops.bisect_plane(bm, geom=faces + edges + verts, plane_co=co, plane_no=no, dist=1e-4)
                cut_edges += [g for g in res["geom_cut"] if isinstance(g, bmesh.types.BMEdge)]
                new_faces = [g for g in res["geom"] if isinstance(g, bmesh.types.BMFace)]
                cand = [f for f in cand if f.is_valid] + new_faces
                cand = list({f for f in cand if len(f.verts) > 3 and ((lambda e: e[1] - e[0] > step)(ext(f)) or f in new_faces)})
        # large connected flat areas (road networks, wide roofs) are one island even with short faces: split the
        # edges between faces whose centres fall in different grid cells
        cell = {f: (math.floor(f.calc_center_median().x / step), math.floor(f.calc_center_median().y / step))
                for f in bm.faces}
        cut_edges += [e for e in bm.edges if len(e.link_faces) == 2 and
                      cell[e.link_faces[0]] != cell[e.link_faces[1]]]
        cut_edges = [e for e in set(cut_edges) if e.is_valid]
        if cut_edges:
            bmesh.ops.split_edges(bm, edges=cut_edges)
            n_cut += len(cut_edges)
        bm.to_mesh(o.data)
        bm.free()
    return n_cut
