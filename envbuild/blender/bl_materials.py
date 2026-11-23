"""Procedural PBR materials for the Kandla world (Blender 5.2). All colours are plausible, not measured."""

from __future__ import annotations

import bpy


class NT:
    """Small helper over a shader node tree."""

    def __init__(self, mat: bpy.types.Material):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def n(self, kind: str, **props):
        node = self.nt.nodes.new(kind)
        node.location = (self.x, 0)
        self.x -= 220
        for k, v in props.items():
            setattr(node, k, v)
        return node

    @staticmethod
    def sock(sockets, name, typ=None):
        cands = [s for s in sockets if s.name == name and (typ is None or s.type == typ)]
        if not cands:
            raise KeyError(f"socket {name} ({typ}) not in {[s.name for s in sockets]}")
        en = [s for s in cands if s.enabled]
        return (en or cands)[0]

    def link(self, a, b):
        self.nt.links.new(a, b)

    def set(self, node, name, value, typ=None):
        self.sock(node.inputs, name, typ).default_value = value

    def bsdf(self, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0):
        out = self.n("ShaderNodeOutputMaterial")
        p = self.n("ShaderNodeBsdfPrincipled")
        self.set(p, "Base Color", (*color, 1.0))
        self.set(p, "Roughness", rough)
        self.set(p, "Metallic", metal)
        self.link(p.outputs["BSDF"], out.inputs["Surface"])
        return p

    def coord(self, which="Object"):
        return self.n("ShaderNodeTexCoord").outputs[which]

    def noise(self, vec, scale, detail=4.0, rough=0.55, dims="3D", w=None):
        t = self.n("ShaderNodeTexNoise", noise_dimensions=dims)
        self.link(vec, t.inputs["Vector"])
        self.set(t, "Scale", scale)
        self.set(t, "Detail", detail)
        self.set(t, "Roughness", rough)
        if w is not None:
            self.set(t, "W", w)
        return t

    def mixc(self, fac, a, b):
        m = self.n("ShaderNodeMix", data_type="RGBA")
        self._in(m, "Factor", fac, "VALUE")
        self._in(m, "A", a, "RGBA")
        self._in(m, "B", b, "RGBA")
        return self.sock(m.outputs, "Result", "RGBA")

    def mixf(self, fac, a, b):
        m = self.n("ShaderNodeMix", data_type="FLOAT")
        self._in(m, "Factor", fac, "VALUE")
        self._in(m, "A", a, "VALUE")
        self._in(m, "B", b, "VALUE")
        return self.sock(m.outputs, "Result", "VALUE")

    def _in(self, node, name, v, typ):
        s = self.sock(node.inputs, name, typ)
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, s)
        else:
            s.default_value = (*v, 1.0) if typ == "RGBA" and len(v) == 3 else v

    def math(self, op, a, b=0.0, clamp=False):
        m = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if isinstance(v, bpy.types.NodeSocket):
                self.link(v, m.inputs[i])
            else:
                m.inputs[i].default_value = v
        return m.outputs[0]

    def maprange(self, v, fmin, fmax, tmin=0.0, tmax=1.0):
        m = self.n("ShaderNodeMapRange", clamp=True)
        self._in(m, "Value", v, "VALUE")
        for k, val in (("From Min", fmin), ("From Max", fmax), ("To Min", tmin), ("To Max", tmax)):
            self._in(m, k, val, "VALUE")
        return self.sock(m.outputs, "Result", "VALUE")

    def attr(self, name, typ="GEOMETRY"):
        a = self.n("ShaderNodeAttribute", attribute_name=name, attribute_type=typ)
        return a.outputs["Fac"]

    def bump(self, p, height, strength=0.2, dist=0.1):
        b = self.n("ShaderNodeBump")
        self.set(b, "Strength", strength)
        self.set(b, "Distance", dist)
        self.link(height, b.inputs["Height"])
        self.link(b.outputs["Normal"], p.inputs["Normal"])

    def xyz(self, vec):
        s = self.n("ShaderNodeSeparateXYZ")
        self.link(vec, s.inputs[0])
        return s.outputs

    def vscale(self, vec, sx, sy, sz):
        m = self.n("ShaderNodeVectorMath", operation="MULTIPLY")
        self.link(vec, m.inputs[0])
        m.inputs[1].default_value = (sx, sy, sz)
        return m.outputs[0]

    def rand(self):
        return self.n("ShaderNodeObjectInfo").outputs["Random"]


def _new(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    return m, NT(m)


S2_IMAGE = None      # set by build_world to the Sentinel-2 drape (bpy image), +-S2_HALF metres around the origin
S2_HALF = 15000.0
S2_GROUND = None     # same drape with roofs/tanks removed (envbuild.blender.s2_ground); used for the ground


def _s2(t, image=None):
    """Sentinel-2 true colour sampled by object XY (ENU metres)."""
    co = t.coord("Object")
    m = t.n("ShaderNodeMapping")
    t.link(co, m.inputs["Vector"])
    m.inputs["Location"].default_value = (0.5, 0.5, 0.0)
    m.inputs["Scale"].default_value = (1 / (2 * S2_HALF), 1 / (2 * S2_HALF), 1.0)
    img = t.n("ShaderNodeTexImage", interpolation="Cubic", extension="EXTEND")
    img.image = image or S2_IMAGE
    t.link(m.outputs["Vector"], img.inputs["Vector"])
    return img.outputs["Color"]


def ground():
    """Port land: Sentinel-2 colour (real, 10 m) modulated by procedural micro detail; wet banks darker."""
    m, t = _new("ground")
    p = t.bsdf()
    co = t.coord("Object")
    mid = t.noise(co, 0.05, 8)
    fine = t.noise(co, 1.5, 6)
    base = _s2(t, S2_GROUND) if S2_IMAGE else (0.3, 0.27, 0.22)
    detail = t.maprange(t.math("ADD", t.math("MULTIPLY", mid.outputs["Fac"], 0.6),
                                   t.math("MULTIPLY", fine.outputs["Fac"], 0.4)), 0.3, 0.7, 0.82, 1.15)
    cc = t.n("ShaderNodeCombineColor")
    for k in range(3):
        t.link(detail, cc.inputs[k])
    mul = t.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    t._in(mul, "Factor", 1.0, "VALUE")
    t._in(mul, "A", base, "RGBA")
    t._in(mul, "B", cc.outputs[0], "RGBA")
    col = t.sock(mul.outputs, "Result", "RGBA")
    sd = t.attr("sd")
    built0 = t.math("MAXIMUM", t.attr("built"), t.attr("industrial"))
    grav = t.mixc(t.maprange(fine.outputs["Fac"], 0.3, 0.7), (0.22, 0.21, 0.19), (0.36, 0.35, 0.32))
    grav = t.mixc(t.maprange(mid.outputs["Fac"], 0.45, 0.6), grav, (0.15, 0.14, 0.13))
    tint = t.n("ShaderNodeMix", data_type="RGBA", blend_type="OVERLAY")
    t._in(tint, "Factor", 0.5, "VALUE")
    t._in(tint, "A", grav, "RGBA")
    t._in(tint, "B", col, "RGBA")
    col = t.mixc(t.math("MULTIPLY", built0, 0.8), col, t.sock(tint.outputs, "Result", "RGBA"))
    # close-range detail: slab joints in yards, drying cracks on the banks, tyre/oil stains
    x, y, _z = t.xyz(co)
    jx = t.maprange(t.math("FRACT", t.math("DIVIDE", x, 6.0)), 0.0, 0.025, 1.0, 0.0)
    jy = t.maprange(t.math("FRACT", t.math("DIVIDE", y, 6.0)), 0.0, 0.025, 1.0, 0.0)
    joints = t.math("MULTIPLY", t.math("MAXIMUM", jx, jy), built0)
    col = t.mixc(t.math("MULTIPLY", joints, 0.35), col, (0.06, 0.06, 0.055))
    vor = t.n("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
    t.link(co, vor.inputs["Vector"])
    t.set(vor, "Scale", 0.7)
    crack = t.maprange(t.sock(vor.outputs, "Distance"), 0.0, 0.04, 1.0, 0.0)
    bank = t.maprange(sd, 30.0, 4.0)
    col = t.mixc(t.math("MULTIPLY", t.math("MULTIPLY", crack, bank), 0.5), col, (0.05, 0.045, 0.035))
    stain = t.maprange(t.noise(co, 0.35, 3).outputs["Fac"], 0.62, 0.72)
    col = t.mixc(t.math("MULTIPLY", t.math("MULTIPLY", stain, built0), 0.45), col, (0.04, 0.04, 0.04))
    wet = t.maprange(sd, 14.0, -1.0)
    col = t.mixc(t.math("MULTIPLY", wet, 0.45), col, (0.08, 0.07, 0.055))
    t.link(col, p.inputs["Base Color"])
    built = t.math("MAXIMUM", t.attr("built"), t.attr("industrial"))
    t.link(t.mixf(wet, t.mixf(built, 0.95, 0.8), 0.3), p.inputs["Roughness"])
    t.bump(p, t.math("ADD", fine.outputs["Fac"], t.math("MULTIPLY", mid.outputs["Fac"], 0.5)), 0.25, 0.2)
    return m


def far_ground():
    """Beyond the detailed tile: Sentinel-2 colour only (water areas a little glossier)."""
    m, t = _new("far_ground")
    p = t.bsdf((0.3, 0.27, 0.22), 0.9)
    if S2_IMAGE:
        x, y, _z = t.xyz(t.coord("Object"))
        edge = t.maprange(t.math("MAXIMUM", t.math("ABSOLUTE", x), t.math("ABSOLUTE", y)),
                          S2_HALF - 2500.0, S2_HALF - 200.0)
        t.link(t.mixc(edge, _s2(t), (0.13, 0.12, 0.09)), p.inputs["Base Color"])   # fade to mean colour, no smear
    return m


def water():
    """Kandla Creek: turbid, sediment-laden tidal water with animated small waves."""
    m, t = _new("water")
    p = t.bsdf((0.10, 0.11, 0.085), 0.06, 0.0)
    t.set(p, "IOR", 1.333)
    t.set(p, "Specular IOR Level", 0.4)
    co = t.coord("Object")
    plume = t.noise(co, 0.0015, 3)
    if S2_IMAGE:                       # turbid creek colour taken from the Sentinel-2 drape itself
        hs = t.n("ShaderNodeHueSaturation")
        t.link(_s2(t), hs.inputs["Color"])
        hs.inputs["Saturation"].default_value = 0.35            # kill roof/land colour bleeding at 10 m pixels
        t.link(t.mixc(0.45, hs.outputs["Color"], (0.12, 0.105, 0.075)), p.inputs["Base Color"])
    else:
        t.link(t.mixc(plume.outputs["Fac"], (0.085, 0.095, 0.075), (0.16, 0.15, 0.11)), p.inputs["Base Color"])
    w1 = t.noise(t.vscale(co, 1.0, 2.2, 1.0), 0.18, 6, 0.6, "4D")
    w2 = t.noise(co, 0.9, 4, 0.5, "4D")
    for node, speed in ((w1, 0.04), (w2, 0.12)):
        fc = t.sock(node.inputs, "W").driver_add("default_value")
        fc.driver.expression = f"frame*{speed}"
    t.bump(p, t.math("ADD", w1.outputs["Fac"], t.math("MULTIPLY", w2.outputs["Fac"], 0.35)), 0.18, 0.4)
    return m


def tank_paint():
    m, t = _new("tank_paint")
    p = t.bsdf((0.82, 0.82, 0.80), 0.42)
    co = t.coord("Object")
    gz = t.xyz(t.coord("Generated"))[2]
    streak = t.noise(t.vscale(co, 1.0, 1.0, 0.04), 1.2, 4)
    dirt = t.maprange(gz, 0.18, 0.0)
    base = t.mixc(t.rand(), (0.84, 0.84, 0.82), (0.72, 0.73, 0.72))
    c = t.mixc(t.math("MULTIPLY", t.maprange(streak.outputs["Fac"], 0.55, 0.75), 0.35), base, (0.52, 0.47, 0.40))
    c = t.mixc(t.math("MULTIPLY", dirt, 0.6), c, (0.48, 0.42, 0.34))
    t.link(c, p.inputs["Base Color"])
    seams = t.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z")
    t.link(co, seams.inputs["Vector"])
    t.set(seams, "Scale", 0.4)
    t.set(seams, "Distortion", 0.0)
    t.bump(p, t.maprange(seams.outputs["Fac"], 0.0, 0.08), 0.15, 0.02)
    return m


def flat(name, color, rough=0.6, metal=0.0, noise_amt=0.15, scale=0.5):
    m, t = _new(name)
    p = t.bsdf(color, rough, metal)
    nz = t.noise(t.coord("Object"), scale, 6)
    dark = tuple(max(0.0, c * (1 - noise_amt)) for c in color)
    t.link(t.mixc(nz.outputs["Fac"], dark, color), p.inputs["Base Color"])
    t.bump(p, nz.outputs["Fac"], 0.08, 0.05)
    return m


def corrugated(name="roof_metal"):
    m, t = _new(name)
    p = t.bsdf((0.6, 0.62, 0.63), 0.55, 0.35)
    r = t.rand()
    c = t.mixc(t.maprange(r, 0.0, 0.5), (0.20, 0.32, 0.46), (0.62, 0.64, 0.65))
    c = t.mixc(t.maprange(r, 0.75, 0.9), c, (0.78, 0.76, 0.70))
    nz = t.noise(t.coord("Object"), 0.15, 5)
    c = t.mixc(t.math("MULTIPLY", t.maprange(nz.outputs["Fac"], 0.55, 0.75), 0.4), c, (0.45, 0.30, 0.20))
    c = _imagery_tint(t, c, nz)
    t.link(c, p.inputs["Base Color"])
    wv = t.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="X", wave_profile="SIN")
    t.link(t.coord("UV"), wv.inputs["Vector"])
    t.set(wv, "Scale", 6.0)
    t.set(wv, "Distortion", 0.0)
    t.bump(p, wv.outputs["Fac"], 0.35, 0.03)
    return m


def _imagery_tint(t, c, nz):
    """Blend in the per-face colour sampled from Sentinel-2 (attribute roof_col; alpha = weight)."""
    a = t.n("ShaderNodeAttribute", attribute_name="roof_col", attribute_type="GEOMETRY")
    var = t.maprange(nz.outputs["Fac"], 0.3, 0.7, 0.85, 1.12)
    tinted = t.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    t._in(tinted, "Factor", 1.0, "VALUE")
    t._in(tinted, "A", a.outputs["Color"], "RGBA")
    cc = t.n("ShaderNodeCombineColor")
    for k in range(3):
        t.link(var, cc.inputs[k])
    t._in(tinted, "B", cc.outputs[0], "RGBA")
    return t.mixc(a.outputs["Alpha"], c, t.sock(tinted.outputs, "Result", "RGBA"))


def roof_flat():
    m, t = _new("roof_flat")
    p = t.bsdf((0.42, 0.41, 0.39), 0.85)
    nz = t.noise(t.coord("Object"), 0.4, 6)
    base = t.mixc(nz.outputs["Fac"], (0.33, 0.32, 0.30), (0.48, 0.47, 0.45))
    t.link(_imagery_tint(t, base, nz), p.inputs["Base Color"])
    t.bump(p, nz.outputs["Fac"], 0.08, 0.05)
    return m


def roof_flat_named(name, color):
    m, t = _new(name)
    p = t.bsdf(color, 0.7, 0.3)
    nz = t.noise(t.coord("Object"), 0.8, 6)
    dark = tuple(c * 0.8 for c in color)
    base = t.mixc(nz.outputs["Fac"], dark, color)
    t.link(_imagery_tint(t, base, nz), p.inputs["Base Color"])
    t.bump(p, nz.outputs["Fac"], 0.1, 0.05)
    return m


def wall():
    """Plaster/panel walls with window bands (object space)."""
    m, t = _new("wall")
    p = t.bsdf()
    co = t.coord("Object")
    x, y, z = t.xyz(co)
    u = t.math("ADD", x, y)
    row = t.maprange(t.math("FRACT", t.math("DIVIDE", z, 3.6)), 0.38, 0.40)
    row2 = t.maprange(t.math("FRACT", t.math("DIVIDE", z, 3.6)), 0.78, 0.76)
    col_ = t.maprange(t.math("FRACT", t.math("DIVIDE", u, 3.2)), 0.18, 0.20)
    col2 = t.maprange(t.math("FRACT", t.math("DIVIDE", u, 3.2)), 0.82, 0.80)
    win = t.math("MINIMUM", t.math("MINIMUM", row, row2), t.math("MINIMUM", col_, col2))
    low = t.maprange(z, 1.2, 1.6)
    win = t.math("MULTIPLY", win, low)
    r = t.rand()
    plaster = t.mixc(r, (0.80, 0.75, 0.64), (0.70, 0.71, 0.70))
    nz = t.noise(co, 0.6, 5)
    plaster = t.mixc(t.math("MULTIPLY", t.maprange(nz.outputs["Fac"], 0.5, 0.8), 0.3), plaster, (0.55, 0.50, 0.42))
    plaster = _imagery_tint(t, plaster, nz)          # per-building facade colour (face attribute roof_col)
    t.link(t.mixc(win, plaster, (0.06, 0.08, 0.10)), p.inputs["Base Color"])
    t.link(t.mixf(win, 0.85, 0.08), p.inputs["Roughness"])
    t.link(t.mixf(win, 0.0, 0.6), p.inputs["Metallic"])
    return m


def asphalt(name="asphalt", marked=True):
    """Road surface; UV u = metres along the road, v = 0..1 across. Centre dashes and edge lines on major roads."""
    m, t = _new(name)
    p = t.bsdf((0.11, 0.11, 0.105), 0.88)
    uv = t.xyz(t.coord("UV"))
    u, v = uv[0], uv[1]
    nz = t.noise(t.coord("Object"), 0.8, 6)
    base = t.mixc(nz.outputs["Fac"], (0.07, 0.07, 0.07), (0.17, 0.165, 0.16))
    dash = t.math("MULTIPLY", t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.018, 0.012),
                  t.maprange(t.math("FRACT", t.math("DIVIDE", u, 9.0)), 0.52, 0.48))
    edge = t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.43, 0.45)
    edge = t.math("MULTIPLY", edge, t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.475, 0.465))
    mark = t.math("MULTIPLY", t.math("MAXIMUM", dash, edge), 1.0 if marked else 0.0)
    t.link(t.mixc(mark, base, (0.78, 0.77, 0.72)), p.inputs["Base Color"])
    t.bump(p, nz.outputs["Fac"], 0.15, 0.02)
    return m


def ballast():
    """Rail bed with concrete sleepers every 0.6 m (UV u = metres along, v across)."""
    m, t = _new("ballast")
    p = t.bsdf((0.36, 0.34, 0.31), 0.95)
    uv = t.xyz(t.coord("UV"))
    sl = t.math("MULTIPLY", t.maprange(t.math("FRACT", t.math("DIVIDE", uv[0], 0.6)), 0.42, 0.38),
                t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", uv[1], 0.5)), 0.36, 0.34))
    nz = t.noise(t.coord("Object"), 6.0, 3)
    base = t.mixc(nz.outputs["Fac"], (0.25, 0.23, 0.21), (0.45, 0.43, 0.40))
    t.link(t.mixc(sl, base, (0.55, 0.54, 0.52)), p.inputs["Base Color"])
    t.bump(p, nz.outputs["Fac"], 0.6, 0.05)
    return m


def vegetation(name, color):
    m, t = _new(name)
    p = t.bsdf(color, 0.75)
    r = t.rand()
    c2 = tuple(min(1.0, c * 1.5) for c in color)
    t.link(t.mixc(r, color, c2), p.inputs["Base Color"])
    t.set(p, "Subsurface Weight", 0.15)
    nz = t.noise(t.coord("Object"), 4.0, 3)
    t.bump(p, nz.outputs["Fac"], 0.5, 0.1)
    return m


def hull():
    """Tanker hull: red antifouling below the boot top, dark grey topsides (object z = 0 at the waterline)."""
    m, t = _new("hull")
    p = t.bsdf((0.1, 0.1, 0.1), 0.45, 0.3)
    z = t.xyz(t.coord("Object"))[2]
    top = t.maprange(z, 0.4, 0.6)
    co = t.coord("Object")
    streak = t.noise(t.vscale(co, 0.15, 1.0, 1.0), 0.8, 4)            # vertical weathering streaks
    topside = t.mixc(t.math("MULTIPLY", t.maprange(streak.outputs["Fac"], 0.55, 0.75), 0.35),
                     (0.035, 0.04, 0.045), (0.12, 0.07, 0.045))
    boot = t.maprange(z, 0.3, 0.55)
    t.link(t.mixc(boot, (0.30, 0.055, 0.04), topside), p.inputs["Base Color"])
    return m


def glow(name, color, strength):
    m, t = _new(name)
    p = t.bsdf(color, 0.4)
    t.set(p, "Emission Color", (*color, 1.0))
    t.set(p, "Emission Strength", strength)
    return m


def build_all() -> dict[str, bpy.types.Material]:
    M = {
        "ground": ground(), "far_ground": far_ground(), "water": water(), "tank_paint": tank_paint(), "roof_metal": corrugated(),
        "wall": wall(), "roof_flat": roof_flat(), "asphalt": asphalt(), "asphalt_plain": asphalt("asphalt_plain", False), "ballast": ballast(),
        "wall_cladding": corrugated("wall_cladding"),
        "container_blue": flat("container_blue", (0.06, 0.16, 0.32), 0.55, 0.3, 0.25, 2.0),
        "container_red": flat("container_red", (0.38, 0.06, 0.04), 0.55, 0.3, 0.25, 2.0),
        "container_orange": flat("container_orange", (0.55, 0.22, 0.04), 0.55, 0.3, 0.25, 2.0),
        "container_grey": flat("container_grey", (0.35, 0.36, 0.36), 0.55, 0.3, 0.25, 2.0),
        "container_green": flat("container_green", (0.07, 0.22, 0.12), 0.55, 0.3, 0.25, 2.0),
        "concrete": flat("concrete", (0.60, 0.59, 0.56), 0.85, 0.0, 0.25, 0.4),
        "concrete_dark": flat("concrete_dark", (0.42, 0.41, 0.39), 0.9, 0.0, 0.25, 0.4),
        "steel": flat("steel", (0.30, 0.31, 0.32), 0.45, 0.7, 0.2, 1.0),
        "galvanised": flat("galvanised", (0.62, 0.63, 0.64), 0.35, 0.9, 0.15, 1.0),
        "rail_steel": flat("rail_steel", (0.55, 0.53, 0.50), 0.25, 1.0, 0.3, 2.0),
        "cladding": flat("cladding", (0.75, 0.76, 0.77), 0.3, 0.85, 0.1, 0.5),
        "yellow": flat("yellow", (0.85, 0.62, 0.08), 0.5, 0.1, 0.15, 1.0),
        "red": flat("red", (0.55, 0.06, 0.04), 0.5, 0.1, 0.2, 1.0),
        "white": flat("white_paint", (0.86, 0.86, 0.84), 0.45, 0.0, 0.1, 0.8),
        "deck_green": flat("deck_green", (0.20, 0.07, 0.05), 0.65, 0.1, 0.25, 0.5),
        "rubber": flat("rubber", (0.03, 0.03, 0.03), 0.8, 0.0, 0.2, 1.0),
        "floating_roof": flat("floating_roof", (0.45, 0.45, 0.44), 0.5, 0.6, 0.3, 0.3),
        "coal": flat("coal", (0.035, 0.034, 0.033), 0.9, 0.0, 0.5, 0.6),
        "bulk_white": flat("bulk_white", (0.78, 0.76, 0.70), 0.95, 0.0, 0.25, 0.5),
        "deck_plate": roof_flat_named("deck_plate", (0.52, 0.47, 0.38)),
        "deck_pontoon": roof_flat_named("deck_pontoon", (0.36, 0.33, 0.28)),
        "roof_cone": roof_flat_named("roof_cone", (0.70, 0.70, 0.68)),
        "deck_seam": flat("deck_seam", (0.30, 0.28, 0.25), 0.6, 0.4, 0.2, 1.0),
        "bund_floor": flat("bund_floor", (0.20, 0.19, 0.17), 0.9, 0.0, 0.35, 0.08),
        "hull": hull(),
        "hull_red": flat("hull_red", (0.26, 0.045, 0.03), 0.5, 0.2, 0.25, 0.4),
        "scrub": vegetation("scrub", (0.13, 0.15, 0.06)),
        "mangrove": vegetation("mangrove", (0.05, 0.10, 0.04)),
        "bark": flat("bark", (0.16, 0.12, 0.09), 0.9, 0.0, 0.3, 3.0),
        "lamp": glow("lamp", (1.0, 0.85, 0.6), 0.0),
        "antenna": flat("antenna", (0.05, 0.05, 0.05), 0.5, 0.0, 0.1, 1.0),
    }
    return M
