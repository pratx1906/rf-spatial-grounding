"""Photo-scanned PBR upgrade for the Kandla world materials (Poly Haven CC0 textures, see fetch_polyhaven.py).

Every surface keeps its large-scale colour from the real data (Sentinel-2 ground, imagery roof colours, facade
tints) and gets photographed micro-detail (albedo variation, roughness, normal maps) at true metre scale.
Tiling is broken by sampling each texture twice (different scale and rotation) and blending with low-frequency
noise. Texture detail is normalised by the texture's mean colour, so it modulates the data colour, never replaces it.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

import bl_materials as BM

PH = Path(__file__).resolve().parents[2] / "data" / "assets" / "polyhaven"
_IMG: dict = {}
_MEAN: dict = {}


def available() -> bool:
    return (PH / "manifest.json").exists()


def _img(aid: str, kind: str):
    key = (aid, kind)
    if key in _IMG:
        return _IMG[key]
    f = next((p for p in (PH / aid).glob(f"{kind}.*")), None)
    if f is None:
        return None
    im = bpy.data.images.get(f"ph_{aid}_{kind}") or bpy.data.images.load(str(f), check_existing=True)
    im.name = f"ph_{aid}_{kind}"
    im.colorspace_settings.name = "sRGB" if kind == "diff" else "Non-Color"
    _IMG[key] = im
    return im


def _mean(aid: str) -> tuple:
    if aid not in _MEAN:
        im = _img(aid, "diff")
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        srgb = px.reshape(-1, 4)[::7, :3]            # byte images hold display-encoded (sRGB) values
        lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        _MEAN[aid] = tuple(float(v) for v in lin.mean(0))    # the shader sees linear colour
    return _MEAN[aid]


def _sample(t: BM.NT, aid: str, kind: str, vec):
    n = t.n("ShaderNodeTexImage", projection="BOX", interpolation="Linear")
    n.projection_blend = 0.3
    n.image = _img(aid, kind)
    t.link(vec, n.inputs["Vector"])
    return n


def _vec(t: BM.NT, scale_m: float, rot: float = 0.0, offset=(0.0, 0.0, 0.0)):
    m = t.n("ShaderNodeMapping")
    t.link(t.coord("Object"), m.inputs["Vector"])
    m.inputs["Scale"].default_value = (1 / scale_m, 1 / scale_m, 1 / scale_m)
    m.inputs["Rotation"].default_value = (0.0, 0.0, rot)
    m.inputs["Location"].default_value = offset
    return m.outputs["Vector"]


def pbr(t: BM.NT, aid: str, scale_m: float, antitile: bool = True):
    """-> dict(col, rough, nor, norm_col): norm_col = colour / texture mean (detail around 1.0)."""
    v1 = _vec(t, scale_m)
    d1, r1, n1 = (_sample(t, aid, k, v1) for k in ("diff", "rough", "nor"))
    col, rough, nor = d1.outputs["Color"], r1.outputs["Color"], n1.outputs["Color"]
    if antitile:
        v2 = _vec(t, scale_m * 2.71, 0.83, (13.7, 4.1, 0.0))
        d2, r2, n2 = (_sample(t, aid, k, v2) for k in ("diff", "rough", "nor"))
        nz = t.noise(t.coord("Object"), 0.6 / scale_m, 2)
        f = t.maprange(nz.outputs["Fac"], 0.4, 0.6)
        col = t.mixc(f, col, d2.outputs["Color"])
        rough = t.mixc(f, rough, r2.outputs["Color"])
        nor = t.mixc(f, nor, n2.outputs["Color"])
    mean = _mean(aid)
    dv = t.n("ShaderNodeVectorMath", operation="DIVIDE")
    t.link(col, dv.inputs[0])
    dv.inputs[1].default_value = tuple(max(m_, 0.02) for m_ in mean)
    return {"col": col, "rough": rough, "nor": nor, "norm_col": dv.outputs[0]}


def _bsdf(t: BM.NT):
    return next(n for n in t.nt.nodes if n.type == "BSDF_PRINCIPLED")


def _normal(t: BM.NT, p, nor, strength=0.8):
    nm = t.n("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = strength
    t.link(nor, nm.inputs["Color"])
    t.link(nm.outputs["Normal"], p.inputs["Normal"])


def _rough(t: BM.NT, p, rough, lo=0.0, hi=1.0):
    sep = t.n("ShaderNodeSeparateColor")
    t.link(rough, sep.inputs[0])
    t.link(t.maprange(sep.outputs[0], 0.0, 1.0, lo, hi), p.inputs["Roughness"])


def _modulate(t: BM.NT, base, detail_norm, amount=1.0):
    """base x detail (detail ~1 around its mean); amount blends from flat base to fully detailed."""
    mul = t.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    t._in(mul, "Factor", amount, "VALUE")
    t._in(mul, "A", base, "RGBA")
    t._in(mul, "B", detail_norm, "RGBA")
    return t.sock(mul.outputs, "Result", "RGBA")


def _near(t: BM.NT, d0=40.0, d1=900.0):
    """1 near the camera, 0 far away (texture detail fades where it would only alias)."""
    cam = t.n("ShaderNodeCameraData")
    return t.maprange(t.sock(cam.outputs, "View Distance"), d0, d1, 1.0, 0.25)


# ------------------------------------------------------------------------------------------------- materials
def ground():
    m, t = BM._new("ground")
    p = t.bsdf()
    base = BM._s2(t, BM.S2_GROUND) if BM.S2_IMAGE else (0.3, 0.27, 0.22)
    dry = pbr(t, "dry_ground_01", 4.0)
    sand = pbr(t, "coast_sand_01", 3.0)
    mud = pbr(t, "brown_mud_02", 3.5)
    crack = pbr(t, "mud_cracked_dry_03", 5.0)
    yard = pbr(t, "concrete_floor_worn_001", 5.0)
    grav = pbr(t, "tarred_gravel", 3.0)
    sd = t.attr("sd")
    built = t.math("MAXIMUM", t.attr("built"), t.attr("industrial"))
    wet = t.maprange(sd, 12.0, -1.0)
    bank = t.math("MULTIPLY", t.maprange(sd, 40.0, 10.0), t.math("SUBTRACT", 1.0, wet))
    nz = t.noise(t.coord("Object"), 0.01, 3)
    land_d = t.mixc(t.maprange(nz.outputs["Fac"], 0.42, 0.58), dry["norm_col"], sand["norm_col"])
    land_d = t.mixc(bank, land_d, crack["norm_col"])
    yard_d = t.mixc(t.maprange(t.noise(t.coord("Object"), 0.02, 3).outputs["Fac"], 0.45, 0.55),
                    yard["norm_col"], grav["norm_col"])
    det = t.mixc(built, land_d, yard_d)
    det = t.mixc(wet, det, mud["norm_col"])
    col = _modulate(t, base, det, _near(t, 40.0, 1800.0))
    # sub-10 m variation the imagery cannot carry: mottling (~30 m) and worn/stained patches (~120 m)
    co = t.coord("Object")
    mott = t.maprange(t.noise(co, 1 / 30.0, 4, 0.6).outputs["Fac"], 0.3, 0.7, 0.84, 1.14)
    patch = t.maprange(t.noise(co, 1 / 120.0, 3, 0.5).outputs["Fac"], 0.58, 0.68, 1.0, 0.78)
    shade = t.math("MULTIPLY", mott, patch)
    sh = t.n("ShaderNodeCombineColor")
    for k in range(3):
        t.link(shade, sh.inputs[k])
    mul = t.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
    t._in(mul, "Factor", 1.0, "VALUE")
    t._in(mul, "A", col, "RGBA")
    t._in(mul, "B", sh.outputs[0], "RGBA")
    col = t.sock(mul.outputs, "Result", "RGBA")
    col = t.mixc(t.math("MULTIPLY", wet, 0.35), col, (0.07, 0.06, 0.05))
    t.link(col, p.inputs["Base Color"])
    rough = t.mixc(built, t.mixc(bank, dry["rough"], crack["rough"]), yard["rough"])
    rough = t.mixc(wet, rough, mud["rough"])
    _rough(t, p, rough, 0.25, 1.0)
    nor = t.mixc(built, t.mixc(bank, dry["nor"], crack["nor"]), yard["nor"])
    nor = t.mixc(wet, nor, mud["nor"])
    _normal(t, p, nor, 1.0)
    return m


def asphalt(name="asphalt", marked=True):
    m, t = BM._new(name)
    p = t.bsdf()
    a = pbr(t, "worn_asphalt", 4.0)
    uv = t.xyz(t.coord("UV"))
    u, v = uv[0], uv[1]
    dash = t.math("MULTIPLY", t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.018, 0.012),
                  t.maprange(t.math("FRACT", t.math("DIVIDE", u, 9.0)), 0.52, 0.48))
    edge = t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.43, 0.45)
    edge = t.math("MULTIPLY", edge, t.maprange(t.math("ABSOLUTE", t.math("SUBTRACT", v, 0.5)), 0.475, 0.465))
    mark = t.math("MULTIPLY", t.math("MAXIMUM", dash, edge), 1.0 if marked else 0.0)
    worn = t.maprange(t.noise(t.coord("Object"), 0.4, 4).outputs["Fac"], 0.35, 0.65)
    base = _modulate(t, (0.12, 0.115, 0.11), a["norm_col"], 1.0)
    t.link(t.mixc(t.math("MULTIPLY", mark, worn), base, (0.70, 0.69, 0.64)), p.inputs["Base Color"])
    _rough(t, p, a["rough"], 0.55, 1.0)
    _normal(t, p, a["nor"], 0.8)
    return m


def concrete(name, tint, aid="concrete_floor_worn_001", scale=4.0):
    m, t = BM._new(name)
    p = t.bsdf()
    c = pbr(t, aid, scale)
    base = BM._imagery_tint(t, tint, t.noise(t.coord("Object"), 0.4, 4))   # imagery roof colour where present
    t.link(_modulate(t, base, c["norm_col"]), p.inputs["Base Color"])
    _rough(t, p, c["rough"], 0.5, 1.0)
    _normal(t, p, c["nor"], 0.8)
    return m


def cladding(name):
    """Corrugated metal sheds: real corrugation normals, imagery/palette tint, weathering streaks."""
    m, t = BM._new(name)
    p = t.bsdf((0.6, 0.62, 0.63), 0.5, 0.45)
    c = pbr(t, "corrugated_iron_02", 3.0, antitile=False)
    w = pbr(t, "worn_corrugated_iron", 3.0, antitile=False)
    r = t.rand()
    pal = t.mixc(t.maprange(r, 0.0, 0.5), (0.20, 0.32, 0.46), (0.62, 0.64, 0.65))
    pal = t.mixc(t.maprange(r, 0.75, 0.9), pal, (0.78, 0.76, 0.70))
    nz = t.noise(t.coord("Object"), 0.15, 5)
    pal = BM._imagery_tint(t, pal, nz)
    streak = t.maprange(t.noise(t.vscale(t.coord("Object"), 1.0, 1.0, 0.05), 0.5, 4).outputs["Fac"], 0.55, 0.75)
    det = t.mixc(t.math("MULTIPLY", streak, 0.6), c["norm_col"], w["norm_col"])
    t.link(_modulate(t, pal, det, 0.9), p.inputs["Base Color"])
    _rough(t, p, t.mixc(streak, c["rough"], w["rough"]), 0.3, 0.9)
    _normal(t, p, c["nor"], 1.0)
    return m


def plaster_wall():
    """Painted plaster facades (tint per building from roof_col) with window bands and dirt at the base."""
    m, t = BM._new("wall")
    p = t.bsdf()
    pl = pbr(t, "white_rough_plaster", 2.5)
    co = t.coord("Object")
    x, y, z = t.xyz(co)
    u = t.math("ADD", x, y)
    row = t.maprange(t.math("FRACT", t.math("DIVIDE", z, 3.6)), 0.38, 0.40)
    row2 = t.maprange(t.math("FRACT", t.math("DIVIDE", z, 3.6)), 0.78, 0.76)
    col_ = t.maprange(t.math("FRACT", t.math("DIVIDE", u, 3.2)), 0.22, 0.24)
    col2 = t.maprange(t.math("FRACT", t.math("DIVIDE", u, 3.2)), 0.78, 0.76)
    win = t.math("MINIMUM", t.math("MINIMUM", row, row2), t.math("MINIMUM", col_, col2))
    win = t.math("MULTIPLY", win, t.maprange(z, 1.2, 1.6))
    nz = t.noise(co, 0.6, 5)
    base = BM._imagery_tint(t, (0.66, 0.64, 0.60), nz)
    fade = t.n("ShaderNodeHueSaturation")                     # tropical sun bleaches paint
    t.link(base, fade.inputs["Color"])
    fade.inputs["Saturation"].default_value = 0.62
    fade.inputs["Value"].default_value = 0.86
    base = fade.outputs["Color"]
    grime = t.maprange(z, 0.0, 1.8, 1.0, 0.0)
    base = t.mixc(t.math("MULTIPLY", grime, 0.4), base, (0.30, 0.26, 0.20))
    # monsoon stains: streaks running down from window sills and the roof parapet
    stn = t.noise(t.vscale(co, 1.0, 1.0, 0.12), 2.2, 4, 0.6).outputs["Fac"]
    sill = t.maprange(t.math("FRACT", t.math("DIVIDE", z, 3.6)), 0.38, 0.05)
    stain = t.math("MULTIPLY", t.maprange(stn, 0.48, 0.70), t.math("ADD", t.math("MULTIPLY", sill, 0.6), 0.25))
    base = t.mixc(t.math("MULTIPLY", stain, 0.45), base, (0.24, 0.22, 0.19))
    plaster = _modulate(t, base, pl["norm_col"])
    # glass: reflective, per-window variation (some curtains/blinds, some open)
    cell = _combine(t, t.math("FLOOR", t.math("DIVIDE", u, 3.2)), t.math("FLOOR", t.math("DIVIDE", z, 3.6)))
    wn = t.n("ShaderNodeTexWhiteNoise", noise_dimensions="3D")
    t.link(cell, wn.inputs["Vector"])
    wv = t.sock(wn.outputs, "Value")
    glass = t.mixc(t.maprange(wv, 0.70, 0.72), (0.075, 0.09, 0.105), (0.30, 0.27, 0.22))       # curtains
    glass = t.mixc(t.maprange(wv, 0.90, 0.91), glass, (0.02, 0.02, 0.02))                     # open/dark
    t.link(t.mixc(win, plaster, glass), p.inputs["Base Color"])
    t.link(t.mixf(win, 0.9, t.mixf(t.maprange(wv, 0.70, 0.72), 0.05, 0.6)), p.inputs["Roughness"])
    t.link(t.mixf(win, 0.0, 0.7), p.inputs["Metallic"])
    _normal(t, p, pl["nor"], 0.6)
    return m


def tank_paint():
    """Tank shells (white / light grey / aluminium / cream paints by tank), plate seams from scanned painted metal,
    run-off streaks from the roof edge, dust and rust at the base. Weathering is illustrative."""
    m, t = BM._new("tank_paint")
    p = t.bsdf((0.66, 0.66, 0.64), 0.42)
    mp = pbr(t, "metal_plate_02", 3.0, antitile=False)
    rust = pbr(t, "rusty_metal_02", 2.0)
    co = t.coord("Object")
    gz = t.xyz(t.coord("Generated"))[2]
    r = t.rand()
    paint = t.mixc(t.maprange(r, 0.0, 0.55), (0.70, 0.70, 0.68), (0.58, 0.60, 0.60))      # white -> light grey
    paint = t.mixc(t.maprange(r, 0.78, 0.80), paint, (0.50, 0.51, 0.50))                  # aluminium
    paint = t.mixc(t.maprange(r, 0.92, 0.94), paint, (0.66, 0.60, 0.48))                  # cream
    # run-off streaks: vertical, strongest just below the roof edge, fading downwards
    st = t.noise(t.vscale(co, 1.0, 1.0, 0.03), 1.4, 5, 0.6)
    runs = t.math("MULTIPLY", t.maprange(st.outputs["Fac"], 0.52, 0.72), t.maprange(gz, 0.55, 0.98))
    dirt = t.mixc(t.math("MULTIPLY", runs, 0.55), paint, (0.30, 0.28, 0.24))
    # dust skirt and splash at the base, broken up by noise
    dn = t.noise(co, 0.35, 4).outputs["Fac"]
    skirt = t.math("MULTIPLY", t.maprange(gz, 0.16, 0.0), t.maprange(dn, 0.3, 0.7, 0.5, 1.0))
    dirt = t.mixc(t.math("MULTIPLY", skirt, 0.75), dirt, (0.42, 0.34, 0.25))
    # rust weeping from seams (sparse)
    rs = t.maprange(t.noise(t.vscale(co, 1.0, 1.0, 0.06), 2.0, 5).outputs["Fac"], 0.66, 0.80)
    painted = _modulate(t, dirt, mp["norm_col"], 0.5)
    rusty = _modulate(t, (0.30, 0.17, 0.10), rust["norm_col"])
    rw = t.math("MAXIMUM", t.math("MULTIPLY", rs, 0.5), t.math("MULTIPLY", skirt, 0.25))
    t.link(t.mixc(rw, painted, rusty), p.inputs["Base Color"])
    _rough(t, p, t.mixc(rw, mp["rough"], rust["rough"]), 0.32, 0.9)
    _normal(t, p, mp["nor"], 0.5)
    return m


def wharf_yard():
    """Cargo wharf / yard apron: real Sentinel-2 ground colour with worn concrete and tarred gravel detail."""
    m, t = BM._new("wharf_yard")
    p = t.bsdf()
    base = BM._s2(t, BM.S2_GROUND) if BM.S2_IMAGE else (0.35, 0.34, 0.32)
    yard = pbr(t, "concrete_floor_worn_001", 5.0)
    grav = pbr(t, "tarred_gravel", 3.0)
    f = t.maprange(t.noise(t.coord("Object"), 0.02, 3).outputs["Fac"], 0.45, 0.55)
    det = t.mixc(f, yard["norm_col"], grav["norm_col"])
    t.link(_modulate(t, base, det, _near(t)), p.inputs["Base Color"])
    _rough(t, p, t.mixc(f, yard["rough"], grav["rough"]), 0.5, 1.0)
    _normal(t, p, t.mixc(f, yard["nor"], grav["nor"]), 0.9)
    return m


def painted_steel(name, color, rust_amt=0.25):
    """Painted steel (ship superstructures, hulls): plate normals, rust streaks running down."""
    m, t = BM._new(name)
    p = t.bsdf(color, 0.45, 0.2)
    mp = pbr(t, "metal_plate_02", 2.5, antitile=False)
    rust = pbr(t, "rusty_metal_02", 2.0)
    co = t.coord("Object")
    streak = t.maprange(t.noise(t.vscale(co, 1.0, 1.0, 0.08), 1.6, 5).outputs["Fac"], 0.60, 0.80)
    painted = _modulate(t, color, mp["norm_col"], 0.4)
    rusty = _modulate(t, (0.30, 0.16, 0.09), rust["norm_col"])
    t.link(t.mixc(t.math("MULTIPLY", streak, rust_amt), painted, rusty), p.inputs["Base Color"])
    _rough(t, p, t.mixc(streak, mp["rough"], rust["rough"]), 0.3, 0.9)
    _normal(t, p, mp["nor"], 0.5)
    return m


def ship_hull():
    """Hull: red antifouling below the boot top, dark topsides with rust weeping (object z = 0 at waterline)."""
    m, t = BM._new("hull")
    p = t.bsdf((0.1, 0.1, 0.1), 0.5, 0.25)
    mp = pbr(t, "metal_plate_02", 3.0, antitile=False)
    rust = pbr(t, "rusty_metal_02", 2.5)
    co = t.coord("Object")
    z = t.xyz(co)[2]
    boot = t.maprange(z, 0.3, 0.55)
    streak = t.maprange(t.noise(t.vscale(co, 0.6, 1.0, 0.07), 1.4, 5).outputs["Fac"], 0.58, 0.78)
    r = t.rand()                                                     # topside colour per ship (illustrative)
    tc = t.mixc(t.maprange(r, 0.30, 0.31), (0.035, 0.04, 0.045), (0.02, 0.045, 0.11))     # black -> navy
    tc = t.mixc(t.maprange(r, 0.55, 0.56), tc, (0.03, 0.08, 0.05))                       # dark green
    tc = t.mixc(t.maprange(r, 0.75, 0.76), tc, (0.20, 0.21, 0.22))                       # grey
    tc = t.mixc(t.maprange(r, 0.90, 0.91), tc, (0.22, 0.035, 0.025))                     # maroon
    top = t.mixc(t.math("MULTIPLY", streak, 0.45), tc, _modulate(t, (0.25, 0.13, 0.07), rust["norm_col"]))
    col = t.mixc(boot, (0.30, 0.055, 0.04), top)
    t.link(_modulate(t, col, mp["norm_col"], 0.35), p.inputs["Base Color"])
    _rough(t, p, mp["rough"], 0.35, 0.8)
    _normal(t, p, mp["nor"], 0.6)
    return m


def upgrade_all(M: dict) -> dict:
    """Replace the procedural node trees of the main materials with the photo-scanned PBR versions (in place)."""
    if not available():
        return M
    M["ground"] = ground()
    M["asphalt"] = asphalt("asphalt", True)
    M["asphalt_plain"] = asphalt("asphalt_plain", False)
    M["concrete"] = concrete("concrete", (0.40, 0.385, 0.35))
    M["concrete_dark"] = concrete("concrete_dark", (0.26, 0.25, 0.235), "concrete_wall_006", 3.0)
    M["bund_floor"] = concrete("bund_floor", (0.22, 0.21, 0.19), "gravel_concrete_03", 4.0)
    M["roof_flat"] = concrete("roof_flat", (0.36, 0.35, 0.33), "gravel_concrete_03", 3.0)
    M["roof_metal"] = cladding("roof_metal")
    M["wall_cladding"] = cladding("wall_cladding")
    M["wall"] = plaster_wall()
    M["tank_paint"] = tank_paint()
    M["wharf_yard"] = wharf_yard()
    M["hull"] = ship_hull()
    M["hull_red"] = painted_steel("hull_red", (0.26, 0.045, 0.03), 0.35)
    M["white"] = painted_steel("white_paint", (0.66, 0.66, 0.64), 0.22)
    M["red"] = painted_steel("red", (0.50, 0.05, 0.035), 0.2)
    M["yellow"] = painted_steel("yellow", (0.62, 0.40, 0.04), 0.2)
    M["steel"] = painted_steel("steel", (0.28, 0.29, 0.30), 0.3)
    return M


# ------------------------------------------------------------------------------------------------- vegetation
SHRUBS = {"scrub": [("shrub_03", 1.6), ("shrub_02", 1.2), ("wild_rooibos_bush", 1.8), ("shrub_04", 1.4)],
          "mangrove": [("shrub_04", 1.9), ("shrub_02", 2.1), ("shrub_03", 2.4)]}


def import_shrub(aid: str, scale: float, coll, tag: str, target_tris: int = 1200):
    """Import a Poly Haven shrub, join, decimate to ~target_tris, base at z=0, link into `coll`."""
    f = next((PH / aid).glob("*.gltf"))
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(f))
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for o in [o for o in bpy.data.objects if o not in before and o.type != "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)
    ctx = bpy.context
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in new:
        o.select_set(True)
    ctx.view_layer.objects.active = new[0]
    if len(new) > 1:
        bpy.ops.object.join()
    ob = ctx.view_layer.objects.active
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    if tris > target_tris:
        mod = ob.modifiers.new("dec", "DECIMATE")
        mod.ratio = target_tris / tris
        bpy.ops.object.modifier_apply(modifier=mod.name)
    zmin = min(v.co.z for v in ob.data.vertices)
    for v in ob.data.vertices:
        v.co.z -= zmin
        v.co *= scale
    ob.name = f"veg_{tag}_{aid}"
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------------------------------------- world
def world_hdri(sun_az_deg: float, strength: float = 1.0):
    """Captured sky (Poly Haven, CC0) rotated so its sun sits at our sun azimuth. Returns the HDRI sun elevation."""
    import math
    f = next((p for p in PH.rglob("*.hdr")), None)
    if f is None:
        return None
    img = bpy.data.images.get(f.name) or bpy.data.images.load(str(f))
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    lum = px.reshape(h, w, 4)[..., :3].sum(-1)
    j, i = np.unravel_index(int(np.argmax(lum)), lum.shape)
    u, v = (i + 0.5) / w, (j + 0.5) / h                       # pixels: row 0 = bottom
    phi = (u - 0.5) * 2 * math.pi
    el = (v - 0.5) * math.pi
    d0_az = math.atan2(math.sin(phi), -math.cos(phi))          # direction (x=-cos phi, y=sin phi) as atan2(y, x)
    target = math.atan2(math.cos(math.radians(sun_az_deg)), math.sin(math.radians(sun_az_deg)))
    rot = target - d0_az
    wd = bpy.context.scene.world
    nt = wd.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    mp = nt.nodes.new("ShaderNodeMapping")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    mp.inputs["Rotation"].default_value = (0.0, 0.0, -rot)
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    for attr, val in (("sun_threshold", 5.0), ("sun_angle", math.radians(1.0)), ("use_sun_shadow", True)):
        if hasattr(wd, attr):
            setattr(wd, attr, val)
    return math.degrees(el)


# ------------------------------------------------------------------------------------------------- live water
def _drive(sock, expr: str, index: int | None = None):
    fc = sock.driver_add("default_value", index) if index is not None else sock.driver_add("default_value")
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = expr          # simple expression: evaluated without Python, safe and fast
    return fc


def flow_image(sd: np.ndarray, flow_npz: Path):
    """Pack the creek current (vx, vy m/s), the wake mask and the shore distance into one float image over the
    tile (10 m pixels): R = vx, G = vy, B = wake, A = signed shore distance (m). See envbuild/blender/water_flow.py."""
    f = np.load(flow_npz)
    vx, vy, wk = f["vx"], f["vy"], f["wake"]
    h, w = vx.shape
    sdd = sd[::2, ::2][:h, :w]
    img = bpy.data.images.get("water_flow")
    if img is None or tuple(img.size) != (w, h):
        if img:
            bpy.data.images.remove(img)
        img = bpy.data.images.new("water_flow", w, h, alpha=True, float_buffer=True, is_data=True)
    img.colorspace_settings.name = "Non-Color"
    img.alpha_mode = "CHANNEL_PACKED"
    px = np.stack([vx, vy, wk, sdd.astype(np.float32)], -1).astype(np.float32)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img, float(f["x0"]), float(f["y0"]), float(f["step"])


def _combine(t, x, y, z=0.0):
    c = t.n("ShaderNodeCombineXYZ")
    for k, v in enumerate((x, y, z)):
        if isinstance(v, bpy.types.NodeSocket):
            t.link(v, c.inputs[k])
        else:
            c.inputs[k].default_value = v
    return c.outputs[0]


def _vmath(t, op, a, b=None, scale=None):
    m = t.n("ShaderNodeVectorMath", operation=op)
    t.link(a, m.inputs[0])
    if b is not None:
        if isinstance(b, bpy.types.NodeSocket):
            t.link(b, m.inputs[1])
        else:
            m.inputs[1].default_value = b
    if scale is not None:
        if isinstance(scale, bpy.types.NodeSocket):
            t.link(scale, t.sock(m.inputs, "Scale"))
        else:
            t.sock(m.inputs, "Scale").default_value = scale
    return m.outputs[0]


def water_live(flow, period_s: float = 6.0):
    """Turbid tidal creek water, animated with the timeline (24 fps):
    - current: flow-map advection along the solved creek field (two phases cross-faded, period `period_s`), so
      ripples and foam drift downstream at the local current speed;
    - waves: swell (~35 m), wind chop (~5 m), ripples (~1 m); chop modulated by wind patches and raised in wakes;
      amplitudes are in noise units (noise Fac has ~0.15 contrast), tuned for surface slopes of ~0.05-0.1;
    - colour: Sentinel-2 creek colour (turbid, sediment-laden), lighter and muddier on the shallows;
    - foam: wake streaks behind piles/hulls and a breaking band along the banks; long smooth slicks along the flow;
    - distance-aware roughness so far water does not sparkle (no aliasing at 4K)."""
    img, x0, y0, step = flow
    m, t = BM._new("water")
    p = t.bsdf((0.10, 0.10, 0.075), 0.05, 0.0)
    t.set(p, "IOR", 1.333)
    t.set(p, "Specular IOR Level", 0.5)
    co = t.coord("Object")
    tv = t.n("ShaderNodeValue")
    tv.label = "time_s"
    _drive(tv.outputs[0], "frame/24")
    T = tv.outputs[0]
    # flow image lookup
    w, h = img.size
    uvm = t.n("ShaderNodeMapping")
    t.link(co, uvm.inputs["Vector"])
    uvm.inputs["Location"].default_value = ((-x0 + step / 2) / (w * step), (-y0 + step / 2) / (h * step), 0.0)
    uvm.inputs["Scale"].default_value = (1 / (w * step), 1 / (h * step), 1.0)
    ft = t.n("ShaderNodeTexImage", interpolation="Cubic", extension="EXTEND")
    ft.image = img
    t.link(uvm.outputs["Vector"], ft.inputs["Vector"])
    fx = t.xyz(ft.outputs["Color"])
    vx, vy, wake = fx[0], fx[1], fx[2]
    sd = ft.outputs["Alpha"]
    vel = _combine(t, vx, vy)
    # two-phase flow map; spatial jitter hides the cross-fade pulse
    jit = t.noise(co, 0.004, 1).outputs["Fac"]
    base = t.math("ADD", t.math("DIVIDE", T, period_s), t.math("MULTIPLY", jit, 3.0))
    ph0 = t.math("FRACT", base)
    ph1 = t.math("FRACT", t.math("ADD", base, 0.5))
    blend = t.math("ABSOLUTE", t.math("SUBTRACT", 1.0, t.math("MULTIPLY", ph0, 2.0)))   # 1 -> sample 1 only
    pos0 = _vmath(t, "SUBTRACT", co, _vmath(t, "SCALE", vel, scale=t.math("MULTIPLY", ph0, period_s)))
    pos1 = _vmath(t, "SUBTRACT", co, _vmath(t, "SCALE", vel, scale=t.math("MULTIPLY", ph1, period_s)))
    pos1 = _vmath(t, "ADD", pos1, (37.1, 11.3, 0.0))                    # decorrelate the two phases

    wind = t.maprange(t.noise(co, 1 / 320.0, 3).outputs["Fac"], 0.36, 0.64, 0.35, 1.0)    # gust patches
    turb = t.math("ADD", 1.0, t.math("MULTIPLY", wake, 1.8))

    def small_waves(pos):
        chop = t.noise(pos, 1 / 5.0, 3, 0.55, "4D")
        _drive(t.sock(chop.inputs, "W"), "frame/24*0.22")
        rip = t.noise(pos, 1 / 1.1, 2, 0.5, "4D")
        _drive(t.sock(rip.inputs, "W"), "frame/24*0.8")
        return t.math("ADD", t.math("MULTIPLY", chop.outputs["Fac"], 0.42),
                      t.math("MULTIPLY", rip.outputs["Fac"], 0.07))

    small = t.mixf(blend, small_waves(pos0), small_waves(pos1))
    swell = t.noise(t.vscale(co, 1.0, 1.7, 1.0), 1 / 35.0, 2, 0.5, "4D")
    _drive(t.sock(swell.inputs, "W"), "frame/24*0.05")
    height = t.math("ADD", t.math("MULTIPLY", swell.outputs["Fac"], 1.1),
                    t.math("MULTIPLY", small, t.math("MULTIPLY", wind, turb)))
    # slicks: long smooth streaks along the (mostly N-S) current
    slick_n = t.noise(t.vscale(pos0, 1.0, 0.12, 1.0), 1 / 18.0, 2)
    slick = t.math("MULTIPLY", t.maprange(slick_n.outputs["Fac"], 0.56, 0.64), 0.7)
    height = t.math("MULTIPLY", height, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", slick, 0.6)))
    t.bump(p, height, 1.0, 1.0)

    # colour: imagery creek colour, muddier on the shallows
    depth = t.maprange(sd, -4.0, -70.0)
    if BM.S2_IMAGE:
        hs = t.n("ShaderNodeHueSaturation")
        t.link(BM._s2(t), hs.inputs["Color"])
        hs.inputs["Saturation"].default_value = 0.6
        s2c = hs.outputs["Color"]
    else:
        s2c = (0.12, 0.11, 0.08)
    shallow = t.mixc(0.7, s2c, (0.105, 0.088, 0.062))
    deep = t.mixc(0.7, s2c, (0.042, 0.047, 0.038))
    col = t.mixc(depth, shallow, deep)
    # foam: wake streaks (advected) + breaking band along the bank
    fz0 = t.noise(t.vscale(pos0, 1.0, 0.45, 1.0), 1 / 3.0, 5, 0.65, "4D")
    fz1 = t.noise(t.vscale(pos1, 1.0, 0.45, 1.0), 1 / 3.0, 5, 0.65, "4D")
    for fz in (fz0, fz1):
        _drive(t.sock(fz.inputs, "W"), "frame/24*0.15")
    fzn = t.mixf(blend, fz0.outputs["Fac"], fz1.outputs["Fac"])
    wake_foam = t.math("MULTIPLY", t.maprange(wake, 0.08, 0.7), t.maprange(fzn, 0.50, 0.66))
    band = t.math("MULTIPLY", t.maprange(sd, -14.0, -3.0), t.maprange(sd, 0.5, -1.5))
    shore_foam = t.math("MULTIPLY", band, t.maprange(fzn, 0.47, 0.60))
    foam = t.math("MINIMUM", t.math("ADD", wake_foam, shore_foam), 1.0)
    col = t.mixc(t.math("MULTIPLY", foam, 0.85), col, (0.50, 0.48, 0.43))
    t.link(col, p.inputs["Base Color"])
    # roughness: slicks smoother, foam rough, far water rougher (micro-waves below pixel size, no sparkle)
    cam = t.n("ShaderNodeCameraData")
    far = t.maprange(cam.outputs["View Distance"], 150.0, 3500.0, 0.0, 1.0)
    r = t.mixf(far, 0.05, 0.24)
    r = t.mixf(slick, r, 0.015)
    r = t.mixf(foam, r, 0.75)
    t.link(r, p.inputs["Roughness"])
    m["source"] = "ILLUSTRATIVE ebb-tide current (water_flow.py); colour from Sentinel-2; waves procedural"
    return m
