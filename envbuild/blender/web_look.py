"""Export the Blender world's camera look for the three.js console (CPU only, no render; the .blend is never saved).

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/web_look.py
    -> data/site_kandla/web/look_lut.png   Blender's own view transform as a 3D LUT (AgX + its look; exposure NOT
                                           included, the console applies it): 65^3 strip, width = 65 * 65 (x = r + 65 b),
                                           height 65 (y = g, row 0 = top = g 64). Input coordinate per channel:
                                           t = (log2(max(c, 1e-10) / 0.18) - LO) / (HI - LO), scene-linear Rec.709.
       data/site_kandla/web/sky_1k.hdr     the world HDRI as the camera sees it: the World Mapping node's rotation baked
                                           in (a pure longitude shift), 1024 x 512 area-averaged, three.js equirect
                                           convention (equal to Blender's at rotation 0), strength NOT included. Note: the
                                           HDRI's own sun (azimuth 65, el 47.8) is opposite the KW_sun lamp (245, 47.8);
                                           Blender lights with both (the HDRI sun as soft world light), so does the console.
       data/site_kandla/web/manifest.json  "look": exposure, world strength, sun direction / colour / strength, mist,
                                           bloom, LUT domain (the console reads these instead of hard-coded guesses).
Why: the console used three.js AgX (no "Medium High Contrast" look), a guessed exposure, the HDRI unrotated and at full
strength as background, and the sun at 28 deg elevation while the .blend's KW_sun stands at ~48 deg.
"""
import json
import math
from pathlib import Path

import bpy
import numpy as np

REPO = Path(bpy.data.filepath).resolve().parents[3]
OUT = REPO / "data" / "site_kandla" / "web"
N, LO, HI = 65, -12.5, 4.5


def bake_lut(sc) -> None:
    k = np.arange(N)
    lin = 0.18 * np.exp2(LO + (HI - LO) * k / (N - 1))
    r, g, b = np.meshgrid(lin, lin, lin, indexing="ij")          # [r, g, b]
    W, H = N * N, N
    px = np.zeros((H, W, 4), np.float32)                          # Blender pixel rows: row 0 = bottom
    for bi in range(N):
        # x = r + N * b, y = g
        px[:, bi * N:(bi + 1) * N, 0] = r[:, :, bi].T
        px[:, bi * N:(bi + 1) * N, 1] = g[:, :, bi].T
        px[:, bi * N:(bi + 1) * N, 2] = b[:, :, bi].T
    px[..., 3] = 1.0
    img = bpy.data.images.new("look_lut_in", W, H, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709"
    img.pixels.foreach_set(px.ravel())
    vs = sc.view_settings
    keep = (vs.exposure, vs.gamma)
    vs.exposure, vs.gamma = 0.0, 1.0
    s = sc.render.image_settings
    s.file_format, s.color_mode, s.color_depth = "PNG", "RGB", "8"
    f = OUT / "look_lut.png"
    img.save_render(str(f), scene=sc)
    vs.exposure, vs.gamma = keep
    print("LUT", f, vs.view_transform, vs.look, f.stat().st_size)


def write_hdr(path: Path, rgb: np.ndarray) -> None:
    """Radiance RGBE, flat scanlines (three.js RGBELoader reads them). Blender's own HDR writer clamps a generated
    float image to ~96, which would cut the HDRI's sun (75 000) and two thirds of the sky's irradiance."""
    h, w, _ = rgb.shape
    m = rgb.max(-1)
    mant, ex = np.frexp(m)
    sc = np.where(m > 1e-32, mant * 256.0 / np.maximum(m, 1e-32), 0.0)
    q = np.zeros((h, w, 4), np.uint8)
    q[..., :3] = np.clip(rgb * sc[..., None], 0, 255).astype(np.uint8)
    q[..., 3] = np.where(m > 1e-32, ex + 128, 0).astype(np.uint8)
    path.write_bytes(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode() + q.tobytes())


def bake_sky(sc) -> dict:
    wn = sc.world.node_tree.nodes
    env = next(n for n in wn if n.type == "TEX_ENVIRONMENT")
    mp = next((n for n in wn if n.type == "MAPPING"), None)
    bg = next(n for n in wn if n.type == "BACKGROUND")
    rot = mp.inputs["Rotation"].default_value if mp else (0, 0, 0)
    assert abs(rot[0]) < 1e-6 and abs(rot[1]) < 1e-6, rot
    th = float(rot[2])
    src = env.image
    sw, sh = src.size
    a = np.empty(sw * sh * 4, np.float32)
    src.pixels.foreach_get(a)
    a = a.reshape(sh, sw, 4)[..., :3]
    W, H = 1024, 512
    f = sw // W                                                    # area-average to 1024 x 512 (keeps the sun's energy)
    a = a.reshape(H, sh // H, W, f, 3).mean((1, 3))
    # A Z rotation of the lookup only shifts longitude: Blender reads the HDRI at R_z(th) d, and both equirect
    # conventions put azimuth atan2(N, E) at u = 0.5 - az / 2pi, so out(u) = src(u - th / 2pi). Exact, energy-preserving
    # linear column interpolation (checked: the HDRI sun lands at azimuth 65 deg, elevation 47.8 deg, as a Cycles probe
    # of the .blend's world shows).
    sx = (th / (2 * np.pi)) % 1.0 * W                               # np.roll(a, k)[j] = a[j - k]
    i0 = int(np.floor(sx))
    fr = sx - i0
    out = np.roll(a, i0, axis=1) * (1 - fr) + np.roll(a, i0 + 1, axis=1) * fr
    write_hdr(OUT / "sky_1k.hdr", out[::-1])                      # file rows top-down
    print("SKY", src.filepath, "rot_z_deg", math.degrees(th), "max", float(out.max()))
    k = float(bg.inputs["Strength"].default_value)
    lat = ((np.arange(H) + 0.5) / H - 0.5) * np.pi                   # rows bottom-up
    w = np.cos(lat) * np.clip(np.sin(lat), 0, None) * (np.pi / H) * (2 * np.pi / W)
    e_up = (out * w[:, None, None]).sum((0, 1)) * k                 # irradiance on an upward surface, HDRI sun included
    return {"hdri": Path(bpy.path.abspath(src.filepath)).name, "rotation_z_deg": round(math.degrees(th), 3),
            "strength": round(k, 4), "irradiance_up": [round(float(x), 4) for x in e_up]}


def main() -> None:
    sc = bpy.context.scene
    vs = sc.view_settings
    bake_lut(sc)
    sky = bake_sky(sc)
    sun = next(o for o in bpy.data.objects if o.type == "LIGHT" and o.data.type == "SUN" and not o.hide_render)
    from mathutils import Vector
    d = -(sun.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()          # toward the sun, ENU
    az = (math.degrees(math.atan2(d.x, d.y)) + 360) % 360
    el = math.degrees(math.asin(d.z))
    mist = sc.world.mist_settings
    look = {"source": "envbuild/blender/web_look.py from kandla_world.blend",
            "view_transform": vs.view_transform, "look": vs.look, "exposure_stops": round(vs.exposure, 4),
            "lut": {"file": "look_lut.png", "n": N, "log2_lo": LO, "log2_hi": HI, "mid_grey": 0.18},
            "world": sky,
            "sun": {"azimuth_deg": round(az, 2), "elevation_deg": round(el, 2), "strength": round(sun.data.energy, 3),
                    "colour": [round(c, 4) for c in sun.data.color], "angle_deg": round(math.degrees(sun.data.angle), 3)},
            "mist": {"start_m": mist.start, "depth_m": mist.depth, "falloff": mist.falloff}}
    ct = getattr(sc, "compositing_node_group", None)
    if ct:
        mix = next((n for n in ct.nodes if n.type == "MIX"), None)
        mth = next((n for n in ct.nodes if n.type == "MATH"), None)
        gl = next((n for n in ct.nodes if n.type == "GLARE"), None)
        if mix and mth:
            look["mist"].update({"colour": [round(c, 4) for c in mix.inputs["B"].default_value[:3]],
                                 "factor": round(mth.inputs[1].default_value, 4)})
        if gl:
            look["bloom"] = {k.lower(): round(gl.inputs[k].default_value, 4) for k in ("Threshold", "Strength", "Size")}
    mp = OUT / "manifest.json"
    man = json.loads(mp.read_text())
    man["look"] = look
    mp.write_text(json.dumps(man, indent=1))
    print("LOOK", json.dumps(look))


main()
