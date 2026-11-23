"""Render one demo beat as a 4K cinematic clip: the Blender world (drones flying on the scenario's own trajectories at
the question's sim time) with SceneScout's evidence built as glowing 3D objects, revealed in two phases.

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/render_beat.py -- \
        --beat runs/blender/beats/<task_id>.json [--pct 100] [--frames 192] [--samples 48] [--step 1] [--preview]

Phase 1 (working, frames 1-70): evidence draws in bone white, as the agent's tool calls found it.
Phase 2 (answer, 70+): the accepted claim turns sodium orange, rejected evidence turns ash, the fix appears; anything
the operator must decide on turns signal orange. Colours follow docs/design/FINAL.md. Frames are resumable.
Output: runs/blender/beats/<task_id>/frames/f_####.png
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drones  # noqa: E402

REPO = drones.REPO
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


B = json.loads(Path(arg("--beat", "")).read_text())
FPS = 24
NF = int(arg("--frames", "192"))
F_ANSWER = 70
OUT = REPO / "runs" / "blender" / "beats" / B["task_id"] / ("preview" if "--preview" in argv else "frames")
OUT.mkdir(parents=True, exist_ok=True)

HEX = {"sodium": "#F2A33A", "ash": "#9A958C", "bone": "#ECE8E1", "signal": "#FF5A1F", "friendly": "#7DB0CC"}
STRENGTH = {"sodium": 2.6, "ash": 1.1, "bone": 1.8, "signal": 3.2, "friendly": 1.8}   # AgX keeps hue below ~3


def lin(hexs):
    h = hexs.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


coll = bpy.data.collections.new("SS_evidence")
bpy.context.scene.collection.children.link(coll)


def emission(name, color, alpha=1.0):
    """Emission material whose colour/strength can be keyframed (one material per animated object)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (*lin(HEX[color]), 1)
    em.inputs["Strength"].default_value = STRENGTH[color]
    if alpha < 1.0:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader")
        mx.inputs[0].default_value = alpha
        nt.links.new(tr.outputs[0], mx.inputs[1])
        nt.links.new(em.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs["Surface"])
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        m.blend_method = "BLEND" if hasattr(m, "blend_method") else None
    else:
        nt.links.new(em.outputs[0], out.inputs["Surface"])
    m["em"] = em.name
    return m


def recolor(mat, color, frame, ramp=8):
    """Keyframe a material from its current colour to `color` over `ramp` frames starting at `frame`."""
    em = mat.node_tree.nodes[mat["em"]]
    for f, col in ((frame, None), (frame + ramp, color)):
        if col:
            em.inputs["Color"].default_value = (*lin(HEX[col]), 1)
            em.inputs["Strength"].default_value = STRENGTH[col]
        em.inputs["Color"].keyframe_insert("default_value", frame=f)
        em.inputs["Strength"].keyframe_insert("default_value", frame=f)


def curve(name, pts, radius, mat, grow=None):
    """Polyline tube. grow=(f0, f1) animates bevel_factor_end 0->1 (the line draws itself)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1)
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    coll.objects.link(ob)
    if grow:
        cu.bevel_factor_end = 0.0
        cu.keyframe_insert("bevel_factor_end", frame=grow[0])
        cu.bevel_factor_end = 1.0
        cu.keyframe_insert("bevel_factor_end", frame=grow[1])
    return ob


def dashed(name, p0, p1, radius, mat, dash=40.0, gap=28.0, grow=None):
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    d = (p1 - p0).normalized()
    s, k, obs = 0.0, 0, []
    while s < L:
        e = min(s + dash, L)
        ob = curve(f"{name}_{k}", [p0 + d * s, p0 + d * e], radius, mat)
        if grow:                                   # dashes appear progressively along the line
            f = int(grow[0] + (grow[1] - grow[0]) * s / L)
            ob.hide_render = True
            ob.keyframe_insert("hide_render", frame=1)
            ob.hide_render = False
            ob.keyframe_insert("hide_render", frame=max(f, 2))
        obs.append(ob)
        s += dash + gap
        k += 1
    return obs


def ellipse_pts(cx, cy, z, a, b, theta_deg, n=72):
    th = math.radians(theta_deg or 0.0)
    out = []
    for i in range(n + 1):
        g = 2 * math.pi * i / n
        ca, sb = a * math.cos(g), b * math.sin(g)
        out.append((cx + ca * math.sin(th) + sb * math.cos(th), cy + ca * math.cos(th) - sb * math.sin(th), z))
    return out


def sphere(name, p, r, mat, appear=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=p, segments=24, ring_count=12)
    ob = bpy.context.active_object
    ob.name = name
    for c in ob.users_collection:
        c.objects.unlink(ob)
    coll.objects.link(ob)
    ob.data.materials.append(mat)
    if appear:
        ob.scale = (0.001,) * 3
        ob.keyframe_insert("scale", frame=appear)
        ob.scale = (1, 1, 1)
        ob.keyframe_insert("scale", frame=appear + 10)
    return ob


def scale_in(obs, frame, dur=10):
    for ob in obs:
        ob.hide_render = True
        ob.keyframe_insert("hide_render", frame=1)
        ob.hide_render = False
        ob.keyframe_insert("hide_render", frame=frame)


FONT = bpy.data.fonts.load(str(REPO / "data" / "assets" / "fonts" / "B612-Bold.ttf"))
CAM = None
LABELS = []


def label(text, p, color="bone", size=1.0, appear=1):
    cu = bpy.data.curves.new("lbl", "FONT")
    cu.body = text
    cu.font = FONT
    cu.align_x = "LEFT"
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new(f"lbl_{text[:12]}", cu)
    ob.location = Vector(p) + Vector((0, 0, 45))      # above tank roofs and sheds so structures never hide a label
    ob.data.materials.append(emission(f"lblmat_{len(LABELS)}", color))
    coll.objects.link(ob)
    LABELS.append((ob, size))
    if appear > 1:
        scale_in([ob], appear)
    return ob


# ------------------------------------------------------------------------------------------- evidence per beat
S = {k: Vector(v) for k, v in B["sensors"].items()}
pts_of_interest = []
t0 = min(float(B.get("t_s", 300.0)), 590.0)


def ray_end(s, bearing, L=3000.0):
    b = math.radians(bearing)
    return s + Vector((L * math.sin(b), L * math.cos(b), 0.0))


tmpl = B["template"]
if tmpl == "direct_path":
    for i, r in enumerate(B["rays"]):
        s = S[r["sensor"]]
        e = ray_end(s, r["bearing_deg"])
        m = emission(f"ray{i}", "bone")
        if r["status"] == "rejected":
            dashed(f"ray{i}", s, e, 2.2, m, grow=(6 + 6 * i, 40 + 6 * i))
            recolor(m, "ash", F_ANSWER)
            label(f"{r['bearing_deg']:05.1f}° reflection, rejected", s.lerp(e, 0.22) + Vector((0, 0, 30)), "ash", 1.0, F_ANSWER)
        elif r["status"] == "chosen":
            curve(f"ray{i}", [s, e], 3.2, m, grow=(6 + 6 * i, 40 + 6 * i))
            recolor(m, "sodium", F_ANSWER)
            label(f"{r['bearing_deg']:05.1f}° direct path", s.lerp(e, 0.30) + Vector((0, 0, 34)), "sodium", 1.15, F_ANSWER)
        elif r["sensor"] != B.get("focus_sensor"):
            curve(f"ray{i}", [s, e], 1.4, emission(f"rayo{i}", "ash", 0.6), grow=(10, 50))
        pts_of_interest += [s, s.lerp(e, 0.45)]
    if B.get("fit"):
        f = B["fit"]
        p = Vector(f["pos"])
        m = emission("fit", "bone")
        sphere("fix", p, 9, m, appear=46)
        curve("fix_stalk", [Vector((p.x, p.y, 0.5)), p], 1.2, m, grow=(46, 58))
        el = f["ell"]
        curve("fix_ell", ellipse_pts(p.x, p.y, 1.0, el["a_m"], el["b_m"], el.get("theta_deg")), 2.0, m, grow=(50, 66))
        recolor(m, "sodium", F_ANSWER + 6)
        label(f"fix {f['ref']}, 1σ ±{el['a_m']:.0f} m", p + Vector((0, 0, 40)), "sodium", 1.0, F_ANSWER + 6)
        pts_of_interest.append(p)
    fs = B.get("focus_sensor")
    if fs:
        label(f"{fs} AIR7311", S[fs] + Vector((0, 0, 40)), "friendly", 1.0)
        pts_of_interest.append(S[fs])
elif tmpl == "genuine_drone":
    tr = B["track"]
    p = Vector(tr["pos"])
    m = emission("trk", "bone")
    el = tr["ell"]
    curve("trk_ell", ellipse_pts(p.x, p.y, 1.0, el["a_m"], el["b_m"], el["theta_deg"]), 2.2, m, grow=(8, 40))
    curve("trk_stalk", [Vector((p.x, p.y, 0.5)), p], 1.2, m, grow=(20, 34))
    ring = curve("trk_ring", ellipse_pts(p.x, p.y, p.z, 14, 14, 0, 48), 1.6, m, grow=(30, 46))
    recolor(m, "sodium", F_ANSWER)
    probs = tr.get("klass_probs") or {}
    top = max(probs, key=probs.get) if probs else "drone"
    label(f"{tr['ref']}  {top} {probs.get(top, 0):.0%}, track confidence {tr.get('confidence', 0):.2f}", p + Vector((24, 0, 30)),
          "sodium", 1.0, F_ANSWER)
    pts_of_interest += [p, p + Vector((el["a_m"], 0, 0)), p - Vector((el["a_m"], 0, 0))]
elif tmpl == "ds_hover":
    D = B["drone"]
    path = [Vector(x) for x in D["path"]]
    mp = emission("dpath", "friendly")
    n = len(path)
    for i in range(n - 1):
        curve(f"dpath{i}", [path[i], path[i + 1]], 2.4, mp, grow=(8 + 18 * i, 24 + 18 * i))
    for i, ms in enumerate(D["measurements"]):
        a = Vector(ms["from"])
        e = ray_end(a, ms["bearing_deg"], (a - Vector(D["final"]["pos"])).length * 1.25)
        dashed(f"meas{i}", a, e, 1.4, emission(f"meas{i}", "bone", 0.8), dash=24, gap=18, grow=(24 + 18 * i, 36 + 18 * i))
        label(f"D1 {ms['wp']}  {ms['bearing_deg']:05.1f}°", a + Vector((0, 0, 24)), "friendly", 0.9, 24 + 18 * i)
    ests = D["estimates"]
    for i, es in enumerate(ests[:-1]):
        p, el = Vector(es["pos"]), es["ell"]
        m = emission(f"est{i}", "bone", 0.9)
        curve(f"est{i}", ellipse_pts(p.x, p.y, 2.0, el["a_m"], el["b_m"], el.get("theta_deg")), 1.6, m, grow=(6 + 18 * i, 18 + 18 * i))
        recolor(m, "ash", 18 + 18 * (i + 1))
    fin = D["final"]
    p, el = Vector(fin["pos"]), fin["ell"]
    m = emission("final", "bone")
    curve("final_ell", ellipse_pts(p.x, p.y, 2.0, el["a_m"], el["b_m"], el.get("theta_deg")), 2.6, m, grow=(F_ANSWER - 6, F_ANSWER + 6))
    sphere("final_pt", p + Vector((0, 0, 3)), 4, m, appear=F_ANSWER - 4)
    recolor(m, "sodium", F_ANSWER)
    label(f"emitter fixed ±{el['a_m']:.0f} m", p + Vector((0, 0, 34)), "sodium", 1.15, F_ANSWER)
    pts_of_interest += path + [p]
elif tmpl == "suggest_one":
    z = B["zone"]
    ringp = [(x, y, 2.0) for x, y in z["ring"]]
    curve("zone", ringp, 2.4, emission("zone", "ash"), grow=(6, 40))
    label(z["name"], Vector((sum(p[0] for p in ringp) / len(ringp), sum(p[1] for p in ringp) / len(ringp), 30)), "ash", 1.0)
    nd = B["node"]
    p = Vector(nd["pos"])
    m = emission("node", "bone")
    curve("mast", [Vector((p.x, p.y, 0.5)), p + Vector((0, 0, 18))], 2.6, m, grow=(30, 50))
    sphere("node_top", p + Vector((0, 0, 20)), 5, m, appear=48)
    recolor(m, "sodium", F_ANSWER)
    label(f"new DF node, DF error {nd['err_before']:.0f}° to {nd['err_after']:.0f}°", p + Vector((10, 0, 44)), "sodium", 1.1, F_ANSWER)
    pts_of_interest += [p] + [Vector(q) for q in ringp[::4]]
    for sid, s in S.items():
        if sid != "D1" and (s - p).length < 1500:
            label(sid, s + Vector((0, 0, 30)), "friendly", 0.8)
elif tmpl == "jam_links":
    fp = B["footprint"]
    mf = emission("fp", "bone", 0.55)
    for i, r in enumerate(fp["rings"][:40]):
        if len(r) < 3:
            continue
        curve(f"fp{i}", [(x, y, 3.0) for x, y in r], 1.8, mf, grow=(6, 46))
    recolor(mf, "sodium", F_ANSWER)
    ex, ey = fp["effector"][:2]
    e = Vector((ex, ey, 0))
    label(f"proposed 868/915 MHz from {B['effector']}, {fp['eirp_dbm']:.0f} dBm, not active", e + Vector((0, 0, 70)), "bone", 1.4)
    pts_of_interest += [e]
    for i, v in enumerate(B["victims"]):
        p = Vector((v["pos"][0], v["pos"][1], 25))
        m = emission(f"vic{i}", "bone")
        bpy.ops.mesh.primitive_cube_add(size=16, location=p)
        cub = bpy.context.active_object
        cub.rotation_euler = (math.radians(45), math.radians(35), 0)
        for c in cub.users_collection:
            c.objects.unlink(cub)
        coll.objects.link(cub)
        cub.data.materials.append(m)
        curve(f"vicstalk{i}", [Vector((p.x, p.y, 0.5)), p], 1.2, m)
        recolor(m, "signal", F_ANSWER + 4 * i)
        over = v["level_dbm"] - fp["threshold_dbm"]
        label(f"{v['ref'].split(':')[1]}  {v['level_dbm']:.0f} dBm, {over:.0f} dB over", p + Vector((14, 0, 26 + 30 * i)), "signal", 1.1,
              F_ANSWER + 4 * i)
        pts_of_interest.append(p)
    for r in fp["rings"][:40]:
        for q in r[::max(1, len(r) // 6)]:
            pts_of_interest.append(Vector((q[0], q[1], 0)))

for ob in coll.objects:                      # evidence glows; it must not cast shadows onto the world
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False

# ------------------------------------------------------------------------------------------- camera
c = sum(pts_of_interest, Vector()) / max(1, len(pts_of_interest))
rad = max(220.0, max((p - c).length for p in pts_of_interest))
if tmpl == "jam_links":
    rad *= 0.55
cam_data = bpy.data.cameras.new("SS_cam")
cam_data.lens = 32
cam_data.clip_end = 60000
cam = bpy.data.objects.new("SS_cam", cam_data)
coll.objects.link(cam)
tgt = bpy.data.objects.new("SS_tgt", None)
tgt.location = c
coll.objects.link(tgt)
con = cam.constraints.new("TRACK_TO")
con.target = tgt
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"
az0 = math.radians(float(arg("--az", "200")))
for f, az, dist, h in ((1, az0, rad * 2.7, rad * 1.25), (NF, az0 + math.radians(22), rad * 1.85, rad * 0.78)):
    cam.location = c + Vector((dist * math.sin(az), dist * math.cos(az), h))
    cam.keyframe_insert("location", frame=f)
if cam.animation_data and cam.animation_data.action:
    for fc in getattr(cam.animation_data.action, "fcurves", []):
        for kp in fc.keyframe_points:
            kp.interpolation = "BEZIER"
            kp.easing = "EASE_IN_OUT"
sc = bpy.context.scene
sc.camera = cam
CAM = cam

# labels face the camera and keep a constant screen size
def _labels(scn, depsgraph=None):
    cp = cam.matrix_world.translation
    for ob, size in LABELS:
        d = (ob.location - cp).length
        s = size * d * 0.021
        ob.scale = (s, s, s)
        ob.rotation_euler = (cp - ob.location).to_track_quat("Z", "Y").to_euler()


# ------------------------------------------------------------------------------------------- drones at sim time
Sx = drones.Scenario()
drones.setup(Sx)


def _frame(scn, depsgraph=None):
    f = scn.frame_current
    drones.place_drones(Sx, t0 + (f - 1) / FPS, f)
    _labels(scn)


for h in list(bpy.app.handlers.frame_change_pre):
    if getattr(h, "__name__", "") in ("_kw_frame", "_frame"):
        bpy.app.handlers.frame_change_pre.remove(h)
bpy.app.handlers.frame_change_pre.append(_frame)

sc.render.engine = "BLENDER_EEVEE"
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, NF
sc.render.resolution_x, sc.render.resolution_y = 3840, 2160
sc.render.resolution_percentage = int(arg("--pct", "100"))
sc.render.image_settings.file_format = "PNG"
if hasattr(sc.eevee, "taa_render_samples"):
    sc.eevee.taa_render_samples = int(arg("--samples", "48"))
step = int(arg("--step", "1"))
frames = [int(x) for x in arg("--only", "").split(",") if x] or list(range(1, NF + 1, step))
t_all, n = time.time(), 0
for f in frames:
    png = OUT / f"f_{f:04d}.png"
    if png.exists():
        continue
    sc.frame_set(f)
    _frame(sc)
    sc.render.filepath = str(png)
    bpy.ops.render.render(write_still=True)
    n += 1
    if n % 12 == 0 or len(frames) < 12:
        print(f"[render_beat] {B['task_id']} frame {f}/{NF}  {(time.time() - t_all) / n:.1f} s/frame", flush=True)
print(f"[render_beat] done {n} frames in {time.time() - t_all:.0f} s -> {OUT}", flush=True)
