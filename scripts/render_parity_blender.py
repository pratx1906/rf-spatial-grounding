"""Render parity (Task B): Blender reference frames at the views of docs/design/audit/shots_render/views.json, with the
.blend's own engine settings, world, sun, view transform and compositor (mist + bloom). The .blend is never saved.

    blender -b data/site_kandla/blender/kandla_world.blend -P scripts/render_parity_blender.py -- OUTDIR [W H] [views...]

40 deg vertical fov, roll 0 (same as the console camera). KW_sensors is hidden: the console does not draw the sensor
masts from the world tiles. Run on an idle GPU (python scripts/gpu_lock.py run bench -- blender ...).
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

REPO = Path(bpy.data.filepath).resolve().parents[3]
a = sys.argv[sys.argv.index("--") + 1:]
out = Path(a[0]).resolve()
out.mkdir(parents=True, exist_ok=True)
w, h = (int(a[1]), int(a[2])) if len(a) > 2 else (1600, 900)
views = {k: v for k, v in json.loads((REPO / "docs/design/audit/shots_render/views.json").read_text()).items() if not k.startswith("_")}
pick = a[3:] or list(views)
sc = bpy.context.scene
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
sc.render.image_settings.file_format = "PNG"
col = bpy.data.collections.get("KW_sensors")
if col:
    col.hide_render = True
cd = bpy.data.cameras.new("parity")
cd.sensor_fit = "VERTICAL"
cd.angle_y = math.radians(40)
cd.clip_start, cd.clip_end = 1.0, 60000.0
cam = bpy.data.objects.new("parity", cd)
sc.collection.objects.link(cam)
sc.camera = cam
for name in pick:
    pos, tgt = views[name]
    cam.location = pos
    cam.rotation_euler = (Vector(tgt) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = str(out / f"{name}.png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {name} {sc.render.engine} {w}x{h} {time.time() - t:.1f}s", flush=True)
