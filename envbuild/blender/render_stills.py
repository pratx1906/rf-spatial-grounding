"""Render 4K stills of the Kandla world with Cycles on the GPU (headless).

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/render_stills.py -- \
        [--cams cam_aerial,cam_tanker] [--samples 160] [--pct 100] [--out runs/blender/stills]
"""

import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_world  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


cams = arg("--cams", "cam_aerial,cam_tankfarm,cam_tankfarm_close,cam_tanker,cam_creek").split(",")
out = Path(arg("--out", str(build_world.REPO / "runs" / "blender" / "stills")))
out.mkdir(parents=True, exist_ok=True)
build_world.setup_cycles(int(arg("--samples", "160")))
sc = bpy.context.scene
sc.render.resolution_percentage = int(arg("--pct", "100"))
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
for c in cams:
    sc.camera = bpy.data.objects[c]
    sc.render.filepath = str(out / f"{c}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render_stills] {c}: {time.time() - t0:.1f}s -> {sc.render.filepath}", flush=True)
