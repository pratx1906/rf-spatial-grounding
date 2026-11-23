"""Render the Kandla flythrough (EEVEE, GPU) frame by frame with a HUD sidecar per frame. Resumable.

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/render_video.py -- \
        [--out runs/blender/video] [--pct 100] [--start 1] [--end N] [--step 1]
"""

import json
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drones  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


out = Path(arg("--out", str(drones.REPO / "runs" / "blender" / "video")))
(out / "frames").mkdir(parents=True, exist_ok=True)
(out / "hud").mkdir(parents=True, exist_ok=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 3840, 2160
sc.render.resolution_percentage = int(arg("--pct", "100"))
sc.render.image_settings.file_format = "PNG"
D = drones.install(sc)
f0, f1, step = int(arg("--start", "1")), int(arg("--end", str(D.last))), int(arg("--step", "1"))
(out / "shots.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "cam"} for s in D.shots], indent=1))
t_all = time.time()
n = 0
for f in range(f0, f1 + 1, step):
    png = out / "frames" / f"f_{f:04d}.png"
    if png.exists():
        continue
    sc.frame_set(f)
    sh = D.shot_at(f)
    t_s = sh["t0"] + (f - sh["f0"]) / drones.FPS
    sc.render.filepath = str(png)
    bpy.ops.render.render(write_still=True)
    hud = drones.hud(sc, sc.camera)
    hud.update({"frame": f, "shot": sh["name"], "t_s": round(t_s, 2)})
    (out / "hud" / f"f_{f:04d}.json").write_text(json.dumps(hud))
    n += 1
    if n % 25 == 0:
        el = time.time() - t_all
        print(f"[render_video] frame {f}/{f1}  {el / n:.1f} s/frame", flush=True)
print(f"[render_video] done {n} frames in {time.time() - t_all:.0f} s", flush=True)
