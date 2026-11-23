"""Ground assets for the three.js console, from the Blender world bundle (same sources, same ENU frame).

    python scripts/web_ground.py [--out DIR] [--veg-only | --cls-only | --flow-only]
        -> data/site_kandla/web/{terrain.bin, terrain.json, ground.jpg, detail.jpg, veg_*.bin}

terrain.bin: float32 heights (m) on a 20 m grid over the detailed tile, row 0 = south (y0), column 0 = west (x0).
ground.jpg: Sentinel-2 L2A colour with structures removed (s2_ground.png), 30 km square centred on the site origin,
row 0 = north. "Contains modified Copernicus Sentinel data 2026".
veg_<asset>.bin: the vegetation instances of veg_<asset>.json (export_web.py) as int16 rows
[x / 0.25 m, y / 0.25 m, z / 0.01 m, rot_z / 1e-4 rad, scale / 1e-3]: 10 bytes per instance instead of ~40 bytes of JSON,
no JSON parse in the browser; manifest.vegetation.<asset>.bin names it.
ground_cls.webp (--cls-only builds just this): ground classes on the 5 m terrain grid for the console's near-camera
detail textures, row 0 = north over the detailed tile: R = paved (OSM industrial / port landuse, built-up), G = bank
and mudflat (within ~40 m of the waterline, from the terrain's signed shore distance), B = under water. Smoothed
over ~15 m so class edges blend; manifest.ground.cls names it.
water_flow.png (--flow-only builds just this): the Blender water's ILLUSTRATIVE ebb-tide flow field
(data/site_kandla/blender/water_flow.npz, envbuild/blender/water_flow.py) for the console's wake foam, as in
bl_pbr.water_live: R, G = velocity east / north (v / VSCALE + 0.5), B = wake (0-1); row 0 = north; manifest.water.
water_shore.png (with --flow-only): the terrain's signed shore distance on the same 10 m grid (the alpha channel of
Blender's flow image, bl_pbr.flow_image: sd[::2, ::2]), grey = (sd + SHORE_LO) / SHORE_SPAN, so -20 m (water) .. +5 m
(land); the console draws Blender's breaking band along the banks from it (manifest.water.shore, shore_lo, shore_span).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BUN = REPO / "data" / "site_kandla" / "blender"
OUT = REPO / "data" / "site_kandla" / "web"
if "--out" in sys.argv:
    OUT = (REPO / sys.argv[sys.argv.index("--out") + 1]).resolve()
Image.MAX_IMAGE_PIXELS = None


def veg_binary() -> None:
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text())
    for name, v in (man.get("vegetation") or {}).items():
        rows = np.array(json.loads((OUT / v["instances"]).read_text()), np.float64)
        q = np.round(rows / np.array([0.25, 0.25, 0.01, 1e-4, 1e-3])).astype(np.int64)
        assert np.abs(q).max() < 32767, (name, np.abs(q).max(0))
        f = OUT / v["instances"].replace(".json", ".bin")
        q.astype("<i2").tofile(f)
        v["bin"] = f.name
        print(f"veg {name}: {len(rows)} instances -> {f.name} {f.stat().st_size / 1e6:.2f} MB")
    man_path.write_text(json.dumps(man, indent=1))


def ground_classes() -> None:
    from PIL import ImageFilter
    d = np.load(BUN / "terrain.npz")
    sd = d["sd"].astype(np.float32)
    paved = (d["industrial"] | d["built"]).astype(np.float32)
    bank = np.clip(1.0 - (sd - 4.0) / 36.0, 0.0, 1.0) * (sd > -3.0)
    under = (sd < -1.0).astype(np.float32)
    rgb = np.stack([paved, bank, under], -1)[::-1]                 # row 0 = north
    im = Image.fromarray((rgb * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5))
    f = OUT / "ground_cls.webp"
    im.save(f, quality=85)
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text())
    man.setdefault("ground", {})["cls"] = f.name
    man_path.write_text(json.dumps(man, indent=1))
    print(f"ground classes {im.size} -> {f.name} {f.stat().st_size / 1e3:.0f} kB")


SHORE_LO, SHORE_SPAN = 20.0, 25.0                                  # shore distance -20 .. +5 m -> 0 .. 1


def water_flow() -> None:
    d = np.load(BUN / "water_flow.npz")
    vs = 4.0                                                        # m/s full scale (max speed ~1.5)
    rgb = np.stack([d["vx"] / vs + 0.5, d["vy"] / vs + 0.5, d["wake"]], -1)[::-1]
    f = OUT / "water_flow.png"
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f, optimize=True)
    n = d["vx"].shape
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text())
    t = np.load(BUN / "terrain.npz")
    sd = t["sd"][::2, ::2][:n[0], :n[1]].astype(np.float32)        # as bl_pbr.flow_image
    fs = OUT / "water_shore.png"
    Image.fromarray((np.clip((sd + SHORE_LO) / SHORE_SPAN, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1], "L").save(fs, optimize=True)
    man["water"] = {"flow": f.name, "x0": float(d["x0"]), "y0": float(d["y0"]), "step_m": float(d["step"]), "nx": n[1], "ny": n[0],
                    "vscale": vs, "note": str(d["note"]), "shore": fs.name, "shore_lo": SHORE_LO, "shore_span": SHORE_SPAN}
    man_path.write_text(json.dumps(man, indent=1))
    print(f"water flow {n} -> {f.name} {f.stat().st_size / 1e3:.0f} kB")


if "--flow-only" in sys.argv:
    water_flow()
    sys.exit(0)
if "--cls-only" in sys.argv:
    ground_classes()
    sys.exit(0)
veg_binary()
if "--veg-only" in sys.argv:
    sys.exit(0)

d = np.load(BUN / "terrain.npz")
x, y, z = d["x"], d["y"], d["z"]
step = 4
zs = np.ascontiguousarray(z[::step, ::step].astype(np.float32))
# orientation check: the creek (east of the jetties) must be below quay level
ix, iy = int(np.argmin(abs(x - 1600))), int(np.argmin(abs(y - 0)))
print("z at (1600, 0):", float(z[iy, ix]), "z at (-1500, -1000):", float(z[int(np.argmin(abs(y + 1000))), int(np.argmin(abs(x + 1500)))]))
zs.tofile(OUT / "terrain.bin")
meta = {"nx": zs.shape[1], "ny": zs.shape[0], "x0": float(x[0]), "x1": float(x[::step][-1]), "y0": float(y[0]),
        "y1": float(y[::step][-1]), "water_z_m": -3.5, "ground_half_m": 15000.0,
        "attribution": "Contains modified Copernicus Sentinel data 2026; OpenStreetMap contributors"}
(OUT / "terrain.json").write_text(json.dumps(meta))
Image.open(BUN / "s2_ground.png").convert("RGB").save(OUT / "ground.jpg", quality=90, optimize=True)
det = REPO / "data" / "assets" / "polyhaven"
cand = sorted(det.glob("**/dry_ground_01_diff*")) + sorted(det.glob("**/*dry_ground*diff*"))
if cand:
    Image.open(cand[0]).convert("RGB").resize((1024, 1024)).save(OUT / "detail.jpg", quality=88)
print(meta, [f.name for f in OUT.glob("*.jpg")])
ground_classes()
water_flow()
