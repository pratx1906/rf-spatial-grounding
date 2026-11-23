"""Draw the C-UAS style HUD over rendered frames and encode the MP4.

    uv run --no-project --with pillow python -m envbuild.blender.hud_encode [--video runs/blender/video]
          [--only-hud] [--crf 16]

Everything the HUD shows comes from the frame sidecars written by render_video.py (projected drone positions,
scenario clock). The badge and the attribution line are always drawn: drones, ships and sensor sites are synthetic;
imagery is Copernicus Sentinel-2; geometry is OpenStreetMap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_SANS = "C:/Windows/Fonts/segoeui.ttf"
FONT_SANS_B = "C:/Windows/Fonts/segoeuib.ttf"
HOSTILE = (255, 178, 36)
FRIENDLY = (96, 226, 140)
INK = (236, 240, 244)
DIM = (170, 178, 186)

TITLES = {
    "establish": ("Deendayal Port, Kandla Creek", "Gulf of Kutch, Gujarat"),
    "swarm1_chase": ("Swarm SW-1 inbound", "8 UAS, V formation, from the north-east over the mangroves"),
    "swarm1_tankers": ("SW-1 approaching the oil jetties", "moored tankers at J2-J4"),
    "swarm2_tankfarm": ("Swarm SW-2 over the tank farms", "5 UAS, line abreast, westbound track"),
    "df_node_S1": ("DF node S1, jetty head", "AIR7311, 4 coherent channels, 2x2 omni array"),
    "finale": ("SW-1 holding over the jetties", "18 tracks in the scenario, 2 swarms"),
}


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def label(d) -> str:
    if d["kind"] == "friendly_drone":
        return "FRIENDLY"
    if d.get("swarm"):
        sw = d["swarm"].replace("swarm_", "SW-")
        m = d["id"].split("_m")[-1]
        return f"{sw}.{m}"
    return "UAS-" + d["id"].split("_")[-1].upper()


def draw(im: Image.Image, hud: dict, shot_f0: int) -> Image.Image:
    W, H = im.size
    s = W / 3840
    shade = Image.new("L", (1, H))
    for yy in range(H):                    # soft top and bottom shade so HUD text reads on bright ground
        u = yy / H
        shade.putpixel((0, yy), int(150 * max(0.0, 1 - u / 0.16) ** 1.6 + 170 * max(0.0, (u - 0.86) / 0.14) ** 1.6))
    im = Image.composite(Image.new("RGB", im.size, (8, 10, 12)), im, shade.resize(im.size))
    dr = ImageDraw.Draw(im, "RGBA")
    mono = _font(FONT_MONO, int(26 * s))
    mono_s = _font(FONT_MONO, int(22 * s))
    sans_b = _font(FONT_SANS_B, int(54 * s))
    sans = _font(FONT_SANS, int(32 * s))
    small = _font(FONT_SANS, int(22 * s))
    # drone brackets
    for d in sorted(hud["drones"], key=lambda d: -d["range_m"]):
        x, y = d["x"] * W, d["y"] * H
        r = max(18 * s, min(140 * s, 2600 * s / max(d["range_m"], 1)))
        c = FRIENDLY if d["kind"] == "friendly_drone" else HOSTILE
        k = r * 0.45
        w = max(2, int(3 * s))
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = x + sx * r, y + sy * r
                dr.line([(cx, cy), (cx - sx * k, cy)], fill=c, width=w)
                dr.line([(cx, cy), (cx, cy - sy * k)], fill=c, width=w)
        if d["range_m"] < 1500:
            txt = f"{label(d)}  ALT {d['alt_m']:.0f} m  RNG {d['range_m']:.0f} m"
            dr.text((x - r, y - r - 34 * s), txt, font=mono_s, fill=c)
    # title block, fades in over the first 1.2 s of a shot
    f = hud["frame"]
    a = max(0.0, min(1.0, (f - shot_f0) / 36.0))
    t1, t2 = TITLES.get(hud["shot"], ("", ""))
    x0, y0 = 110 * s, 96 * s
    dr.rectangle([x0 - 26 * s, y0 - 6 * s, x0 - 20 * s, y0 + 112 * s], fill=(*HOSTILE, int(255 * a)))
    dr.text((x0, y0), t1, font=sans_b, fill=(*INK, int(255 * a)))
    dr.text((x0, y0 + 70 * s), t2, font=sans, fill=(*DIM, int(255 * a)))
    # clock and status, top right
    t = hud["t_s"]
    clock = f"T+{int(t // 60):02d}:{int(t % 60):02d}"
    tw = dr.textlength(clock, font=mono)
    dr.text((W - 110 * s - tw, y0), clock, font=mono, fill=INK)
    n_h = sum(1 for d in hud["drones"] if d["kind"] != "friendly_drone")
    st = f"{n_h} hostile in view"
    dr.text((W - 110 * s - dr.textlength(st, font=mono_s), y0 + 40 * s), st, font=mono_s, fill=DIM)
    # honesty badge + attribution, bottom
    badge = "SIMULATION  synthetic drones, ships and sensor sites"
    bw = dr.textlength(badge, font=mono_s)
    dr.rectangle([110 * s - 14 * s, H - 150 * s, 110 * s + bw + 14 * s, H - 112 * s], fill=(0, 0, 0, 120))
    dr.text((110 * s, H - 146 * s), badge, font=mono_s, fill=HOSTILE)
    attr = ("SceneScout  |  Imagery: contains modified Copernicus Sentinel data 2026  |  "
            "Map data (c) OpenStreetMap contributors, ODbL  |  Heights estimated")
    dr.text((110 * s, H - 92 * s), attr, font=small, fill=(*DIM, 220))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO / "runs" / "blender" / "video"))
    ap.add_argument("--only-hud", action="store_true")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()
    v = Path(a.video)
    shots = {s["name"]: s for s in json.loads((v / "shots.json").read_text())}
    out = v / "final"
    out.mkdir(exist_ok=True)
    frames = sorted((v / "frames").glob("f_*.png"))
    for p in frames:
        q = out / p.name
        if q.exists():
            continue
        hud = json.loads((v / "hud" / (p.stem + ".json")).read_text())
        im = Image.open(p).convert("RGB")
        draw(im, hud, shots[hud["shot"]]["f0"]).save(q, compress_level=1)
    print(f"hud: {len(frames)} frames")
    if a.only_hud:
        return
    mp4 = v / "kandla_flythrough_4k.mp4"
    cmd = ["ffmpeg", "-y", "-framerate", "30", "-start_number", "1", "-i", str(out / "f_%04d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(mp4)]
    subprocess.run(cmd, check=True)
    print("video:", mp4)


if __name__ == "__main__":
    main()
