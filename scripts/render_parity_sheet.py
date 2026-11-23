"""Render parity (Task B): triptychs console before | console after | Blender per view, plus simple image statistics
(mean luminance, RMS contrast, mean saturation, and the after/before mean absolute difference to Blender).

    uv run --no-project --with pillow --with numpy python scripts/render_parity_sheet.py
        --before runs/render_parity/before --after runs/render_parity/after --blender runs/render_parity/blender
        --out docs/design/audit/shots_render
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TW = 800


def font(sz):
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def stats(im: Image.Image) -> dict:
    a = np.asarray(im.convert("RGB"), np.float32) / 255
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return {"lum": round(float(lum.mean()), 3), "contrast": round(float(lum.std()), 3), "sat": round(float(sat.mean()), 3),
            "rgb": [round(float(x), 3) for x in a.reshape(-1, 3).mean(0)]}


def mad(x: Image.Image, y: Image.Image) -> float:
    s = (400, 225)                                   # low-pass: compare tone and colour, not sub-pixel alignment
    p = np.asarray(x.convert("RGB").resize(s, Image.BOX), np.float32)
    q = np.asarray(y.convert("RGB").resize(s, Image.BOX), np.float32)
    return round(float(np.abs(p - q).mean()), 2)


ap = argparse.ArgumentParser()
ap.add_argument("--before", required=True)
ap.add_argument("--after", required=True)
ap.add_argument("--blender", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", default="jetties,tankfarm,creek,top")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
F = font(20)
res = {}
for v in a.views.split(","):
    cols = [("Console before", Path(a.before) / f"{v}.png"), ("Console after", Path(a.after) / f"{v}.png"),
            ("Blender EEVEE (kandla_world.blend)", Path(a.blender) / f"{v}.png")]
    ims = [(lab, Image.open(p).convert("RGB")) for lab, p in cols if p.exists()]
    th = round(TW * ims[0][1].height / ims[0][1].width)
    pad, top = 8, 34
    sheet = Image.new("RGB", (len(ims) * TW + (len(ims) + 1) * pad, th + top + pad), (16, 18, 20))
    d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(ims):
        x = pad + i * (TW + pad)
        sheet.paste(im.resize((TW, th), Image.LANCZOS), (x, top))
        d.text((x, 6), f"{lab} - {v}", fill=(230, 230, 230), font=F)
    sheet.save(out / f"triptych_{v}.jpg", quality=88)
    blend = dict(ims)["Blender EEVEE (kandla_world.blend)"] if len(ims) == 3 else None
    r = {lab: stats(im) for lab, im in ims}
    if blend is not None:
        r["mad_to_blender"] = {"before": mad(ims[0][1], blend), "after": mad(ims[1][1], blend)}
    res[v] = r
    print(v, json.dumps(r))
(out / "stats.json").write_text(json.dumps(res, indent=1))
