"""Ground-only Sentinel-2 layer: remove roofs and tanks from the drape so the twin's 3D structures carry their own
colour and the ground under and around them reads as yard/soil (WSL venv).

    python -m envbuild.blender.s2_ground   -> data/site_kandla/blender/s2_ground.png

Every footprint (OSM tanks/buildings/jetties, Overture buildings), dilated by one 10 m pixel, is masked and filled
by diffusion from the surrounding unmasked pixels (a harmonic in-fill; no new information is invented).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[2]
B = REPO / "data" / "site_kandla" / "blender"


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur (edge-clamped) via cumulative sums; works on float arrays of any channel count."""
    def ax(x, axis):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(x, pad, mode="edge"), axis=axis)
        hi = np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)
        return (hi - lo) / (2 * r + 1)
    return ax(ax(a, 0), 1)


def main() -> None:
    meta = json.loads((B / "s2_albedo.json").read_text())
    half, pix = meta["half_m"], meta["pix_m"]
    img = np.asarray(Image.open(B / "s2_albedo.png").convert("RGB"), np.float32)
    n = img.shape[0]
    bundle = json.loads((B / "scene_input.json").read_text())
    mask = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(mask)
    to_px = lambda x, y: ((x + half) / pix, (half - y) / pix)  # noqa: E731  (row 0 = north)
    k = 0
    for o in bundle["objects"]:
        if o["kind"] in ("tank", "building", "silo", "substation", "jetty") and o.get("outer"):
            d.polygon([to_px(x, y) for x, y in o["outer"]], fill=255)
            k += 1
    sf = B / "s2_structures.json"
    if sf.exists():
        st = json.loads(sf.read_text())
        for r in st["roofs"]:
            d.polygon([to_px(x, y) for x, y in r["outer"]], fill=255)
            k += 1
    m = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0
    out = img.copy()
    out[m] = 0
    w = (~m).astype(np.float32)
    acc = out * w[..., None]
    # diffusion fill: repeatedly blur value and weight, divide, keep known pixels fixed
    for r in (2, 4, 8, 16):
        for _ in range(6):
            num = _blur(acc * w[..., None], r)
            den = _blur(w, r)
            fill = num / np.maximum(den, 1e-4)[..., None]
            upd = m & (den > 1e-3)
            acc = np.where(upd[..., None], fill * 1.0, acc)
            w = np.where(upd, 1.0, w).astype(np.float32)
            acc = np.where(m[..., None], acc, img)
    res = np.where(m[..., None], acc, img)
    Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).save(B / "s2_ground.png")
    print(json.dumps({"footprints": k, "masked_px": int(m.sum()), "out": str(B / "s2_ground.png")}))


if __name__ == "__main__":
    main()
