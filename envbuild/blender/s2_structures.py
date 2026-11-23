"""Structures no dataset has, detected from the Sentinel-2 drape (10 m): large roofs inside the port and ships on
the creek (snapshot of 2026-10-01). Results are approximations and are tagged `estimated: Sentinel-2 detection`.

    python -m envbuild.blender.s2_structures   -> data/site_kandla/blender/s2_structures.json

Roofs: bright, blue or red pixel clusters inside built/industrial areas, not already covered by a known footprint,
at least 6 pixels (~600 m2); each becomes an oriented rectangle shed (height estimated 12 m).
Ships: bright or red clusters inside the creek water, 3+ pixels, elongated; oriented rectangle hull, length from the
cluster plus one pixel. These replace nothing in the RF scene or simulation (visual layer only).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
B = REPO / "data" / "site_kandla" / "blender"


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, bool)
    out = []
    H, W = mask.shape
    for j0, i0 in zip(*np.nonzero(mask)):
        if seen[j0, i0]:
            continue
        q = deque([(j0, i0)])
        seen[j0, i0] = True
        comp = []
        while q:
            j, i = q.popleft()
            comp.append((j, i))
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = j + dj, i + di
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    q.append((a, b))
        out.append(comp)
    return out


def main() -> None:
    meta = json.loads((B / "s2_albedo.json").read_text())
    half, pix = meta["half_m"], meta["pix_m"]
    img = np.asarray(Image.open(B / "s2_albedo.png").convert("RGB"), np.float32)
    bundle = json.loads((B / "scene_input.json").read_text())
    terr = np.load(B / "terrain.npz")
    hm = float(bundle["half_m"])
    tx0, tx1, ty0, ty1 = bundle.get("tile") or (-hm, hm, -hm, hm)
    # restrict to the detailed tile
    c0, c1 = int((half + tx0) / pix), int((half + tx1) / pix)        # columns (x)
    r0, r1 = int((half - ty1) / pix), int((half - ty0) / pix)        # rows (row 0 = north)
    sub = img[r0:r1, c0:c1]
    nr, nc = sub.shape[:2]
    to_xy = lambda j, i: (tx0 + (i + 0.5) * pix, ty1 - (j + 0.5) * pix)  # noqa: E731
    to_px = lambda x, y: ((x - tx0) / pix, (ty1 - y) / pix)  # noqa: E731
    # masks on the S2 grid: known footprints, built/industrial, water (from the 5 m terrain grid, nearest sample)
    wharves = []
    known = Image.new("L", (nc, nr), 0)
    d = ImageDraw.Draw(known)
    for o in bundle["objects"]:
        if not o.get("outer"):
            continue
        if o["kind"] == "jetty" and (o.get("wharf") or o.get("area_m2", 0) > 100000):
            wharves.append(shapely.Polygon(o["outer"]))       # the cargo wharf is a yard, not a structure
            continue
        if o["kind"] in ("building", "silo", "substation", "jetty"):
            d.polygon([to_px(x, y) for x, y in o["outer"]], fill=255)
        elif o["kind"] in ("tank", "tank_farm"):   # tank bunds are bright bare earth: exclude 30 m around tanks
            c = np.mean(o["outer"], 0)
            r = (o.get("diameter_m") or 20) / 2 + 30 if o["kind"] == "tank" else 0
            if r:
                d.ellipse([to_px(c[0] - r, c[1] + r), to_px(c[0] + r, c[1] - r)], fill=255)
            else:
                d.polygon([to_px(x, y) for x, y in o["outer"]], fill=255)
    known = np.asarray(known) > 0
    tx, ty = terr["x"], terr["y"]
    jj, ii = np.mgrid[0:nr, 0:nc]
    X = tx0 + (ii + 0.5) * pix
    Y = ty1 - (jj + 0.5) * pix
    gi = np.clip(np.searchsorted(tx, X), 0, len(tx) - 1)
    gj = np.clip(np.searchsorted(ty, Y), 0, len(ty) - 1)
    sd = terr["sd"][gj, gi]
    builtish = (sd > 8) & ~terr["wet"][gj, gi]
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    blue = (b > r + 18) & (b > g + 5)
    red = (r > g + 45) & (r > b + 45)
    bright = lum > 175
    salt = (r - b > 55) & (g - b > 35) & (lum > 120)            # yellow salt pans, not roofs
    roof = builtish & ~known & ~salt & (bright | blue | red)
    roofs = []
    for comp in _components(roof):
        if len(comp) < 6:
            continue
        pts = shapely.MultiPoint([to_xy(j, i) for j, i in comp])
        rect = pts.buffer(pix / 2, cap_style=3).minimum_rotated_rectangle
        if rect.area < 600 or rect.area > 30000 or len(comp) * pix * pix / rect.area < 0.6:
            continue
        rr = np.median(sub[tuple(np.array(comp).T)], axis=0)
        roofs.append({"outer": [[round(x, 2), round(y, 2)] for x, y in list(rect.exterior.coords)[:-1]],
                      "area_m2": round(rect.area, 1), "rgb": [int(v) for v in rr], "pixels": len(comp)})
    water = sd < -4
    ship = water & (red | (lum > 140))
    ships = []
    for comp in _components(ship):
        if len(comp) < 3:
            continue
        pts = shapely.MultiPoint([to_xy(j, i) for j, i in comp])
        rect = pts.buffer(pix / 2, cap_style=3).minimum_rotated_rectangle
        q = list(rect.exterior.coords)
        e = sorted(((float(np.hypot(q[k + 1][0] - q[k][0], q[k + 1][1] - q[k][1])), k) for k in range(4)))
        w, ln, k = e[0][0], e[-1][0], e[-1][1]
        if ln < 60 or ln > 300 or w > 55 or ln / max(w, 1) < 2.5:
            continue
        c = rect.centroid
        if any(shapely.Polygon(o["outer"]).buffer(12).intersects(rect) for o in bundle["objects"]
               if o["kind"] == "jetty" and o.get("outer")):
            if not red[tuple(np.array(comp).T)].mean() > 0.3:      # bright blobs on jetties are structures
                continue
        ships.append({"centroid": [round(c.x, 1), round(c.y, 1)], "length_m": round(ln + pix, 1),
                      "beam_m": round(max(w, 15.0), 1), "red_hull": bool(red[tuple(np.array(comp).T)].mean() > 0.3),
                      "axis_deg": round(float(np.degrees(np.arctan2(q[k + 1][1] - q[k][1], q[k + 1][0] - q[k][0]))), 1)})
    # stockpiles: dark (coal/coke) non-vegetated, non-water clusters in yards
    veg = (g - r > 6) & (g > b)
    wh = Image.new("L", (nc, nr), 0)
    dw = ImageDraw.Draw(wh)
    for w_ in wharves:
        dw.polygon([to_px(x, y) for x, y in w_.exterior.coords], fill=255)
    wharf = np.asarray(wh) > 0
    yard = (terr['built'][gj, gi] | terr['industrial'][gj, gi] | wharf)
    dark = (lum < 60) & ((sd > 25) | wharf) & ~veg & ~known & yard
    piles = []
    for comp in _components(dark):
        if len(comp) < 8:
            continue
        if len(comp) * pix * pix > 20000:                 # a yard: split into heaps on a 50 m grid
            cs = set(comp)
            js = [c[0] for c in comp]
            is_ = [c[1] for c in comp]
            # windrows: 3-pixel-wide rows along the yard's long axis, split where the dark run breaks
            hull_c = shapely.MultiPoint([to_xy(j, i) for j, i in comp]).minimum_rotated_rectangle
            q = list(hull_c.exterior.coords)
            e = max(range(4), key=lambda k: np.hypot(q[k + 1][0] - q[k][0], q[k + 1][1] - q[k][1]))
            ang = float(np.arctan2(q[e + 1][1] - q[e][1], q[e + 1][0] - q[e][0]))
            u = np.array([np.cos(ang), np.sin(ang)])
            v = np.array([-u[1], u[0]])
            P = np.array([to_xy(j, i) for j, i in comp])
            su, sv = P @ u, P @ v
            for v0 in np.arange(sv.min(), sv.max() + 1, 45.0):              # 30 m rows, 15 m gaps
                sel = (sv >= v0) & (sv < v0 + 30.0)
                if sel.sum() < 6:
                    continue
                us = np.sort(su[sel])
                breaks = np.nonzero(np.diff(us) > 25.0)[0]
                for seg in np.split(us, breaks + 1):
                    if seg.max() - seg.min() < 40:
                        continue
                    a0, a1 = seg.min() + 4, seg.max() - 4
                    corners = [a0 * u + (v0 + 3) * v, a1 * u + (v0 + 3) * v, a1 * u + (v0 + 27) * v,
                               a0 * u + (v0 + 27) * v]
                    piles.append({"outer": [[round(float(c[0]), 1), round(float(c[1]), 1)] for c in corners],
                                  "area_m2": round(float((a1 - a0) * 24), 1), "material": "coal", "windrow": True})
            continue
        pts = shapely.MultiPoint([to_xy(j, i) for j, i in comp])
        hull = pts.buffer(pix * 0.6).simplify(pix * 0.5)
        if hull.geom_type != "Polygon" or hull.area < 2000 or hull.area > 120000:
            continue
        piles.append({"outer": [[round(x, 1), round(y, 1)] for x, y in list(hull.exterior.coords)[:-1]],
                      "area_m2": round(hull.area, 1), "material": "coal"})
    # bright irregular bulk (salt, fertiliser, sand) inside yards: piles, not roofs (roofs are rectangular)
    roof_px = Image.new("L", (nc, nr), 0)
    dr_ = ImageDraw.Draw(roof_px)
    for r_ in roofs:
        dr_.polygon([to_px(x, y) for x, y in r_["outer"]], fill=255)
    bulk = (lum > 170) & yard & ~known & ~(np.asarray(roof_px) > 0) & ~salt & (sd > 15)
    for comp in _components(bulk):
        if len(comp) < 6:
            continue
        pts = shapely.MultiPoint([to_xy(j, i) for j, i in comp])
        hull = pts.buffer(pix * 0.6).simplify(pix * 0.5)
        if hull.geom_type != "Polygon" or hull.area < 800 or hull.area > 80000:
            continue
        if len(comp) * pix * pix / hull.minimum_rotated_rectangle.area > 0.8:      # too rectangular: a roof
            continue
        piles.append({"outer": [[round(x, 1), round(y, 1)] for x, y in list(hull.exterior.coords)[:-1]],
                      "area_m2": round(hull.area, 1), "material": "bulk_white"})
    out = {"source": f"Sentinel-2 {meta['item']} detection (10 m); estimated", "roofs": roofs, "ships": ships,
           "stockpiles": piles}
    (B / "s2_structures.json").write_text(json.dumps(out))
    print(json.dumps({"roofs": len(roofs), "roof_area_m2": round(sum(x["area_m2"] for x in roofs)),
                      "ships": len(ships), "ship_lengths": [s["length_m"] for s in ships],
                      "stockpiles": len(piles), "pile_area_m2": round(sum(p["area_m2"] for p in piles))}))


if __name__ == "__main__":
    main()
