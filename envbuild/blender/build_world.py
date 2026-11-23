"""Build the high-detail Kandla Creek world in Blender from the twin bundle (envbuild.blender.export_scene).

Headless:  blender --background --python envbuild/blender/build_world.py -- [--save PATH] [--stages all]
Live (MCP): import build_world; build_world.run(stages=[...])

Every object carries custom properties `fidelity` (osm | estimated | illustrative) and `source`. Footprints and
centrelines are OSM; heights, widths, shore slope and water level are estimated; all procedural detail (stairs,
loading arms, piles, pipe racks, vegetation, ships, sensor masts) is illustrative. Same ENU frame as the RF scene.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bl_geometry as G  # noqa: E402
import bl_materials  # noqa: E402
import bl_pbr  # noqa: E402
import mound as MD  # noqa: E402

REPO = HERE.parents[1]
BUNDLE = REPO / "data" / "site_kandla" / "blender"

STAGES = ["reset", "materials", "terrain", "water", "tanks", "farms", "jetties", "ships", "buildings", "stockpiles", "cranes", "yard", "roads",
          "rail", "traffic", "pipelines", "power", "fences", "sensors", "vegetation", "lighting", "compositor", "cameras"]


class Ctx:
    def __init__(self):
        self.B = json.loads((BUNDLE / "scene_input.json").read_text())
        t = np.load(BUNDLE / "terrain.npz")
        self.tx, self.ty, self.tz, self.sd = t["x"], t["y"], t["z"], t["sd"]
        self.wet, self.ind, self.built = t["wet"], t["industrial"], t["built"]
        self.builtf, self.indf = self.built.astype(np.float32), self.ind.astype(np.float32)
        self.E = self.B["estimated"]
        h = self.B["half_m"]
        self.tile = tuple(self.B.get("tile") or (-h, h, -h, h))
        self.M: dict = {}
        self.cols: dict = {}

    # bilinear terrain height and signed shore distance
    def _bil(self, grid, P):
        P = np.atleast_2d(np.asarray(P, float))
        g = self.tx[1] - self.tx[0]
        fx = np.clip((P[:, 0] - self.tx[0]) / g, 0, len(self.tx) - 1.001)
        fy = np.clip((P[:, 1] - self.ty[0]) / g, 0, len(self.ty) - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        return ((1 - u) * (1 - v) * grid[j, i] + u * (1 - v) * grid[j, i + 1] + (1 - u) * v * grid[j + 1, i]
                + u * v * grid[j + 1, i + 1])

    def z(self, P):
        return self._bil(self.tz, P)

    def sdist(self, P):
        return self._bil(self.sd, P)

    def col(self, name):
        if name not in self.cols:
            c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
            if c.name not in bpy.context.scene.collection.children:
                bpy.context.scene.collection.children.link(c)
            self.cols[name] = c
        return self.cols[name]


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))


def _props(fid, src, **kw):
    return {"fidelity": fid, "source": src, **kw}


# ------------------------------------------------------------------------------------------------------------
def reset(C):
    for c in list(bpy.data.collections):
        if c.name.startswith("KW_"):
            for o in list(c.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(c)
    for o in list(bpy.data.objects):
        if o.name in ("Cube", "Light", "Camera"):
            bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes) + list(bpy.data.curves):
        if m.users == 0:
            (bpy.data.meshes if isinstance(m, bpy.types.Mesh) else bpy.data.curves).remove(m)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.length_unit = "METERS"


def _s2_image():
    f = BUNDLE / "s2_albedo.png"
    if not f.exists():
        return None
    img = bpy.data.images.get("s2_albedo") or bpy.data.images.load(str(f))
    img.name = "s2_albedo"
    img.colorspace_settings.name = "sRGB"
    return img


def _s2_ground_image():
    f = BUNDLE / "s2_ground.png"
    if not f.exists():
        return None
    img = bpy.data.images.get("s2_ground") or bpy.data.images.load(str(f))
    img.name = "s2_ground"
    img.reload()
    return img


def _s2_struct():
    f = BUNDLE / "s2_structures.json"
    return json.loads(f.read_text()) if f.exists() else {"roofs": [], "ships": []}


def materials(C):
    bl_materials.S2_IMAGE = _s2_image()
    bl_materials.S2_GROUND = _s2_ground_image()
    C.M = bl_materials.build_all()
    C.M = bl_pbr.upgrade_all(C.M)          # photo-scanned CC0 PBR detail when the assets are present


def terrain(C):
    nx, ny = len(C.tx), len(C.ty)
    X, Y = np.meshgrid(C.tx, C.ty)
    V = np.column_stack([X.ravel(), Y.ravel(), C.tz.ravel()]).astype(np.float32)
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    a = (jj * nx + ii).ravel()
    F = np.column_stack([a, a + 1, a + 1 + nx, a + nx]).astype(np.int32)
    me = bpy.data.meshes.new("KW_terrain")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.shade_smooth()
    uvl = me.uv_layers.new(name="UVMap")
    loop_co = V[F.ravel()][:, :2]
    x0, x1, y0, y1 = C.tile
    uvl.data.foreach_set("uv", ((loop_co - [x0, y0]) / [x1 - x0, y1 - y0]).astype(np.float32).ravel())

    # attributes for the ground shader and the vegetation scatter
    road_mask = _road_mask(C)
    built = C.built.astype(np.float32)
    ind = C.ind.astype(np.float32)
    wet = C.wet.astype(np.float32)
    sd = C.sd
    clump = 0.5 + 0.25 * np.sin(X / 37.0 + np.sin(Y / 53.0) * 2.1) + 0.25 * np.sin(Y / 29.0 + np.cos(X / 61.0) * 1.7)
    open_land = (sd > 9) & (built < 0.5) & (ind < 0.5) & ~road_mask
    scrub = np.where(open_land, np.clip(clump, 0, 1) ** 2, 0.0)
    bank = (sd > -7) & (sd < 16) & (built < 0.5) & (ind < 0.5) & ~road_mask
    mang = np.where(bank | ((wet > 0.5) & (sd > -7) & ~road_mask), np.clip(0.4 + clump, 0, 1), 0.0)
    g = _s2_green(X, Y)
    px = _s2_pixels()
    if px is not None:                      # dark, unvegetated pixels are ponds / salt-pan water: no shrubs there
        hh, ww, _ = px.shape
        half = bl_materials.S2_HALF
        ii = np.clip(((X + half) / (2 * half) * ww).astype(int), 0, ww - 1)
        jj = np.clip(((Y + half) / (2 * half) * hh).astype(int), 0, hh - 1)
        rgb = px[jj, ii]
        lum = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
        pond = (lum < 0.25) & (rgb[..., 1] - rgb[..., 0] < 0.03)
        scrub = np.where(pond, 0.0, scrub)
        mang = np.where(pond, 0.0, mang)
    if g is not None:                       # Sentinel-2: dark green = mangrove / dense scrub on the ground truth
        free = (sd > -6) & (built < 0.5) & (ind < 0.5) & ~road_mask
        mang = np.maximum(mang * 0.4, np.where(free & (g > 0.5), g, 0.0))
        scrub = np.where(open_land, np.clip(scrub * 0.5 + 0.8 * np.clip(g - 0.15, 0, 1), 0, 1), 0.0)
    for name, arr in (("sd", sd), ("wet", wet), ("industrial", ind), ("built", built),
                      ("veg_scrub", scrub), ("veg_mangrove", mang), ("road", road_mask.astype(np.float32))):
        at = me.attributes.new(name, "FLOAT", "POINT")
        at.data.foreach_set("value", np.asarray(arr, np.float32).ravel())
    me.materials.append(C.M["ground"])
    ob = bpy.data.objects.new("KW_terrain", me)
    C.col("KW_terrain").objects.link(ob)
    for k, v in _props("estimated", "OSM coastline + estimated bank slope and water level",
                       grid_m=C.B["grid_m"]).items():
        ob[k] = v
    _skirt(C, nx, ny, V)


def _skirt(C, nx, ny, V):
    """Far field around the detailed tile: a ground sheet just below quay level carrying the Sentinel-2 colour
    out to 40 km, so the port sits in its real surroundings (creek, mangroves, salt pans) instead of a void."""
    x0, x1, y0, y1 = C.tile
    R = 40000.0
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    outer = [(-R, -R), (R, -R), (R, R), (-R, R)]
    verts = [(x, y, -0.4) for x, y in inner + outer]
    faces = [(k, 4 + k, 4 + (k + 1) % 4, (k + 1) % 4) for k in range(4)]
    me = bpy.data.meshes.new("KW_terrain_skirt")
    me.from_pydata(verts, [], faces)
    me.materials.append(C.M["far_ground"])
    ob = bpy.data.objects.new("KW_terrain_skirt", me)
    C.col("KW_terrain").objects.link(ob)
    ob["fidelity"] = "illustrative"
    ob["source"] = "flat far field, colour from Sentinel-2 L2A 2026-10-01 (Copernicus)"


def _s2_green(X, Y):
    """0..1 'dark vegetation' index from the Sentinel-2 drape at ENU points (None without the drape)."""
    img = _s2_image()
    if img is None:
        return None
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    half = bl_materials.S2_HALF
    i = np.clip(((X + half) / (2 * half) * w).astype(int), 0, w - 1)
    j = np.clip(((Y + half) / (2 * half) * h).astype(int), 0, h - 1)
    r, g, b = px[j, i, 0], px[j, i, 1], px[j, i, 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    green = (g - r) / (g + r + 1e-3)
    return np.clip((green + 0.02) * 12, 0, 1) * np.clip((0.36 - lum) * 8, 0, 1)


def _road_mask(C):
    """Grid cells within (half width + 2 m) of a road centreline."""
    g = C.tx[1] - C.tx[0]
    m = np.zeros(C.tz.shape, bool)
    for r in C.B["lines"]["road"]:
        P = np.asarray(r["xy"], float)
        hw = r["width_m"] / 2 + 2.0
        for p0, p1 in zip(P[:-1], P[1:]):
            lo = np.minimum(p0, p1) - hw
            hi = np.maximum(p0, p1) + hw
            i0, i1 = np.searchsorted(C.tx, [lo[0], hi[0]])
            j0, j1 = np.searchsorted(C.ty, [lo[1], hi[1]])
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(C.tx[i0:i1], C.ty[j0:j1])
            d = p1 - p0
            t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d + 1e-9), 0, 1)
            dist = np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1]))
            m[j0:j1, i0:i1] |= dist <= max(hw, g / 2)
    return m


def water(C):
    # only over the detailed tile: beyond it the far-field sheet carries the creek colour, and a 60 km water plane
    # 3 m under that sheet z-fights in the 24-bit viewport depth buffer at range
    x0, x1, y0, y1 = C.tile
    me = bpy.data.meshes.new("KW_water")
    z = C.E["water_z_m"]
    me.from_pydata([(x0 - 50, y0 - 50, z), (x1 + 50, y0 - 50, z), (x1 + 50, y1 + 50, z), (x0 - 50, y1 + 50, z)], [],
                   [(0, 1, 2, 3)])
    if bl_pbr.available() and (BUNDLE / "water_flow.npz").exists():     # python -m envbuild.blender.water_flow
        C.M["water"] = bl_pbr.water_live(bl_pbr.flow_image(C.sd, BUNDLE / "water_flow.npz"))
    me.materials.append(C.M["water"])
    ob = bpy.data.objects.new("KW_water", me)
    C.col("KW_water").objects.link(ob)
    for k, v in _props("estimated", f"mean water level {z} m below quay (estimated)").items():
        ob[k] = v


# ------------------------------------------------------------------------------------------------------------
def _tank(C, o):
    rng = _rng(o["id"])
    cx, cy = o["centroid"]
    d = o["diameter_m"]
    r = d / 2
    h = float(o["height_m"])
    z0 = float(C.z([[cx, cy]])[0])
    seg = int(np.clip(round(d * 1.6), 32, 96))
    mb = G.MB()
    mb.cylinder(cx, cy, z0 - 0.3, z0 + 0.45, r + 0.6, seg, "concrete")
    floating = d >= 28
    top = z0 + h
    deck_col = _disc_colour(cx, cy, r * 0.75)
    if floating:
        mb.cylinder(cx, cy, z0 + 0.45, top, r, seg, "tank_paint", top=False)
        mb.cylinder(cx, cy, z0 + 0.45, top, r - 0.02, seg, "tank_paint", inner=True)
        fill = top - 1.2 - rng.uniform(0.0, 0.45) * (h - 3.0)          # roofs ride high when tanks are in use
        rd = r - 0.35
        mb.cylinder(cx, cy, fill - 0.6, fill - 0.2, rd, seg, "floating_roof", top=False)
        mb.cur_fc = deck_col
        mb.cylinder(cx, cy, fill - 0.2, fill, rd * 0.86, seg, "deck_plate")            # centre deck
        mb.annulus(cx, cy, fill + 0.12, rd * 0.86, rd, seg, "deck_pontoon", thick=0.32)  # pontoon ring
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        mb.annulus(cx, cy, fill + 0.14, rd, r - 0.02, seg, "rubber")                   # rim seal
        mb.cylinder(cx, cy, fill, fill + 0.6, r - 1.3, seg, "galvanised", top=False)   # foam dam
        nrib = 24 if d > 45 else 16
        for k in range(nrib):                                                         # radial deck seams
            a = 2 * math.pi * k / nrib
            r0, r1 = rd * 0.12, rd * 0.84
            mb.box((cx + (r0 + r1) / 2 * math.cos(a), cy + (r0 + r1) / 2 * math.sin(a), fill + 0.03),
                   (r1 - r0, 0.18, 0.06), "deck_seam", a)
        mb.cylinder(cx, cy, fill, fill + 0.25, 0.9, 12, "steel")                       # centre manhole
        for k in range(8):                                                            # drains / legs / vents
            a = 2 * math.pi * k / 8 + rng.random() * 0.3
            rr = rd * rng.uniform(0.35, 0.78)
            mb.cylinder(cx + rr * math.cos(a), cy + rr * math.sin(a), fill, fill + 0.4, 0.35, 8, "steel")
        mb.annulus(cx, cy, top - 1.0, r, r + 0.9, seg, "steel", thick=0.15)            # wind girder
        mb.annulus(cx, cy, top - 0.02, r - 0.02, r + 0.12, seg, "steel", thick=0.25)   # top angle
        a0 = rng.uniform(0, 2 * math.pi)                                              # rolling ladder
        p_top = (cx + (r - 0.5) * math.cos(a0), cy + (r - 0.5) * math.sin(a0), top + 0.2)
        run = max(2.0, (top - fill) * 0.9)
        p_dn = (cx + (r - 0.5 - run) * math.cos(a0), cy + (r - 0.5 - run) * math.sin(a0), fill + 0.2)
        for off in (-0.45, 0.45):
            o2 = (-math.sin(a0) * off, math.cos(a0) * off, 0)
            mb.tube([p_top[i] + o2[i] for i in range(3)], [p_dn[i] + o2[i] for i in range(3)], 0.07, 4, "galvanised")
        mb.box(((p_dn[0] + cx) / 2 * 0 + p_dn[0], p_dn[1], fill + 0.1), (6.0, 0.25, 0.2), "steel", a0 + math.pi / 2)
    else:
        mb.cylinder(cx, cy, z0 + 0.45, top, r, seg, "tank_paint", top=False)
        mb.cur_fc = deck_col
        mb.cone(cx, cy, top, r + 0.05, r / 6.0, seg, "roof_cone")
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        for k in range(16):                                                           # roof rafters / seams
            a = 2 * math.pi * k / 16
            p_out = (cx + (r - 0.3) * math.cos(a), cy + (r - 0.3) * math.sin(a), top + 0.08)
            p_in = (cx + 0.8 * math.cos(a), cy + 0.8 * math.sin(a), top + r / 6.0 - 0.1)
            mb.tube(p_out, p_in, 0.06, 4, "deck_seam")
        mb.cylinder(cx, cy, top + r / 6.0 - 0.1, top + r / 6.0 + 0.7, 0.45, 12, "steel")   # vent
        a0 = rng.uniform(0, 2 * math.pi)
        for k in range(3):                                                            # PV vents, gauge hatch
            a = a0 + 0.9 + k * 0.5
            rr = r * 0.6
            zz = top + (r - rr) / 6.0
            mb.box((cx + rr * math.cos(a), cy + rr * math.sin(a), zz + 0.3), (0.6, 0.6, 0.6), "steel", a)
    # top handrail ring with posts
    nposts = max(12, int(2 * math.pi * r / 2.5))
    ring = [(cx + (r - 0.25) * math.cos(2 * math.pi * k / nposts), cy + (r - 0.25) * math.sin(2 * math.pi * k / nposts))
            for k in range(nposts)]
    roof_z = top + (0.0 if floating else 0.15)
    for k in range(nposts):
        x0, y0 = ring[k]
        x1, y1 = ring[(k + 1) % nposts]
        mb.tube((x0, y0, roof_z + 1.1), (x1, y1, roof_z + 1.1), 0.03, 4, "yellow")
        mb.tube((x0, y0, roof_z), (x0, y0, roof_z + 1.1), 0.025, 4, "yellow")
    # spiral stair around the shell
    n = max(20, int(h / 0.21))
    sweep = 1.25 * h / (r + 0.6)
    th0 = a0 - sweep
    pts, zs = [], []
    for i in range(n):
        f = (i + 1) / n
        th = th0 + f * sweep
        zz = z0 + 0.45 + f * (h - 0.45)
        rr = r + 0.55
        mb.box((cx + rr * math.cos(th), cy + rr * math.sin(th), zz), (0.30, 0.95, 0.05), "steel", th)
        if i % 2 == 0:
            pts.append((cx + (r + 1.05) * math.cos(th), cy + (r + 1.05) * math.sin(th)))
            zs.append(zz)
    for k in range(len(pts) - 1):
        mb.tube((*pts[k], zs[k] + 1.0), (*pts[k + 1], zs[k + 1] + 1.0), 0.03, 4, "yellow")
        mb.tube((*pts[k], zs[k] - 0.15), (*pts[k + 1], zs[k + 1] - 0.15), 0.08, 4, "steel")
        if k % 3 == 0:
            mb.tube((*pts[k], zs[k] - 0.15), (*pts[k], zs[k] + 1.0), 0.025, 4, "yellow")
    # top landing
    mb.box((cx + (r + 0.9) * math.cos(a0), cy + (r + 0.9) * math.sin(a0), top + 0.05), (2.0, 1.6, 0.12), "steel", a0)
    # outlet nozzle, valve and pipe stub towards the farm manifold
    an = a0 + math.pi * rng.uniform(0.6, 1.4)
    p0 = (cx + r * math.cos(an), cy + r * math.sin(an), z0 + 1.3)
    p1 = (cx + (r + 6.5) * math.cos(an), cy + (r + 6.5) * math.sin(an), z0 + 1.3)
    mb.tube(p0, p1, 0.32, 10, "steel")
    mb.box((cx + (r + 3.0) * math.cos(an), cy + (r + 3.0) * math.sin(an), z0 + 1.3), (0.9, 0.9, 1.1), "red", an)
    mb.tube(p1, (p1[0], p1[1], z0 + 0.2), 0.32, 10, "steel")
    return mb.build(o["id"], C.M, C.col("KW_tanks"), props=_props(
        "osm", "OSM footprint; height estimated; stairs, roof type, rails, nozzle illustrative",
        osm_id=o.get("osm_id") or "", name=o["name"], height_m=h, diameter_m=d, roof="floating" if floating else "cone"))


def _disc_colour(cx, cy, rad):
    """Mean Sentinel-2 colour over a disc (tank roofs are 3-8 pixels across): linear RGB, weight 0.85."""
    px = _s2_pixels()
    if px is None:
        return (0.0, 0.0, 0.0, 0.0)
    h, w, _ = px.shape
    half = bl_materials.S2_HALF
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    pts = np.array([[cx, cy]] + [[cx + rad * f * np.cos(a), cy + rad * f * np.sin(a)] for f in (0.4, 0.8) for a in ang])
    i = np.clip(((pts[:, 0] + half) / (2 * half) * w).astype(int), 0, w - 1)
    j = np.clip(((pts[:, 1] + half) / (2 * half) * h).astype(int), 0, h - 1)
    srgb = np.median(px[j, i], axis=0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lin = np.clip(lin * 1.5, 0.02, 0.85)
    return (float(lin[0]), float(lin[1]), float(lin[2]), 0.85 if rad > 6 else 0.4)


def tanks(C):
    for o in C.B["objects"]:
        if o["kind"] == "tank":
            _tank(C, o)


def farms(C):
    for o in C.B["objects"]:
        if o["kind"] != "tank_farm":
            continue
        ring = G.ccw(o["outer"])
        mb = G.MB()
        mb.wall(ring, 0.45, lambda P: C.z(P), 1.4, "concrete", closed=True)
        # pipe rack along the longest hull edge, inset 9 m
        n = len(ring)
        k = max(range(n), key=lambda i: math.dist(ring[i], ring[(i + 1) % n]))
        a, b = np.array(ring[k]), np.array(ring[(k + 1) % n])
        d = b - a
        L = float(np.linalg.norm(d))
        d /= L
        nrm = np.array([-d[1], d[0]])
        a2, b2 = a + nrm * 9 + d * 6, b + nrm * 9 - d * 6
        L2 = float(np.linalg.norm(b2 - a2))
        rot = math.atan2(d[1], d[0])
        steps = max(2, int(L2 / 6))
        for s in range(steps + 1):
            p = a2 + d * (L2 * s / steps)
            zg = float(C.z([p])[0])
            for side in (-1.5, 1.5):
                q = p + nrm * side
                mb.box((q[0], q[1], zg + 2.75), (0.3, 0.3, 5.5), "steel", rot)
            mb.box((p[0], p[1], zg + 5.5), (0.3, 3.6, 0.35), "steel", rot)
            mb.box((p[0], p[1], zg + 3.2), (0.25, 3.6, 0.3), "steel", rot)
        zg = float(np.mean(C.z([a2, b2])))
        for i, off in enumerate(np.linspace(-1.3, 1.3, 6)):
            r_ = 0.28 if i % 2 else 0.2
            q0, q1 = a2 + nrm * off, b2 + nrm * off
            mb.tube((q0[0], q0[1], zg + 5.7 + r_), (q1[0], q1[1], zg + 5.7 + r_), r_, 10,
                    "cladding" if i % 3 else "steel")
        for i, off in enumerate(np.linspace(-1.2, 1.2, 4)):
            q0, q1 = a2 + nrm * off, b2 + nrm * off
            mb.tube((q0[0], q0[1], zg + 3.5 + 0.25), (q1[0], q1[1], zg + 3.5 + 0.25), 0.25, 10, "steel")
        # per-tank rectangular bunds aligned with the farm grid (as seen in imagery; dimensions estimated)
        hull = np.asarray(ring)
        members = [t for t in C.B["objects"] if t["kind"] == "tank" and G.point_in_poly(*t["centroid"], ring)]
        rot2 = rot
        cr, sr = math.cos(rot2), math.sin(rot2)
        for t in members:
            tr = t["diameter_m"] / 2
            hs = tr + max(6.0, 0.3 * tr)
            tc = np.array(t["centroid"])
            sq = [tuple(tc + np.array([dx * cr - dy * sr, dx * sr + dy * cr])) for dx, dy in
                  ((-hs, -hs), (hs, -hs), (hs, hs), (-hs, hs))]
            zg = float(C.z([tc])[0])
            mb.prism(sq, zg, zg + 0.06, "bund_floor", top=True)
            mb.wall(sq, 0.45, lambda P, zg=zg: np.full(len(P), zg), 1.3, "concrete", closed=True)
        mb.build(o["id"], C.M, C.col("KW_tank_farms"), props=_props(
            "illustrative", "bund wall on derived OSM tank-farm hull; pipe rack illustrative", name=o["name"]))


# ------------------------------------------------------------------------------------------------------------
def _ships_by_jetty(C):
    ships = [o for o in C.B["objects"] if o["kind"] == "ship"]
    out = {}
    for s in ships:
        k = s["id"].split("_s")[-1]
        out[f"jetty_j{k}"] = s
    return out


def jetties(C):
    ships = _ships_by_jetty(C)
    deck = C.E["jetty_deck_z_m"]
    for o in C.B["objects"]:
        if o["kind"] != "jetty":
            continue
        ring = G.ccw(o["outer"])
        mb = G.MB()
        if o.get("wharf") or o.get("area_m2", 0) > 100000:
            # the old cargo port: OSM maps the whole wharf and backup yard as one pier polygon. It is a land-backed
            # quay, not a deck on piles: yard surface at quay level (imagery colour), quay wall on the water side.
            mb.prism(ring, C.E["bed_z_m"], 0.12, "wharf_yard" if "wharf_yard" in C.M else "ground", top=True,
                     side_mat="concrete_dark")
            P = np.asarray(ring)
            n = len(P)
            for i in range(n):
                a, b = P[i], P[(i + 1) % n]
                L = float(np.linalg.norm(b - a))
                mid = (a + b) / 2
                if L < 20 or C.sdist([mid])[0] > -2:          # only edges that face the water
                    continue
                d_ = (b - a) / L
                nrm = np.array([d_[1], -d_[0]])               # outward for a CCW ring
                rot_e = math.atan2(d_[1], d_[0])
                for s_ in np.arange(8, L - 8, 16.0):
                    q = a + d_ * s_
                    f = q + nrm * 0.6
                    mb.box((f[0], f[1], -1.2), (3.0, 1.2, 2.6), "rubber", rot_e)            # fenders
                    bq = q - nrm * 1.5
                    mb.cylinder(bq[0], bq[1], 0.12, 0.9, 0.35, 10, "steel")                  # bollards
                mb.wall([tuple(a - nrm * 0.4), tuple(b - nrm * 0.4)], 0.6, lambda Q: np.full(len(Q), 0.12), 0.25,
                        "concrete")                                                         # cope line
            mb.build(o["id"], C.M, C.col("KW_jetties"), props=_props(
                "osm", "OSM pier polygon of the cargo wharf; quay level and quay furniture estimated/illustrative",
                osm_id=o.get("osm_id") or "", name="Cargo wharf (old Kandla port)"))
            continue
        mb.prism(ring, deck - 1.3, deck, "concrete", top=True, bottom=True, side_mat="concrete_dark")
        # piles on a 7 m grid inside the deck, down to the bed
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        for x in np.arange(min(xs) + 3, max(xs) - 2, 7.0):
            for y in np.arange(min(ys) + 3, max(ys) - 2, 7.0):
                if G.point_in_poly(x, y, ring):
                    zb = float(C.z([[x, y]])[0])
                    if zb < deck - 1.5:
                        mb.cylinder(x, y, zb - 0.5, deck - 1.3, 0.45, 10, "concrete_dark", top=False)
        ang = math.radians(o.get("axis_deg", 0.0))
        ax = np.array([math.cos(ang), math.sin(ang)])
        nm = np.array([-ax[1], ax[0]])
        c = np.array(o["centroid"])
        ship = ships.get(o["id"])
        if ship is not None:
            berth = np.array(ship["centroid"]) - c
        else:
            berth = nm if C.sdist([c + nm * 40])[0] < C.sdist([c - nm * 40])[0] else -nm
        berth = berth / (np.linalg.norm(berth) + 1e-9)
        if abs(berth @ nm) > abs(berth @ ax):
            bn = nm * np.sign(berth @ nm)
            along = ax
        else:
            bn = ax * np.sign(berth @ ax)
            along = nm
        # deck edge on the berth side through the centroid
        edge = c.copy()
        for _ in range(200):
            if not G.point_in_poly(*(edge + bn * 1.0), ring):
                break
            edge = edge + bn * 1.0
        rot = math.atan2(along[1], along[0])
        # loading arms (illustrative marine loading arms) on the berth face
        for k in (-1.5, -0.5, 0.5, 1.5):
            base = edge - bn * 4.0 + along * (k * 9.0)
            if not G.point_in_poly(*base, ring):
                continue
            bz = deck
            top = np.array([*base, bz + 8.0])
            mb.cylinder(base[0], base[1], bz, bz + 8.0, 0.45, 12, "yellow")
            apex = np.array([*(base + bn * 3.0), bz + 16.5])
            tip = np.array([*(base + bn * 11.0), bz + 5.0])
            mb.tube(top, apex, 0.32, 10, "yellow")
            mb.tube(apex, tip, 0.28, 10, "yellow")
            mb.tube(tip, tip - np.array([0, 0, 2.5]), 0.25, 10, "steel")
            cw = top + np.array([*(-bn * 2.2), 4.5])
            mb.tube(top + np.array([0, 0, 0.5]), cw, 0.12, 6, "steel")
            mb.box(tuple(cw), (1.6, 1.6, 1.6), "concrete_dark", rot)
        # pipe rack along the jetty axis, carried on portal trestles ~3.5 m above the deck (illustrative)
        L = o.get("mrr_wl_m", [20, 100])[1]
        p_a = c - ax * (L / 2 - 4)
        p_b = c + ax * (L / 2 - 4)
        rz = deck + 3.6
        for i, off in enumerate(np.linspace(-2.6, 2.6, 7)):
            q0, q1 = p_a + nm * off, p_b + nm * off
            r_ = 0.35 if i % 2 else 0.25
            mb.tube((q0[0], q0[1], rz + r_), (q1[0], q1[1], rz + r_), r_, 10, "cladding" if i % 2 else "steel")
        for i, off in enumerate(np.linspace(-1.8, 1.8, 4)):           # upper tier: smaller lines and cable tray
            q0, q1 = p_a + nm * off, p_b + nm * off
            mb.tube((q0[0], q0[1], rz + 2.3), (q1[0], q1[1], rz + 2.3), 0.16, 8, "steel")
        q0, q1 = p_a - nm * 3.1, p_b - nm * 3.1
        mb.tube((q0[0], q0[1], rz + 2.6), (q1[0], q1[1], rz + 2.6), 0.08, 4, "galvanised")
        steps = max(2, int(L / 7))
        rot_ax = math.atan2(ax[1], ax[0])
        for st in range(steps + 1):
            q = p_a + (p_b - p_a) * st / steps
            for side in (-1, 1):                                       # portal legs
                lg = q + nm * side * 3.1
                mb.box((lg[0], lg[1], deck + (rz + 2.4 - deck) / 2), (0.35, 0.35, rz + 2.4 - deck), "steel", rot_ax)
            mb.box((q[0], q[1], rz - 0.15), (0.4, 6.6, 0.3), "steel", rot_ax)        # lower beam
            mb.box((q[0], q[1], rz + 2.0), (0.3, 6.6, 0.25), "steel", rot_ax)        # upper beam
        # handrails around the deck edge (top + mid rail, posts every ~2.5 m)
        Rg = np.asarray(ring, float)
        for i in range(len(Rg)):
            a_, b_ = Rg[i], Rg[(i + 1) % len(Rg)]
            Le = float(np.linalg.norm(b_ - a_))
            if Le < 1.0:
                continue
            for hz in (0.55, 1.1):
                mb.tube((a_[0], a_[1], deck + hz), (b_[0], b_[1], deck + hz), 0.03, 4, "yellow")
            for f_ in np.arange(0.0, Le, 2.5):
                q = a_ + (b_ - a_) * f_ / Le
                mb.box((q[0], q[1], deck + 0.55), (0.06, 0.06, 1.1), "yellow")
        # fenders and bollards on the berth face, lamp posts
        for k in np.arange(-2.5, 2.6, 1.0):
            p = edge + along * (k * 12.0)
            if G.point_in_poly(*(p - bn * 1.0), ring):
                q = p + bn * 0.6
                mb.box((q[0], q[1], deck - 1.8), (1.2, 3.2, 3.0), "rubber", rot)
                b = p - bn * 1.2
                mb.cylinder(b[0], b[1], deck, deck + 0.8, 0.35, 10, "steel")
        for k in np.arange(-3, 3.1, 1.0):
            for side in (-1, 1):
                p = c + ax * (k * 25.0) + nm * side * 6.0
                if G.point_in_poly(*p, ring):
                    mb.cylinder(p[0], p[1], deck, deck + 12.0, 0.12, 6, "galvanised")
                    mb.box((p[0], p[1], deck + 12.1), (0.9, 0.4, 0.25), "lamp", 0.0)
        mb.build(o["id"], C.M, C.col("KW_jetties"), props=_props(
            "osm", "OSM pier footprint; deck level estimated; piles, loading arms, racks, fenders illustrative",
            osm_id=o.get("osm_id") or "", name=o["name"]))


def _ship(C, o):
    """Procedural ~180 m product tanker (synthetic, illustrative)."""
    W, L = o.get("mrr_wl_m", [30.0, 180.0])
    B = W / 2
    wz = C.E["water_z_m"]
    keel, deckz = -11.5, 7.5
    xs = np.linspace(-L / 2, L / 2, 46)

    def hb(x):
        if x > L / 2 - 0.17 * L:
            t = (x - (L / 2 - 0.17 * L)) / (0.17 * L)
            return max(0.35, B * math.sqrt(max(0.0, 1 - t ** 1.6)))
        if x < -L / 2 + 0.06 * L:
            t = (-L / 2 + 0.06 * L - x) / (0.06 * L)
            return B * (1 - 0.18 * t)
        return B

    zs = [keel, keel + 0.6, keel + 1.6, keel + 3.0, 0.0, deckz]
    fz = [0.80, 0.93, 0.985, 1.0, 1.0, 1.0]
    mb = G.MB()
    rows = []
    for x in xs:
        bow = x > L / 2 - 0.17 * L
        kz = keel + (2.0 * ((x - (L / 2 - 0.17 * L)) / (0.17 * L)) ** 2 if bow else 0.0)
        sec = []
        for z, f in zip(zs, fz):
            zz = max(z, kz) if z < 0 else z
            sec.append((x, -hb(x) * f, zz))
        for z, f in reversed(list(zip(zs, fz))):
            zz = max(z, kz) if z < 0 else z
            sec.append((x, hb(x) * f, zz))
        rows.append(mb.add_v(sec))
    m = len(zs) * 2
    for a, b in zip(rows[:-1], rows[1:]):
        for k in range(m - 1):
            mb.face((a + k, b + k, b + k + 1, a + k + 1), "hull_red" if o.get("red") else "hull")
        mb.face((a + m - 1, b + m - 1, b, a), "deck_green")        # deck
    mb.face([rows[0] + k for k in range(m)], "hull")              # transom
    # superstructure aft: 5 stepped accommodation tiers with window bands, wheelhouse, bridge wings, funnel
    W_ = "white"
    sx = -L / 2 + 0.085 * L
    tier_h, n_t = 2.9, 5
    ln0, bw0 = 17.0, W - 5.0
    for k in range(n_t):
        ln_ = ln0 - k * 0.9
        z0_ = deckz + k * tier_h
        mb.box((sx - k * 0.45, 0, z0_ + tier_h / 2), (ln_, bw0 - k * 0.6, tier_h), W_)
        mb.box((sx - k * 0.45 + ln_ / 2 + 0.03, 0, z0_ + 1.55), (0.06, bw0 - k * 0.6 - 1.5, 0.95), "rubber")  # front
        for sd_ in (-1, 1):                                                                      # side windows
            mb.box((sx - k * 0.45, sd_ * ((bw0 - k * 0.6) / 2 + 0.03), z0_ + 1.55), (ln_ - 2.0, 0.06, 0.8), "rubber")
        mb.box((sx - k * 0.45, 0, z0_ + tier_h - 0.05), (ln_ + 0.6, bw0 - k * 0.6 + 0.6, 0.1), "steel")   # deck edge
    zb = deckz + n_t * tier_h
    mb.box((sx + 1.5, 0, zb + 1.5), (9.0, W + 2.0, 3.0), W_)                                    # wheelhouse + wings
    mb.box((sx + 6.03, 0, zb + 1.9), (0.06, W - 1.0, 1.4), "rubber")                            # bridge windows
    mb.box((sx + 1.5, 0, zb + 3.1), (9.4, W + 2.4, 0.2), "steel")
    mb.tube((sx - 1.0, 0, zb + 3.2), (sx - 1.0, 0, zb + 10.0), 0.18, 8, W_)                     # radar mast
    mb.box((sx - 1.0, 0, zb + 8.0), (0.3, 4.2, 0.15), "steel")
    mb.box((sx - 9.5, 0, deckz + 15.5), (6.5, 8.0, 13.0), W_)                                    # funnel casing
    mb.box((sx - 9.5, 0, deckz + 22.5), (6.0, 7.4, 1.2), "red")                                  # funnel band
    mb.box((sx - 9.5, 0, deckz + 23.6), (5.6, 7.0, 1.0), "rubber")                               # funnel top
    for sd_ in (-1, 1):                                                                          # lifeboats
        mb.box((sx - 3.0, sd_ * (bw0 / 2 + 1.4), deckz + 2 * tier_h + 1.0), (7.5, 2.4, 2.2), "container_orange")
        mb.box((sx - 3.0, sd_ * (bw0 / 2 + 0.6), deckz + 2 * tier_h + 2.6), (8.0, 0.25, 0.25), "steel")
    if o.get("bulk"):
        # bulk carrier: hatch coamings and covers, 4 deck cranes between the hatches
        n_h = 7
        x_a, x_b = sx + ln0 / 2 + 6.0, L / 2 - 0.12 * L - 4.0
        hl = (x_b - x_a) / n_h
        for k in range(n_h):
            hx = x_a + hl * (k + 0.5)
            mb.box((hx, 0, deckz + 0.9), (hl * 0.72, W * 0.62, 1.8), "hull_red" if o.get("red") else "deck_green")
            mb.box((hx, 0, deckz + 2.05), (hl * 0.74, W * 0.64, 0.5), "red" if k % 2 else "container_red")
            for f_ in (-0.25, 0.0, 0.25):
                mb.box((hx + hl * 0.72 * f_, 0, deckz + 2.35), (0.25, W * 0.64, 0.12), "steel")
        for k in (1, 3, 5, 7):
            if k >= n_h:
                continue
            cx_ = x_a + hl * k
            mb.cylinder(cx_, 0, deckz, deckz + 7.5, 1.2, 12, W_)
            mb.box((cx_, 0, deckz + 8.6), (3.2, 3.0, 2.4), W_)
            mb.tube((cx_ + 0.8, 0, deckz + 8.0), (cx_ + 18.0, W * 0.25, deckz + 22.0), 0.45, 8, "yellow")
            mb.tube((cx_ + 18.0, W * 0.25, deckz + 22.0), (cx_ + 18.0, W * 0.25, deckz + 12.0), 0.05, 4, "steel")
    else:
        # tanker: catwalk with cargo lines, manifold, hose cranes
        mb.box((L / 2 - 0.08 * L, 0, deckz + 1.3), (0.12 * L, W * 0.55, 2.6), W_)
        for yoff in (-1.2, 0.0, 1.2):
            mb.tube((sx + 12, yoff, deckz + 1.4), (L / 2 - 0.15 * L, yoff, deckz + 1.4), 0.28, 8, "steel")
        mb.box(((sx + 12 + L / 2 - 0.15 * L) / 2, 0, deckz + 2.2), (L / 2 - 0.15 * L - sx - 12, 1.4, 0.12), "steel")
        for k in range(7):
            y0 = -B + 2.0
            mb.tube((-2 + k * 0.9 - 2.7, y0, deckz + 1.6), (-2 + k * 0.9 - 2.7, -y0, deckz + 1.6), 0.22, 8,
                    "red" if k % 2 else "steel")
        for side in (-1, 1):
            mb.cylinder(0.0, side * (B - 4), deckz, deckz + 9, 0.5, 10, W_)
            mb.tube((0.0, side * (B - 4), deckz + 9), (8.0, side * (B - 9), deckz + 12), 0.25, 8, "yellow")
    mb.tube((L / 2 - 0.1 * L, 0, deckz + 2.6), (L / 2 - 0.1 * L, 0, deckz + 16), 0.35, 8, W_)   # fore mast
    for k in range(-int(L / 20), int(L / 20) + 1):                                             # rail stanchions
        mb.box((k * 10.0, -B + 0.6, deckz + 0.6), (0.1, 0.1, 1.2), W_)
        mb.box((k * 10.0, B - 0.6, deckz + 0.6), (0.1, 0.1, 1.2), W_)
    mb.box((0, -B + 0.6, deckz + 1.15), (L * 0.8, 0.06, 0.06), W_)
    mb.box((0, B - 0.6, deckz + 1.15), (L * 0.8, 0.06, 0.06), W_)
    for xw in (L / 2 - 0.05 * L, L / 2 - 0.09 * L, -L / 2 + 3.0, -L / 2 + 7.0):        # mooring winches
        for sg in (-1, 1):
            mb.cylinder(xw, sg * (B - 4.5), deckz + 0.4, deckz + 1.6, 0.7, 12, "steel")
            mb.box((xw, sg * (B - 6.3), deckz + 0.9), (1.6, 1.2, 1.3), "deck_green")
    # mooring lines to the berth (ships alongside) or the anchor chain (ships at anchor)
    c_ = np.asarray(o["centroid"], float)
    a_ = math.radians(o.get("axis_deg", 0.0))
    ux, nx = np.array([math.cos(a_), math.sin(a_)]), np.array([-math.sin(a_), math.cos(a_)])
    wz = C.E["water_z_m"]
    side = 0
    for sg in (1, -1):
        if C.sdist([c_ + nx * sg * (B + 22)])[0] > -6:
            side = sg
            break
    if side:
        def to_local(xl, yl):
            w = c_ + ux * xl + nx * yl
            zt = (C.E["jetty_deck_z_m"] if C.sdist([w])[0] < 0 else float(C.z([w])[0])) + 0.9
            return (xl, yl, zt - wz)
        lines = [((L / 2 - 4, side * (B - 1)), (L / 2 + 28, side * (B + 14))),     # head lines
                 ((L / 2 - 6, side * (B - 1)), (L / 2 + 18, side * (B + 18))),
                 ((L * 0.3, side * B), (L * 0.05, side * (B + 9))),                # springs
                 ((-L * 0.3, side * B), (-L * 0.05, side * (B + 9))),
                 ((-L / 2 + 3, side * (B - 1)), (-L / 2 - 26, side * (B + 14))),   # stern lines
                 ((-L / 2 + 5, side * (B - 1)), (-L / 2 - 16, side * (B + 18)))]
        for (x0_, y0_), (x1_, y1_) in lines:
            mb.tube((x0_, y0_, deckz + 1.2), to_local(x1_, y1_), 0.06, 4, "container_orange")
    else:
        mb.tube((L / 2 - 2.5, 3.5, deckz - 1.5), (L / 2 + 14.0, 3.5, 0.0), 0.12, 4, "rubber")     # anchor chain
    ob = mb.build(o["id"], C.M, C.col("KW_ships"), props=_props(
        "illustrative", "SYNTHETIC moored tanker (dynamic layer); not a real vessel", name=o["name"]))
    ob.location = (*o["centroid"], wz)
    ob.rotation_euler = (0, 0, math.radians(o.get("axis_deg", 0.0)))
    return ob


def _near_wharf(C, c):
    """Distance (m) from a point to the cargo-wharf outline (ships berthed there are bulk carriers)."""
    best = 1e9
    for o in C.B["objects"]:
        if o["kind"] == "jetty" and (o.get("wharf") or o.get("area_m2", 0) > 100000):
            A = np.asarray(o["outer"], float)
            Bv = np.roll(A, -1, 0)
            AB = Bv - A
            t_ = np.clip(((np.asarray(c) - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
            best = min(best, float(np.min(np.hypot(*(A + AB * t_[:, None] - np.asarray(c)).T))))
    return best


def ships(C):
    obs = _s2_struct()["ships"]
    if obs:      # ships seen in the 2026-10-01 Sentinel-2 image, at their real berths (snapshot)
        for k, sh in enumerate(obs, start=1):
            o = {"id": f"ship_obs_{k}", "name": f"Tanker, observed {k} (Sentinel-2 2026-10-01)",
                 "centroid": sh["centroid"], "mrr_wl_m": [min(sh["beam_m"], 0.16 * sh["length_m"] + 4), sh["length_m"]],
                 "axis_deg": sh["axis_deg"], "red": sh.get("red_hull", False),
                 "bulk": _near_wharf(C, sh["centroid"]) < 80.0}
            ob = _ship(C, o)
            ob["fidelity"] = "estimated"
            ob["source"] = ("position, heading, length from Sentinel-2 2026-10-01 (10 m); hull form illustrative; "
                            + ("bulk carrier (berthed at the cargo wharf, type assumed)" if o["bulk"] else "tanker (type assumed)"))
        return
    for o in C.B["objects"]:
        if o["kind"] == "ship":
            _ship(C, o)


# ------------------------------------------------------------------------------------------------------------
_S2PX = None


def _s2_pixels():
    """Sentinel-2 drape as an (h, w, 3) float array in display sRGB (cached)."""
    global _S2PX
    if _S2PX is None:
        img = _s2_image()
        if img is None:
            return None
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        _S2PX = px.reshape(h, w, 4)[:, :, :3]
    return _S2PX


def _roof_colour(ring, area):
    """Mean imagery colour inside a roof (linear RGB, alpha = weight); weight 0 for roofs under ~4 S2 pixels."""
    px = _s2_pixels()
    if px is None or area < 400:
        return (0.0, 0.0, 0.0, 0.0)
    h, w, _ = px.shape
    half = bl_materials.S2_HALF
    P = np.asarray(ring, float)
    c = P.mean(0)
    sh = (P - c) * 0.6 + c                                  # sample the inner 60 %, away from edges and shadows
    pts = np.vstack([sh, c[None]])
    i = np.clip(((pts[:, 0] + half) / (2 * half) * w).astype(int), 0, w - 1)
    j = np.clip(((pts[:, 1] + half) / (2 * half) * h).astype(int), 0, h - 1)
    srgb = np.median(px[j, i], axis=0)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lin = np.clip(lin * 1.6, 0, 0.9)                        # TCI is compressed (0.3 reflectance = 255)
    wgt = float(np.clip((area - 400) / 1600, 0.35, 0.9))
    return (float(lin[0]), float(lin[1]), float(lin[2]), wgt)


CONTAINER_MATS = ("container_blue", "container_red", "container_orange", "container_grey", "container_green")
FACADES = ((0.70, 0.55, 0.32), (0.66, 0.36, 0.33), (0.78, 0.58, 0.20), (0.34, 0.50, 0.66), (0.50, 0.50, 0.48),
           (0.80, 0.76, 0.66), (0.56, 0.28, 0.18), (0.56, 0.64, 0.44), (0.82, 0.80, 0.74), (0.62, 0.46, 0.56))


def _building_details(mb, rng, ring, z0, h, area, fac):
    """Township articulation (illustrative, typical of Gujarat RCC buildings): floor slab bands that double as
    window sunshades, balconies on the longest facade, rooftop stair room and black water tanks, wall AC units."""
    floors = max(1, int(h / 3.6 + 0.2))
    for k in range(floors):                                            # slab band / chajja above each window row
        zc = z0 + k * 3.6 + 2.86
        if zc < z0 + h - 0.6:
            mb.wall(ring, 0.55, lambda P, zc=zc: np.full(len(P), zc), 0.11, "concrete", closed=True)
    P = np.asarray(ring, float)
    n = len(P)
    lens = [float(np.linalg.norm(P[(i + 1) % n] - P[i])) for i in range(n)]
    i = int(np.argmax(lens))
    a, b = P[i], P[(i + 1) % n]
    L = lens[i]
    d = (b - a) / max(L, 1e-6)
    nrm = np.array([d[1], -d[0]])                                      # outward for a CCW ring
    rot = math.atan2(d[1], d[0])
    if floors >= 2 and L > 7 and area < 900 and rng.random() < 0.65:
        bw = min(L * 0.45, 6.5)
        for f_ in (0.27, 0.73) if L > 16 else (0.5,):
            q = a + d * L * f_ + nrm * 0.65
            for k in range(1, floors):
                zb = z0 + k * 3.6
                mb.box((q[0], q[1], zb - 0.08), (bw, 1.3, 0.16), "concrete", rot)
                e = q + nrm * 0.62
                mb.box((e[0], e[1], zb + 0.5), (bw, 0.08, 1.0), "steel", rot)
                for sg in (-1, 1):
                    sq = q + d * sg * bw / 2
                    mb.box((sq[0], sq[1], zb + 0.5), (0.08, 1.3, 1.0), "steel", rot)
    for _ in range(rng.randint(0, min(4, floors + 1))):               # split-AC outdoor units
        q = a + d * L * rng.uniform(0.1, 0.9) + nrm * 0.25
        zc = z0 + rng.randrange(floors) * 3.6 + rng.choice((0.9, 2.2))
        mb.box((q[0], q[1], zc), (0.85, 0.32, 0.6), "white", rot)
    c = P.mean(0)
    top = z0 + h
    if area > 60 and floors >= 1:                                       # stair room (mumty)
        q = c + d * rng.uniform(-0.2, 0.2) * L * 0.5
        mb.cur_fc = fac
        mb.box((q[0], q[1], top + 1.3), (3.2, 3.6, 2.6), "wall", rot)
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        mb.box((q[0], q[1], top + 2.66), (3.6, 4.0, 0.12), "concrete", rot)
        tq = q + nrm * 0.0
        for k in range(1 + int(area // 300)):                            # black polyethylene water tanks
            r_ = rng.uniform(0.6, 0.85)
            tq = q + d * (1.0 + k * 1.9) - nrm * 0.6
            mb.cylinder(tq[0], tq[1], top + 2.72, top + 2.72 + r_ * 1.6, r_, 14, "rubber")


def _building_into(C, mb, o):
    ring = G.ccw(o["outer"])
    if len(ring) < 3:
        return
    rng = _rng(o["id"])
    z0 = float(np.min(C.z(ring)))
    h = float(o["height_m"])
    c0 = o["centroid"]
    in_yard = (float(C._bil(C.indf, [c0])[0]) > 0.5           # OSM industrial/port landuse only (not the township)
               and float(C.sdist([c0])[0]) < 700)
    if o.get("fidelity") == "ml_footprint" and o.get("area_m2", 0) < 80 and in_yard:
        # small detections inside port/industrial yards are containers, trucks and small plant (illustrative)
        mb.prism(ring, z0, z0 + 2.6 * (1 + (rng.random() < 0.3)), CONTAINER_MATS[rng.randrange(5)], top=True)
        return
    mb.cur_fc = _roof_colour(ring, o.get("area_m2", 0))
    industrial = "corrugated" in (o.get("material") or "")
    wl = o.get("mrr_wl_m")
    rect = wl and o.get("area_m2", 0) / max(1.0, wl[0] * wl[1]) > 0.85
    if industrial and rect and o.get("area_m2", 0) > 250:
        eave = h - min(2.5, wl[0] * 0.12)
        fc = mb.cur_fc
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        mb.prism(ring, z0, z0 + eave, "roof_metal", top=False, side_mat="roof_metal")
        mb.cur_fc = fc
        ang = math.radians(o["axis_deg"])
        ax = np.array([math.cos(ang), math.sin(ang)])
        nm = np.array([-ax[1], ax[0]])
        c = np.array(o["centroid"])
        hl, hw = wl[1] / 2 + 0.6, wl[0] / 2 + 0.6
        ridge = z0 + h
        P = lambda s_, t_, z: (*(c + ax * s_ + nm * t_), z)  # noqa: E731
        v = mb.add_v([P(-hl, -hw, z0 + eave), P(hl, -hw, z0 + eave), P(hl, 0, ridge), P(-hl, 0, ridge),
                      P(-hl, hw, z0 + eave), P(hl, hw, z0 + eave)])
        slope = math.hypot(hw, ridge - z0 - eave)
        mb.face((v, v + 1, v + 2, v + 3), "roof_metal", [(0, 0), (0, 2 * hl), (slope, 2 * hl), (slope, 0)])
        mb.face((v + 5, v + 4, v + 3, v + 2), "roof_metal", [(0, 2 * hl), (0, 0), (slope, 0), (slope, 2 * hl)])
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        for sgn in (-1, 1):
            e = mb.add_v([P(sgn * (hl - 0.6), -sgn * (hw - 0.6), z0 + eave), P(sgn * (hl - 0.6), sgn * (hw - 0.6), z0 + eave),
                          P(sgn * (hl - 0.6), 0, ridge - 0.05)])
            mb.face((e, e + 1, e + 2), "roof_metal")
        n_doors = max(1, int(wl[1] / 25))
        for k in range(n_doors):
            s_ = -wl[1] / 2 + (k + 0.5) * wl[1] / n_doors
            q = c + ax * s_ - nm * (wl[0] / 2 + 0.05)
            mb.box((q[0], q[1], z0 + 2.75), (5.0, 0.12, 5.5), "concrete_dark", ang)
    else:
        fc = mb.cur_fc
        clad = h >= 8.5
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0) if clad else (*FACADES[rng.randrange(len(FACADES))], 0.92)
        mb.prism(ring, z0, z0 + h, "wall_cladding" if clad else "wall", top=False)
        fac = mb.cur_fc
        mb.cur_fc = fc
        mb.prism(ring, z0 + h - 0.01, z0 + h, "roof_flat", top=True)
        mb.cur_fc = (0.0, 0.0, 0.0, 0.0)
        area = o.get("area_m2", 0)
        if area > 40:
            mb.wall(ring, 0.3, lambda P: np.full(len(P), z0 + h), 0.9, "concrete", closed=True)
        if not clad and area > 25:
            _building_details(mb, rng, ring, z0, h, area, fac)
        if o.get("area_m2", 0) > 300:
            c = o["centroid"]
            for _ in range(rng.randint(1, 4)):
                x = c[0] + rng.uniform(-6, 6)
                y = c[1] + rng.uniform(-6, 6)
                if G.point_in_poly(x, y, ring):
                    mb.box((x, y, z0 + h + 0.8), (rng.uniform(1.5, 3.5), rng.uniform(1.2, 2.5), 1.6), "galvanised",
                           rng.uniform(0, 3.14))
    mb.cur_fc = (0.0, 0.0, 0.0, 0.0)


def buildings(C):
    cells: dict = {}
    for o in C.B["objects"]:
        if o["kind"] not in ("building", "silo", "substation", "tower"):
            continue
        ring = G.ccw(o["outer"])
        if len(ring) < 3:
            continue
        z0 = float(np.min(C.z(ring)))
        if o["kind"] == "substation":
            mb = G.MB()
            mb.prism(ring, z0 - 0.2, z0 + 0.15, "concrete_dark")
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            for x in np.arange(min(xs) + 6, max(xs) - 4, 9.0):
                for y in np.arange(min(ys) + 6, max(ys) - 4, 12.0):
                    if G.point_in_poly(x, y, ring):
                        mb.box((x, y, z0 + 1.9), (4.2, 3.0, 3.5), "steel")
                        for f in range(5):
                            mb.box((x - 2.4, y - 1.2 + f * 0.6, z0 + 2.0), (0.6, 0.12, 2.6), "steel")
                        for dy in (-0.8, 0.0, 0.8):
                            mb.cylinder(x + 0.5, y + dy, z0 + 3.6, z0 + 5.4, 0.15, 8, "white")
            mb.wall(ring, 0.06, lambda P: C.z(P), 2.4, "galvanised", closed=True)
            mb.build(o["id"], C.M, C.col("KW_buildings"), props=_props(
                "osm", "OSM substation outline; equipment illustrative", name=o["name"]))
            continue
        if o.get("fidelity") == "ml_footprint":          # thousands of ML footprints: batch by 800 m cell
            c = o["centroid"]
            key = (int(math.floor(c[0] / 800)), int(math.floor(c[1] / 800)))
            mb = cells.setdefault(key, G.MB())
            _building_into(C, mb, o)
            continue
        mb = G.MB()
        _building_into(C, mb, o)
        mb.build(o["id"], C.M, C.col("KW_buildings"), props=_props(
            "osm", "OSM footprint; height estimated; roof form and openings illustrative; roof colour from Sentinel-2",
            name=o["name"], height_m=float(o["height_m"])))
    mbr = G.MB()
    for k, r in enumerate(_s2_struct()["roofs"]):
        ring = r["outer"]
        P = np.asarray(ring, float)
        e = [(float(np.linalg.norm(P[(i + 1) % 4] - P[i])), i) for i in range(4)]
        e.sort()
        w, ln, i = e[0][0], e[-1][0], e[-1][1]
        d = P[(i + 1) % 4] - P[i]
        o = {"id": f"s2roof_{k}", "outer": ring, "centroid": P.mean(0).tolist(), "area_m2": r["area_m2"],
             "height_m": 12.0, "material": "corrugated_metal", "fidelity": "s2_detect",
             "mrr_wl_m": [w, ln], "axis_deg": math.degrees(math.atan2(d[1], d[0]))}
        _building_into(C, mbr, o)
    mbr.build("KW_s2_roofs", C.M, C.col("KW_buildings_ml"), props=_props(
        "estimated", "large roofs detected in Sentinel-2 2026-10-01 not in any footprint dataset; height 12 m estimated"))
    for (i, j), mb in cells.items():
        mb.build(f"KW_mlbuildings_{i}_{j}", C.M, C.col("KW_buildings_ml"), props=_props(
            "ml_footprint", "Overture 2026-09-23.1 (Google Open Buildings CC BY 4.0, Microsoft ML Buildings ODbL); "
            "heights estimated by size and shape; roof colour from Sentinel-2"))


def stockpiles(C):
    C.windrows = []
    """Bulk stockpiles (coal/coke) detected as dark yard areas in Sentinel-2: stepped mounds, heights estimated."""
    mb = G.MB()
    tanks_c = np.array([t["centroid"] for t in C.B["objects"] if t["kind"] == "tank"])
    for k, pl in enumerate(_s2_struct().get("stockpiles", [])):
        ring = G.ccw(pl["outer"])
        if len(ring) < 3:
            continue
        P = np.asarray(ring, float)
        c = P.mean(0)
        if len(tanks_c) and np.min(np.hypot(*(tanks_c - c).T)) < 150:     # dark bund water, not coal
            continue
        if float(C.sdist([c])[0]) < 60:                                     # keep the quay apron clear
            continue
        h = float(np.clip(math.sqrt(pl["area_m2"]) * 0.16, 6.0, 12.0))      # bulk yard stacks are typically 8-14 m
        z0 = float(np.min(C.z(ring)))
        steps = [(1.0, 0.0), (0.82, 0.45), (0.6, 0.78), (0.35, 0.95), (0.12, 1.0)]
        if pl.get("windrow") and len(P) == 4:
            # long stacking windrow: rectangular base, ridge along the long axis, ~37 deg repose slopes
            e01, e12 = P[1] - P[0], P[2] - P[1]
            if np.linalg.norm(e01) < np.linalg.norm(e12):
                P = np.roll(P, -1, axis=0)
                e01, e12 = P[1] - P[0], P[2] - P[1]
            u = e01 / np.linalg.norm(e01)
            v = e12 / np.linalg.norm(e12)
            half_w = np.linalg.norm(e12) / 2
            h = min(h, 0.75 * half_w)
            C.windrows.append({"c": P.mean(0), "u": u, "v": v, "len": float(np.linalg.norm(e01)), "half_w": half_w,
                               "h": h, "z0": z0})
            sgn = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            rings = []
            for f in (0.0, 0.45, 0.78, 0.95, 1.0):
                ins_v = (half_w - 1.0) * f
                ins_u = 1.3 * h * f
                rings.append(mb.add_v([(*(P[k] - sgn[k][0] * u * ins_u - sgn[k][1] * v * ins_v), z0 + h * f)
                                       for k in range(4)]))
        elif not MD.is_convex(ring):
            # concave yard: rings scaled toward the vertex mean cross the footprint (starburst, dark baked wedges);
            # triangulate the footprint and lift by distance to the edge instead (demo-v2 S6, envbuild/blender/mound.py)
            V, T = MD.mound(ring, z0, h)
            b0 = mb.add_v(V)
            for t in T:
                mb.face(tuple(b0 + i for i in t), pl.get("material", "coal"))
            continue
        else:
            rings = [mb.add_v([(*(c + (p - c) * sc), z0 + h * fz) for p in P]) for sc, fz in steps]
        n = len(P)
        for a, b in zip(rings[:-1], rings[1:]):
            for i in range(n):
                j = (i + 1) % n
                mb.face((a + i, a + j, b + j, b + i), pl.get("material", "coal"))
        mb.face([rings[-1] + i for i in range(n)], pl.get("material", "coal"))
    mb.build("KW_stockpiles", C.M, C.col("KW_stockpiles"), smooth=True, props=_props(
        "estimated", "dark bulk stockpiles detected in Sentinel-2 2026-10-01; shape and height estimated"))


def cranes(C):
    """Mobile harbour cranes along the cargo wharf's water edge (ILLUSTRATIVE: type and count not surveyed;
    one every ~220 m of berth line, boom slewed over the water)."""
    mb = G.MB()
    k = 0
    for o in C.B["objects"]:
        if o["kind"] != "jetty" or not (o.get("wharf") or o.get("area_m2", 0) > 100000):
            continue
        P = np.asarray(G.ccw(o["outer"]))
        seg = np.diff(np.vstack([P, P[:1]]), axis=0)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        last = -1e9
        for s_ in np.arange(0, cum[-1], 20.0):
            i = int(np.searchsorted(cum, s_, side="right") - 1)
            L = float(cum[i + 1] - cum[i])
            if L < 1e-6:
                continue
            d_ = seg[i] / L
            q0 = P[i] + d_ * (s_ - cum[i])
            nrm = np.array([d_[1], -d_[0]])                   # towards the water for a CCW ring
            if C.sdist([q0 + nrm * 10])[0] > -3 or s_ - last < 220:      # only along the water-facing berth line
                continue
            last = s_
            q = q0 - nrm * 14.0
            rot = math.atan2(d_[1], d_[0])
            z = 0.12
            for sx in (-1, 1):                                  # outrigger pads
                for sy in (-1, 1):
                    pp = q + d_ * sx * 6.5 + nrm * sy * 6.5
                    mb.box((pp[0], pp[1], z + 0.3), (1.6, 1.6, 0.6), "steel", rot)
            mb.box((q[0], q[1], z + 2.2), (13.0, 9.0, 3.0), "yellow", rot)                    # undercarriage
            mb.box((q[0], q[1], z + 5.0), (7.0, 6.0, 2.6), "white", rot)                      # superstructure
            cw = q - nrm * 3.2
            mb.box((cw[0], cw[1], z + 5.4), (5.0, 2.5, 3.2), "concrete_dark", rot)            # counterweight
            top = np.array([q[0], q[1], z + 30.0])
            mb.tube((q[0], q[1], z + 6.3), top, 1.0, 10, "yellow")                           # tower
            cab = q + nrm * 2.0
            mb.box((cab[0], cab[1], z + 22.0), (2.2, 2.2, 2.4), "white", rot)
            tip = np.array([*(q + nrm * 42.0 + d_ * 6.0), z + 34.0])
            foot = np.array([q[0], q[1], z + 9.0])
            mb.tube(foot, tip, 0.6, 8, "yellow")                                              # boom
            mb.tube(top, tip, 0.08, 4, "steel")                                               # luffing ropes
            mb.tube(tip, tip - np.array([0, 0, 18.0]), 0.05, 4, "steel")                      # hoist
            mb.box(tuple(tip - np.array([0, 0, 18.5])), (1.2, 1.2, 1.0), "yellow", rot)       # hook block
            k += 1
    mb.build("KW_cranes", C.M, C.col("KW_cranes"), props=_props(
        "illustrative", f"{k} mobile harbour cranes along the cargo wharf; type and count not surveyed"))


def yard(C):
    """Elevated bulk-yard machinery along the coal windrows (ILLUSTRATIVE: layout inferred from the windrow axes in
    Sentinel-2, not surveyed): conveyor galleries on trestles, a stacker-reclaimer per windrow on its rail track,
    transfer towers at the berth end, and 30 m high-mast yard lighting."""
    rows = getattr(C, "windrows", None)
    if rows is None:
        C.windrows = []
        stockpiles(C)
        rows = C.windrows
    mb = G.MB()
    rng = _rng("yard")
    n_conv = 0
    masts = []
    for k, w in enumerate(rows):
        c, u, v = w["c"], w["u"], w["v"]
        L, hw, h, z0 = w["len"], w["half_w"], w["h"], w["z0"]
        if L < 100:                                            # galleries only on the long stacking rows
            continue
        side = 1 if k % 2 == 0 else -1
        off = v * side * (hw + 6.0)
        a, b = c - u * (L / 2 + 8) + off, c + u * (L / 2 + 8) + off
        if min(C.sdist([a])[0], C.sdist([b])[0]) < 8:          # never over the water
            continue
        gz = z0 + h * 0.75 + 2.0                               # gallery floor above the stack shoulder
        rot = math.atan2(u[1], u[0])
        mid = (a + b) / 2
        mb.box((mid[0], mid[1], gz + 1.0), (L + 16, 2.6, 2.0), "galvanised", rot)          # covered gallery
        mb.box((mid[0], mid[1], gz + 2.08), (L + 16.4, 3.0, 0.16), "roof_metal", rot)      # gallery roof
        for f_ in np.arange(0, L + 16, 16.0):                                              # A-frame trestles
            q = a + u * f_
            for sgn in (-1, 1):
                foot = q + v * sgn * 2.2
                mb.tube((foot[0], foot[1], z0), (q[0] + v[0] * sgn * 1.1, q[1] + v[1] * sgn * 1.1, gz), 0.22, 6, "steel")
            mb.box((q[0], q[1], gz - 0.15), (0.4, 3.2, 0.3), "steel", rot)
        # rail track of the stacker-reclaimer between the gallery and the pile
        tr = c + v * side * (hw + 2.5)
        for g_ in (-1.6, 1.6):
            r0, r1 = tr - u * L / 2 + v * g_, tr + u * L / 2 + v * g_
            mb.box(((r0[0] + r1[0]) / 2, (r0[1] + r1[1]) / 2, z0 + 0.12), (L, 0.18, 0.24), "rail_steel", rot)
        # stacker-reclaimer at a random station along the track
        s_ = rng.uniform(-0.35, 0.35) * L
        q = tr + u * s_
        mb.box((q[0], q[1], z0 + 1.6), (10.0, 6.0, 2.4), "yellow", rot)                     # bogies/portal
        mb.box((q[0], q[1], z0 + 4.6), (6.0, 5.0, 3.6), "yellow", rot)                      # slewing deck
        mb.box((q[0] - v[0] * side * 1.0, q[1] - v[1] * side * 1.0, z0 + 7.9), (2.6, 2.4, 2.6), "white", rot)
        tip = q - v * side * (hw * 0.9) + u * 6.0
        tz = z0 + h + 2.0
        mb.tube((q[0], q[1], z0 + 6.5), (tip[0], tip[1], tz), 0.8, 8, "yellow")             # boom with belt
        cw = q + v * side * 12.0
        mb.tube((q[0], q[1], z0 + 6.5), (cw[0], cw[1], z0 + 9.0), 0.5, 8, "yellow")          # counterweight arm
        mb.box((cw[0], cw[1], z0 + 8.5), (3.5, 3.0, 3.0), "concrete_dark", rot)
        mb.tube((q[0], q[1], z0 + 6.5), (q[0], q[1], z0 + 15.0), 0.35, 8, "yellow")          # mast
        for e_ in (tip, cw):
            mb.tube((q[0], q[1], z0 + 15.0), (e_[0], e_[1], tz if e_ is tip else z0 + 9.5), 0.05, 4, "steel")
        bw = tip                                                                           # bucket wheel
        mb.tube((bw[0] - u[0] * 0.6, bw[1] - u[1] * 0.6, tz - 1.0), (bw[0] + u[0] * 0.6, bw[1] + u[1] * 0.6, tz - 1.0),
                3.0, 16, "yellow")
        # transfer tower at the berth-side end of the gallery
        e_end = a if C.sdist([a])[0] < C.sdist([b])[0] else b
        mb.box((e_end[0], e_end[1], z0 + (gz + 4.0 - z0) / 2), (7.0, 7.0, gz + 4.0 - z0), "cladding", rot)
        mb.box((e_end[0], e_end[1], gz + 4.15), (7.4, 7.4, 0.3), "roof_metal", rot)
        n_conv += 1
        if rng.random() < 0.5:
            masts.append(c + v * side * (hw + 12.0) + u * rng.uniform(-0.3, 0.3) * L)
    # high masts over the wharf yard as well (every ~260 m inside the yard polygon, clear of the quay edge)
    for o in C.B["objects"]:
        if o["kind"] == "jetty" and (o.get("wharf") or o.get("area_m2", 0) > 100000):
            ring = o["outer"]
            P = np.asarray(ring)
            for x in np.arange(P[:, 0].min() + 60, P[:, 0].max(), 260.0):
                for y in np.arange(P[:, 1].min() + 60, P[:, 1].max(), 260.0):
                    q = np.array([x + rng.uniform(-40, 40), y + rng.uniform(-40, 40)])
                    if G.point_in_poly(*q, ring) and C.sdist([q])[0] > 40 and \
                            all(np.linalg.norm(q - m_) > 120 for m_ in masts):
                        masts.append(q)
    for q in masts:
        z0 = float(C.z([q])[0])
        mb.cylinder(q[0], q[1], z0, z0 + 0.5, 1.0, 12, "concrete")
        mb.tube((q[0], q[1], z0 + 0.5), (q[0], q[1], z0 + 30.0), 0.32, 10, "galvanised")
        mb.annulus(q[0], q[1], z0 + 30.0, 0.3, 2.2, 16, "galvanised", thick=0.3)
        for kk in range(8):
            a_ = 2 * math.pi * kk / 8
            mb.box((q[0] + 1.7 * math.cos(a_), q[1] + 1.7 * math.sin(a_), z0 + 30.6), (0.7, 0.5, 0.4), "lamp", a_)
    mb.build("KW_yard_machinery", C.M, C.col("KW_yard"), props=_props(
        "illustrative", f"{n_conv} conveyor galleries + stacker-reclaimers along Sentinel-2 windrows, {len(masts)} "
        "high masts; layout inferred, not surveyed"))


# ------------------------------------------------------------------------------------------------------------
def roads(C):
    major = {"trunk", "primary", "secondary", "tertiary"}
    mb_maj, mb_min = G.MB(), G.MB()
    zoff = {"trunk": 0.14, "primary": 0.13, "secondary": 0.12, "tertiary": 0.11, "tertiary_link": 0.1,
            "unclassified": 0.09, "service": 0.08}
    for r in C.B["lines"]["road"]:
        P = _densify(r["xy"], 6.0)
        mb = mb_maj if r["class"] in major else mb_min
        mb.ribbon(P, r["width_m"] / 2, lambda Q: C.z(Q), "asphalt" if r["class"] in major else "asphalt_plain",
                  zoff[r["class"]])
    mb_maj.build("KW_roads_major", C.M, C.col("KW_roads"), props=_props(
        "osm", "OSM centrelines; widths estimated by class; markings illustrative"))
    mb_min.build("KW_roads_service", C.M, C.col("KW_roads"), props=_props(
        "osm", "OSM centrelines; widths estimated by class"))


def _densify(pts, step):
    out = [tuple(pts[0])]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        for k in range(1, n + 1):
            out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return out


def rail(C):
    mb = G.MB()
    gauge = 1.676                                   # Indian broad gauge
    for r in C.B["lines"]["rail"]:
        P = _densify(r["xy"], 5.0)
        if len(P) < 2:
            continue
        mb.ribbon(P, 2.4, lambda Q: C.z(Q), "ballast", 0.05)
        mb.ribbon(P, 1.6, lambda Q: C.z(Q), "ballast", 0.32)
        Lft, Rgt = G.offset_polyline(P, gauge / 2)
        for side in (Lft, Rgt):
            mb.ribbon([tuple(p) for p in side], 0.04, lambda Q: C.z(Q), "rail_steel", 0.5)
    mb.build("KW_rail", C.M, C.col("KW_rail"), props=_props(
        "osm", "OSM railway centrelines; broad gauge 1676 mm; ballast and sleepers illustrative"))


def _along(P, s_):
    """Point and unit direction at arc length s_ along a polyline."""
    P = np.asarray(P, float)
    seg = np.diff(P, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(ln)])
    i = int(np.clip(np.searchsorted(cum, s_, side="right") - 1, 0, len(ln) - 1))
    d = seg[i] / max(ln[i], 1e-6)
    return P[i] + d * (s_ - cum[i]), d, float(cum[-1])


TRUCK_CABS = ("container_orange", "yellow", "container_blue", "container_red", "container_green", "white")


def _truck(mb, rng, q, d, z):
    rot = math.atan2(d[1], d[0])
    kind = rng.random()
    cab = TRUCK_CABS[rng.randrange(len(TRUCK_CABS))]
    L = 11.0 if kind < 0.75 else 15.0
    f = q + d * (L / 2 - 1.3)
    mb.box((f[0], f[1], z + 1.75), (2.4, 2.45, 2.3), cab, rot)                          # cab
    mb.box((q[0], q[1], z + 0.75), (L, 2.2, 0.5), "rubber", rot)                        # chassis + wheels
    r = q - d * 1.3
    if kind < 0.4:                                                                    # tipper (coal/bulk)
        mb.box((r[0], r[1], z + 2.0), (L - 3.2, 2.45, 1.9), cab if rng.random() < 0.5 else "steel", rot)
        mb.box((r[0], r[1], z + 2.9), (L - 3.6, 2.2, 0.15), "coal" if rng.random() < 0.7 else "bulk_white", rot)
    elif kind < 0.75:                                                                 # tanker
        a_, b_ = r - d * (L - 3.6) / 2, r + d * (L - 3.6) / 2
        mb.tube((a_[0], a_[1], z + 2.2), (b_[0], b_[1], z + 2.2), 1.15, 14, "white" if rng.random() < 0.6 else "steel")
    else:                                                                             # container trailer
        mb.box((r[0], r[1], z + 2.35), (12.2, 2.44, 2.6), CONTAINER_MATS[rng.randrange(5)], rot)


def _car(mb, rng, q, d, z):
    rot = math.atan2(d[1], d[0])
    col = ("white", "white", "steel", "container_red", "container_grey", "container_blue")[rng.randrange(6)]
    mb.box((q[0], q[1], z + 0.65), (4.3, 1.75, 0.8), col, rot)
    mb.box((q[0] - d[0] * 0.3, q[1] - d[1] * 0.3, z + 1.3), (2.3, 1.6, 0.55), "rubber", rot)


def traffic(C):
    """Road and rail traffic (ILLUSTRATIVE snapshot, not observed): truck queues parked on the shoulders of
    port/industrial service roads, moving trucks and cars in the lanes of major roads, coal rakes (open wagons) with
    diesel locomotives on the yard rail lines."""
    mb = G.MB()
    rng = _rng("traffic")
    major = {"trunk", "primary", "secondary", "tertiary"}
    n_t = n_c = n_w = 0
    for r in C.B["lines"]["road"]:
        P = r["xy"]
        if len(P) < 2:
            continue
        _, _, total = _along(P, 0.0)
        hw = r["width_m"] / 2
        s_ = rng.uniform(0, 60)
        while s_ < total - 8:
            q, d, _ = _along(P, s_)
            if C.sdist([q])[0] < 8:
                s_ += 40
                continue
            ind = float(C._bil(C.indf, [q])[0]) > 0.5
            nrm = np.array([-d[1], d[0]])
            if r["class"] in major:
                lane = rng.choice((-1, 1))
                p_ = q + nrm * lane * hw * 0.45
                dd = d if lane < 0 else -d                                            # drive on the left
                z = float(C.z([p_])[0]) + 0.14
                if rng.random() < (0.6 if ind else 0.35):
                    _truck(mb, rng, p_, dd, z); n_t += 1
                else:
                    _car(mb, rng, p_, dd, z); n_c += 1
                s_ += rng.uniform(70, 160)
            elif ind and rng.random() < 0.35:                                       # parked queue on the shoulder
                k_ = rng.randint(3, 9)
                for j in range(k_):
                    if s_ + j * 17 > total - 8:
                        break
                    qq, dq, _ = _along(P, s_ + j * 17)
                    nq = np.array([-dq[1], dq[0]])
                    p_ = qq + nq * (hw + 1.6)
                    if C.sdist([p_])[0] < 8:
                        break
                    _truck(mb, rng, p_, dq, float(C.z([p_])[0])); n_t += 1
                s_ += k_ * 17 + rng.uniform(120, 300)
            else:
                s_ += rng.uniform(150, 350)
    for r in C.B["lines"]["rail"]:
        P = r["xy"]
        if len(P) < 2:
            continue
        _, _, total = _along(P, 0.0)
        q0, _, _ = _along(P, total / 2)
        if total < 250 or float(C._bil(C.indf, [q0])[0]) < 0.5 or rng.random() < 0.35:
            continue
        n_wag = int(min(45, (total - 60) / 10.7) * rng.uniform(0.5, 1.0))
        s0 = rng.uniform(10, max(11, total - n_wag * 10.9 - 50))
        for j in range(n_wag + 2):
            q, d, _ = _along(P, s0 + j * 10.9 + (0 if j < 2 else 10.0))
            z = float(C.z([q])[0]) + 0.55
            rot = math.atan2(d[1], d[0])
            if j < 2:                                                                 # twin diesel locomotives
                q, d, _ = _along(P, s0 + j * 21.5)
                mb.box((q[0], q[1], z + 0.9), (21.0, 3.1, 1.0), "rubber", rot)
                mb.box((q[0], q[1], z + 2.75), (20.0, 2.9, 2.7), "container_blue" if j == 0 else "yellow", rot)
                continue
            mb.box((q[0], q[1], z + 0.6), (10.4, 2.6, 0.5), "rubber", rot)
            mb.box((q[0], q[1], z + 2.3), (10.3, 3.1, 2.4), "rusty" if "rusty" in C.M else "steel", rot)
            mb.box((q[0], q[1], z + 3.45), (10.0, 2.85, 0.12), "coal", rot)
            n_w += 1
    mb.build("KW_traffic", C.M, C.col("KW_traffic"), props=_props(
        "illustrative", f"snapshot traffic, not observed: {n_t} trucks, {n_c} cars, {n_w} coal wagons"))
    print(f"[traffic] trucks {n_t}, cars {n_c}, wagons {n_w}")


def pipelines(C):
    mb = G.MB()
    for r in C.B["lines"]["pipeline"]:
        P = _densify(r["xy"], 8.0)
        zz = C.z(P)
        for i in range(len(P) - 1):
            mb.tube((*P[i], zz[i] + 1.4), (*P[i + 1], zz[i + 1] + 1.4), 0.4, 10, "cladding")
            mb.box((P[i][0], P[i][1], zz[i] + 0.5), (0.6, 1.2, 1.0), "concrete_dark",
                   math.atan2(P[i + 1][1] - P[i][1], P[i + 1][0] - P[i][0]))
    mb.build("KW_pipelines", C.M, C.col("KW_pipelines"), props=_props(
        "osm", "OSM pipeline routes; diameter and supports illustrative"))


def _tower_template(C, h):
    mb = G.MB()
    levels = np.arange(0, h - 2, 4.0)
    half = lambda z: 3.6 - (3.6 - 0.9) * (z / h)  # noqa: E731
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for (sx, sy) in corners:
        mb.tube((sx * 3.6, sy * 3.6, 0), (sx * 0.9, sy * 0.9, h), 0.13, 4, "galvanised")
    for z0, z1 in zip(levels[:-1], levels[1:]):
        a, b = half(z0), half(z1)
        for i in range(4):
            c0, c1 = corners[i], corners[(i + 1) % 4]
            mb.tube((c0[0] * a, c0[1] * a, z0), (c1[0] * a, c1[1] * a, z0), 0.06, 4, "galvanised")
            mb.tube((c0[0] * a, c0[1] * a, z0), (c1[0] * b, c1[1] * b, z1), 0.05, 4, "galvanised")
            mb.tube((c1[0] * a, c1[1] * a, z0), (c0[0] * b, c0[1] * b, z1), 0.05, 4, "galvanised")
    arms = [h - 11, h - 6, h - 1]
    for z in arms:
        w = half(z)
        for s in (-1, 1):
            mb.tube((s * w, -0.6, z), (s * 7.0, 0, z), 0.08, 4, "galvanised")
            mb.tube((s * w, 0.6, z), (s * 7.0, 0, z), 0.08, 4, "galvanised")
            mb.tube((s * w, 0, z + 1.6), (s * 7.0, 0, z), 0.07, 4, "galvanised")
            mb.tube((s * 6.6, 0, z), (s * 6.6, 0, z - 2.2), 0.12, 6, "white")
    mb.tube((0, 0, h), (0, 0, h + 2.5), 0.08, 4, "galvanised")
    return mb, arms


def power(C):
    h = C.E["power_tower_h_m"]
    mb_t, arms = _tower_template(C, h)
    me_ob = mb_t.build("KW_tower_template", C.M, C.col("KW_power"))
    me = me_ob.data
    bpy.data.objects.remove(me_ob, do_unlink=True)
    lines = C.B["lines"]["power_line"]
    sites = []
    for ln in lines:
        P = np.asarray(ln["xy"], float)
        for i in range(len(P)):
            d = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
            sites.append((P[i], math.atan2(d[1], d[0])))
    for t in C.B["points"]["power_tower"]:
        p = np.asarray(t["xy"], float)
        if all(np.linalg.norm(p - s[0]) > 15 for s in sites):
            sites.append((p, 0.0))
    for k, (p, ang) in enumerate(sites):
        ob = bpy.data.objects.new(f"KW_tower_{k:03d}", me)
        ob.location = (p[0], p[1], float(C.z([p])[0]))
        ob.rotation_euler = (0, 0, ang + math.pi / 2)
        C.col("KW_power").objects.link(ob)
        for kk, v in _props("osm", "OSM power line vertex / tower node; lattice design and height illustrative").items():
            ob[kk] = v
    # conductors: 6 phases + earth wire with parabolic sag
    cu = bpy.data.curves.new("KW_conductors", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.035
    cu.bevel_resolution = 1
    for ln in lines:
        P = np.asarray(ln["xy"], float)
        if len(P) < 2:
            continue
        angs = []
        for i in range(len(P)):
            d = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
            angs.append(math.atan2(d[1], d[0]) + math.pi / 2)
        zg = C.z(P)
        attach = [(s * 6.6, z - 2.2) for z in arms for s in (-1, 1)] + [(0.0, h + 2.5)]
        for (lx, lz) in attach:
            for i in range(len(P) - 1):
                a = np.array([*(P[i] + lx * np.array([math.cos(angs[i]), math.sin(angs[i])])), zg[i] + lz])
                b = np.array([*(P[i + 1] + lx * np.array([math.cos(angs[i + 1]), math.sin(angs[i + 1])])),
                              zg[i + 1] + lz])
                span = float(np.linalg.norm(b[:2] - a[:2]))
                sag = 0.03 * span
                sp = cu.splines.new("POLY")
                n = 14
                sp.points.add(n - 1)
                for j in range(n):
                    t = j / (n - 1)
                    q = a + (b - a) * t
                    q[2] -= 4 * sag * t * (1 - t)
                    sp.points[j].co = (q[0], q[1], q[2], 1.0)
    cu.materials.append(C.M["rubber"])
    ob = bpy.data.objects.new("KW_conductors", cu)
    C.col("KW_power").objects.link(ob)
    ob["fidelity"] = "illustrative"
    ob["source"] = "conductors along OSM power-line routes; sag and geometry illustrative"


def fences(C):
    mb = G.MB()
    for r in C.B["lines"]["fence"]:
        P = _densify(r["xy"], 3.0)
        zz = C.z(P)
        for i, p in enumerate(P):
            mb.box((p[0], p[1], zz[i] + 1.2), (0.08, 0.08, 2.4), "galvanised")
        for hgt in (0.3, 1.2, 2.3):
            for i in range(len(P) - 1):
                mb.tube((*P[i], zz[i] + hgt), (*P[i + 1], zz[i + 1] + hgt), 0.02, 4, "galvanised")
    mb.build("KW_fences", C.M, C.col("KW_fences"), props=_props("osm", "OSM fence lines; fence design illustrative"))


def sensors(C):
    for s in C.B["sensors"]:
        if s["type"] != "fixed_df":
            continue
        x, y, z = s["xyz"]
        zg = float(C.z([[x, y]])[0])
        mb = G.MB()
        base = max(zg, z - 6.0)
        mb.cylinder(x, y, base, z, 0.12, 10, "galvanised")
        mb.box((x, y, z - 0.6), (0.42, 0.3, 0.16), "white")
        side = 0.054
        for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            mb.cylinder(x + dx * side / 2 * 4, y + dy * side / 2 * 4, z, z + 0.45, 0.012, 6, "antenna")
        mb.box((x, y, z - 0.02), (0.3, 0.3, 0.04), "galvanised")
        mb.build(f"KW_sensor_{s['id']}", C.M, C.col("KW_sensors"), props=_props(
            "illustrative", "SYNTHETIC DF sensor placement (AIR7311 node, 2x2 omni array, spacing exaggerated x4)",
            sensor_id=s["id"]))


# ------------------------------------------------------------------------------------------------------------
def _blob(name, r, sx, sy, sz, z, seed, mat, trunk=None, M=None, coll=None):
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=r)
    rng = np.random.default_rng(seed)
    for v in bm.verts:
        n = v.co.normalized()
        v.co = n * r * (1 + 0.22 * rng.standard_normal())
        v.co.x *= sx
        v.co.y *= sy
        v.co.z = v.co.z * sz + z
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M[mat])
    if trunk:
        me.materials.append(M["bark"])
        tmb = G.MB()
        tmb.cylinder(0, 0, 0, trunk[0], trunk[1], 6, "bark")
        tob = tmb.build(name + "_trunk", M, coll)
        tob.data.materials.clear()
        tob.data.materials.append(M["bark"])
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        tob.parent = ob
        return ob
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def vegetation(C):
    assets_scrub = C.col("KW_veg_scrub_assets")
    assets_mang = C.col("KW_veg_mangrove_assets")
    if bl_pbr.available():
        for tag, coll in (("scrub", assets_scrub), ("mangrove", assets_mang)):
            for aid, sc_ in bl_pbr.SHRUBS[tag]:
                bl_pbr.import_shrub(aid, sc_, coll, tag)
        dens = (0.0018, 0.008)
    else:
        dens = (0.0035, 0.02)
        _blobs(C, assets_scrub, assets_mang)
    _veg_nodes(C, assets_scrub, assets_mang, dens)


def _blobs(C, assets_scrub, assets_mang):
    _blob("scrub_tree_a", 1.0, 2.2, 2.0, 0.9, 2.6, 1, "scrub", (2.2, 0.18), C.M, assets_scrub)
    _blob("scrub_tree_b", 1.0, 3.0, 2.7, 1.1, 3.4, 2, "scrub", (2.8, 0.22), C.M, assets_scrub)
    _blob("scrub_bush_c", 1.0, 1.1, 1.0, 0.7, 0.5, 3, "scrub", None, C.M, assets_scrub)
    _blob("scrub_bush_d", 1.0, 1.6, 1.4, 0.8, 0.7, 4, "scrub", None, C.M, assets_scrub)
    _blob("mangrove_a", 1.0, 1.7, 1.6, 1.0, 1.0, 5, "mangrove", None, C.M, assets_mang)
    _blob("mangrove_b", 1.0, 2.4, 2.2, 1.2, 1.3, 6, "mangrove", None, C.M, assets_mang)
    _blob("mangrove_c", 1.0, 1.2, 1.1, 0.8, 0.8, 7, "mangrove", None, C.M, assets_mang)


def _veg_nodes(C, assets_scrub, assets_mang, dens):
    for c in (assets_scrub, assets_mang):
        lc = bpy.context.view_layer.layer_collection.children.get(c.name)
        if lc:
            lc.exclude = True
    ng = bpy.data.node_groups.new("KW_vegetation", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, Lk = ng.nodes, ng.links
    gi = N.new("NodeGroupInput")
    go = N.new("NodeGroupOutput")
    join = N.new("GeometryNodeJoinGeometry")
    Lk.new(gi.outputs[0], join.inputs[0])

    def scatter(attr, coll, dens, seed, smin, smax):
        dist = N.new("GeometryNodeDistributePointsOnFaces")
        na = N.new("GeometryNodeInputNamedAttribute")
        na.data_type = "FLOAT"
        na.inputs["Name"].default_value = attr
        mul = N.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = dens
        Lk.new(na.outputs["Attribute"], mul.inputs[0])
        Lk.new(gi.outputs[0], dist.inputs["Mesh"])
        Lk.new(mul.outputs[0], dist.inputs["Density"])
        dist.inputs["Seed"].default_value = seed
        ci = N.new("GeometryNodeCollectionInfo")
        ci.inputs["Collection"].default_value = coll
        ci.inputs["Separate Children"].default_value = True
        ci.inputs["Reset Children"].default_value = True
        iop = N.new("GeometryNodeInstanceOnPoints")
        Lk.new(dist.outputs["Points"], iop.inputs["Points"])
        Lk.new(ci.outputs[0], iop.inputs["Instance"])
        iop.inputs["Pick Instance"].default_value = True
        ri = N.new("FunctionNodeRandomValue")
        ri.data_type = "INT"
        ri.inputs["Min"].default_value = 0
        ri.inputs["Max"].default_value = 100
        Lk.new(NT_out(ri, "INT"), iop.inputs["Instance Index"])
        rr = N.new("FunctionNodeRandomValue")
        rr.data_type = "FLOAT_VECTOR"
        NT_in(rr, "Max", "VECTOR").default_value = (0, 0, 6.283)
        Lk.new(NT_out(rr, "VECTOR"), iop.inputs["Rotation"])
        rs = N.new("FunctionNodeRandomValue")
        rs.data_type = "FLOAT"
        NT_in(rs, "Min", "VALUE").default_value = smin
        NT_in(rs, "Max", "VALUE").default_value = smax
        Lk.new(NT_out(rs, "VALUE"), iop.inputs["Scale"])
        Lk.new(iop.outputs["Instances"], join.inputs[0])

    scatter("veg_scrub", assets_scrub, dens[0], 11, 0.6, 1.5)
    scatter("veg_mangrove", assets_mang, dens[1], 12, 0.6, 1.4)
    Lk.new(join.outputs[0], go.inputs[0])
    ter = bpy.data.objects["KW_terrain"]
    mod = ter.modifiers.new("vegetation", "NODES")
    mod.node_group = ng


def NT_in(node, name, typ):
    c = [s for s in node.inputs if s.name == name and s.type == typ]
    return ([s for s in c if s.enabled] or c)[0]


def NT_out(node, typ):
    c = [s for s in node.outputs if s.type == typ]
    return ([s for s in c if s.enabled] or c)[0]


# ------------------------------------------------------------------------------------------------------------
SUN_AZ_DEG, SUN_EL_DEG = 245.0, 28.0     # late afternoon at 23 N (illustrative lighting choice)


def lighting(C):
    global SUN_EL_DEG
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(SUN_EL_DEG)
    sky.sun_rotation = math.radians(90.0 - SUN_AZ_DEG)
    sky.air_density = 1.0
    sky.aerosol_density = 2.5
    nt.links.new(sky.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.14
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    if bl_pbr.available():
        el = bl_pbr.world_hdri(SUN_AZ_DEG, 0.55)       # captured sky as fill light; its sun rotated to our azimuth
        if el is not None and 10 < el < 80:
            SUN_EL_DEG = el
    sun = bpy.data.lights.get("KW_sun") or bpy.data.lights.new("KW_sun", "SUN")
    sun.energy = 5.2                                   # hard, hazy coastal Gujarat sun
    sun.angle = math.radians(0.53)
    sun.color = (1.0, 0.93, 0.83)
    ob = bpy.data.objects.get("KW_sun") or bpy.data.objects.new("KW_sun", sun)
    if ob.name not in C.col("KW_lighting").objects:
        C.col("KW_lighting").objects.link(ob)
    # the lamp is the only sun: the captured sky is a soft fill (its sun extraction is disabled in world_hdri)
    ob.hide_render = ob.hide_viewport = False
    if hasattr(w, "sun_threshold"):
        w.sun_threshold = 1e4
    az, el = math.radians(SUN_AZ_DEG), math.radians(SUN_EL_DEG)
    s = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    ob.rotation_euler = s.to_track_quat("Z", "Y").to_euler()
    vs = sc.view_settings
    vs.view_transform = "AgX"
    for want in ("AgX - Medium High Contrast", "Medium High Contrast"):   # look enum is dynamic: try names
        try:
            vs.look = want
            break
        except TypeError:
            continue
    vs.exposure = -0.45                                # hard sun: keep whites (paint, concrete) below clipping
    ee = sc.eevee
    for attr, val in (("use_raytracing", True), ("use_shadows", True), ("shadow_ray_count", 3),
                      ("shadow_step_count", 12), ("use_fast_gi", True), ("taa_render_samples", 128),
                      ("taa_samples", 32), ("use_shadow_jitter_viewport", True), ("horizon_quality", 0.75)):
        if hasattr(ee, attr):
            setattr(ee, attr, val)


def compositor(C=None):
    """Aerial perspective from the mist pass (EEVEE cannot render a world volume over km scales)."""
    sc = bpy.context.scene
    sc.view_layers[0].use_pass_mist = True
    ms = sc.world.mist_settings
    ms.start, ms.depth, ms.falloff = 4000.0, 32000.0, "QUADRATIC"
    ng = bpy.data.node_groups.get("KW_comp") or bpy.data.node_groups.new("KW_comp", "CompositorNodeTree")
    ng.nodes.clear()
    if not ng.interface.items_tree:
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = ng.nodes, ng.links
    rl = N.new("CompositorNodeRLayers")
    out = N.new("NodeGroupOutput")
    k = N.new("ShaderNodeMath")
    k.operation, k.use_clamp = "MULTIPLY", True
    k.inputs[1].default_value = 0.5
    L.new(rl.outputs["Mist"], k.inputs[0])
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L.new(k.outputs[0], NT_in(mix, "Factor", "VALUE"))
    img = rl.outputs["Image"]
    try:                                    # bloom so nav lights and sun glints read on screen
        gl = N.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Threshold"].default_value = 1.5
        gl.inputs["Strength"].default_value = 0.5
        gl.inputs["Size"].default_value = 0.4
        L.new(img, gl.inputs[0])
        img = gl.outputs[0]
    except Exception as e:  # noqa: BLE001
        print("glare unavailable", e)
    L.new(img, NT_in(mix, "A", "RGBA"))
    NT_in(mix, "B", "RGBA").default_value = (0.70, 0.71, 0.72, 1)      # pale dusty haze
    L.new(NT_out(mix, "RGBA"), out.inputs[0])
    sc.compositing_node_group = ng


def _look_at(cam, eye, target):
    cam.location = eye
    d = Vector(target) - Vector(eye)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def cameras(C):
    objs = {o["id"]: o for o in C.B["objects"]}
    farmsl = sorted([o for o in C.B["objects"] if o["kind"] == "tank_farm"], key=lambda o: -o.get("area_m2", 0))
    shipl = [o for o in C.B["objects"] if o["kind"] == "ship"]
    jet = objs.get("jetty_j1")
    shots = {"cam_aerial": ((-2300.0, -2600.0, 1150.0), (250.0, 300.0, 0.0), 24.0)}
    if farmsl:
        f = farmsl[0]["centroid"]
        shots["cam_tankfarm"] = ((f[0] - 330, f[1] - 280, 120.0), (f[0], f[1], 5.0), 35.0)
        shots["cam_tankfarm_close"] = ((f[0] - 120, f[1] - 60, 28.0), (f[0], f[1], 10.0), 28.0)
    if shipl:
        s = shipl[0]["centroid"]
        shots["cam_tanker"] = ((s[0] + 160, s[1] - 230, 55.0), (s[0], s[1], 4.0), 32.0)
    if jet:
        j = jet["centroid"]
        shots["cam_creek"] = ((j[0] - 700, j[1] - 500, 6.0), (j[0], j[1], 4.0), 50.0)
    for name, (eye, tgt, lens) in shots.items():
        cd = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
        cd.lens = lens
        cd.clip_start = 0.5
        cd.clip_end = 60000
        ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, cd)
        if ob.name not in C.col("KW_cameras").objects:
            C.col("KW_cameras").objects.link(ob)
        _look_at(ob, eye, tgt)
    bpy.context.scene.camera = bpy.data.objects["cam_aerial"]
    r = bpy.context.scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = 3840, 2160, 100


def setup_cycles(samples=192):
    sc = bpy.context.scene
    try:
        sc.render.engine = "CYCLES"
    except TypeError as e:
        print("engine", e)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dt in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dt
            break
        except TypeError:
            continue
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.use_persistent_data = True


def run(stages=None):
    C = Ctx()
    stages = stages or STAGES
    if "materials" not in stages:
        C.M = {m.name: m for m in bpy.data.materials}
        C.M.update({"white": bpy.data.materials.get("white_paint")})
    import time
    for s in stages:
        t0 = time.time()
        globals()[s](C)
        print(f"[build_world] {s}: {time.time() - t0:.1f}s")
    return C


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    save = None
    if "--save" in argv:
        save = argv[argv.index("--save") + 1]
    run()
    setup_cycles()
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=save, compress=True)
