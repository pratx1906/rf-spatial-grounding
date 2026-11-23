"""Bake the Blender Kandla world's look into per-collection texture atlases and export Draco GLB for the console.

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/bake_web.py -- [--only KW_tanks,...]
            [--plan] [--texel 0.12] [--max 4096] [--ao-samples 64] [--out DIR] [--ground] [--gi [--gi-samples 128]]
    -> data/site_kandla/web/<group>.glb, <group>_albedo.webp, <group>_ao.webp, manifest.json (tiles.<group>.baked)
    --gi: also <group>_gi.webp, Cycles' indirect diffuse light (bounce off the ground and neighbouring structures, the
          light EEVEE's fast GI adds and three.js has no real-time equivalent for), lighting only, / GI_SCALE, linear;
          the console adds albedo x gi x GI_SCALE to the indirect term (manifest tiles.<group>.baked.gi, gi_scale)
    --ground: the terrain tile's ground instead -> ground_albedo.webp (8192, 1.2 m), ground_ao.webp, manifest.ground

Why: glTF keeps only image textures wired straight into Principled Base Color, so the procedural Blender materials
(Poly Haven sets in object-space box projection, Sentinel-2 roof tints, weathering streaks, rust, window bands) used
to arrive in the browser as one flat colour per material. Here Cycles bakes, per structure collection:
- albedo: the DIFFUSE colour pass of every procedural material (no lighting), sRGB;
- ao: ambient occlusion (6 m reach, terrain included), the glTF occlusion convention (1 = open), linear.
The browser lights the albedo with the same sun and sky as the Blender build (real-time, so overlays, shadows and the
time of day stay consistent) and applies the baked AO to the ambient term (Needle-style lightmap split).

The objects of a collection are joined (see join()), then get a new UV map `bake` (Smart UV Project, one atlas per
collection, so the texel density is uniform inside a collection). The atlas size follows the collection's surface area at the target
texel size, clamped to [512, --max]. The original UV map stays the render UV, so materials that read UVs bake as
designed. Only identity-transform objects (or many small parts) are joined, so object-space procedural coordinates
stay as authored. The .blend is never saved.

Geometry in the GLB: positions (Draco 14 bit per object), normals (10 bit), the bake UV (12 bit); no vertex colours.
One material per collection, so the browser draws each collection in one call. Fidelity custom properties travel in
glTF extras as before. Run on the GPU, alone (`python scripts/gpu_lock.py status`).
"""

import json
import math
import sys
import time
from pathlib import Path

import bpy
import bmesh

REPO = Path(bpy.data.filepath).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from envbuild.blender import web_detail as WD  # noqa: E402  (per-object stamps, markings, edge definition)
from envbuild.blender import mound as MD  # noqa: E402  (concave stockpile mounds, demo-v2 S6)

OUT = REPO / "data" / "site_kandla" / "web"
SKIP = {"KW_terrain", "KW_water", "KW_cameras", "KW_lighting", "KW_veg_scrub_assets", "KW_veg_mangrove_assets", "KW_drones",
        "KW_sensors", "SS_evidence"}
PACK_EFF = 0.62                       # share of the atlas the packed islands cover (Smart UV Project + margins)
# atlas caps for collections whose look is near-uniform (asphalt, ballast, galvanised steel): the byte budget goes
# where the procedural detail is (tanks, buildings, jetties, ships, yard)
CAP = {"roads": 2048, "rail": 2048, "stockpiles": 2048, "pipelines": 2048, "power": 2048, "traffic": 2048,
       "tank_farms": 4096, "cranes": 1024, "fences": 512}
# where the console camera usually looks (jetties, tank farms, berths, roads with markings): a finer texel than the
# --max cap allows elsewhere (8192 atlas: tanks ~0.13 m, jetties ~0.09 m, township ~0.28 m; roads 4096, ~0.3 m: 8192 cost 13 MB)
KEY = {"tanks": 8192, "jetties": 8192, "buildings": 8192, "buildings_ml": 8192, "roads": 4096}


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    get = lambda k, d: a[a.index(k) + 1] if k in a else d
    return {"only": get("--only", None) and get("--only", None).split(","), "plan": "--plan" in a,
            "texel": float(get("--texel", 0.12)), "max": int(get("--max", 4096)), "ao_samples": int(get("--ao-samples", 64)),
            "col_samples": int(get("--col-samples", 4)), "gi": "--gi" in a, "gi_samples": int(get("--gi-samples", 128)), "out": (REPO / get("--out", OUT)).resolve(), "ground": "--ground" in a}


def setup_cycles(a: dict) -> None:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                print("CYCLES device", kind, [d.name for d in devs], flush=True)
                break
        except Exception as e:                                   # noqa: BLE001
            print("device", kind, "unavailable", e)
    sc.cycles.device = "GPU"
    sc.cycles.use_denoising = False
    sc.render.bake.margin = 6
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    sc.world.light_settings.distance = 6.0                      # AO reach (m)
    # the scatter (234k shrub instances) and the water plane do not belong in structure AO
    for o in bpy.data.objects:
        if o.type == "MESH" and any(m.type == "NODES" for m in o.modifiers) and o.users_collection and \
                o.users_collection[0].name == "KW_terrain":
            for m in o.modifiers:
                if m.type == "NODES":
                    m.show_render = False
                    m.show_viewport = False
    for name in ("KW_water", "KW_veg_scrub_assets", "KW_veg_mangrove_assets"):
        c = bpy.data.collections.get(name)
        if c:
            for o in c.all_objects:
                if o is not None:
                    o.hide_render = True


def world_area(objs) -> float:
    dg = bpy.context.evaluated_depsgraph_get()
    tot = 0.0
    for o in objs:
        me = o.evaluated_get(dg).to_mesh()
        mw = o.matrix_world
        s = abs(mw.determinant()) ** (2 / 3) if mw.determinant() else 1.0
        tot += sum(p.area for p in me.polygons) * s
        o.evaluated_get(dg).to_mesh_clear()
    return tot


def atlas_size(area: float, a: dict, group: str) -> int:
    side = math.sqrt(area / PACK_EFF) / a["texel"]
    cap = KEY.get(group) or min(a["max"], CAP.get(group, a["max"]))
    return int(min(cap, max(512, 2 ** math.ceil(math.log2(max(side, 1))))))


def realise(coll) -> list:
    """Mesh objects of a collection, curves converted to meshes (the .blend is not saved)."""
    out = []
    for o in list(coll.all_objects):
        if o.type == "CURVE":
            bpy.ops.object.select_all(action="DESELECT")
            o.hide_set(False)
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.convert(target="MESH")
            o = bpy.context.view_layer.objects.active
        if o.type == "MESH" and len(o.data.polygons):
            out.append(o)
    return out


def fix_concave_stockpiles(objs: list) -> int:
    """Web-export patch (demo-v2 S6 Ruling): kandla_world.blend was built before build_world.stockpiles triangulated
    concave yards, so its mound over a concave Sentinel-2 footprint is a set of rings scaled toward the vertex mean
    (starburst, dark baked wedges). The .blend is not rebuilt (that would move the frozen v2 RF geometry); here the
    island of each concave, non-windrow pile is replaced in memory by envbuild.blender.mound, the same function the
    world build now uses, with the island's own base height, top height and material. Never saved."""
    s2 = json.loads((REPO / "data" / "site_kandla" / "blender" / "s2_structures.json").read_text())
    piles = [p for p in s2.get("stockpiles", []) if len(p["outer"]) >= 3 and not p.get("windrow")
             and not MD.is_convex(p["outer"])]
    n_fix = 0
    for o in objs:
        if o.matrix_world != o.matrix_world.Identity(4):
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.faces.ensure_lookup_table()
        seen, islands = set(), []
        for f in bm.faces:                                        # face islands (linked by shared vertices)
            if f.index in seen:
                continue
            stack, isl = [f], []
            seen.add(f.index)
            while stack:
                g = stack.pop()
                isl.append(g)
                for v in g.verts:
                    for h in v.link_faces:
                        if h.index not in seen:
                            seen.add(h.index)
                            stack.append(h)
            islands.append(isl)
        uv = bm.loops.layers.uv.active
        for p in piles:
            ring = [tuple(map(float, q)) for q in p["outer"]]
            cx = sum(q[0] for q in ring) / len(ring)
            cy = sum(q[1] for q in ring) / len(ring)
            for isl in islands:
                if not all(f.is_valid for f in isl):              # already replaced
                    continue
                vs = {v for f in isl for v in f.verts}
                if len(vs) != 5 * len(ring):                     # the old stepped mound: 5 rings of the footprint
                    continue
                mx = sum(v.co.x for v in vs) / len(vs)
                my = sum(v.co.y for v in vs) / len(vs)
                if abs(mx - cx) > 0.5 or abs(my - cy) > 0.5:
                    continue
                z0 = min(v.co.z for v in vs)
                h = max(v.co.z for v in vs) - z0
                mat = isl[0].material_index
                bmesh.ops.delete(bm, geom=list(vs), context="VERTS")
                V, T = MD.mound(ring, z0, h)
                nv = [bm.verts.new(q) for q in V]
                for t in T:
                    try:
                        f = bm.faces.new((nv[t[0]], nv[t[1]], nv[t[2]]))
                    except ValueError:                           # duplicate face
                        continue
                    f.material_index = mat
                    f.smooth = True
                    if uv is not None:
                        for lp in f.loops:
                            lp[uv].uv = (lp.vert.co.x, lp.vert.co.y)
                n_fix += 1
                print(f"  stockpile fix: {len(ring)}-vertex concave yard at ({cx:.0f}, {cy:.0f}): {len(vs)} verts "
                      f"(scaled rings) -> {len(V)} verts, {len(T)} triangles, z0 {z0:.1f} m, h {h:.1f} m", flush=True)
                break
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()
    return n_fix


def join(objs: list, name: str) -> list:
    """Cycles bakes selected objects one after another, each over the whole atlas (~5 s per object at 4096), so a
    collection of hundreds of tanks would take an hour. Objects whose transform is identity (geometry already in
    site ENU) join without changing their object-space procedural coordinates; collections of many instanced parts
    (power towers) join too, which only re-seeds their noise. Ships (13, placed by transform) stay separate."""
    ident = all(o.matrix_world == o.matrix_world.Identity(4) for o in objs)
    if len(objs) < 2 or not (ident or len(objs) > 40):
        return objs
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        if o.data.users > 1:
            o.data = o.data.copy()
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    j = bpy.context.view_layer.objects.active
    j.name = f"web_{name}"
    return [j]


def unwrap(objs: list, size: int) -> None:
    for o in objs:
        me = o.data
        if me.users > 1:                                          # shared mesh data: give each object its own copy
            o.data = me.copy()
            me = o.data
        render = next((u for u in me.uv_layers if u.active_render), None)
        uv = me.uv_layers.get("bake") or me.uv_layers.new(name="bake")
        me.uv_layers.active = uv
        if render is not None:
            render.active_render = True
        elif len(me.uv_layers) == 1:
            uv.active_render = True
    t0 = time.time()
    print(f"  cut_long {WD.cut_long(objs)} edges, {time.time() - t0:.0f} s", flush=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    margin = 4.0 / size
    t0 = time.time()
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=margin, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    print(f"  smart_project {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    bpy.ops.uv.select_all(action="SELECT")
    # bounding-box packing: the concave packer is minutes-slow on thousands of window/balcony islands
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="CARDINAL", scale=True,
                            margin_method="FRACTION", margin=margin, shape_method="AABB")
    print(f"  pack_islands {time.time() - t0:.0f} s", flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def materials_of(objs: list) -> list:
    seen, out = set(), []
    for o in objs:
        for s in o.material_slots:
            m = s.material
            if m and m.name not in seen and m.use_nodes:
                seen.add(m.name)
                out.append(m)
    return out


def target_node(m, img):
    n = m.node_tree.nodes.get("WEB_BAKE") or m.node_tree.nodes.new("ShaderNodeTexImage")
    n.name = "WEB_BAKE"
    n.image = img
    for x in m.node_tree.nodes:
        x.select = False
    n.select = True
    m.node_tree.nodes.active = n
    return n


def bake(objs: list, mats: list, kind: str, img, samples: int) -> float:
    # wide extend margin: islands' colour fills the gaps, so far mip levels do not blend in the black background
    margin = max(6, img.size[0] // 256)
    for m in mats:
        target_node(m, img)
    bpy.context.scene.cycles.samples = samples
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    t0 = time.time()
    if kind == "GI":                                              # indirect diffuse light only (no colour)
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"INDIRECT"}, uv_layer="bake", target="IMAGE_TEXTURES",
                            use_clear=True, margin=margin)
    elif kind == "DIFFUSE":
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, uv_layer="bake", target="IMAGE_TEXTURES",
                            use_clear=True, margin=margin)
    else:
        bpy.ops.object.bake(type="AO", uv_layer="bake", target="IMAGE_TEXTURES", use_clear=True, margin=margin)
    return time.time() - t0


def save_webp(img, path: Path, quality: int, bw: bool) -> None:
    st = bpy.context.scene.render.image_settings
    st.file_format = "WEBP"
    st.quality = quality
    st.color_mode = "RGB"                                       # WebP writer has no BW mode
    try:
        st.color_management = "OVERRIDE"
        st.view_settings.view_transform = "Standard"
        st.view_settings.look = "None"
        st.view_settings.exposure = 0.0
        st.view_settings.gamma = 1.0
    except Exception:                                             # noqa: BLE001
        pass
    img.save_render(str(path), scene=bpy.context.scene)


GI_SCALE = 1.5                        # gi webp value = indirect light / GI_SCALE (8 bit, linear)


def gi_to_8bit(src, name: str):
    """The float GI bake / GI_SCALE, clamped, into an 8-bit Non-Color image for the WebP writer."""
    import numpy as np
    w, h = src.size
    px = np.empty(w * h * 4, np.float32)
    src.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    print(f"  gi light: median {float(np.median(px[:, :3])):.3f}, p99 {float(np.percentile(px[:, :3], 99)):.3f}, "
          f"clipped {float((px[:, :3] > GI_SCALE).mean()) * 100:.2f} %", flush=True)
    px[:, :3] = np.clip(px[:, :3] / GI_SCALE, 0.0, 1.0)
    px[:, 3] = 1.0
    out = bpy.data.images.new(name, w, h, alpha=False, float_buffer=False)
    out.colorspace_settings.name = "Non-Color"
    out.pixels.foreach_set(px.ravel())
    return out


def export_group(coll, objs: list, img_name: str, f: Path) -> None:
    mat = bpy.data.materials.get(f"baked_{img_name}") or bpy.data.materials.new(f"baked_{img_name}")
    mat.use_nodes = True
    WD.strip_attrs(objs)
    for o in objs:
        me = o.data
        for u in [u for u in me.uv_layers if u.name != "bake"]:
            me.uv_layers.remove(u)
        for ca in list(me.color_attributes):
            me.color_attributes.remove(ca)
        me.materials.clear()
        me.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.gltf(filepath=str(f), export_format="GLB", use_selection=True, export_apply=True,
                              export_extras=True, export_yup=True, export_draco_mesh_compression_enable=True,
                              export_draco_mesh_compression_level=10, export_draco_position_quantization=14,
                              export_draco_normal_quantization=10, export_draco_texcoord_quantization=12,
                              export_image_format="NONE", export_cameras=False, export_lights=False,
                              export_vertex_color="NONE", export_materials="EXPORT")


def bake_ground(a: dict, size: int = 8192) -> dict:
    """The detailed terrain tile's Blender ground (Sentinel-2 colour with the build's mottling, worn patches, wet banks)
    as one north-up texture over the tile, plus AO that carries the structures' contact occlusion onto the ground.
    UV: planar over the tile bounds (u east, v north), the same mapping the browser derives from world position."""
    import numpy as np
    meta = json.loads((REPO / "data" / "site_kandla" / "web" / "terrain.json").read_text())
    o = bpy.data.objects["KW_terrain"]
    me = o.data
    uv = me.uv_layers.get("bake") or me.uv_layers.new(name="bake")
    me.uv_layers.active = uv
    for u in me.uv_layers:
        if u.name == "UVMap":
            u.active_render = True
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    mw = np.array(o.matrix_world)
    wx = co[:, 0] * mw[0, 0] + mw[0, 3]
    wy = co[:, 1] * mw[1, 1] + mw[1, 3]
    uvs = np.stack([(wx - meta["x0"]) / (meta["x1"] - meta["x0"]), (wy - meta["y0"]) / (meta["y1"] - meta["y0"])], 1)[vi]
    uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    mats = [s.material for s in o.material_slots if s.material]
    alb = bpy.data.images.new("ground_albedo", size, size, alpha=False, float_buffer=False)
    alb.colorspace_settings.name = "sRGB"
    t_c = bake([o], mats, "DIFFUSE", alb, 2)
    ao = bpy.data.images.new("ground_ao", size // 2, size // 2, alpha=False, float_buffer=False)
    ao.colorspace_settings.name = "Non-Color"
    bpy.context.scene.world.light_settings.distance = 12.0
    t_ao = bake([o], mats, "AO", ao, a["ao_samples"])
    save_webp(alb, OUT / "ground_albedo.webp", 80, False)
    save_webp(ao, OUT / "ground_ao.webp", 78, True)
    nb = (OUT / "ground_albedo.webp").stat().st_size + (OUT / "ground_ao.webp").stat().st_size
    print(f"BAKED ground: colour {t_c:.0f} s, ao {t_ao:.0f} s, {nb / 1e6:.1f} MB, texel {(meta['x1'] - meta['x0']) / size:.2f} m", flush=True)
    return {"albedo": "ground_albedo.webp", "ao": "ground_ao.webp", "size": size,
            "texel_m": round((meta["x1"] - meta["x0"]) / size, 2), "bytes": nb}


def main() -> None:
    global OUT
    a = args()
    OUT = a["out"]
    if a["ground"]:
        setup_cycles(a)
        man_path = OUT / "manifest.json"
        manifest = json.loads(man_path.read_text())
        manifest["ground"] = bake_ground(a)
        man_path.write_text(json.dumps(manifest, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    setup_cycles(a)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"frame": "site ENU (m), glTF Y-up", "tiles": {}}
    total = 0
    for coll in list(bpy.data.collections):
        if not coll.name.startswith("KW_") or coll.name in SKIP or (a["only"] and coll.name not in a["only"]):
            continue
        objs = realise(coll)
        if not objs:
            continue
        g = coll.name[3:]
        if g == "stockpiles" and not a["plan"]:
            print(f"  concave stockpiles fixed: {fix_concave_stockpiles(objs)}", flush=True)
        fidelity = sorted({str(o.get("fidelity", "?")) for o in objs})
        n_src = len(objs)
        area = world_area(objs)
        size = atlas_size(area, a, g)
        texel = math.sqrt(area / PACK_EFF) / size
        print(f"PLAN {g}: {n_src} objs, {sum(len(o.data.polygons) for o in objs)} faces, area {area / 1e3:.0f}k m2, "
              f"atlas {size}, texel ~{texel:.2f} m", flush=True)
        if a["plan"]:
            continue
        t0 = time.time()
        detail = WD.prepare(objs, g)
        objs = join(objs, g)
        for o in objs:                                            # glTF extras of the joined mesh
            o["markings"] = WD.MARKINGS
            if len(objs) == 1 and n_src > 1:
                o["fidelity"] = "+".join(fidelity)
                o["source"] = f"{n_src} objects of KW_{g} joined for the web bake; per-object fidelity and source in kandla_world.blend"
                for k in ("name", "osm_id", "height_m", "diameter_m", "roof"):   # the first object's, not the group's
                    if k in o.keys():
                        del o[k]
        unwrap(objs, size)
        t_uv = time.time() - t0
        mats = materials_of(objs)
        patched = WD.patch(mats)
        print(f"  detail {detail}, patched {patched}", flush=True)
        alb = bpy.data.images.new(f"{g}_albedo", size, size, alpha=False, float_buffer=False)
        alb.colorspace_settings.name = "sRGB"
        t_c = bake(objs, mats, "DIFFUSE", alb, a["col_samples"])
        asz = max(512, size // 2)
        ao = bpy.data.images.new(f"{g}_ao", asz, asz, alpha=False, float_buffer=False)
        ao.colorspace_settings.name = "Non-Color"
        t_ao = bake(objs, mats, "AO", ao, a["ao_samples"])
        t_gi = 0.0
        if a["gi"]:
            gif = bpy.data.images.new(f"{g}_gi_f", asz, asz, alpha=False, float_buffer=True)
            gif.colorspace_settings.name = "Non-Color"
            t_gi = bake(objs, mats, "GI", gif, a["gi_samples"])
            save_webp(gi_to_8bit(gif, f"{g}_gi"), OUT / f"{g}_gi.webp", 85, True)
        save_webp(alb, OUT / f"{g}_albedo.webp", 82, False)
        save_webp(ao, OUT / f"{g}_ao.webp", 80, True)
        for m in mats:                                            # leave the shared materials as they were
            n = m.node_tree.nodes.get("WEB_BAKE")
            if n:
                m.node_tree.nodes.remove(n)
        f = OUT / f"{g}.glb"
        export_group(coll, objs, g, f)
        files = [f, OUT / f"{g}_albedo.webp", OUT / f"{g}_ao.webp"] + ([OUT / f"{g}_gi.webp"] if a["gi"] else [])
        nbytes = sum(x.stat().st_size for x in files)
        total += nbytes
        manifest["tiles"][g] = {"file": f.name, "objects": n_src, "bytes": f.stat().st_size, "fidelity": fidelity,
                                "baked": {"albedo": f"{g}_albedo.webp", "ao": f"{g}_ao.webp", "atlas": size,
                                          "texel_m": round(texel, 3), "bytes": nbytes,
                                          **({"gi": f"{g}_gi.webp", "gi_scale": GI_SCALE} if a["gi"] else {})},
                                "markings": WD.MARKINGS}
        man_path.write_text(json.dumps(manifest, indent=1))
        print(f"BAKED {g}: uv {t_uv:.0f} s, colour {t_c:.0f} s, ao {t_ao:.0f} s, gi {t_gi:.0f} s, {nbytes / 1e6:.1f} MB", flush=True)
    print(f"TOTAL baked bytes this run {total / 1e6:.1f} MB", flush=True)


main()
