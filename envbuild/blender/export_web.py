"""Export the Blender Kandla world's structures to Draco-compressed GLB tiles for the console (flat colours).

Superseded for structures by envbuild/blender/bake_web.py (baked albedo + AO atlases, demo v2); this script still
exports the vegetation assets and their instance transforms (`--only veg`).

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/export_web.py -- [--only KW_tanks,...]
    -> data/site_kandla/web/<collection>.glb + manifest.json

Coordinates stay in the site ENU frame (metres; glTF is Y-up, Cesium converts back to Z-up), so the console places
every tile with one east-north-up model matrix at the site origin. Terrain, water, vegetation scatter, cameras and
lights are not exported: Cesium supplies imagery, terrain and the water surface. Fidelity custom properties
(osm / estimated / illustrative) are written into glTF extras.
"""

import json
import re
import sys
import time
from pathlib import Path

import bpy

REPO = Path(bpy.data.filepath).resolve().parents[3]
OUT = REPO / "data" / "site_kandla" / "web"
SKIP = {"KW_terrain", "KW_water", "KW_cameras", "KW_lighting", "KW_veg_scrub_assets", "KW_veg_mangrove_assets", "KW_drones",
        "SS_evidence"}


def flat_colours() -> dict:
    """name -> linear RGB of the procedural materials (bl_materials.py `flat(...)` / `roof_flat_named(...)`)."""
    src = (REPO / "envbuild" / "blender" / "bl_materials.py").read_text(encoding="utf-8")
    pat = r'"(\w+)":\s*(?:flat|roof_flat_named)\("\w+",\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)'
    return {m[0]: tuple(float(x) for x in m[1:]) for m in re.findall(pat, src)}


def bake_flat_colours() -> list[str]:
    """glTF keeps only image textures wired to Principled Base Color; procedural colour networks export as white.
    For those materials, cut the network and set the Base Color constant to the material's defining colour."""
    table, fixed = flat_colours(), []
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        inp = bsdf.inputs["Base Color"]
        direct_img = inp.is_linked and inp.links[0].from_node.type == "TEX_IMAGE"
        if direct_img:
            continue
        rgb = table.get(m.name) or tuple(m.diffuse_color[:3])
        for link in list(inp.links):
            m.node_tree.links.remove(link)
        inp.default_value = (*rgb, 1.0)
        fixed.append(m.name)
    return fixed


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    only = a[a.index("--only") + 1].split(",") if "--only" in a else None
    return {"only": only}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = args()["only"]
    print("FLAT", bake_flat_colours(), flush=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"frame": "site ENU (m), glTF Y-up", "tiles": {}}
    for coll in bpy.data.collections:
        if not coll.name.startswith("KW_") or coll.name in SKIP or (only and coll.name not in only):
            continue
        objs = [o for o in coll.all_objects if o.type in {"MESH", "CURVE"}]
        if not objs:
            continue
        for o in objs:                               # per-building imagery tint (roof_col) travels as vertex colour
            if o.type == "MESH" and "roof_col" in o.data.attributes:
                me = o.data
                src = me.attributes["roof_col"]
                if src.domain == "FACE":                 # glTF carries only point/corner colours: expand per corner
                    import numpy as np
                    fc = np.zeros(len(me.polygons) * 4, np.float32)
                    src.data.foreach_get("color", fc)
                    fc = fc.reshape(-1, 4)
                    li = np.zeros(len(me.loops), np.int32)
                    me.polygons.foreach_get("loop_total", tot := np.zeros(len(me.polygons), np.int32))
                    li = np.repeat(np.arange(len(me.polygons)), tot)
                    dst = me.color_attributes.get("web_col") or me.color_attributes.new("web_col", "BYTE_COLOR", "CORNER")
                    dst.data.foreach_set("color", fc[li].ravel())
                    me.color_attributes.active_color = dst
                    me.color_attributes.render_color_index = me.color_attributes.find("web_col")   # what glTF exports
                else:
                    me.color_attributes.active_color = me.color_attributes["roof_col"]
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.hide_set(False)
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        f = OUT / f"{coll.name[3:]}.glb"
        t0 = time.time()
        bpy.ops.export_scene.gltf(filepath=str(f), export_format="GLB", use_selection=True, export_apply=True,
                                  export_extras=True, export_yup=True, export_draco_mesh_compression_enable=True,
                                  export_draco_mesh_compression_level=6, export_draco_position_quantization=16,
                                  export_draco_normal_quantization=10, export_draco_texcoord_quantization=12,
                                  export_image_format="WEBP", export_cameras=False, export_lights=False,
                                  export_vertex_color="ACTIVE")
        manifest["tiles"][coll.name[3:]] = {"file": f.name, "objects": len(objs), "bytes": f.stat().st_size,
                                            "fidelity": sorted({str(o.get("fidelity", "?")) for o in objs})}
        print(f"EXPORTED {f.name} {len(objs)} objs {f.stat().st_size / 1e6:.1f} MB {time.time() - t0:.0f} s", flush=True)
    if not only or "veg" in only:
        manifest["vegetation"] = export_vegetation()
    man_path.write_text(json.dumps(manifest, indent=1))


def export_vegetation() -> dict:
    """Vegetation is a geometry-nodes scatter on the terrain: write each asset once (GLB) plus its instance
    transforms (site ENU, metres) so the browser draws them as instanced meshes."""
    dg = bpy.context.evaluated_depsgraph_get()
    inst: dict[str, list] = {}
    for di in dg.object_instances:
        if not di.is_instance or not di.parent or di.parent.original.name != "KW_terrain":
            continue
        name = di.object.original.name if di.object.original else di.object.name
        m = di.matrix_world
        loc, rot, sc = m.decompose()
        inst.setdefault(name, []).append([round(loc.x, 2), round(loc.y, 2), round(loc.z, 2), round(rot.to_euler().z, 3),
                                          round(sc.x, 3)])
    out = {}
    for name, rows in inst.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        linked = ob.name not in bpy.context.view_layer.objects
        if linked:                                   # scatter sources sit in an excluded collection
            bpy.context.scene.collection.objects.link(ob)
        loc0 = ob.location.copy()
        ob.location = (0, 0, 0)
        ob.hide_set(False)
        ob.hide_render = False
        ob.select_set(True)
        f = OUT / f"veg_{name}.glb"
        bpy.ops.export_scene.gltf(filepath=str(f), export_format="GLB", use_selection=True, export_apply=True,
                                  export_yup=True, export_draco_mesh_compression_enable=True, export_cameras=False,
                                  export_lights=False, export_image_format="WEBP")
        ob.location = loc0
        if linked:
            bpy.context.scene.collection.objects.unlink(ob)
        (OUT / f"veg_{name}.json").write_text(json.dumps(rows))
        import numpy as np                           # browser format: int16 rows, see scripts/web_ground.py
        q = np.round(np.array(rows, np.float64) / np.array([0.25, 0.25, 0.01, 1e-4, 1e-3])).astype("<i2")
        q.tofile(OUT / f"veg_{name}.bin")
        out[name] = {"file": f.name, "instances": f"veg_{name}.json", "bin": f"veg_{name}.bin", "count": len(rows)}
        print(f"EXPORTED veg {name}: {len(rows)} instances", flush=True)
    return out


main()
