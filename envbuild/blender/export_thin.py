"""Re-export the thin-member structure groups the console draws flat (power: lattice towers + the KW_conductors curve)
with fine Draco position quantization (CPU only; the .blend is never saved).

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/export_thin.py
    -> data/site_kandla/web/power.glb (manifest tiles.power.bytes updated)

Why: bake_web.py quantizes positions to 14 bits per object. KW_conductors is one curve object spanning the whole
line route (kilometres), so 14 bits leave ~0.3 m steps on wires a few centimetres thick: the tubes collapsed into
metre-wide flat ribbons that read as white blades near the camera (render-parity check, docs/design/audit/
render_parity.md). 20 bits keep them as thin tubes (~5 mm steps). No UVs, no textures: ss3d.js draws the group flat.
"""
import json
from pathlib import Path

import bpy

REPO = Path(bpy.data.filepath).resolve().parents[3]
OUT = REPO / "data" / "site_kandla" / "web"
GROUPS = {"power": "KW_power"}

for g, cn in GROUPS.items():
    col = bpy.data.collections[cn]
    bpy.ops.object.select_all(action="DESELECT")
    objs = [o for o in col.all_objects if o.type in {"MESH", "CURVE"}]
    for o in objs:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    f = OUT / f"{g}.glb"
    bpy.ops.export_scene.gltf(filepath=str(f), export_format="GLB", use_selection=True, export_apply=True,
                              export_extras=True, export_yup=True, export_draco_mesh_compression_enable=True,
                              export_draco_mesh_compression_level=10, export_draco_position_quantization=20,
                              export_draco_normal_quantization=10, export_texcoords=False, export_image_format="NONE",
                              export_cameras=False, export_lights=False, export_vertex_color="NONE",
                              export_materials="PLACEHOLDER")
    mp = OUT / "manifest.json"
    man = json.loads(mp.read_text())
    man["tiles"][g]["bytes"] = f.stat().st_size
    man["tiles"][g]["position_bits"] = 20
    mp.write_text(json.dumps(man, indent=1))
    print("THIN", g, len(objs), f.stat().st_size)
