"""Export the Blender Kandla world's structures as the RF scene geometry (RF phase 2; STRICT rule: the detailed
Blender world is the only production geometry: RF, sim, tools, console, tests).

    blender -b data/site_kandla/blender/kandla_world.blend -P envbuild/blender/export_rf.py -- --allow-demo-roots
    -> data/site_kandla/rf_scene_blender/meshes/<shape>.ply + objects.json          (the demo-v2 export)
    blender -b ... -P envbuild/blender/export_rf.py -- --simplify --out data/site_kandla/rf_scene_blender_rf
    -> the simplified RF copy (RF convergence plan, Task 3; envbuild/blender/rf_simplify.simplify_bmesh)
    blender -b ... -P envbuild/blender/export_rf.py -- --rf-relevance --min-cross-m 0.05 --out data/site_kandla/rf_scene_blender_rfr
    -> plus the RF-relevance pass (Task 3b): components with cross-section < 0.05 m dropped, ML-building roof slabs
       collapsed to their top faces (envbuild/blender/rf_relevance)

Arguments after `--`: --out (default: the data version's rf_scene, envbuild.variant_roots.data_roots), --simplify,
--angle-deg (1.0), --min-extent-m (0.10), --weld-m (0.0005), --angle-override COLLECTION=DEG (repeatable),
--rf-relevance (implies --simplify), --min-cross-m (0.05), --slab-top COLLECTION (repeatable; default
KW_buildings_ml), --allow-demo-roots. The demo export dir is written only with
--allow-demo-roots, and never with --simplify. Only <out>/meshes/*.ply is wiped. With --simplify the mesh is
simplified in the site frame (after the world transform), before triangulation; objects.json records the parameters
and triangle counts under "simplify".

Every mesh object of the structure collections is evaluated (modifiers applied), placed in the site ENU frame
(Z up, metres), triangulated and split by face material into RF material classes (steel, concrete, corrugated
metal, dry ground). One PLY per (object, RF material). objects.json keeps per shape: Blender object, collection,
RF material, fidelity/source custom properties, OSM id and name where present, triangle count and bounding box.
Not exported: terrain and water (the RF ground comes from terrain.npz, water per tide variant: envbuild.rf_variants),
vegetation and road paint (sub-wavelength / flat on the ground), DF sensor masts (the node antenna sits on top),
and traffic (dynamic clutter: kept as a separate `clutter` group for hidden worlds only).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import bmesh
    import bpy
except ImportError:            # imported outside Blender (tests of the argument handling)
    bmesh = bpy = None

REPO = Path(bpy.data.filepath).resolve().parents[3] if bpy is not None else Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from envbuild.variant_roots import DemoRootError, add_allow_flag, data_roots, guard_write, is_demo_path  # noqa: E402

OUT = REPO / "data" / "site_kandla" / "rf_scene_blender"          # the demo-v2 export
STRUCTURES = ["KW_tanks", "KW_buildings", "KW_buildings_ml", "KW_jetties", "KW_tank_farms", "KW_pipelines", "KW_power",
              "KW_rail", "KW_fences", "KW_stockpiles", "KW_ships", "KW_cranes", "KW_yard"]
CLUTTER = ["KW_traffic"]
# Blender material -> RF material (radio properties in envbuild/config/materials.yaml)
STEEL = {"steel", "tank_paint", "deck_seam", "galvanised", "rail_steel", "hull", "deck_green", "cladding",
         "container_grey", "yellow", "antenna"}
CONCRETE = {"concrete", "concrete_dark", "wall", "roof_flat", "bund_floor", "rubber"}
METAL_SHEET = {"roof_metal"}
GROUND = {"ballast", "coal", "bulk_white", "asphalt", "asphalt_plain"}
COLLECTION_DEFAULT = {"KW_tanks": "steel", "KW_ships": "steel", "KW_power": "steel", "KW_pipelines": "steel",
                      "KW_cranes": "steel", "KW_yard": "steel", "KW_fences": "steel", "KW_rail": "steel",
                      "KW_traffic": "steel", "KW_stockpiles": "ground_dry"}


def rf_material(blender_mat: str | None, collection: str) -> str:
    base = (blender_mat or "").split(".")[0]
    if base in STEEL:
        return "steel"
    if base in CONCRETE:
        return "concrete"
    if base in METAL_SHEET:
        return "corrugated_metal"
    if base in GROUND:
        return "ground_dry"
    return COLLECTION_DEFAULT.get(collection, "concrete")       # e.g. white_paint: on the collection's substrate


def write_ply(path: Path, verts, faces) -> None:
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        for v in verts:
            f.write(struct.pack("<fff", *v))
        for a, b, c in faces:
            f.write(struct.pack("<Biii", 3, a, b, c))


SUM_KEYS = ("thin_removed", "thin_removed_tris", "thin_removed_area_m2", "slabs_collapsed", "slab_removed_tris",
            "slab_removed_area_m2", "slab_top_area_m2")


def merge_relevance(acc: dict, st: dict) -> dict:
    """Add one rf_relevance_pass result (or a per-collection sum) into `acc`."""
    for k in SUM_KEYS:
        acc[k] = acc.get(k, 0) + st.get(k, 0)
    acc["thin_max_cross_m"] = max(acc.get("thin_max_cross_m", 0.0), st.get("thin_max_cross_m", 0.0))
    by = acc.setdefault("thin_by_class", {})
    for c, d in st.get("thin_by_class", {}).items():
        e = by.setdefault(c, {"n": 0, "tris": 0, "area_m2": 0.0, "max_cross_m": 0.0})
        for k in ("n", "tris", "area_m2"):
            e[k] += d[k]
        e["max_cross_m"] = max(e["max_cross_m"], d["max_cross_m"])
    return acc


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Arguments after `--` (Blender passes its own before it); no `--`: all defaults."""
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(prog="export_rf.py")
    ap.add_argument("--out", default=None, help="export dir (default: the data version's rf_scene)")
    ap.add_argument("--simplify", action="store_true", help="simplified RF copy (never into the demo export)")
    ap.add_argument("--angle-deg", type=float, default=1.0, help="planar dissolve angle limit")
    ap.add_argument("--min-extent-m", type=float, default=0.10, help="drop components smaller than this (bbox max)")
    ap.add_argument("--weld-m", type=float, default=0.0005, help="weld distance (< 1 mm: thin double skins)")
    ap.add_argument("--angle-override", action="append", default=[], metavar="COLLECTION=DEG",
                    help="per-collection dissolve angle (e.g. KW_tanks=3); only with a passing rf_geom_check")
    ap.add_argument("--rf-relevance", action="store_true",
                    help="drop components with cross-section < --min-cross-m and collapse roof slabs (implies --simplify)")
    ap.add_argument("--min-cross-m", type=float, default=0.05, help="RF-relevance cross-section threshold")
    ap.add_argument("--slab-top", action="append", default=None, metavar="COLLECTION",
                    help="collections whose closed roof slabs keep only their top faces (default KW_buildings_ml)")
    add_allow_flag(ap)
    a = ap.parse_args(rest)
    if a.rf_relevance:
        a.simplify = True
    if a.slab_top is None:
        a.slab_top = ["KW_buildings_ml"] if a.rf_relevance else []
    return a


def resolve_out(a: argparse.Namespace) -> Path:
    out = Path(a.out) if a.out else data_roots()["rf_scene"]
    out = out if out.is_absolute() else REPO / out
    if a.simplify and is_demo_path(out) is not None:
        raise DemoRootError(f"refusing to write a simplified export into the demo root {out}: use --out "
                            f"{data_roots('v3')['rf_scene']}")
    return guard_write(out, a.allow_demo_roots)


def export_object(o, collection: str, depsgraph, meshes: Path, simplify: dict | None = None,
                  stats: dict | None = None) -> list[dict]:
    ev = o.evaluated_get(depsgraph)
    me = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    if simplify is None:
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.transform(o.matrix_world)
    else:                       # simplify in the site frame (metres), so extents and weld act on world sizes
        from envbuild.blender.rf_simplify import simplify_bmesh

        bm.transform(o.matrix_world)
        st = simplify_bmesh(bm, **simplify)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        if stats is not None:
            for k in ("tris_before", "tris_after", "removed_components", "restored_components"):
                stats[k] = stats.get(k, 0) + int(st.get(k, 0))
            if "thin_removed" in st:
                merge_relevance(stats, st)
    groups: dict[str, tuple[list, list, dict]] = {}
    for f in bm.faces:
        mat = o.material_slots[f.material_index].material.name if (o.material_slots and f.material_index < len(o.material_slots)
                                                                     and o.material_slots[f.material_index].material) else None
        rm = rf_material(mat, collection)
        verts, faces, index = groups.setdefault(rm, ([], [], {}))
        tri = []
        for v in f.verts:
            if v.index not in index:
                index[v.index] = len(verts)
                verts.append(tuple(v.co))
            tri.append(index[v.index])
        faces.append(tri)
    bm.free()
    ev.to_mesh_clear()
    out = []
    for rm, (verts, faces, _) in sorted(groups.items()):
        if not faces:
            continue
        sid = f"{o.name}__{rm}".replace(" ", "_").replace(".", "_")
        write_ply(meshes / f"{sid}.ply", verts, faces)
        xs, ys, zs = zip(*verts)
        out.append({"shape": sid, "object": o.name, "collection": collection, "rf_material": rm,
                    "fidelity": str(o.get("fidelity", "")), "source": str(o.get("source", "")),
                    "osm_id": str(o.get("osm_id", "")) or None, "name": str(o.get("name", "")) or None,
                    "tris": len(faces), "bbox": [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]})
    return out


def main(argv: list[str]) -> None:
    a = parse_args(argv)
    OUT = resolve_out(a)
    simplify = ({"angle_deg": a.angle_deg, "min_extent_m": a.min_extent_m, "weld_m": a.weld_m}
                if a.simplify else None)
    relevance = {"min_cross_m": a.min_cross_m} if a.rf_relevance else None
    overrides = {k: float(v) for k, v in (o.split("=", 1) for o in a.angle_override)}
    if overrides and simplify is None:
        raise SystemExit("--angle-override needs --simplify")
    stats: dict = {}
    per_col: dict = {}
    meshes = OUT / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for f in meshes.glob("*.ply"):
        f.unlink()
    dg = bpy.context.evaluated_depsgraph_get()
    rows = []
    for group, cols in (("structure", STRUCTURES), ("clutter", CLUTTER)):
        for cname in cols:
            col = bpy.data.collections.get(cname)
            if col is None:
                print(f"[export_rf] missing collection {cname}", flush=True)
                continue
            sp = simplify if simplify is None or cname not in overrides else {**simplify, "angle_deg": overrides[cname]}
            if sp is not None and relevance is not None:
                sp = {**sp, "rf_relevance": {**relevance, "slab_top": cname in a.slab_top}}
            cst = per_col.setdefault(cname, {"angle_deg": sp["angle_deg"]} if sp else {})
            for o in col.all_objects:
                if o.type != "MESH":
                    continue
                for r in export_object(o, cname, dg, meshes, sp, cst):
                    r["group"] = group
                    rows.append(r)
            print(f"[export_rf] {cname}: {sum(r['collection'] == cname for r in rows)} shapes", flush=True)
    meta = {"source": "data/site_kandla/blender/kandla_world.blend", "frame": "site ENU, metres, Z up",
            "water_level_m": -3.5, "land_level_m": 0.0, "shapes": rows,
            "tris": {"structure": sum(r["tris"] for r in rows if r["group"] == "structure"),
                     "clutter": sum(r["tris"] for r in rows if r["group"] == "clutter")}}
    if simplify is not None:
        for c in per_col.values():
            for k in ("tris_before", "tris_after", "removed_components", "restored_components"):
                stats[k] = stats.get(k, 0) + c.get(k, 0)
            if relevance is not None:
                merge_relevance(stats, c)
        params = {**simplify, "angle_override": overrides}
        if relevance is not None:
            params["rf_relevance"] = {**relevance, "slab_top": a.slab_top}
        meta["simplify"] = {"params": params, **stats, "per_collection": per_col}
    (OUT / "objects.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["tris"]), len(rows), "shapes", flush=True)


if bpy is not None:
    main(sys.argv)
    sys.exit(0)
