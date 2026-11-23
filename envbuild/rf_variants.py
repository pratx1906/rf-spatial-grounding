"""RF living-world variants: parametrised Sionna scenes for tide, berth occupancy, materials and wet weather.

    python -m envbuild.rf_variants --export-all            (WSL; geometry from the same OSM master as export_rf)
    python -m envbuild.rf_variants --only W1 B2

Spec: configs/rf_variants.yaml; plan docs/superpowers/plans/2026-10-20-rf-living-world.md (rev 2); DECISIONS D-031.
The nominal scene (envbuild/export_rf.py) is never modified: variants reuse its static meshes and rewrite only the
water plane, a bank skirt along the coastline, the ships and the materials, with one scene file per band so every
band gets its own ITU-R P.2040 material values (scaled by the variant's material draw).
World variants are written under data/site_kandla/world_cache/ (world side only); belief variants under
rf_cache/ensemble_scene/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import LineString, Polygon, box

from envbuild.export_rf import MATERIALS, extrude, itu_params, plane, write_ply
from envbuild.site_kandla import OUT

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs" / "rf_variants.yaml"
NOMINAL = OUT / "rf_scene"
BANDS = {"sub_ghz_868_915": 900e6, "ism_2400": 2.44e9, "ism_5800": 5.8e9}   # = rf_precompute.BAND_HZ


@dataclass
class Variant:
    id: str
    side: str
    water_z_m: float
    ships: dict | str
    material_seed: int | None
    weather: dict
    extra_ships: list = field(default_factory=list)
    wet: bool = False
    note: str = ""
    material_scale: dict = field(default_factory=dict)
    stack: str = "v1"
    geometry: str | None = None          # v2 Task 3 features (envbuild/geometry_v2.py)
    tank_fill: dict | None = None
    mudflat: bool = False
    clutter: dict | None = None
    diffuse: bool = False
    sea_state: int = 0                   # v2: 0-4, Ament roughness at trace time (S7.4; sim.rf.materials)


def _draw_scale(seed: int | None) -> dict[str, dict[str, float]]:
    out = {}
    rng = np.random.default_rng(seed) if seed is not None else None
    for m in sorted(MATERIALS):
        p = float(MATERIALS[m].get("perturb_pct", 0))
        if rng is None:
            out[m] = {"eps": 1.0, "sigma": 1.0}
            continue
        eps, sig = 1 + rng.uniform(-p, p) / 100, 1 + rng.uniform(-p, p) / 100
        if MATERIALS[m]["sionna"] == "itu_metal":
            eps = 1.0                                   # metal: eps = 1 by definition, never scaled
        out[m] = {"eps": round(eps, 4), "sigma": round(sig, 4)}
    return out


def load_variants(path: Path = SPEC, stack: str = "v1") -> dict[str, Variant]:
    raw = yaml.safe_load(path.read_text())
    out = {}
    for vid, v in raw["variants" if stack == "v1" else "variants_v2"].items():
        var = Variant(id=vid, side=v["side"], water_z_m=float(v["water_z_m"]), ships=v.get("ships", "all"),
                      material_seed=v.get("material_seed"), weather=v.get("weather", {}),
                      extra_ships=v.get("extra_ships", []), wet=bool(v.get("wet", False)), note=v.get("note", ""),
                      geometry=v.get("geometry"), tank_fill=v.get("tank_fill"), mudflat=bool(v.get("mudflat", False)),
                      clutter=v.get("clutter"), diffuse=bool(v.get("diffuse", False)),
                      sea_state=int(v.get("sea_state", 0)))
        var.material_scale = _draw_scale(var.material_seed)
        var.stack = stack
        out[vid] = var
    return out


def scene_dir(v: Variant) -> Path:
    from envbuild.variant_roots import scene_dir as _sd

    return _sd(v.id, v.side, v.stack)


def cache_root(v: Variant) -> Path:
    """Where the traced AIR7311 caches of a variant go."""
    from envbuild.variant_roots import cache_root as _cr

    return _cr(v.id, v.side, v.stack)


def _material_values(m: str, band_hz: float, v: Variant, cfg: dict) -> tuple[float, float]:
    spec = MATERIALS[m]
    f_ghz = band_hz / 1e9
    if m == "sea_water" and v.stack == "v2":
        from sim.rf.materials import sea_water

        er, sg = sea_water(band_hz, 25.0, 35.0)
    elif spec["sionna"].startswith("itu_"):
        kind = spec["sionna"][4:]
        if v.wet and m == "ground_dry":
            kind = cfg["wet"]["ground_dry_as"]
        er, sg = itu_params(kind, f_ghz)
    else:
        er, sg = float(spec["relative_permittivity"]), float(spec["conductivity_s_per_m"])
    sc = v.material_scale.get(m, {"eps": 1.0, "sigma": 1.0})
    er, sg = er * sc["eps"], sg * sc["sigma"]
    if v.wet and m in cfg["wet"] and isinstance(cfg["wet"][m], dict):
        er, sg = er * cfg["wet"][m]["eps"], sg * cfg["wet"][m]["sigma"]
    return max(1.0, er), sg


def _nominal_shapes() -> list[tuple[str, str]]:
    xml = (NOMINAL / "scene.xml").read_text()
    return re.findall(r'<shape type="ply" id="mesh-([\w-]+)">.*?<ref id="mat-sc_(\w+)"', xml, flags=re.S)


def _ship_hull(cx: float, cy: float, ang: float, L: float, W: float) -> Polygon:
    c, s = math.cos(ang), math.sin(ang)
    return Polygon([(cx + dx * c - dy * s, cy + dx * s + dy * c)
                    for dx, dy in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])


def _major_axis(p: Polygon) -> float:
    pts = list(p.minimum_rotated_rectangle.exterior.coords)
    a, b = max(((pts[i], pts[i + 1]) for i in range(4)), key=lambda ab: math.dist(*ab))
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _geometry():
    """Extent, water/land split and the dynamic objects, exactly as envbuild.export_rf builds them."""
    from envbuild.site_kandla import _load_osm, build, sea_polygon

    b = build()
    ext = json.loads((NOMINAL / "manifest.json").read_text())["extent_m"]
    hh = ext[2]
    extent = box(-hh, -hh, hh, hh)
    els, _ = _load_osm()
    water = sea_polygon(els, b["enu"], hh).intersection(extent)
    ships = {o.id: o for o in b["objs"] if o.kind == "ship"}
    jetties = [o.geom for o in b["objs"] if o.kind == "jetty"]
    _geometry.tanks = {o.id: (o.geom, float(o.height_m)) for o in b["objs"] if o.kind == "tank"}
    return extent, water, ships, jetties


def _skirt(water, extent, water_z: float, jetties, near_m: float):
    """Vertical bank strip along the land/water boundary from water_z up to the quay (z = 0), so no ray passes
    under the land sheet; concrete near jetties/berths, wet mud elsewhere."""
    inner = water.boundary.difference(extent.exterior.buffer(1.0))
    lines = list(inner.geoms) if hasattr(inner, "geoms") else [inner]
    groups = {"concrete": ([], []), "mudflat_wet": ([], [])}
    for ln in lines:
        if ln.is_empty or not isinstance(ln, LineString):
            continue
        pts = list(ln.simplify(1.0).coords)
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            mid = LineString([(ax, ay), (bx, by)]).interpolate(0.5, normalized=True)
            mat = "concrete" if any(j.distance(mid) <= near_m for j in jetties) else "mudflat_wet"
            verts, faces = groups[mat]
            k = len(verts)
            verts += [(ax, ay, water_z), (bx, by, water_z), (bx, by, 0.0), (ax, ay, 0.0)]
            faces += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return {m: (np.array(v), np.array(f)) for m, (v, f) in groups.items() if v}


def export_variant(v: Variant, out_dir: Path | None = None) -> dict:
    cfg = yaml.safe_load(SPEC.read_text())
    out_dir = Path(out_dir or scene_dir(v))
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    for f in (out_dir / "meshes").glob("*.ply"):
        f.unlink()
    extent, water, ships, jetties = _geometry()
    shapes: list[tuple[str, str, str]] = []           # (shape id, material, filename relative to out_dir)

    hidden: dict[str, tuple] = {}                     # Task 3 shapes the belief model never has
    meta: dict = {}
    if v.stack == "v2":
        raise ValueError("v2 scenes come from envbuild.blender_rf (Blender world geometry), not the OSM exporter")
    replaced = {k[: -len("_wall")] for k in hidden if k.endswith("_wall")}
    for sid, mat in _nominal_shapes():
        if sid.startswith("ship_") or sid == "ground_water" or sid in replaced:
            continue                                   # variant-specific below
        rel = Path(os.path.relpath(NOMINAL / "meshes" / f"{sid}.ply", out_dir)).as_posix()
        shapes.append((sid, mat, rel))

    wz = v.water_z_m
    tidal = wz < -0.01
    wgeom = water.buffer(cfg["bank"]["water_buffer_m"]).intersection(extent) if tidal else water
    verts, faces = plane(wgeom, wz)
    write_ply(out_dir / "meshes" / "ground_water.ply", verts, faces)
    shapes.append(("ground_water", "sea_water", "meshes/ground_water.ply"))
    if tidal:
        for mat, (sv, sf) in _skirt(water, extent, wz, jetties, cfg["bank"]["concrete_within_m"]).items():
            name = f"bank_{mat}"
            write_ply(out_dir / "meshes" / f"{name}.ply", sv, sf)
            shapes.append((name, mat, f"meshes/{name}.ply"))

    spec = v.ships if isinstance(v.ships, dict) else {}
    absent = set(spec.get("absent", []))
    shift = spec.get("shift_along_m", {})
    sc = cfg["ship"]
    hulls = {}
    for sid, o in sorted(ships.items()):
        if sid in absent:
            continue
        g = o.geom
        if sid in shift:
            ang = _major_axis(g)
            from shapely.affinity import translate
            g = translate(g, shift[sid] * math.cos(ang), shift[sid] * math.sin(ang))
        hulls[sid] = g
    for k, e in enumerate(v.extra_ships, start=1):
        hulls[f"ship_x{k}"] = _ship_hull(e["x_m"], e["y_m"], math.radians(90 - e["heading_deg"]),
                                         sc["length_m"], sc["beam_m"])
    for sid, g in hulls.items():
        sv, sf = extrude(g, wz, wz + sc["freeboard_m"])
        write_ply(out_dir / "meshes" / f"{sid}.ply", sv, sf)
        shapes.append((sid, "steel", f"meshes/{sid}.ply"))

    for sid, (mat, hv, hf) in sorted(hidden.items()):
        if len(hv):
            write_ply(out_dir / "meshes" / f"{sid}.ply", hv, hf)
            shapes.append((sid, mat, f"meshes/{sid}.ply"))
    used = sorted({m for _, m, _ in shapes})
    for band, f_hz in BANDS.items():
        xml = ['<scene version="2.1.0">', "", f"<!-- RF variant {v.id} ({v.side}), band {band}: D-031 -->"]
        for m in used:
            er, sg = _material_values(m, f_hz, v, cfg)
            xml += [f'\t<bsdf type="radio-material" id="mat-sc_{m}">',
                    f'\t\t<float name="relative_permittivity" value="{er:.6g}"/>',
                    f'\t\t<float name="conductivity" value="{sg:.6g}"/>',
                    f'\t\t<float name="thickness" value="{MATERIALS[m]["thickness_m"]}"/>']
            if v.diffuse:      # Sionna also scales specular reflection by sqrt(1 - S^2): only in diffuse worlds
                xml.append(f'\t\t<float name="scattering_coefficient" value="{MATERIALS[m].get("scattering", 0.0)}"/>')
            xml.append("\t</bsdf>")
        xml += [""]
        for sid, m, rel in shapes:
            xml += [f'\t<shape type="ply" id="mesh-{sid}">', f'\t\t<string name="filename" value="{rel}"/>',
                    '\t\t<boolean name="face_normals" value="true"/>', f'\t\t<ref id="mat-sc_{m}" name="bsdf"/>',
                    "\t</shape>"]
        xml.append("</scene>")
        (out_dir / f"scene_{band}.xml").write_text("\n".join(xml))
    manifest = {"id": v.id, "side": v.side, "water_z_m": wz, "ships": sorted(hulls), "weather": v.weather,
                "wet": v.wet, "material_seed": v.material_seed, "material_scale": v.material_scale,
                "shapes": len(shapes), "bands": list(BANDS), "fidelity": "estimated (D-031)", "note": v.note,
                "stack": v.stack, "hidden_features": meta, "diffuse": v.diffuse}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--export-all", action="store_true")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--stack", choices=["v1", "v2"], default="v1")
    a = p.parse_args()
    if a.stack == "v2":
        raise SystemExit("v2 scenes are built from the Blender world only: python -m envbuild.blender_rf --export-all "
                         "(STRICT rule: no OSM-extrusion geometry in production)")
    vs = load_variants(stack=a.stack)
    for vid in (list(vs) if a.export_all else a.only):
        m = export_variant(vs[vid])
        print(f"{vid:4s} {m['side']:6s} water {m['water_z_m']:+.1f} m  ships {len(m['ships'])}  shapes {m['shapes']}"
              f"  -> {scene_dir(vs[vid])}")


if __name__ == "__main__":
    main()
