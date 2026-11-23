"""E8 RF export: Sionna/Mitsuba scene (XML + one PLY per object) from the same master as the visual twin.

    python -m envbuild.export_rf [--margin-m 1200]

ENV spec 11: metres, ENU (x east, y north, z up), closed volumes, outward normals, one shape per object with the
object id in the shape id, every shape has exactly one material. Tanks are tessellated cylinders (32 segments).
Ground: real OSM water polygons as sea water, the rest of the scene square as dry ground (no overlap).
Writes data/site_kandla/rf_scene/{scene.xml, meshes/*.ply, manifest.json}.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from envbuild.site_kandla import OUT, build

REPO = Path(__file__).resolve().parents[1]
MATERIALS = yaml.safe_load((REPO / "envbuild" / "config" / "materials.yaml").read_text())["materials"]
SEGMENTS = 32
MIN_FEATURE_M = 0.5


# ITU-R P.2040 parameters (a, b, c, d): eps_r = a * f^b, sigma = c * f^d, f in GHz (as tabulated in sionna.rt).
ITU = {"concrete": (5.24, 0.0, 0.0462, 0.7822), "metal": (1.0, 0.0, 1e7, 0.0), "brick": (3.91, 0.0, 0.0238, 0.16),
       "medium_dry_ground": (15.0, -0.1, 0.035, 1.63), "wet_ground": (30.0, -0.4, 0.15, 1.30),
       "very_dry_ground": (3.0, 0.0, 0.00015, 2.52)}


def itu_params(kind: str, f_ghz: float) -> tuple[float, float]:
    a, b, c, d = ITU[kind]
    return a * f_ghz ** b, c * f_ghz ** d


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(np.asarray(verts, "<f4").tobytes())
        for a, b, c in faces:
            f.write(struct.pack("<Biii", 3, int(a), int(b), int(c)))


def _cap(poly: Polygon) -> list[tuple]:
    tris = []
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        c = list(t.exterior.coords)[:3]
        # orient each triangle counter-clockwise (seen from above)
        (x0, y0), (x1, y1), (x2, y2) = c
        if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0:
            c = [c[0], c[2], c[1]]
        tris.append(c)
    return tris


def extrude(poly: Polygon, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed prism with outward normals (CCW exterior; holes ignored for RF, ENV spec 11 min-feature rule)."""
    poly = orient(Polygon(poly.exterior).simplify(MIN_FEATURE_M), 1.0)
    verts, faces = [], []

    def v(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1
    ring = list(poly.exterior.coords)[:-1]
    for i in range(len(ring)):
        (ax, ay), (bx, by) = ring[i], ring[(i + 1) % len(ring)]
        a0, b0, b1, a1 = v(ax, ay, z0), v(bx, by, z0), v(bx, by, z1), v(ax, ay, z1)
        faces += [(a0, b0, b1), (a0, b1, a1)]
    for tri in _cap(poly):
        top = [v(x, y, z1) for x, y in tri]
        faces.append(tuple(top))
        bot = [v(x, y, z0) for x, y in tri]
        faces.append((bot[0], bot[2], bot[1]))
    return np.array(verts), np.array(faces)


def cylinder(cx: float, cy: float, r: float, h: float, n: int = SEGMENTS) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in ang]
    return extrude(Polygon(ring), 0.0, h)


def plane(poly: Polygon, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    for p in polys:
        for tri in _cap(orient(p, 1.0)):
            base = len(verts)
            verts += [(x, y, z) for x, y in tri]
            faces.append((base, base + 1, base + 2))
    return np.array(verts), np.array(faces)


def export(margin_m: float = 1200.0) -> dict:
    b = build()
    hh = b["hero_half_m"] + margin_m
    extent = box(-hh, -hh, hh, hh)
    out = OUT / "rf_scene"
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for f in (out / "meshes").glob("*.ply"):
        f.unlink()

    shapes, used = [], set()
    tri_total = 0
    for o in b["objs"]:
        if o.kind in ("tank_farm", "road") or not extent.intersects(o.geom) or o.height_m < 1:
            continue                     # farms are groupings (their tanks are meshed); roads are ground
        if o.kind == "tank":
            c = o.geom.centroid
            r = math.sqrt(o.geom.area / math.pi)
            verts, faces = cylinder(c.x, c.y, r, o.height_m)
        else:
            g = o.geom.intersection(extent)
            if g.is_empty or g.area < 4:
                continue
            g = max(g.geoms, key=lambda p: p.area) if hasattr(g, "geoms") else g
            verts, faces = extrude(g, 0.0, o.height_m)
        mat = o.material if o.material in MATERIALS else "concrete"
        write_ply(out / "meshes" / f"{o.id}.ply", verts, faces)
        shapes.append((o.id, mat))
        used.add(mat)
        tri_total += len(faces)

    # Water from OSM coastlines (Kandla Creek) + small water polygons, same function as the creek zone.
    from envbuild.site_kandla import _load_osm, sea_polygon
    els, _ = _load_osm()
    water = sea_polygon(els, b["enu"], hh).intersection(extent)
    land = extent.difference(water)
    for name, geom, mat in (("ground_land", land, "ground_dry"), ("ground_water", water, "sea_water")):
        if geom.is_empty:
            continue
        verts, faces = plane(geom, 0.0)
        write_ply(out / "meshes" / f"{name}.ply", verts, faces)
        shapes.append((name, mat))
        used.add(mat)
        tri_total += len(faces)

    xml = ['<scene version="2.1.0">', "", "<!-- Materials (envbuild/config/materials.yaml) -->"]
    for m in sorted(used):
        spec = MATERIALS[m]
        if spec["sionna"].startswith("itu_"):
            xml += [f'\t<bsdf type="itu-radio-material" id="mat-sc_{m}">',
                    f'\t\t<string name="type" value="{spec["sionna"][4:]}"/>',
                    f'\t\t<float name="thickness" value="{spec["thickness_m"]}"/>', "\t</bsdf>"]
        else:
            xml += [f'\t<bsdf type="radio-material" id="mat-sc_{m}">',
                    f'\t\t<float name="relative_permittivity" value="{spec["relative_permittivity"]}"/>',
                    f'\t\t<float name="conductivity" value="{spec["conductivity_s_per_m"]}"/>',
                    f'\t\t<float name="thickness" value="{spec["thickness_m"]}"/>', "\t</bsdf>"]
    xml += ["", "<!-- Shapes: shape id = mesh-<object id> -->"]
    for sid, m in shapes:
        xml += [f'\t<shape type="ply" id="mesh-{sid}">', f'\t\t<string name="filename" value="meshes/{sid}.ply"/>',
                '\t\t<boolean name="face_normals" value="true"/>', f'\t\t<ref id="mat-sc_{m}" name="bsdf"/>',
                "\t</shape>"]
    xml.append("</scene>")
    (out / "scene.xml").write_text("\n".join(xml))
    # Sub-GHz variant: Sionna's ITU table starts at 1 GHz, so write the same materials as custom radio-materials
    # evaluated with the ITU-R P.2040 formulas at 0.9 GHz (slight extrapolation; DECISIONS D-022).
    sub = []
    for line in xml:
        sub.append(line)
    sub_xml = "\n".join(sub)
    for m in sorted(used):
        spec = MATERIALS[m]
        if not spec["sionna"].startswith("itu_"):
            continue
        er, sg = itu_params(spec["sionna"][4:], 0.9)
        old = (f'\t<bsdf type="itu-radio-material" id="mat-sc_{m}">\n\t\t<string name="type" '
               f'value="{spec["sionna"][4:]}"/>\n\t\t<float name="thickness" value="{spec["thickness_m"]}"/>\n\t</bsdf>')
        new = (f'\t<bsdf type="radio-material" id="mat-sc_{m}">\n'
               f'\t\t<float name="relative_permittivity" value="{er:.4f}"/>\n'
               f'\t\t<float name="conductivity" value="{sg:.6g}"/>\n'
               f'\t\t<float name="thickness" value="{spec["thickness_m"]}"/>\n\t</bsdf>')
        sub_xml = sub_xml.replace(old, new)
    (out / "scene_subghz.xml").write_text(sub_xml)
    manifest = {"extent_m": [-hh, -hh, hh, hh], "shapes": len(shapes), "triangles": tri_total,
                "materials": sorted(used), "tank_segments": SEGMENTS, "min_feature_m": MIN_FEATURE_M,
                "objects": [s for s, _ in shapes], "origin": b["enu"].meta()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--margin-m", type=float, default=1200.0)
    a = p.parse_args()
    m = export(a.margin_m)
    print(json.dumps({k: m[k] for k in ("extent_m", "shapes", "triangles", "materials")}))
