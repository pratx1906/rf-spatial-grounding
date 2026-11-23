"""RF scenes from the Blender world (RF phase 2; STRICT: the Blender world is the only production geometry).

    python -m envbuild.blender_rf --export-all       (WSL, CPU; after envbuild/blender/export_rf.py in Blender)

Builds one Sionna scene directory per v2 variant (configs/rf_variants.yaml `variants_v2`):
  structures  every Blender structure shape (data/site_kandla/rf_scene_blender/meshes, objects.json), referenced in
              place; Sentinel-2 ships (`ship_obs_*`) can be removed or shifted along their hull heading (principal
              axis of the steel hull mesh) per variant, and they FLOAT: every shape of a ship moves by
              water_z_m - MSL_M (constant draft; the Blender world has them at mean sea level -3.5 m)
  terrain     data/site_kandla/blender/terrain.npz (land 0 m, creek bed to -11 m) resampled to TERRAIN_STEP_M; split into
              land (ground_dry) and bed (mudflat_wet): at low tide the bed between the water line and the shore is
              exposed intertidal ground, at high tide it is under water
  water       a sea-water plane at the variant's water level (Blender mean sea level -3.5 m; tide -6 .. 0 m);
              sea_state 0-4 (manifest; the Ament roughness factor is applied per path at trace time,
              envbuild.air7311_precompute --sea-state, since Sionna 2.2 has no angle-dependent roughness)
  tank fill   floating-roof tanks (diameter >= 40 m, assumed type, [HUMAN]) replaced by an open steel shell and a roof
              disc at the drawn fill fraction (synthetic)
  clutter     the Blender traffic snapshot (ILLUSTRATIVE) for hidden worlds that declare it
Materials: band-centre ITU-R P.2040 / Klein-Swift values x the variant's material draw (envbuild.rf_variants).

Data versions (SCENESCOUT_DATA_VERSION, envbuild.variant_roots.data_roots): the structures come from the version's
Blender RF export (v2/v21 rf_scene_blender, v3 rf_scene_blender_rf or, with SCENESCOUT_RF_GEOM=rfr, rf_scene_blender_rfr) and the scenes go to the version's cache roots.
The demo scene dirs are written only with --allow-demo-roots (v2), never under v3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

from envbuild.export_rf import MATERIALS, write_ply
from envbuild.variant_roots import (add_allow_flag, convergence_scene_dir, data_roots, data_version,  # noqa: E402
                                   guard_write, scene_dir)

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "data" / "site_kandla"
BL = SITE / "rf_scene_blender"          # demo (v2) export; export_v2 reads the data version's export (blender_src)
SPEC = REPO / "configs" / "rf_variants.yaml"
BANDS = {"sub_ghz_868_915": 900e6, "ism_2400": 2.44e9, "ism_5800": 5.8e9}
TERRAIN_STEP_M = 20.0
MSL_M = -3.5


def terrain_meshes(step: float = TERRAIN_STEP_M) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = np.load(SITE / "blender" / "terrain.npz")
    xs, ys, h = z["x"], z["y"], z["z"]
    k = max(1, int(round(step / (xs[1] - xs[0]))))
    xs, ys, h = xs[::k], ys[::k], h[::k, ::k]
    ny, nx = h.shape
    X, Y = np.meshgrid(xs, ys)
    verts = np.column_stack([X.ravel(), Y.ravel(), h.ravel()])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    zc = verts[tris, 2].mean(1)
    out = {}
    for name, sel in (("terrain_land", zc >= -0.25), ("terrain_bed", zc < -0.25)):
        t = tris[sel]
        used, inv = np.unique(t, return_inverse=True)
        out[name] = (verts[used], inv.reshape(-1, 3))
    return out


def _ply_read(p: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = p.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    lines = head.decode().splitlines()
    nv = int(next(x for x in lines if x.startswith("element vertex")).split()[-1])
    nf = int(next(x for x in lines if x.startswith("element face")).split()[-1])
    v = np.frombuffer(body[: nv * 12], "<f4").reshape(nv, 3).astype(float)
    f = np.frombuffer(body[nv * 12: nv * 12 + nf * 13], dtype=np.dtype([("n", "u1"), ("i", "<i4", 3)]))["i"]
    return v, f


def _cylinder(cx, cy, r, z0, z1, n=48, cap: float | None = None):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    v, f = [], []
    for i in range(n):
        j = (i + 1) % n
        base = len(v)
        v += [(*ring[i], z0), (*ring[j], z0), (*ring[j], z1), (*ring[i], z1)]
        f += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    if cap is not None:
        c0 = len(v)
        v.append((cx, cy, cap))
        v += [(*p, cap) for p in ring]
        f += [(c0, c0 + 1 + i, c0 + 1 + (i + 1) % n) for i in range(n)]
    return np.array(v, float), np.array(f, int)


def hull_axis(vtx: np.ndarray) -> np.ndarray:
    """Unit hull heading [3] (z = 0) of a ship mesh: the principal axis of its horizontal vertex positions. Sign:
    positive along the hull's dominant bbox axis (x if the x extent >= y extent, else y), the direction the old
    bbox-axis shift used, so `shift_along_m` signs keep their meaning."""
    xy = np.asarray(vtx, float)[:, :2]
    xy = xy - xy.mean(0)
    w, v = np.linalg.eigh(xy.T @ xy)
    ax = v[:, int(np.argmax(w))]
    ext = np.ptp(np.asarray(vtx, float)[:, :2], axis=0)
    dom = 0 if ext[0] >= ext[1] else 1
    if ax[dom] < 0:
        ax = -ax
    return np.array([ax[0], ax[1], 0.0])


def ship_offset(hull_vtx: np.ndarray, shift_m: float, water_z_m: float) -> np.ndarray:
    """Translation [3] of every shape of one ship: `shift_m` along the hull heading plus the tide (constant draft:
    the hull rises and falls with the water, water_z_m - MSL_M)."""
    return hull_axis(hull_vtx) * float(shift_m) + np.array([0.0, 0.0, float(water_z_m) - MSL_M])


def blender_src(version: str | None = None, rf_geom: str | None = None) -> Path:
    """The Blender RF export the scenes reference (v2/v21: rf_scene_blender; v3: rf_scene_blender_rf, or
    rf_scene_blender_rfr with SCENESCOUT_RF_GEOM=rfr / rf_geom="rfr")."""
    return data_roots(version, rf_geom)["rf_scene"]


def export_v2(vid: str, spec: dict, cfg: dict, out: Path | None = None, allow_demo_roots: bool = False) -> dict:
    out = guard_write(Path(out) if out is not None else scene_dir(vid, spec["side"], "v2"), allow_demo_roots)
    BL = blender_src()
    from envbuild.rf_variants import Variant, _draw_scale, _material_values
    from sim.rf.materials import sea_state_sigma_h

    v = Variant(id=vid, side=spec["side"], water_z_m=float(spec["water_z_m"]), ships=spec.get("ships", "all"),
                material_seed=spec.get("material_seed"), weather=spec.get("weather", {}),
                extra_ships=spec.get("extra_ships", []), wet=bool(spec.get("wet", False)), note=spec.get("note", ""))
    v.material_scale, v.stack, v.diffuse = _draw_scale(v.material_seed), "v2", bool(spec.get("diffuse", False))
    sea_state = int(spec.get("sea_state", 0))
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for f in (out / "meshes").glob("*.ply"):
        f.unlink()
    objs = json.loads((BL / "objects.json").read_text())["shapes"]
    ships = spec.get("ships") if isinstance(spec.get("ships"), dict) else {}
    absent, shift = set(ships.get("absent", [])), ships.get("shift_along_m", {})
    fills, rng = {}, np.random.default_rng(spec["tank_fill"]["seed"]) if spec.get("tank_fill") else None
    shapes, meta = [], {"structures": 0, "clutter": 0, "tank_fill": 0, "ships_absent": sorted(absent),
                        "ships_shifted": sorted(shift), "ships_floating_dz_m": round(v.water_z_m - MSL_M, 3)}
    offsets = {}            # ship object -> translation (heading shift + tide), from its steel hull mesh
    for r in objs:
        if r["object"].startswith("ship_obs_") and r["rf_material"] == "steel" and r["object"] not in absent:
            hv, _ = _ply_read(BL / "meshes" / f"{r['shape']}.ply")
            offsets[r["object"]] = ship_offset(hv, shift.get(r["object"], 0.0), v.water_z_m)
    missing = set(shift) - set(offsets) - absent
    if missing:
        raise ValueError(f"{vid}: shift_along_m names unknown ships {sorted(missing)}")
    meta["ship_offsets_m"] = {k: [round(float(x), 3) for x in o] for k, o in sorted(offsets.items()) if np.any(o)}
    for r in objs:
        if r["group"] == "clutter" and not spec.get("clutter"):
            continue
        if r["object"] in absent:
            continue
        rel = Path(os.path.relpath(BL / "meshes" / f"{r['shape']}.ply", out)).as_posix()
        off = offsets.get(r["object"])
        if off is not None and np.any(off != 0):
            vtx, fc = _ply_read(BL / "meshes" / f"{r['shape']}.ply")
            write_ply(out / "meshes" / f"{r['shape']}.ply", vtx + off, fc)
            rel = f"meshes/{r['shape']}.ply"
        if rng is not None and r["collection"] == "KW_tanks" and r["rf_material"] == "steel":
            b = r["bbox"]
            diam = max(b[3] - b[0], b[4] - b[1])
            if diam >= 40.0:
                fill = float(rng.uniform(0.2, 0.95))
                vv, ff = _cylinder((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, diam / 2, b[2], b[5], cap=b[2] + fill * (b[5] - b[2]))
                name = f"{r['shape']}_open"
                write_ply(out / "meshes" / f"{name}.ply", vv, ff)
                shapes.append((name, "steel", f"meshes/{name}.ply"))
                fills[r["object"]] = round(fill, 3)
                meta["tank_fill"] += 1
                continue
        shapes.append((r["shape"], r["rf_material"], rel))
        meta["clutter" if r["group"] == "clutter" else "structures"] += 1
    for name, (vv, ff) in terrain_meshes().items():
        write_ply(out / "meshes" / f"{name}.ply", vv, ff)
        shapes.append((name, "ground_dry" if name == "terrain_land" else "mudflat_wet", f"meshes/{name}.ply"))
    bb = np.array([r["bbox"] for r in objs])
    x0, y0, x1, y1 = bb[:, 0].min() - 3000, bb[:, 1].min() - 3000, bb[:, 3].max() + 3000, bb[:, 4].max() + 3000
    wz = v.water_z_m
    water = np.array([[x0, y0, wz], [x1, y0, wz], [x1, y1, wz], [x0, y1, wz]])
    write_ply(out / "meshes" / "water.ply", water, np.array([[0, 1, 2], [0, 2, 3]]))
    shapes.append(("water", "sea_water", "meshes/water.ply"))
    for k, e in enumerate(v.extra_ships, start=1):
        sc = cfg["ship"]
        L, W, H = sc["length_m"], sc["beam_m"], sc["freeboard_m"]
        from envbuild.export_rf import extrude
        from envbuild.rf_variants import _ship_hull

        vv, ff = extrude(_ship_hull(e["x_m"], e["y_m"], math.radians(90 - e["heading_deg"]), L, W), wz - 10, wz + H)
        write_ply(out / "meshes" / f"ship_x{k}.ply", vv, ff)
        shapes.append((f"ship_x{k}", "steel", f"meshes/ship_x{k}.ply"))
    used = sorted({m for _, m, _ in shapes})
    for band, f_hz in BANDS.items():
        xml = ['<scene version="2.1.0">', f"<!-- RF variant {vid} ({v.side}), band {band}: Blender world geometry -->"]
        for m in used:
            er, sg = _material_values(m, f_hz, v, cfg)
            xml += [f'\t<bsdf type="radio-material" id="mat-sc_{m}">', f'\t\t<float name="relative_permittivity" value="{er:.6g}"/>',
                    f'\t\t<float name="conductivity" value="{sg:.6g}"/>',
                    f'\t\t<float name="thickness" value="{MATERIALS[m]["thickness_m"]}"/>']
            if v.diffuse:
                xml.append(f'\t\t<float name="scattering_coefficient" value="{MATERIALS[m].get("scattering", 0.0)}"/>')
            xml.append("\t</bsdf>")
        for sid, m, rel in shapes:
            xml += [f'\t<shape type="ply" id="mesh-{sid}">', f'\t\t<string name="filename" value="{rel}"/>',
                    '\t\t<boolean name="face_normals" value="true"/>', f'\t\t<ref id="mat-sc_{m}" name="bsdf"/>', "\t</shape>"]
        xml.append("</scene>")
        (out / f"scene_{band}.xml").write_text("\n".join(xml))
    man = {"id": vid, "side": v.side, "stack": "v2", "geometry": "blender", "water_z_m": wz, "weather": v.weather,
           "wet": v.wet, "material_seed": v.material_seed, "material_scale": v.material_scale, "diffuse": v.diffuse,
           "sea_state": sea_state, "sea_sigma_h_m": sea_state_sigma_h(sea_state),
           "shapes": len(shapes), "hidden_features": {**meta, "fills": fills,
                                                       "clutter_fidelity": "illustrative" if spec.get("clutter") else None},
           "source": str(BL / "objects.json"), "note": v.note}
    if data_version() != "v2":
        man["data_version"] = data_version()          # v2 manifests stay byte-identical
        man["rf_scene"] = BL.name                      # v3: rf_scene_blender_rf (Task 3) or _rfr (Task 3b)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--convergence-scenes", action="store_true",
                    help="v3 only: write to the per-geometry convergence scene dirs (variant_roots."
                         "convergence_scene_dir; rf and rfr side by side for scripts/convergence_sweep.py)")
    add_allow_flag(ap)
    a = ap.parse_args()
    cfg = yaml.safe_load(SPEC.read_text())
    vs = cfg["variants_v2"]
    todo = list(vs) if a.export_all else a.only
    if a.convergence_scenes and data_version() != "v3":
        raise SystemExit("--convergence-scenes needs SCENESCOUT_DATA_VERSION=v3")
    dest = {vid: (convergence_scene_dir(vid, vs[vid]["side"]) if a.convergence_scenes
                  else scene_dir(vid, vs[vid]["side"], "v2")) for vid in todo}
    for vid in todo:                                  # refuse demo roots before writing anything
        guard_write(dest[vid], a.allow_demo_roots)
    if not (blender_src() / "objects.json").exists():
        raise SystemExit(f"no Blender RF export at {blender_src()}: run envbuild/blender/export_rf.py first")
    for vid in todo:
        m = export_v2(vid, vs[vid], cfg, out=dest[vid], allow_demo_roots=a.allow_demo_roots)
        print(vid, m["side"], "water", m["water_z_m"], "shapes", m["shapes"], json.dumps({k: v for k, v in m["hidden_features"].items() if k != "fills"}))


if __name__ == "__main__":
    main()
