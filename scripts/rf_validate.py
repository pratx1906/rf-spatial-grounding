"""RF validation suite, Sionna RT side (Task 1 of docs/superpowers/plans/2026-10-20-rf-phase2.md).

    .venv-sionna-win\\Scripts\\python scripts\\rf_validate.py      -> runs/rfval/sionna.json

Canonical scenes, each traced with the same Sionna RT 2.2 PathSolver settings family as the site caches, compared
with the closed forms in sim/rf/refs.py:
  A  free space (no geometry)                      vs Friis
  B  flat ground, one specular bounce, V and H pol  vs two-ray with the slab coefficient P.2040 (sea water,
     steel, 0.3 m concrete) at 0.9 / 2.44 / 5.8 GHz
  D  thin steel screen, edge diffraction            vs ITU-R P.526 knife-edge J(v)
The narrowband channel is h = sum_paths a * exp(-j 2 pi f tau) (paths.a excludes propagation phase, CLAUDE.md 5).
"""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.rf import refs  # noqa: E402

OUT = REPO / "runs" / "rfval"
BANDS = {"sub_ghz_868_915": 0.9e9, "ism_2400": 2.44e9, "ism_5800": 5.8e9}
# ITU-R P.2040-3 Table 3 (eps = a f^b, sigma = c f^d, f in GHz); sea water from P.527 at 20 C, 35 psu (estimated)
MATS = {"steel": lambda f: (1.0, 1e7), "concrete": lambda f: (5.24, 0.0462 * (f / 1e9) ** 0.7822)}
SEA = {0.9e9: (70.2, 4.40), 2.44e9: (70.2, 5.12), 5.8e9: (64.7, 8.50)}
THICK = {"steel": 0.02, "concrete": 0.3, "sea_water": 10.0}


def _ply(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(np.asarray(v, "<f4").tobytes())
        for tri in f:
            fh.write(struct.pack("<Biii", 3, *map(int, tri)))


def _box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    f = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
         [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]
    return v, np.array(f)


def _scene(tmp: Path, shapes: list[tuple[str, np.ndarray, np.ndarray, str]], mats: dict[str, tuple]) -> Path:
    xml = ['<scene version="2.1.0">']
    for m, (eps, sig, th) in mats.items():
        xml += [f'<bsdf type="radio-material" id="mat-sc_{m}">', f'<float name="relative_permittivity" value="{eps}"/>',
                f'<float name="conductivity" value="{sig}"/>', f'<float name="thickness" value="{th}"/>', "</bsdf>"]
    for name, v, f, m in shapes:
        _ply(tmp / f"{name}.ply", v, f)
        xml += [f'<shape type="ply" id="mesh-{name}">', f'<string name="filename" value="{name}.ply"/>',
                f'<ref id="mat-sc_{m}" name="bsdf"/>', "</shape>"]
    if not shapes:     # an empty scene still needs one object far away (Sionna needs geometry to build a BVH)
        v, f = _box(1e5, 1e5 + 1, 1e5, 1e5 + 1, 0, 1)
        _ply(tmp / "far.ply", v, f)
        xml += ['<bsdf type="radio-material" id="mat-sc_far"><float name="relative_permittivity" value="1.0"/>'
                '<float name="conductivity" value="0.0"/><float name="thickness" value="0.01"/></bsdf>',
                '<shape type="ply" id="mesh-far"><string name="filename" value="far.ply"/><ref id="mat-sc_far" name="bsdf"/>'
                "</shape>"]
    xml.append("</scene>")
    p = tmp / "scene.xml"
    p.write_text("\n".join(xml))
    return p


def _trace(xml: Path, f_hz: float, tx, rxs, pol: str, **kw) -> np.ndarray:
    """Narrowband channel h per receiver (single iso antennas, polarisation pol at both ends)."""
    import sionna.rt as rt

    scene = rt.load_scene(str(xml), merge_shapes=False)
    scene.frequency = f_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization=pol)
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization=pol)
    scene.add(rt.Transmitter(name="tx", position=[float(x) for x in tx]))
    for i, p in enumerate(rxs):
        scene.add(rt.Receiver(name=f"rx{i}", position=[float(x) for x in p]))
    args = dict(max_depth=1, los=True, specular_reflection=True, diffuse_reflection=False, refraction=False,
                diffraction=False, synthetic_array=True, seed=1)
    args.update(kw)
    paths = rt.PathSolver()(scene, **args)
    ar, ai = (np.array(x) for x in paths.a)
    tau = np.array(paths.tau)
    n = len(rxs)
    a = (ar + 1j * ai).reshape(n, -1)
    t = tau.reshape(n, -1)
    a = np.where(np.isfinite(t) & (t >= 0), a, 0)
    return (a * np.exp(-2j * np.pi * f_hz * np.nan_to_num(t))).sum(1)


def case_free_space(tmp: Path) -> dict:
    d = np.array([10.0, 30, 100, 300, 1000, 3000])
    rows = {}
    for band, f in BANDS.items():
        h = _trace(_scene(tmp, [], {}), f, (0, 0, 25), [(x, 0, 25) for x in d], "V", max_depth=0)
        sim = -20 * np.log10(np.abs(h))
        ref = refs.fspl_db(d, f)
        rows[band] = {"d_m": d.tolist(), "sim_db": sim.round(3).tolist(), "ref_db": ref.round(3).tolist(),
                      "max_abs_err_db": round(float(np.abs(sim - ref).max()), 4)}
    return {"accept_db": 0.1, "bands": rows, "pass": all(r["max_abs_err_db"] < 0.1 for r in rows.values())}


def case_two_ray(tmp: Path) -> dict:
    d = np.linspace(60, 3000, 160)
    ground = np.array([[-5000, -5000, 0], [5000, -5000, 0], [5000, 5000, 0], [-5000, 5000, 0]], float)
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    out = {}
    for band, f in BANDS.items():
        for mat in ("sea_water", "steel", "concrete"):
            eps, sig = SEA[f] if mat == "sea_water" else MATS[mat](f)
            xml = _scene(tmp, [("ground", ground, tri, mat)], {mat: (eps, sig, THICK[mat])})
            for pol in ("V", "H"):
                h = _trace(xml, f, (0, 0, 25), [(x, 0, 40) for x in d], pol)
                sim = -20 * np.log10(np.abs(h))
                ref = _two_ray_slab(d, 25.0, 40.0, f, eps, sig, THICK[mat], pol)
                # error away from nulls: points where the reference is within 10 dB of free space
                fs = refs.fspl_db(np.hypot(d, 15.0), f)
                ok = ref - fs < 10
                err = np.abs(sim - ref)[ok]
                out[f"{band}/{mat}/{pol}"] = {"n": int(ok.sum()), "median_err_db": round(float(np.median(err)), 4),
                                              "p95_err_db": round(float(np.percentile(err, 95)), 4),
                                              "max_err_db": round(float(err.max()), 4),
                                              "d_m": d.tolist(), "sim_db": sim.round(3).tolist(),
                                              "ref_db": ref.round(3).tolist()}
    return {"accept": "median < 0.1 dB and p95 < 0.5 dB where the reference is within 10 dB of free space "
                      "(the max sits on the steep slopes beside interference nulls and is reported)",
            "cases": out, "pass": all(c["median_err_db"] < 0.1 and c["p95_err_db"] < 0.5 for c in out.values())}


def _two_ray_slab(d, ht, hr, f, eps, sig, thick, pol):
    lam = refs.C0 / f
    k = 2 * math.pi / lam
    r1, r2 = np.hypot(d, ht - hr), np.hypot(d, ht + hr)
    th = 90.0 - np.degrees(np.arctan2(ht + hr, d))
    g = np.array([refs.slab_gamma(eps, sig, f, t, thick)[1 if pol == "V" else 0] for t in th])
    e = lam / (4 * math.pi) * (np.exp(-1j * k * r1) / r1 + g * np.exp(-1j * k * r2) / r2)
    return -20 * np.log10(np.abs(e))


def case_wedge(tmp: Path) -> dict:
    """Thick 90-degree steel edge (block x in [-1000, 0], z in [-300, 20], 4 km along y; edge at x = 0, z = 20).
    TX at (-500, 0, 30) above the roof; RX at x = +500 swept in z through the shadow boundary. Reference: UTD
    perfectly-conducting wedge (Kouyoumjian & Pathak 1974), n = 1.5, hard polarisation (V-pol: E normal to the
    edge), plus the incident field on the lit side; specular reflection is off in the trace, so points near the
    reflection boundary are excluded. Also reported: the scalar knife edge of ITU-R P.526 (n = 2), which a 90-degree
    edge is NOT, for scale.
    Measured limitation (runs/dbg/diff_dbg3.py): Sionna finds no diffracted path behind a 0.2 m thin screen
    (its edge search needs ray hits on the faces next to the edge); site objects are tanks, buildings and hulls."""
    out = {}
    v_v, f_ = _box(-1000, 0, -2000, 2000, -300, 20)
    tx = np.array([-500.0, 30.0])
    edge = np.array([0.0, 20.0])
    phi_p = math.atan2(tx[1] - edge[1], -(tx[0] - edge[0]))
    sp = float(np.hypot(*(tx - edge)))
    for band, f in BANDS.items():
        lam = refs.C0 / f
        k = 2 * math.pi / lam
        xml = _scene(tmp, [("block", v_v, f_, "steel")], {"steel": (1.0, 1e7, 0.02)})
        zr = np.linspace(-150, 45, 79)
        h = _trace(xml, f, (tx[0], 0.0, tx[1]), [(500.0, 0.0, z) for z in zr], "V", max_depth=1,
                   specular_reflection=False, diffraction=True, edge_diffraction=True,
                   max_num_paths_per_src=4_000_000)
        rows = []
        for z, hv in zip(zr, h):
            phi = math.atan2(z - edge[1], -(500.0 - edge[0])) % (2 * math.pi)
            s_ = float(np.hypot(500.0, z - edge[1]))
            r = float(np.hypot(1000.0, z - tx[1]))
            e = refs.utd_point_source_field(1.5, phi, phi_p, sp, s_, k, hard=True)
            lit = phi < math.pi + phi_p
            if lit:
                e += complex(math.cos(-k * r), math.sin(-k * r)) / r
            ref = -20 * math.log10(abs(e) * r)
            sim = -20 * math.log10(abs(hv)) - float(refs.fspl_db(r, f)) if abs(hv) > 0 else float("nan")
            # knife-edge v for the same geometry (edge height above the TX-RX line)
            h_e = edge[1] - (tx[1] + (z - tx[1]) * 0.5)
            v = h_e * math.sqrt(2 / lam * 1000.0 / 250000.0)
            rows.append({"z": float(z), "phi_deg": math.degrees(phi), "lit": lit, "sim_db": sim, "utd_db": ref,
                         "knife_db": float(refs.knife_edge_db(v)), "v": v,
                         "near_rsb": abs(phi - (math.pi - phi_p)) < math.radians(3)})
        sh = [r for r in rows if not r["lit"] and r["phi_deg"] > math.degrees(math.pi + phi_p) + 1]
        lt = [r for r in rows if r["lit"] and not r["near_rsb"]]
        err = lambda rs: [abs(r["sim_db"] - r["utd_db"]) for r in rs if r["sim_db"] == r["sim_db"]]  # noqa: E731
        out[band] = {"rows": rows, "no_path_points": sum(r["sim_db"] != r["sim_db"] for r in rows),
                     "shadow_n": len(sh), "shadow_median_err_db": round(float(np.median(err(sh))), 3),
                     "shadow_p95_err_db": round(float(np.percentile(err(sh), 95)), 3),
                     "lit_median_err_db": round(float(np.median(err(lt))), 3) if err(lt) else None}
    ok = all(b["shadow_median_err_db"] < 1.0 and b["no_path_points"] == 0 for b in out.values())
    return {"accept": "shadow region: median |Sionna - UTD(hard; V-pol at a horizontal edge)| < 1 dB, no missing paths",
            "bands": out, "pass": ok}


def case_wedge_polarisation(tmp: Path) -> dict:
    """Which UTD coefficient (soft: E parallel to the edge; hard: E normal to the edge; Kouyoumjian & Pathak 1974)
    does Sionna apply? A 90-degree steel edge, horizontal (roof) and vertical (corner), V and H polarisation at
    2.44 GHz, three receivers 20-170 m into the shadow. Physics: V-pol is HARD at a horizontal edge and SOFT at a
    vertical edge; H-pol the reverse."""
    f = 2.44e9
    k = 2 * math.pi * f / refs.C0
    phi_p, sp = math.atan2(10, 500), math.hypot(500, 10)
    edges = {"horizontal": (_box(-1000, 0, -2000, 2000, -300, 20), (-500, 0, 30), lambda o: (500, 0, 20 + o)),
             "vertical": (_box(-1000, 0, -1000, 0, -2000, 2000), (-500, 10, 0), lambda o: (500, o, 0))}
    expect = {("horizontal", "V"): "hard", ("horizontal", "H"): "soft", ("vertical", "V"): "soft",
              ("vertical", "H"): "hard"}
    offs = [-170.0, -80.0, -40.0]
    rows = {}
    for edge, ((v_v, f_), tx, rxf) in edges.items():
        xml = _scene(tmp, [("block", v_v, f_, "steel")], {"steel": (1.0, 1e7, 0.02)})
        for pol in ("V", "H"):
            h = _trace(xml, f, tx, [rxf(o) for o in offs], pol, max_depth=1, specular_reflection=False,
                       diffraction=True, edge_diffraction=True, max_num_paths_per_src=4_000_000)
            res = []
            for o, hv in zip(offs, h):
                phi = math.atan2(o, -500) % (2 * math.pi)
                s_, r = math.hypot(500, o), math.hypot(1000, o - 10)
                sim = -20 * math.log10(abs(hv)) - float(refs.fspl_db(r, f))
                hard = -20 * math.log10(abs(refs.utd_point_source_field(1.5, phi, phi_p, sp, s_, k, True)) * r)
                soft = -20 * math.log10(abs(refs.utd_point_source_field(1.5, phi, phi_p, sp, s_, k, False)) * r)
                res.append({"offset_m": o, "sionna_db": round(sim, 2), "utd_hard_db": round(hard, 2),
                            "utd_soft_db": round(soft, 2)})
            e_h = float(np.median([abs(x["sionna_db"] - x["utd_hard_db"]) for x in res]))
            e_s = float(np.median([abs(x["sionna_db"] - x["utd_soft_db"]) for x in res]))
            rows[f"{edge}/{pol}"] = {"rows": res, "sionna_matches": "hard" if e_h < e_s else "soft",
                                     "physics_expects": expect[(edge, pol)], "median_err_to_match_db": round(min(e_h, e_s), 2)}
    swapped = all(r["sionna_matches"] != r["physics_expects"] for r in rows.values())
    return {"cases": rows, "soft_hard_exchanged": swapped,
            "note": "If exchanged: for paths whose interactions are all diffractions on (near-)conducting wedges, "
                    "the physical V-V channel equals Sionna's H-H channel (90-degree rotation of the Jones basis)."}


def case_wedge_patched(tmp: Path) -> dict:
    """Case E repeated with envbuild.sionna_fix applied, plus a 0.3 m concrete wedge against Luebbers' lossy UTD
    (soft: R_TE, hard: R_TM, single-layer slab coefficients as Sionna uses; R_0 at grazing phi', R_n at grazing
    n pi - phi). Run after the unpatched cases (the patch is process-global)."""
    from envbuild import sionna_fix

    pid = sionna_fix.apply()
    f = 2.44e9
    k = 2 * math.pi * f / refs.C0
    phi_p, sp = math.atan2(10, 500), math.hypot(500, 10)
    eps_c, sig_c = MATS["concrete"](f)
    edges = {"horizontal": (_box(-1000, 0, -2000, 2000, -300, 20), (-500, 0, 30), lambda o: (500, 0, 20 + o)),
             "vertical": (_box(-1000, 0, -1000, 0, -2000, 2000), (-500, 10, 0), lambda o: (500, o, 0))}
    physics = {("horizontal", "V"): True, ("horizontal", "H"): False, ("vertical", "V"): False, ("vertical", "H"): True}
    offs = [-170.0, -120.0, -80.0, -40.0, -20.0]
    out = {}
    for mat, (eps, sig, th) in {"steel": (1.0, 1e7, 0.02), "concrete": (eps_c, sig_c, 0.3)}.items():
        for edge, ((v_v, f_), tx, rxf) in edges.items():
            xml = _scene(tmp, [("block", v_v, f_, mat)], {mat: (eps, sig, th)})
            for pol in ("V", "H"):
                h = _trace(xml, f, tx, [rxf(o) for o in offs], pol, max_depth=1, specular_reflection=False,
                           diffraction=True, edge_diffraction=True, max_num_paths_per_src=4_000_000)
                hard = physics[(edge, pol)]
                errs, res = [], []
                for o, hv in zip(offs, h):
                    phi = math.atan2(o, -500) % (2 * math.pi)
                    s_, r = math.hypot(500, o), math.hypot(1000, o - 10)
                    L = s_ * sp / (s_ + sp)
                    g0 = refs.slab_gamma(eps, sig, f, 90 - math.degrees(phi_p), th)
                    gn = refs.slab_gamma(eps, sig, f, 90 - math.degrees(1.5 * math.pi - phi), th)
                    i = 1 if hard else 0
                    d = refs.luebbers_d(1.5, phi, phi_p, L, k, g0[i], gn[i])
                    e = d * complex(math.cos(-k * sp), math.sin(-k * sp)) / sp * math.sqrt(sp / (s_ * (s_ + sp)))                         * complex(math.cos(-k * s_), math.sin(-k * s_))
                    ref = -20 * math.log10(abs(e) * r)
                    sim = -20 * math.log10(abs(hv)) - float(refs.fspl_db(r, f))
                    errs.append(abs(sim - ref))
                    res.append({"offset_m": o, "sionna_fixed_db": round(sim, 2), "ref_db": round(ref, 2)})
                out[f"{mat}/{edge}/{pol}"] = {"coefficient": "hard" if hard else "soft", "rows": res,
                                              "median_err_db": round(float(np.median(errs)), 3),
                                              "max_err_db": round(float(np.max(errs)), 3)}
    # phase / sign: lit region between the shadow and reflection boundaries, LOS + diffraction vs UTD total field
    v_v, f_ = _box(-1000, 0, -2000, 2000, -1000, 20)
    xml = _scene(tmp, [("block", v_v, f_, "steel")], {"steel": (1.0, 1e7, 0.02)})
    txz = 120.0
    php, spl = math.atan2(txz - 20, 500), math.hypot(500, txz - 20)
    zs = np.linspace(-300, 100, 81)
    h = _trace(xml, f, (-500, 0, txz), [(500, 0, z) for z in zs], "V", max_depth=1, specular_reflection=False,
               diffraction=True, edge_diffraction=True, max_num_paths_per_src=4_000_000)
    lit_err, sh_err = [], []
    for z, hv in zip(zs, h):
        phi = math.atan2(z - 20, -500) % (2 * math.pi)
        if abs(phi - (math.pi + php)) < math.radians(1.0) or abs(phi - (math.pi - php)) < math.radians(3.0):
            continue
        s_, r = math.hypot(500, z - 20), math.hypot(1000, z - txz)
        e = refs.utd_point_source_field(1.5, phi, php, spl, s_, k, hard=True)
        lit = phi < math.pi + php
        if lit:
            e += complex(math.cos(-k * r), math.sin(-k * r)) / r
        err = abs((-20 * math.log10(abs(hv)) - float(refs.fspl_db(r, f))) - (-20 * math.log10(abs(e) * r)))
        (lit_err if lit else sh_err).append(err)
    out["phase_check_steel_horizontal_V"] = {"coefficient": "hard", "lit_n": len(lit_err),
                                             "median_err_db": round(float(np.median(lit_err)), 3),
                                             "max_err_db": round(float(np.max(lit_err)), 3),
                                             "shadow_median_err_db": round(float(np.median(sh_err)), 3)}
    two_ray = case_two_ray(tmp)
    ok = all(c["median_err_db"] < (1.0 if not k_.startswith("concrete") else 2.0) for k_, c in out.items())
    return {"patch": pid, "cases": out, "two_ray_unchanged_pass": two_ray["pass"], "pass": ok and two_ray["pass"],
            "accept": "steel median < 1 dB vs UTD; concrete median < 2 dB vs Luebbers (both heuristics differ in "
                      "the lossy face terms); two-ray reflection still passes"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in (("A_free_space", case_free_space), ("B_two_ray", case_two_ray), ("D_wedge", case_wedge), ("E_wedge_polarisation", case_wedge_polarisation), ("F_wedge_patched", case_wedge_patched)):
            try:
                res[name] = fn(tmp)
                print(name, "pass" if res[name].get("pass", True) else "FAIL",
                      {k: v for k, v in res[name].items() if k in ("pass", "accept_db")}, flush=True)
            except Exception as e:  # noqa: BLE001 - record and continue so one case cannot hide the others
                import traceback

                res[name] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
                print(name, "ERROR", repr(e), flush=True)
    (OUT / "sionna.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
