"""Sionna RT precompute for one AIR7311 DF node (4-channel coherent receiver) at a site sensor. Windows Sionna venv.

    .venv-sionna-win\\Scripts\\python -m envbuild.air7311_precompute --sensor S4 [--validate]

Per propagation setup (los | spec | spec_diff) and band, traces paths from the node (transmitter at the node,
reciprocity) to a grid of drone positions within --radius-m, and stores per path the COMPLEX amplitude, delay and
arrival direction at the node (compass az, elevation), the interaction type and the first object hit.
The 4-antenna response is then exact plane-wave steering of these paths (what Sionna's synthetic_array does), so
any array geometry can be evaluated from one run. `--validate` checks that claim against Sionna's own
per-antenna output (synthetic_array=True and False) for a 4-element square array.
Output: data/site_kandla/rf_cache/air7311_<sensor>/<setup>/<band>.npz + manifest.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.rf_precompute import BAND_HZ, SCENE, SITE  # noqa: E402
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402

HEIGHTS_M = (5.0, 15.0, 40.0, 80.0)
TOP_K = 24              # captured-energy fraction per cell is stored (kept_energy) so truncation is measured
SETUPS = {
    "los":       dict(max_depth=0, specular_reflection=False, diffraction=False),
    "spec":      dict(max_depth=2, specular_reflection=True, diffraction=False),
    "spec_diff": dict(max_depth=2, specular_reflection=True, diffraction=True, edge_diffraction=True),
    # RF phase 2 diffuse worlds: plus diffuse reflection (materials carry scattering_coefficient in their XML)
    "spec_diff_scat": dict(max_depth=2, specular_reflection=True, diffraction=True, edge_diffraction=True,
                           diffuse_reflection=True, samples_per_src=300_000,
                           max_num_paths_per_src=10_000_000),     # diffuse buffers scale with samples x cap, not chunk
}
C = 299792458.0


def load(frequency_hz: float, scene_dir: Path | None = None, band: str | None = None, dual_pol: bool = False):
    """Nominal scene (scene.xml / scene_subghz.xml) by default; a living-world variant directory (D-031) holds one
    scene_<band>.xml per band with band-centre material values. dual_pol: the grid (drone-end) antenna is a
    co-located V+H pair, so every path yields VV and VH coefficients (RF phase 2, Task 2)."""
    import sionna.rt as rt

    if scene_dir is not None:
        xml = Path(scene_dir) / f"scene_{band}.xml"
    else:
        xml = SCENE.with_name("scene_subghz.xml") if frequency_hz < 1e9 else SCENE
    scene = rt.load_scene(str(xml), merge_shapes=False)
    scene.frequency = frequency_hz
    iso = dict(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array = rt.PlanarArray(**iso)
    scene.rx_array = rt.PlanarArray(**{**iso, "polarization": "VH"}) if dual_pol else rt.PlanarArray(**iso)
    names = {int(o.object_id): n.removeprefix("mesh-") for n, o in scene.objects.items()}
    return scene, names


def place(scene, tx, rx: np.ndarray):
    import sionna.rt as rt

    for n in list(scene.transmitters):
        scene.remove(n)
    for n in list(scene.receivers):
        scene.remove(n)
    scene.add(rt.Transmitter(name="tx", position=[float(v) for v in tx]))
    for i, p in enumerate(rx):
        scene.add(rt.Receiver(name=f"rx{i}", position=[float(v) for v in p]))


def lattice_shift_uv(seed: int | None) -> np.ndarray | None:
    """The seeded Cranley-Patterson rotation of one shifted ray lattice: a uniform (u, v) in [0, 1)^2 (None: none).
    Sionna 2.2 launches rays on a deterministic Fibonacci lattice and its `seed` only drives stochastic samplers, so
    a different seed alone traces the same rays (rf_convergence.md S7.4)."""
    return None if seed is None else np.random.default_rng(int(seed)).uniform(0, 1, 2)


def rotate_unit(x, y, uv):
    """(x, y) in the unit square rotated by uv, mod 1 (numpy reference of the Dr.Jit rotation in shifted_lattice)."""
    return np.mod(np.asarray(x) + float(uv[0]), 1.0), np.mod(np.asarray(y) + float(uv[1]), 1.0)


@contextlib.contextmanager
def shifted_lattice(seed: int | None):
    """Trace on a shifted Fibonacci ray lattice (an equally valid ray set of the same density; None: unchanged).
    Used by scripts/convergence_sweep.py (S7.4 / Task 4) and by --lattice-shift (Task 5 fallback)."""
    uv = lattice_shift_uv(seed)
    if uv is None:
        yield
        return
    import drjit as dr
    import mitsuba as mi
    from sionna.rt.path_solvers import sb_candidate_generator as sb

    orig = sb.fibonacci_lattice

    def lat(n):
        p = orig(n)
        x, y = p.x + float(uv[0]), p.y + float(uv[1])
        return mi.Point2f(x - dr.floor(x), y - dr.floor(y))

    sb.fibonacci_lattice = lat
    try:
        yield
    finally:
        sb.fibonacci_lattice = orig


MAX_PATHS = 40_000_000      # Sionna's default cap (1e6 per source) silently truncates large receiver grids


def solve(scene, setup: str, synthetic: bool = True, max_paths: int = MAX_PATHS, seed: int = 1,
          overrides: dict | None = None):
    """overrides: solver keywords replacing the setup's (e.g. samples_per_src, max_depth; convergence sweep)."""
    import sionna.rt as rt

    kw = dict(los=True, diffuse_reflection=False, refraction=False, synthetic_array=synthetic, seed=seed,
              max_num_paths_per_src=max_paths)
    kw.update(SETUPS[setup])
    kw.update(overrides or {})
    return rt.PathSolver()(scene, **kw)


class GridScene:
    """Pass a whole receiver grid to the solver as ONE GPU tensor (context manager on a real Scene).

    Sionna's Scene.targets() loops over receiver objects in Python with per-device GPU scatters, and scene.add()
    is per receiver: with tens of thousands of receivers that loop, not ray tracing, dominates the runtime. The
    solver and Paths only need target positions/orientations (targets()) and the receiver count (len(receivers)).
    The solver requires a genuine Scene instance, so the scene's class is swapped for a subclass overriding just
    those two members for the duration of the call. The scene keeps one transmitter and one placeholder receiver.
    """

    def __init__(self, scene, positions: np.ndarray):
        import drjit as dr
        import mitsuba as mi

        self.scene = scene
        p = np.asarray(positions, np.float32)
        n = len(p)
        pos = mi.Point3f(mi.Float(p[:, 0]), mi.Float(p[:, 1]), mi.Float(p[:, 2]))
        zero, vel = dr.zeros(mi.Point3f, n), dr.zeros(mi.Vector3f, n)
        base = type(scene)

        class _Grid(base):
            @property
            def receivers(self):
                return dict.fromkeys(range(n))

            def targets(self, synthetic_array: bool, return_velocities: bool):
                rel = dr.zeros(mi.Point3f, n * self.rx_array.array_size) if synthetic_array else None
                return pos, zero, rel, (vel if return_velocities else None)

        self._base, self._grid = base, _Grid

    def __enter__(self):
        self.scene.__class__ = self._grid
        return self.scene

    def __exit__(self, *exc):
        self.scene.__class__ = self._base


def solve_grid(scene, tx: np.ndarray, pts: np.ndarray, setup: str, max_paths: int | None = None, seed: int = 1,
               overrides: dict | None = None):
    """One trace from the node to every point in `pts` [N,3]."""
    import sionna.rt as rt

    for n in list(scene.transmitters):
        scene.remove(n)
    if not scene.receivers:
        scene.add(rt.Receiver(name="placeholder", position=[0.0, 0.0, 1.0]))
    scene.add(rt.Transmitter(name="tx", position=[float(v) for v in tx]))
    with GridScene(scene, pts) as gs:
        return solve(gs, setup, max_paths=max_paths or MAX_PATHS, seed=seed, overrides=overrides)


def _rx_paths(t: np.ndarray, n_rx: int, n_p: int, ant: int = 0) -> np.ndarray:
    """Reduce a [rx, (rx_ant), tx, (tx_ant), paths] tensor to [rx, paths] for receive antenna `ant` (single tx).
    Tensors without an antenna dimension (tau, angles with synthetic arrays) have one slot: ant is ignored."""
    r = t.reshape(n_rx, -1, n_p)
    return r[:, min(ant, r.shape[1] - 1), :]


def _pull(paths, dual_pol: bool = False, vertices: bool = False) -> dict[str, np.ndarray]:
    """Device -> host copies of everything extract needs (main thread; the GPU can trace the next chunk while
    _reduce runs on these numpy arrays in a worker thread). vertices: also the interaction points (sea roughness)."""
    keys = ["phi_t", "theta_t", "tau", "interactions", "objects"] + (["phi_r", "theta_r"] if dual_pol else []) + \
        (["vertices"] if vertices else [])
    raw = {k: np.array(getattr(paths, k)) for k in keys}
    raw["ar"], raw["ai"] = (np.array(x) for x in paths.a)
    return raw


SPECULAR = 1      # sionna.rt.constants.InteractionType.SPECULAR


def sea_roughness_factor(inter: np.ndarray, objs: np.ndarray, verts: np.ndarray, tx, sea_ids,
                         sigma_h_m: float, f_hz: float) -> np.ndarray:
    """Field factor [rx, paths] of the Ament specular reduction for every specular bounce on the horizontal sea plane.

    inter/objs [depth, rx, paths] (interaction type and object id per depth), verts [depth, rx, paths, 3] (interaction
    points), tx [3] (the trace source: the node). The grazing angle psi of a bounce on the horizontal water plane
    is the elevation of the incoming segment (previous vertex or tx -> bounce point): sin psi = |dz| / |d|.
    rho = exp(-2 (2 pi sigma_h sin psi / lambda)^2) per bounce (sim.rf.materials.ament_rho); paths with several sea
    bounces get the product. Sionna 2.2 has no angle-dependent roughness (its scattering_coefficient only scales
    specular by a constant sqrt(1 - S^2)), so the factor is applied after the trace; path discovery is unchanged.
    """
    from sim.rf.materials import ament_rho

    depth = inter.shape[0]
    fac = np.ones(inter.shape[1:], float)
    if sigma_h_m <= 0 or depth == 0:
        return fac
    sea = np.isin(objs, np.asarray(list(sea_ids), dtype=objs.dtype)) & (inter == SPECULAR)
    prev = np.broadcast_to(np.asarray(tx, float), verts.shape[1:])
    for d in range(depth):
        seg = verts[d] - prev
        n = np.linalg.norm(seg, axis=-1)
        sin_psi = np.where(n > 0, np.abs(seg[..., 2]) / np.maximum(n, 1e-12), 0.0)
        fac = np.where(sea[d], fac * ament_rho(sin_psi, sigma_h_m, f_hz), fac)
        prev = np.where((inter[d] != 0)[..., None], verts[d], prev)
    return fac


def _reduce(raw: dict, names: dict[int, str], n_rx: int, f_hz: float, dual_pol: bool = False,
            top_k: int | None = None, rough: dict | None = None) -> dict[str, np.ndarray]:
    """Top-K paths per receiver with the FULL narrowband coefficient h = a * exp(-j 2 pi f tau) (Sionna's `a`
    excludes the propagation phase; validated: with it, analytic steering matches per-element tracing).
    top_k: paths kept per cell (default TOP_K). rough: {"sigma_h_m", "sea_objects" (names), "tx"} applies the sea
    roughness factor (sea_roughness_factor; needs raw["vertices"]) before the top-K selection."""
    TOP_K = top_k or globals()["TOP_K"]  # noqa: N806
    ar, ai = raw["ar"], raw["ai"]
    n_p = ar.shape[-1]
    phi, theta, tau = (_rx_paths(raw[v], n_rx, n_p) for v in ("phi_t", "theta_t", "tau"))
    ph = np.exp(-1j * 2 * np.pi * f_hz * tau)
    a = (_rx_paths(ar, n_rx, n_p) + 1j * _rx_paths(ai, n_rx, n_p)) * ph
    if dual_pol:      # receive antenna 1 = H: the VH coefficient of the same paths
        a_vh = (_rx_paths(ar, n_rx, n_p, 1) + 1j * _rx_paths(ai, n_rx, n_p, 1)) * ph
        phi_r, theta_r = (_rx_paths(raw[v], n_rx, n_p) for v in ("phi_r", "theta_r"))
    depth = max(raw["interactions"].shape[0], 1)
    inter = raw["interactions"].reshape(depth, n_rx, -1, n_p)[:, :, 0, :]
    objs = raw["objects"].reshape(depth, n_rx, -1, n_p)[:, :, 0, :]
    if rough and rough.get("sigma_h_m", 0) > 0:
        verts = raw["vertices"].reshape(depth, n_rx, -1, n_p, 3)[:, :, 0]
        sea_ids = [i for i, nm in names.items() if nm in set(rough["sea_objects"])]
        fac = sea_roughness_factor(inter, objs, verts, rough["tx"], sea_ids, rough["sigma_h_m"], f_hz)
        a = a * fac
        if dual_pol:
            a_vh = a_vh * fac
    power = np.abs(a) ** 2
    k = min(TOP_K, n_p)
    order = np.argsort(-power, axis=1)[:, :k]
    tk = lambda x: np.take_along_axis(x, order, 1)  # noqa: E731
    ok = tk(power) > 0
    first = np.take_along_axis(inter[0], order, 1)
    ptype = np.where(ok, np.where(first == 0, 0, first), 255).astype("u1")
    fobj = np.take_along_axis(objs[0], order, 1)
    ids = np.array(sorted(names))
    pos = np.clip(np.searchsorted(ids, fobj), 0, max(len(ids) - 1, 0))
    known = (ids[pos] == fobj) if len(ids) else np.zeros(fobj.shape, bool)
    via = np.where(known & ok & (ptype != 0), pos, -1).astype("i4")
    total = power.sum(1)
    out = {
        "h_re": np.where(ok, tk(a).real, 0).astype("f4"), "h_im": np.where(ok, tk(a).imag, 0).astype("f4"),
        "kept_energy": np.where(total > 0, tk(power).sum(1) / np.maximum(total, 1e-40), 1.0).astype("f4"),
        "tau_ns": np.where(ok, tk(tau) * 1e9, np.nan).astype("f4"),
        "az_deg": np.where(ok, (90.0 - np.degrees(tk(phi))) % 360.0, np.nan).astype("f4"),
        "el_deg": np.where(ok, 90.0 - np.degrees(tk(theta)), np.nan).astype("f4"),
        "ptype": ptype, "via": via,
    }
    if dual_pol:
        out["h_vh_re"] = np.where(ok, tk(a_vh).real, 0).astype("f4")
        out["h_vh_im"] = np.where(ok, tk(a_vh).imag, 0).astype("f4")
        # drone-end direction toward the incoming wave (compass az, elevation); LOS: bearing drone -> node
        out["az_r_deg"] = np.where(ok, (90.0 - np.degrees(tk(phi_r))) % 360.0, np.nan).astype("f4")
        out["el_r_deg"] = np.where(ok, 90.0 - np.degrees(tk(theta_r)), np.nan).astype("f4")
    if k < TOP_K:
        fill = {"ptype": 255, "via": -1, "h_re": 0.0, "h_im": 0.0, "h_vh_re": 0.0, "h_vh_im": 0.0}
        for key, arr in out.items():
            if arr.ndim == 2:
                out[key] = np.concatenate([arr, np.full((n_rx, TOP_K - k), fill.get(key, np.nan), arr.dtype)], 1)
    out["_names"] = np.array([names[int(i)] for i in ids])
    return out


def extract(paths, names: dict[int, str], n_rx: int, f_hz: float, dual_pol: bool = False, top_k: int | None = None,
            rough: dict | None = None) -> dict[str, np.ndarray]:
    return _reduce(_pull(paths, dual_pol, vertices=bool(rough)), names, n_rx, f_hz, dual_pol, top_k, rough)


def steer(az_deg, el_deg, pos_m: np.ndarray, f_hz: float) -> np.ndarray:
    """Plane-wave response at antenna positions pos_m [N,3] (ENU) for arrivals from (az, el) [...]: [..., N].
    Direction toward the source u = (sin az cos el, cos az cos el, sin el); a_n = exp(+j k u.p_n)."""
    az, el = np.radians(az_deg)[..., None], np.radians(el_deg)[..., None]
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el) * np.ones_like(az)], -1)
    return np.exp(1j * 2 * np.pi * f_hz / C * (u @ pos_m.T)[..., 0, :] if u.ndim > 2 else
                  1j * 2 * np.pi * f_hz / C * (u @ pos_m.T))


def square_array(side_m: float) -> np.ndarray:
    h = side_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])     # channels 0..3 (0,1: chip A; 2,3: B)


def validate(sensor: dict, n: int = 32) -> dict:
    """Analytic plane-wave steering of the single-antenna paths (ALL paths, with propagation phase) vs Sionna's
    own 4-antenna output: synthetic_array=True and exact per-element tracing, for each propagation setup."""
    import mitsuba as mi
    import sionna.rt as rt
    from sionna.rt.antenna_pattern import antenna_pattern_registry

    f = BAND_HZ["ism_2400"]
    lam = C / f
    pos = square_array(0.061)
    scene, _ = load(f)
    tx = np.array([sensor["x_m"], sensor["y_m"], sensor["z_m"]])
    rng = np.random.default_rng(0)
    ang, r = rng.uniform(0, 2 * np.pi, n), rng.uniform(200, 1200, n)
    rx = np.column_stack([tx[0] + r * np.sin(ang), tx[1] + r * np.cos(ang), np.full(n, 40.0)])
    place(scene, tx, rx)
    res = {}
    for setup in SETUPS:
        p1 = solve(scene, setup)
        h1 = (np.array(p1.a[0]) + 1j * np.array(p1.a[1])).reshape(n, -1) * \
            np.exp(-1j * 2 * np.pi * f * np.array(p1.tau).reshape(n, -1))
        ph, th = np.array(p1.phi_t).reshape(n, -1), np.array(p1.theta_t).reshape(n, -1)
        u = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
        y_an = (h1[..., None] * np.exp(1j * 2 * np.pi / lam * (u @ pos.T))).sum(1)          # [rx, 4]
        for synthetic in (True, False):
            pat = antenna_pattern_registry.get("iso")(polarization="V")
            scene.tx_array = rt.AntennaArray(pat, mi.Point3f(*(pos[:, i] / lam for i in range(3))))
            p4 = solve(scene, setup, synthetic=synthetic)
            a4 = np.array(p4.a[0]) + 1j * np.array(p4.a[1])                                 # [rx,1,1,4,P]
            t4 = np.array(p4.tau)
            t4 = t4.reshape(a4.shape) if t4.ndim == 5 else np.broadcast_to(t4.reshape(n, 1, 1, 1, -1), a4.shape)
            y = (a4 * np.exp(-1j * 2 * np.pi * f * t4))[:, 0, 0].sum(-1)                     # [rx, 4]
            d = np.degrees(np.abs(np.angle(y[:, 1:] * np.conj(y[:, :1]) *
                                           np.conj(y_an[:, 1:] * np.conj(y_an[:, :1])))))
            res[f"{setup}/{'synthetic' if synthetic else 'per_element'}"] = {
                "median_dphase_err_deg": round(float(np.median(d)), 3),
                "p95_dphase_err_deg": round(float(np.percentile(d, 95)), 3)}
            scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    return res


@exit_if_busy
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="S4")
    ap.add_argument("--sensors", nargs="*", default=None, help="several sensors in one process (one scene load per band)")
    ap.add_argument("--scene-dir", default=None, help="living-world variant scene dir (envbuild.rf_variants)")
    ap.add_argument("--out-root", default=None, help="cache root instead of data/site_kandla/rf_cache")
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--step-m", type=float, default=20.0)
    ap.add_argument("--chunk", type=int, default=12000)
    ap.add_argument("--setups", nargs="*", default=list(SETUPS))
    ap.add_argument("--bands", nargs="*", default=list(BAND_HZ))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--heights", nargs="*", type=float, default=list(HEIGHTS_M))
    ap.add_argument("--dual-pol", action="store_true", help="V+H grid antenna: adds h_vh, az_r/el_r (RF phase 2)")
    ap.add_argument("--fix-diffraction", action="store_true",
                    help="apply envbuild.sionna_fix (textbook UTD/Luebbers; docs/RESULTS/rf_validation.md)")
    ap.add_argument("--sea-state", type=int, default=0,
                    help="sea state 0-4: Ament roughness on specular sea bounces (sim.rf.materials; 0 = flat sea)")
    ap.add_argument("--sea-objects", nargs="*", default=["water"], help="shape ids of the sea plane")
    ap.add_argument("--samples-per-src", type=int, default=None, help="override the setup's ray count")
    ap.add_argument("--max-depth", type=int, default=None, help="override the setup's max_depth")
    ap.add_argument("--top-k", type=int, default=None, help=f"paths kept per cell (default {TOP_K})")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lattice-shift", type=int, default=None,
                    help="trace on a seeded shifted ray lattice (shifted_lattice; RF convergence Task 4/5)")
    a = ap.parse_args()
    overrides = {k: v for k, v in (("samples_per_src", a.samples_per_src), ("max_depth", a.max_depth)) if v is not None}
    top_k = a.top_k or TOP_K
    all_sensors = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]}
    ids = a.sensors or [a.sensor]
    root = Path(a.out_root) if a.out_root else SITE / "rf_cache"
    scene_dir = Path(a.scene_dir) if a.scene_dir else None
    heights = tuple(float(h) for h in a.heights)
    patch = None
    if a.fix_diffraction:
        from envbuild import sionna_fix

        patch = sionna_fix.apply()
    with gpu_lock("sionna_precompute"), shifted_lattice(a.lattice_shift):
        if a.validate:
            out = root / f"air7311_{ids[0]}"
            out.mkdir(parents=True, exist_ok=True)
            r = validate(all_sensors[ids[0]])
            (out / "validation.json").write_text(json.dumps(r, indent=1))
            print(json.dumps(r, indent=1))
            return 0
        for band in a.bands:
            scene, names = load(BAND_HZ[band], scene_dir, band, dual_pol=a.dual_pol)
            for sid in ids:
                sensor = all_sensors[sid]
                out = root / f"air7311_{sid}"
                out.mkdir(parents=True, exist_ok=True)
                offs = np.arange(-a.radius_m, a.radius_m + 1e-6, a.step_m)
                xs, ys = sensor["x_m"] + offs, sensor["y_m"] + offs
                gx, gy = np.meshgrid(xs, ys)
                tx = np.array([sensor["x_m"], sensor["y_m"], sensor["z_m"]])
                mpath = out / "manifest.json"
                manifest = json.loads(mpath.read_text()) if mpath.exists() and scene_dir is not None else {
                    "sensor": sid, "pose": sensor, "radius_m": a.radius_m, "step_m": a.step_m,
                    "heights_m": list(heights), "top_k": top_k, "runs": [],
                    "note": "transmitter at the node (reciprocity); az/el = arrival at the node"}
                manifest["setups"] = {k: {**SETUPS[k], **overrides} for k in a.setups}
                manifest["top_k"] = top_k
                rough = None
                if a.sea_state > 0:
                    from sim.rf.materials import sea_state_sigma_h

                    rough = {"sigma_h_m": sea_state_sigma_h(a.sea_state), "sea_objects": list(a.sea_objects),
                             "tx": [sensor["x_m"], sensor["y_m"], sensor["z_m"]]}
                manifest["sea_roughness"] = None if rough is None else {
                    "sea_state": a.sea_state, "model": "ament (post-trace, envbuild.air7311_precompute)",
                    "sigma_h_m": rough["sigma_h_m"], "sea_objects": list(a.sea_objects)}
                manifest["heights_m"] = list(heights)
                manifest["dual_pol"] = bool(a.dual_pol)
                manifest["diffraction_patch"] = patch
                manifest["lattice_shift"] = None if a.lattice_shift is None else {
                    "seed": a.lattice_shift, "uv": [float(x) for x in lattice_shift_uv(a.lattice_shift)],
                    "model": "Cranley-Patterson rotation of the Fibonacci launch lattice"}
                if scene_dir is not None:
                    manifest["scene_dir"] = scene_dir.name
                for setup in a.setups:
                    t0 = time.time()
                    parts: dict[str, list] = {}
                    pending: list = []

                    def collect(fut) -> dict:
                        res = fut.result()
                        for k, v in res.items():
                            if not k.startswith("_"):
                                parts.setdefault(k, []).append(v)
                        return res

                    # the GPU traces chunk i+1 while one CPU thread reduces chunk i (order preserved; <= 2 in flight)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        for h in heights:
                            pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, h)])
                            for i in range(0, len(pts), a.chunk):
                                chunk = pts[i:i + a.chunk]
                                raw = _pull(solve_grid(scene, tx, chunk, setup, seed=a.seed, overrides=overrides),
                                            a.dual_pol, vertices=rough is not None)
                                pending.append(pool.submit(_reduce, raw, names, len(chunk), BAND_HZ[band], a.dual_pol,
                                                           top_k, rough))
                                if len(pending) > 1:
                                    e = collect(pending.pop(0))
                        while pending:
                            e = collect(pending.pop(0))
                    n = len(offs)
                    arrays = {k: np.concatenate(v).reshape(len(heights), n, n, *v[0].shape[1:])
                              for k, v in parts.items()}
                    (out / setup).mkdir(exist_ok=True)
                    np.savez_compressed(out / setup / f"{band}.npz", xs=xs, ys=ys, heights=np.array(heights),
                                        f_hz=BAND_HZ[band], names=e["_names"], **arrays)
                    dt = round(time.time() - t0, 1)
                    manifest["runs"].append({"band": band, "setup": setup, "seconds": dt,
                                             "cells": int(gx.size) * len(heights)})
                    print(f"[air7311] {scene_dir.name if scene_dir else 'nominal'} {sid} {band} {setup}: {dt} s",
                          flush=True)
                    mpath.write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
