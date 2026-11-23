"""Sionna RT precompute for the Kandla RF scene (main spec 4, 4A; ENV spec 11). Runs on native Windows (D-008).

    .venv-sionna-win\\Scripts\\python -m envbuild.rf_precompute --smoke
    .venv-sionna-win\\Scripts\\python -m envbuild.rf_precompute [--bands ism_2400 ...] [--step-m 40]

Reciprocity: a transmitter is placed AT each DF sensor and receivers on a 3D grid of candidate drone positions.
The departure angles at the sensor equal the arrival angles the sensor would measure from a drone at that cell.
Per (sensor, band) the top-K paths per cell are cached: compass bearing, elevation, gain, delay, path type and the
first object hit as seen from the sensor (the object a ghost bearing points at).
Output: data/site_kandla/rf_cache/sionna_rt/<band>/aoa_<sensor>.npz + manifest.json (belief side, nominal model).
Only numpy/yaml/sionna here: this module must import cleanly in the Windows Sionna venv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.gpu_lock import exit_if_busy, gpu_lock  # noqa: E402

SITE = REPO / "data" / "site_kandla"
SCENE = SITE / "rf_scene" / "scene.xml"
CACHE = SITE / "rf_cache" / "sionna_rt"
BAND_HZ = {"sub_ghz_868_915": 900e6, "ism_2400": 2.44e9, "ism_5800": 5.8e9}
HEIGHTS_M = (5.0, 15.0, 40.0, 80.0, 120.0)   # low fliers / ground decoys (NLOS, ghost-only) up to transit height
TOP_K = 6
# path type codes stored in the cache
LOS, SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION = 0, 1, 2, 4, 8


def _np(t) -> np.ndarray:
    return np.array(t)


def load(frequency_hz: float):
    import sionna.rt as rt

    xml = SCENE.with_name("scene_subghz.xml") if frequency_hz < 1e9 else SCENE   # ITU table starts at 1 GHz
    scene = rt.load_scene(str(xml), merge_shapes=False)
    scene.frequency = frequency_hz
    arr = dict(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso",
               polarization="V")
    scene.tx_array = rt.PlanarArray(**arr)
    scene.rx_array = rt.PlanarArray(**arr)
    names = {}
    for name, obj in scene.objects.items():
        names[int(obj.object_id)] = name.removeprefix("mesh-")
    return scene, names


def trace(scene, tx_pos, rx_pos: np.ndarray, max_depth: int = 2, edge_diffraction: bool = True):
    import sionna.rt as rt

    for n in list(scene.transmitters):
        scene.remove(n)
    for n in list(scene.receivers):
        scene.remove(n)
    scene.add(rt.Transmitter(name="tx", position=[float(v) for v in tx_pos]))
    for i, p in enumerate(rx_pos):
        scene.add(rt.Receiver(name=f"rx{i}", position=[float(v) for v in p]))
    solver = rt.PathSolver()
    return solver(scene, max_depth=max_depth, los=True, specular_reflection=True, diffuse_reflection=False,
                  refraction=False, diffraction=True, edge_diffraction=edge_diffraction, synthetic_array=True,
                  seed=1)


def extract(paths, names: dict[int, str], top_k: int = TOP_K) -> dict[str, np.ndarray]:
    """Top-K paths per receiver: compass bearing / elevation at the transmitter (the sensor), gain, delay, type."""
    ar, ai = (_np(x) for x in paths.a)
    gain = ar ** 2 + ai ** 2                                   # [rx, (rx_ant), tx, (tx_ant), paths]
    while gain.ndim > 3:
        gain = gain.sum(axis=1) if gain.ndim == 5 else gain.sum(axis=2)
    gain = gain[:, 0, :]                                        # single transmitter
    # synthetic arrays give [rx, tx, paths]; full arrays add antenna axes. Reduce both to [rx, paths].
    phi = _np(paths.phi_t)
    theta = _np(paths.theta_t)
    tau = _np(paths.tau)
    phi = phi.reshape(gain.shape[0], -1, gain.shape[-1])[:, 0, :]
    theta = theta.reshape(gain.shape[0], -1, gain.shape[-1])[:, 0, :]
    tau = tau.reshape(gain.shape[0], -1, gain.shape[-1])[:, 0, :]
    inter = _np(paths.interactions)                            # [depth, rx, (..), tx, (..), paths]
    objs = _np(paths.objects)
    depth = inter.shape[0]
    inter = inter.reshape(depth, gain.shape[0], -1, gain.shape[-1])[:, :, 0, :]
    objs = objs.reshape(depth, gain.shape[0], -1, gain.shape[-1])[:, :, 0, :]
    valid = gain > 0
    n_rx, n_p = gain.shape
    k = min(top_k, n_p)
    order = np.argsort(-np.where(valid, gain, -1.0), axis=1)[:, :k]
    take = np.take_along_axis
    g = take(gain, order, 1)
    ok = g > 0
    out = {
        "az_deg": np.where(ok, (90.0 - np.degrees(take(phi, order, 1))) % 360.0, np.nan).astype("f4"),
        "el_deg": np.where(ok, 90.0 - np.degrees(take(theta, order, 1)), np.nan).astype("f4"),
        "gain_db": np.where(ok, 10 * np.log10(np.maximum(g, 1e-30)), np.nan).astype("f4"),
        "tau_ns": np.where(ok, take(tau, order, 1) * 1e9, np.nan).astype("f4"),
    }
    first_type = take(inter[0], order, 1)                      # interaction nearest the sensor (tx side)
    n_bounce = (take(inter.transpose(1, 2, 0).reshape(n_rx, n_p, depth).max(axis=2) > 0, order, 1)).astype("u1")
    first_obj = take(objs[0], order, 1)
    ptype = np.where(first_type == 0, LOS, first_type).astype("u1")
    out["ptype"] = np.where(ok, ptype, 255).astype("u1")
    out["n_bounce"] = n_bounce
    via = np.full(first_obj.shape, -1, "i4")
    ids = sorted(names)
    lut = {oid: i for i, oid in enumerate(ids)}
    for oid, i in lut.items():
        via[(first_obj == oid) & ok & (ptype != LOS)] = i
    out["via"] = via
    if k < top_k:                                   # pad so every chunk has exactly top_k columns
        fill = {"ptype": 255, "n_bounce": 0, "via": -1}
        for key, arr in out.items():
            pad = np.full((n_rx, top_k - k), fill.get(key, np.nan), dtype=arr.dtype)
            out[key] = np.concatenate([arr, pad], axis=1)
    out["_names"] = np.array([names[i] for i in ids])
    return out


def smoke() -> dict:
    """Scene loads; a clear-LOS link matches free-space path loss (ENV spec 11 smoke test)."""
    scene, names = load(2.44e9)
    sensors = yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
    s = next(x for x in sensors if x["type"] == "fixed_df")["pose"]
    tx = np.array([s["x_m"], s["y_m"], s["z_m"]])
    rx = tx + np.array([0.0, 0.0, 400.0])                      # straight up: guaranteed clear line of sight
    t0 = time.time()
    p = trace(scene, tx, rx[None, :])
    e = extract(p, names)
    d = 400.0
    fspl = 20 * math.log10(4 * math.pi * d * 2.44e9 / 3e8)
    los_gain = float(e["gain_db"][0][e["ptype"][0] == LOS][0])
    return {"objects": len(names), "trace_s": round(time.time() - t0, 2), "los_gain_db": round(los_gain, 2),
            "fspl_db": round(fspl, 2), "delta_db": round(-los_gain - fspl, 2),
            "ok": abs(-los_gain - fspl) < 1.0}


@exit_if_busy
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bands", nargs="*", default=list(BAND_HZ))
    ap.add_argument("--step-m", type=float, default=40.0)
    ap.add_argument("--half-extent-m", type=float, default=3200.0)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--sensors", nargs="*", default=None)
    a = ap.parse_args()
    with gpu_lock("sionna_precompute"):
        if a.smoke:
            r = smoke()
            print(json.dumps(r))
            return 0 if r["ok"] else 1
        sensors = [s for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
                   if s["type"] == "fixed_df" and (a.sensors is None or s["id"] in a.sensors)]
        xs = np.arange(-a.half_extent_m, a.half_extent_m + 1e-6, a.step_m)
        gx, gy = np.meshgrid(xs, xs)                          # [ny, nx]
        manifest = {"backend": "sionna_rt", "scene": str(SCENE.relative_to(REPO)), "grid_step_m": a.step_m,
                    "x_m": [float(xs[0]), float(xs[-1])], "heights_m": list(HEIGHTS_M), "top_k": TOP_K,
                    "max_depth": 2, "diffraction": True, "runs": []}
        for band in a.bands:
            scene, names = load(BAND_HZ[band])
            (CACHE / band).mkdir(parents=True, exist_ok=True)
            for s in sensors:
                pose = s["pose"]
                tx = np.array([pose["x_m"], pose["y_m"], pose["z_m"]])
                t0 = time.time()
                # fast grid solver + raised path cap (validated in air7311_precompute); full narrowband h kept
                from envbuild.air7311_precompute import extract as extract_h
                from envbuild.air7311_precompute import solve_grid
                parts = {k: [] for k in ("az_deg", "el_deg", "gain_db", "tau_ns", "ptype", "via", "h_re", "h_im",
                                         "kept_energy")}
                for h in HEIGHTS_M:
                    pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, h)])
                    for i in range(0, len(pts), a.chunk):
                        chunk = pts[i:i + a.chunk]
                        e = extract_h(solve_grid(scene, tx, chunk, "spec_diff"), names, len(chunk), BAND_HZ[band])
                        hp = e["h_re"].astype("f8") ** 2 + e["h_im"].astype("f8") ** 2
                        e["gain_db"] = np.where(e["ptype"] != 255, 10 * np.log10(np.maximum(hp, 1e-30)),
                                                np.nan).astype("f4")
                        for k in parts:
                            parts[k].append(e[k])
                ny = nx = len(xs)
                arrays = {k: np.concatenate(v).reshape(len(HEIGHTS_M), ny, nx, *v[0].shape[1:])
                          for k, v in parts.items()}
                np.savez_compressed(CACHE / band / f"aoa_{s['id']}.npz", xs=xs, ys=xs,
                                    heights=np.array(HEIGHTS_M), names=e["_names"], **arrays)
                dt = round(time.time() - t0, 1)
                manifest["runs"].append({"band": band, "sensor": s["id"], "seconds": dt, "cells": int(gx.size) *
                                         len(HEIGHTS_M)})
                print(f"[rf] {band} {s['id']}: {dt} s", flush=True)
                (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
