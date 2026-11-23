"""Numerical convergence of the v2 AIR7311 caches (demo-v2 S7.4, audit A Q6): rays x depth x kept paths.

    .venv-sionna-win\\Scripts\\python scripts\\convergence_sweep.py trace          (Windows Sionna, GPU lock, ~1 h)
    python scripts/convergence_sweep.py report                                     (numpy only; any venv)

Grid: v2 nominal scene (rf_cache_v2/scene, Blender geometry), 2.44 GHz, sensors S4 and S1, drone heights 5 and 40 m,
the production 20 m grid within 1500 m (or a sub-tile with --radius-m), spec_diff with the UTD fix (production setup).
Settings: samples_per_src {3e5, 1e6, 3e6} x max_depth {1, 2, 3}, every run reduced to the strongest 64 paths (TOP_K
24 is the first 24 of them: both are evaluated from one trace). Solver noise floor: the 1e6 / depth-2 baseline is
re-run with the same seed (run-to-run determinism) and with two randomly shifted ray lattices (seeds 2, 3).
Sionna 2.2 launches rays on a deterministic Fibonacci lattice (its `seed` only drives stochastic samplers), so a
different seed alone traces the same rays; the shifted runs apply a seeded Cranley-Patterson rotation (uniform
shift mod 1 of the lattice in the unit square), an equally valid ray set of the same density.
Reference = the densest run (3e6, depth 3, 64 paths). Metrics per run vs the reference, per cell with a path in
the reference: strongest-path type agreement, strongest-path azimuth agreement (|d az| <= 1 deg, > 5 deg), total
power error |10 log10(sum_K |h|^2 / sum_64,ref |h|^2)| (median, p95, > 1 dB), kept energy (share of the traced
energy in the K kept paths). For scale, the same metrics between the current caches of the nominal world N and
world variants (W0 control, W1t tide only, B2, W3) are added (report).
Outputs: runs/convergence/<sensor>_h<h>_<tag>.npz, runs/convergence/metrics.json, docs/RESULTS/rf_convergence.md
tables are written from metrics.json by `report`.
Data versions (SCENESCOUT_DATA_VERSION, envbuild.variant_roots.data_roots): the default scene, the production cache
and the variant caches come from the version's roots; the default output is runs/convergence (v2) or
runs/convergence_<version>.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SITE = REPO / "data" / "site_kandla"
from envbuild.variant_roots import cache_root, data_roots, data_version  # noqa: E402

OUT = REPO / "runs" / ("convergence" if data_version() == "v2" else f"convergence_{data_version()}")
BAND = "ism_2400"
F_HZ = 2.44e9
SAMPLES = (300_000, 1_000_000, 3_000_000)
DEPTHS = (1, 2, 3)
KS = (24, 64)
REF = "s3000k_d3"            # densest of the requested grid
REF_DENSE = "s10000k_d3"     # optional 1e7-ray check of the reference itself (used as reference when present)
BASE = "s1000k_d2"
VARIANT_SIDES = {"W0": "world", "W1t": "world", "B2": "belief", "W3": "world"}


def variant_caches() -> dict[str, Path]:
    """Production caches of the world variants compared for scale (the data version's roots)."""
    return {vid: cache_root(vid, side, "v2") for vid, side in VARIANT_SIDES.items()}


CEILING_MB = 22 * 1024


def runs() -> list[dict]:
    out = [{"tag": f"s{s // 1000}k_d{d}", "samples": s, "depth": d, "seed": 1, "shift": None}
           for s in SAMPLES for d in DEPTHS]
    out += [{"tag": f"{BASE}_rerun", "samples": 1_000_000, "depth": 2, "seed": 1, "shift": None},
            {"tag": f"{BASE}_shift2", "samples": 1_000_000, "depth": 2, "seed": 2, "shift": 2},
            {"tag": f"{BASE}_shift3", "samples": 1_000_000, "depth": 2, "seed": 3, "shift": 3}]
    out += [{"tag": f"s10000k_d{d}", "samples": 10_000_000, "depth": d, "seed": 1, "shift": None} for d in (2, 3)]
    out += [{"tag": "s10000k_d2_shift2", "samples": 10_000_000, "depth": 2, "seed": 2, "shift": 2},
            {"tag": "s3000k_d2_shift2", "samples": 3_000_000, "depth": 2, "seed": 2, "shift": 2}]
    return out


def _vram(stop: threading.Event, box: list) -> None:
    while not stop.is_set():
        with contextlib.suppress(Exception):
            mb = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5).stdout.split()[0])
            box[0] = max(box[0], mb)
            if mb > CEILING_MB:
                box.append(mb)
        stop.wait(1)


def trace(a) -> None:
    import gc

    import drjit as dr

    from envbuild import sionna_fix
    from envbuild.air7311_precompute import _pull, _reduce, load, shifted_lattice, solve_grid
    from scripts.gpu_lock import gpu_lock

    sionna_fix.apply()
    sensors = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]}
    OUT.mkdir(parents=True, exist_ok=True)
    stop, box = threading.Event(), [0]
    threading.Thread(target=_vram, args=(stop, box), daemon=True).start()
    log = OUT / "trace_log.jsonl"
    todo = [r for r in runs() if not a.only or r["tag"] in a.only]
    with gpu_lock("sionna_precompute"):
        scene, names = load(F_HZ, Path(a.scene_dir) if a.scene_dir else data_roots()["rf_cache"] / "scene", BAND)
        for sid in a.sensors:
            s = sensors[sid]
            tx = np.array([s["x_m"], s["y_m"], s["z_m"]])
            offs = np.arange(-a.radius_m, a.radius_m + 1e-6, a.step_m)
            gx, gy = np.meshgrid(s["x_m"] + offs, s["y_m"] + offs)
            for h in a.heights:
                pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, h)])
                for r in todo:
                    f = OUT / f"{a.prefix}{sid}_h{int(h)}_{r['tag']}.npz"
                    if f.exists() and not a.force:
                        continue
                    chunk = a.chunk if r["depth"] <= 2 and r["samples"] <= 1_000_000 else a.chunk_heavy
                    if r["samples"] > 3_000_000:
                        chunk = max(500, a.chunk_heavy // 3)
                    t0, parts, n_p = time.time(), {}, 0
                    with shifted_lattice(r["shift"]):
                        for i in range(0, len(pts), chunk):
                            c = pts[i:i + chunk]
                            p = solve_grid(scene, tx, c, "spec_diff", seed=r["seed"],
                                           overrides={"samples_per_src": r["samples"], "max_depth": r["depth"]})
                            raw = _pull(p)
                            del p
                            n_p = max(n_p, raw["ar"].shape[-1])
                            e = _reduce(raw, names, len(c), F_HZ, top_k=max(KS))
                            for k in ("h_re", "h_im", "az_deg", "el_deg", "ptype", "via", "kept_energy"):
                                parts.setdefault(k, []).append(e[k])
                    arr = {k: np.concatenate(v) for k, v in parts.items()}
                    tmp = f.with_suffix(".part.npz")
                    np.savez_compressed(tmp, xs=s["x_m"] + offs, ys=s["y_m"] + offs, height=h, **arr)
                    tmp.replace(f)
                    gc.collect()
                    dr.flush_malloc_cache()        # Dr.Jit keeps freed buffers cached: without this VRAM creeps up
                    rec = {"sensor": sid, "h": h, **r, "seconds": round(time.time() - t0, 1), "cells": len(pts),
                           "chunk": chunk, "max_paths_per_cell": int(n_p), "peak_vram_mb": box[0],
                           "over_ceiling": box[1:]}
                    with log.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    if len(box) > 1:
                        raise SystemExit(f"VRAM ceiling exceeded: {box[1:]}")
    stop.set()


# ---------------------------------------------------------------- metrics (numpy only)
def _first(d: dict) -> tuple[np.ndarray, np.ndarray]:
    return d["ptype"][:, 0], d["az_deg"][:, 0]


def compare(test: dict, ref: dict, k: int, k_ref: int = 64) -> dict:
    """Metrics of one run (first k paths) against a reference run (first k_ref paths), per cell."""
    pt, at = _first(test)
    pr, ar = _first(ref)
    valid = pr != 255
    pw = lambda d, kk: (d["h_re"][:, :kk].astype(float) ** 2 + d["h_im"][:, :kk].astype(float) ** 2).sum(1)  # noqa: E731
    p_t, p_r = pw(test, k), pw(ref, k_ref)
    ok = valid & (p_t > 0) & (p_r > 0)
    err = np.abs(10 * np.log10(np.where(ok, p_t, 1) / np.where(ok, p_r, 1)))[ok]
    daz = np.abs((at - ar + 180) % 360 - 180)
    daz = np.where(np.isnan(daz), 180.0, daz)[valid]
    # kept energy for k paths: share of the traced energy (from the stored top-64 share)
    tot = pw(test, 64) / np.maximum(test["kept_energy"].astype(float), 1e-12)
    kept = (pw(test, k) / np.maximum(tot, 1e-40))[pt != 255]
    # strongest reflected (non-LOS) path: azimuth agreement; NLOS power error
    def refl(d, kk):
        pt_, az_ = d["ptype"][:, :kk], d["az_deg"][:, :kk]
        m = (pt_ != 0) & (pt_ != 255)
        j = np.argmax(m, 1)
        has = m[np.arange(len(m)), j]
        p_n = np.where(m, d["h_re"][:, :kk].astype(float) ** 2 + d["h_im"][:, :kk].astype(float) ** 2, 0).sum(1)
        return has, az_[np.arange(len(m)), j], p_n
    hr, azr, pnr = refl(ref, k_ref)
    ht, azt, pnt = refl(test, k)
    dr_ = np.abs((azt - azr + 180) % 360 - 180)
    dr_ = np.where(ht & hr, dr_, 180.0)[hr]
    okn = hr & ht & (pnr > 0) & (pnt > 0)
    errn = np.abs(10 * np.log10(np.where(okn, pnt, 1) / np.where(okn, pnr, 1)))[okn]
    rec = _dominant_recall(test, ref, k, k_ref)
    return {"cells": int(valid.sum()),
            "refl_az_agree_1deg": round(float((dr_ <= 1.0).mean()), 4) if len(dr_) else None,
            "nlos_power_err_db_median": round(float(np.median(errn)), 3) if len(errn) else None,
            "nlos_power_err_db_p95": round(float(np.percentile(errn, 95)), 3) if len(errn) else None,
            "dominant_recall": rec,
            "type_agree": round(float((pt[valid] == pr[valid]).mean()), 4),
            "az_agree_1deg": round(float((daz <= 1.0).mean()), 4),
            "az_off_5deg": round(float((daz > 5.0).mean()), 4),
            "power_err_db_median": round(float(np.median(err)), 3),
            "power_err_db_p95": round(float(np.percentile(err, 95)), 3),
            "power_err_gt1db": round(float((err > 1.0).mean()), 4),
            "kept_energy_p1": round(float(np.percentile(kept, 1)), 4),
            "kept_energy_median": round(float(np.median(kept)), 4)}


def _dominant_recall(test: dict, ref: dict, k: int, k_ref: int, db: float = 10.0, chunk: int = 2000) -> float:
    """Share of the reference's dominant paths (within `db` of the cell's strongest) that the run reproduces: a
    path within 1 deg azimuth and 3 dB power among the run's first k paths (what a DF estimator sees)."""
    hit = tot = 0
    for i in range(0, len(ref["ptype"]), chunk):
        sl = slice(i, i + chunk)
        pr = 10 * np.log10(ref["h_re"][sl, :k_ref].astype(float) ** 2 + ref["h_im"][sl, :k_ref].astype(float) ** 2 + 1e-30)
        pt = 10 * np.log10(test["h_re"][sl, :k].astype(float) ** 2 + test["h_im"][sl, :k].astype(float) ** 2 + 1e-30)
        dom = (ref["ptype"][sl, :k_ref] != 255) & (pr >= pr[:, :1] - db)
        okt = test["ptype"][sl, :k] != 255
        daz = np.abs((test["az_deg"][sl, None, :k] - ref["az_deg"][sl, :k_ref, None] + 180) % 360 - 180)
        m = (daz <= 1.0) & (np.abs(pt[:, None, :] - pr[:, :, None]) <= 3.0) & okt[:, None, :]
        hit += int((m.any(2) & dom).sum())
        tot += int(dom.sum())
    return round(hit / max(tot, 1), 4)


def _cache_cells(root: Path, sid: str, h: float) -> dict | None:
    f = root / f"air7311_{sid}" / "spec_diff" / f"{BAND}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    i = int(np.argmin(np.abs(z["heights"] - h)))
    if abs(float(z["heights"][i]) - h) > 1e-6:
        return None
    return {k: z[k][i].reshape(-1, z[k].shape[-1]) for k in ("h_re", "h_im", "az_deg", "ptype", "kept_energy")
            if k != "kept_energy"} | {"kept_energy": z["kept_energy"][i].reshape(-1)}


def report(a) -> dict:
    res = {"settings": {}, "noise_floor": {}, "world_variants": {}}
    for sid in a.sensors:
        for h in a.heights:
            key = f"{sid}_h{int(h)}"
            files = {r["tag"]: OUT / f"{key}_{r['tag']}.npz" for r in runs()}
            have = {t: dict(np.load(f)) for t, f in files.items() if f.exists()}
            ref_tag = REF_DENSE if REF_DENSE in have else REF
            if ref_tag not in have:
                continue
            ref = have[ref_tag]
            res.setdefault("reference", {})[key] = ref_tag
            for t, d in have.items():
                if t.startswith(BASE + "_") or t.endswith("_shift2"):
                    continue
                for k in KS:
                    res["settings"].setdefault(f"{t}_K{k}", {})[key] = compare(d, ref, k)
            if BASE in have:
                for t in (f"{BASE}_rerun", f"{BASE}_shift2", f"{BASE}_shift3"):
                    if t in have:
                        res["noise_floor"].setdefault(t, {})[key] = compare(have[t], have[BASE], 24, 24)
                for base in ("s3000k_d2", "s10000k_d2"):        # lattice noise at higher ray counts
                    if base in have and f"{base}_shift2" in have:
                        res["noise_floor"].setdefault(f"{base}_shift2", {})[key] = compare(
                            have[f"{base}_shift2"], have[base], 24, 24)
                if f"{BASE}_shift2" in have and f"{BASE}_shift3" in have:
                    res["noise_floor"].setdefault("shift2_vs_shift3", {})[key] = compare(
                        have[f"{BASE}_shift3"], have[f"{BASE}_shift2"], 24, 24)
            # is a world-variant effect reproducible on another ray lattice? (W1t traced on lattices A and B)
            wa, wb = OUT / f"W1t_{key}_{BASE}.npz", OUT / f"W1t_{key}_{BASE}_shift2.npz"
            if wa.exists() and wb.exists() and f"{BASE}_shift2" in have:
                res.setdefault("gap_lattice", {})[key] = gap_lattice(
                    (have[BASE], dict(np.load(wa))), (have[f"{BASE}_shift2"], dict(np.load(wb))))
            # world-variant effect scale on the production caches (N vs variant), same sub-grid
            prod = data_roots()["rf_cache"]
            n = _cache_cells(prod, sid, h)
            if n is not None:
                sel = _subgrid_index(have[REF], prod, sid)
                for vid, root in variant_caches().items():
                    v = _cache_cells(root, sid, h)
                    if v is not None:
                        res["world_variants"].setdefault(f"N_vs_{vid}", {})[key] = compare(
                            {k: x[sel] for k, x in v.items()}, {k: x[sel] for k, x in n.items()}, 24, 24)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(_summary(res), indent=1))
    return res


def _change(n: dict, w: dict, k: int = 24, either: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per cell of n -> w: strongest path changed (type, or az > 1 deg; cells where n has a path, or with `either`
    where either has one: a path found on one side only is a change) and the total-power change in dB over the first
    k paths (nan where either has no power). S7.4's gap_lattice used either=False."""
    pn, pw_ = n["ptype"][:, 0], w["ptype"][:, 0]
    daz = np.abs((n["az_deg"][:, 0] - w["az_deg"][:, 0] + 180) % 360 - 180)
    ch = (pn != pw_) | (np.nan_to_num(daz, nan=180.0) > 1.0)
    p = lambda d: (d["h_re"][:, :k].astype(float) ** 2 + d["h_im"][:, :k].astype(float) ** 2).sum(1)  # noqa: E731
    ok = (p(n) > 0) & (p(w) > 0)
    has = ((pn != 255) | (pw_ != 255)) if either else (pn != 255)
    return ch & has, np.where(ok, 10 * np.log10(np.where(ok, p(w), 1) / np.where(ok, p(n), 1)), np.nan)


def gap_lattice(a: tuple, b: tuple) -> dict:
    """N -> W1t change per cell on lattice A and on lattice B: strongest path changed (type, or az > 1 deg) and the
    total-power change in dB; agreement of the two lattices' change maps (Jaccard, power-change correlation)."""
    ca, da = _change(*a)
    cb, db = _change(*b)
    ok = ~np.isnan(da) & ~np.isnan(db) & ((np.abs(da) > 0.1) | (np.abs(db) > 0.1))
    return {"changed_A": round(float(ca.mean()), 4), "changed_B": round(float(cb.mean()), 4),
            "jaccard": round(float((ca & cb).sum() / max((ca | cb).sum(), 1)), 4),
            "dpower_db_corr": round(float(np.corrcoef(da[ok], db[ok])[0, 1]), 4) if ok.sum() > 2 else None,
            "dpower_cells": int(ok.sum())}


def _subgrid_index(ref: dict, root: Path, sid: str) -> np.ndarray:
    """Flat indices of the sweep grid inside the production cache grid (same 20 m lattice)."""
    z = np.load(root / f"air7311_{sid}" / "spec_diff" / f"{BAND}.npz")
    xs, ys = z["xs"], z["ys"]
    ix = np.searchsorted(xs, ref["xs"] - 1e-3)
    iy = np.searchsorted(ys, ref["ys"] - 1e-3)
    gy, gx = np.meshgrid(iy, ix, indexing="ij")
    return (gy * len(xs) + gx).ravel()


def _summary(res: dict) -> dict:
    """Mean over the four sensor/height cases per run (for the table)."""
    out = {}
    for sec, d in res.items():
        if sec in ("reference", "gap_lattice"):
            continue
        for t, per in d.items():
            vals = list(per.values())
            out[f"{sec}/{t}"] = {k: (round(float(np.mean([v[k] for v in vals if v[k] is not None])), 4)
                                     if any(v[k] is not None for v in vals) else None) for k in vals[0]} | {"n": len(vals)}
    return out


# ================================================================ v3 grid (RF convergence plan Task 4, P1-1)
# Configurable grid: geometries x samples x depths, every setting traced on P >= 3 SHIFTED lattices (never the
# unshifted one, which the S7.4 reference shared: Review Focus 4). Gate metrics are shift vs shift (all pattern
# pairs), K 24 and 48 from one top-64 trace, and the N -> W3 world effect over the lattice noise (changed-cell rate),
# with W3 traced on the same settings and patterns.
GEOMS = ("full", "rf", "rfr")       # full = the demo v2 scenes (read only); rf / rfr = the v3 convergence scenes
PATTERN_SEED0 = 100                 # pattern i (1..P) = shifted_lattice(PATTERN_SEED0 + i)
GRID_KS = (24, 48)
GATE = {"type": ("type_agree", ">=", 0.97), "az": ("az_agree_1deg", ">=", 0.95),
        "power": ("power_err_db_p95", "<=", 3.0), "effect_ratio": ("ratio", ">=", 3.0)}
PROD_HOURS_18 = 1.9                 # S7.4 re-trace: 18 v2 caches at 1e6 / d2 / K 24, full mesh (rf_convergence.md S7)
VARIANT_SPEC = REPO / "configs" / "rf_variants.yaml"
MIN_FREE_GB = 20.0                  # Review Focus 5


def setting_tag(samples: int, depth: int) -> str:
    return f"s{int(samples) // 1000}k_d{int(depth)}"


def grid_runs(samples, depths, patterns: int) -> list[dict]:
    return [{"tag": f"{setting_tag(s, d)}_p{i}", "setting": setting_tag(s, d), "samples": int(s), "depth": int(d),
             "seed": 1, "shift": PATTERN_SEED0 + i, "pattern": i}
            for s in samples for d in depths for i in range(1, patterns + 1)]


def _side(vid: str) -> str:
    return yaml.safe_load(VARIANT_SPEC.read_text())["variants_v2"][vid]["side"]


def sweep_scene_dir(geom: str, vid: str) -> Path:
    """full: the demo v2 scene (read only); rf / rfr: the v3 convergence scene of that Blender RF source."""
    from envbuild.variant_roots import convergence_scene_dir, scene_dir

    side = _side(vid)
    if geom == "full":
        return scene_dir(vid, side, "v2", "v2")
    return convergence_scene_dir(vid, side, geom, "v3")


def grid_file(out: Path, geom: str, vid: str, sid: str, h: float, tag: str) -> Path:
    return out / geom / f"{vid}_{sid}_h{int(h)}_{tag}.npz"


def _free_gb(p: Path) -> float:
    import shutil

    return shutil.disk_usage(p).free / 2 ** 30


def _status(task: str | None, **kw) -> None:
    if not task:
        return
    with contextlib.suppress(Exception):
        from scripts.pipeline_status import update

        update(task, **kw)


def trace_grid(a) -> None:
    import gc

    import drjit as dr

    from envbuild import sionna_fix
    from envbuild.air7311_precompute import _pull, _reduce, load, shifted_lattice, solve_grid
    from scripts.gpu_lock import gpu_lock

    sionna_fix.apply()
    sensors = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]}
    OUT.mkdir(parents=True, exist_ok=True)
    if _free_gb(OUT) < MIN_FREE_GB:
        raise SystemExit(f"only {_free_gb(OUT):.1f} GB free (< {MIN_FREE_GB} GB): stop")
    stop, box, cur = threading.Event(), [0], [0]

    def mon():
        while not stop.is_set():
            with contextlib.suppress(Exception):
                mb = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5).stdout.split()[0])
                box[0], cur[0] = max(box[0], mb), max(cur[0], mb)
                if mb > CEILING_MB:
                    box.append(mb)
            stop.wait(1)

    threading.Thread(target=mon, daemon=True).start()
    log = OUT / "trace_log.jsonl"
    rs = grid_runs(a.samples, a.depths, a.patterns)
    jobs = [(g, v, sid, h, r) for g in a.geoms for v in a.variants for sid in a.sensors for h in a.heights for r in rs]
    todo = [j for j in jobs if a.force or not grid_file(OUT, j[0], j[1], j[2], j[3], j[4]["tag"]).exists()]
    print(json.dumps({"jobs": len(jobs), "todo": len(todo)}), flush=True)
    done, t_all = 0, time.time()
    with gpu_lock("sionna_precompute"):
        for geom in a.geoms:
            for vid in a.variants:
                mine = [j for j in todo if j[0] == geom and j[1] == vid]
                if not mine:
                    continue
                sd = sweep_scene_dir(geom, vid)
                man = json.loads((sd / "manifest.json").read_text())
                t_load = time.time()
                scene, names = load(F_HZ, sd, BAND)
                t_load = round(time.time() - t_load, 1)
                rough = None
                if int(man.get("sea_state", 0)) > 0:
                    from sim.rf.materials import sea_state_sigma_h

                    rough = {"sigma_h_m": sea_state_sigma_h(int(man["sea_state"])), "sea_objects": ["water"]}
                for _, _, sid, h, r in mine:
                    s = sensors[sid]
                    tx = np.array([s["x_m"], s["y_m"], s["z_m"]])
                    offs = np.arange(-a.radius_m, a.radius_m + 1e-6, a.step_m)
                    gx, gy = np.meshgrid(s["x_m"] + offs, s["y_m"] + offs)
                    pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, h)])
                    f = grid_file(OUT, geom, vid, sid, h, r["tag"])
                    f.parent.mkdir(parents=True, exist_ok=True)
                    chunk = a.chunk if r["depth"] <= 2 and r["samples"] <= 1_000_000 else a.chunk_heavy
                    if r["samples"] > 3_000_000:
                        chunk = max(500, a.chunk_heavy // 3)
                    rr = None if rough is None else {**rough, "tx": [float(v) for v in tx]}
                    t0, parts, n_p, cur[0] = time.time(), {}, 0, 0
                    with shifted_lattice(r["shift"]):
                        for i in range(0, len(pts), chunk):
                            c = pts[i:i + chunk]
                            p = solve_grid(scene, tx, c, "spec_diff", seed=r["seed"],
                                           overrides={"samples_per_src": r["samples"], "max_depth": r["depth"]})
                            raw = _pull(p, vertices=rr is not None)
                            del p
                            n_p = max(n_p, raw["ar"].shape[-1])
                            e = _reduce(raw, names, len(c), F_HZ, top_k=max(KS), rough=rr)
                            for k in ("h_re", "h_im", "az_deg", "el_deg", "ptype", "via", "kept_energy"):
                                parts.setdefault(k, []).append(e[k])
                    arr = {k: np.concatenate(v) for k, v in parts.items()}
                    tmp = f.with_suffix(".part.npz")
                    np.savez_compressed(tmp, xs=s["x_m"] + offs, ys=s["y_m"] + offs, height=h, **arr)
                    tmp.replace(f)
                    gc.collect()
                    dr.flush_malloc_cache()
                    done += 1
                    rec = {"geom": geom, "variant": vid, "sensor": sid, "h": h, **r,
                           "seconds": round(time.time() - t0, 1), "cells": len(pts), "chunk": chunk,
                           "scene_load_s": t_load, "sea_state": int(man.get("sea_state", 0)),
                           "max_paths_per_cell": int(n_p), "peak_vram_mb_trace": cur[0], "peak_vram_mb": box[0],
                           "over_ceiling": box[1:]}
                    with log.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    el = time.time() - t_all
                    _status(a.status_task, state="running", pct=int(5 + 75 * done / max(len(todo), 1)),
                            eta_min=int(el / done * (len(todo) - done) / 60) + 10,
                            note=f"sweep {done}/{len(todo)}: {geom} {vid} {sid} h{int(h)} {r['tag']}")
                    if len(box) > 1:
                        raise SystemExit(f"VRAM ceiling exceeded: {box[1:]}")
                    if _free_gb(OUT) < MIN_FREE_GB:
                        raise SystemExit(f"only {_free_gb(OUT):.1f} GB free: stop")
                del scene
                gc.collect()
                dr.flush_malloc_cache()
    stop.set()


def _mean(ms: list[dict]) -> dict:
    out = {}
    for k in ms[0]:
        vals = [m[k] for m in ms if m.get(k) is not None
                and not (isinstance(m[k], float) and np.isnan(m[k]))]
        out[k] = round(float(np.mean(vals)), 4) if vals and not isinstance(vals[0], str) else None
    return out


def shift_vs_shift(pats: list[dict], k: int) -> dict:
    """compare() over every pair of shifted patterns (j vs i, i < j), averaged: the lattice noise of one setting."""
    ms = [compare(pats[j], pats[i], k, k) for i in range(len(pats)) for j in range(i + 1, len(pats))]
    return _mean(ms) | {"pairs": len(ms)}


def effect_noise(n_pats: list[dict], w_pats: list[dict], k: int = 24) -> dict:
    """N -> W3 world effect vs lattice noise, in changed-cell rate (strongest path type or az > 1 deg).
    world_changed: mean over patterns p of the N_p -> W3_p change rate (same lattice); noise_changed: mean over
    pattern pairs of the N_p -> N_q change rate (same world, other lattice); ratio = world / noise. Reproducibility:
    Jaccard of the world change maps of two patterns and the correlation of their per-cell power changes."""
    wc = [_change(n, w, k, either=True) for n, w in zip(n_pats, w_pats)]
    nc = [_change(n_pats[i], n_pats[j], k, either=True)[0]
          for i in range(len(n_pats)) for j in range(i + 1, len(n_pats))]
    world = float(np.mean([c.mean() for c, _ in wc]))
    noise = float(np.mean([c.mean() for c in nc])) if nc else float("nan")
    jac, cor = [], []
    for i in range(len(wc)):
        for j in range(i + 1, len(wc)):
            (ca, da), (cb, db) = wc[i], wc[j]
            jac.append((ca & cb).sum() / max((ca | cb).sum(), 1))
            ok = ~np.isnan(da) & ~np.isnan(db) & ((np.abs(da) > 0.1) | (np.abs(db) > 0.1))
            if ok.sum() > 2:
                cor.append(np.corrcoef(da[ok], db[ok])[0, 1])
    dp = np.concatenate([d[~np.isnan(d)] for _, d in wc]) if wc else np.array([])
    ratio = (float("inf") if world > 0 else float("nan")) if noise == 0 else world / noise    # 0/0: undefined
    return {"world_changed": round(world, 4), "noise_changed": round(noise, 4),
            "ratio": ratio if np.isinf(ratio) else round(ratio, 3),
            "jaccard_mean": round(float(np.mean(jac)), 4) if jac else None,
            "dpower_corr_mean": round(float(np.nanmean(cor)), 4) if cor else None,
            "world_dpower_db_abs_median": round(float(np.median(np.abs(dp))), 3) if len(dp) else None}


def gate(row: dict) -> dict:
    fails = []
    for name, (key, op, thr) in GATE.items():
        v = row.get(key)
        if v is None or not (v >= thr if op == ">=" else v <= thr):
            fails.append(name)
    return {"pass": not fails, "fails": fails}


def cheapest_passing(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if gate(r)["pass"]]
    return min(ok, key=lambda r: (r["seconds"], r["K"])) if ok else None


def task5_hours(time_ratio: float, caches_ratio: float = 1.0, patterns: int = 1) -> float:
    """Task 5 GPU hours: the S7.4 production re-trace (PROD_HOURS_18 for 18 caches at 1e6 / d2 on the full mesh)
    scaled by the setting's sweep trace time over the full-mesh 1e6 / d2 sweep time, the cache count and the
    number of shifted patterns per cache (fallback)."""
    return PROD_HOURS_18 * time_ratio * caches_ratio * patterns


def report_grid(a) -> dict:
    times: dict = {}
    if (OUT / "trace_log.jsonl").exists():
        for line in (OUT / "trace_log.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "geom" in r:
                times.setdefault((r["geom"], r["variant"], r["setting"]), []).append(r)
    rows, per_case = [], {}
    settings = [(s, d) for s in a.samples for d in a.depths]
    for geom in a.geoms:
        for s, d in settings:
            st = setting_tag(s, d)
            tags = [f"{st}_p{i}" for i in range(1, a.patterns + 1)]
            for k in a.ks:
                cases, effs = {}, {}
                for sid in a.sensors:
                    for h in a.heights:
                        key = f"{sid}_h{int(h)}"
                        fn = [grid_file(OUT, geom, "N", sid, h, t) for t in tags]
                        if not all(f.exists() for f in fn):
                            continue
                        n = [dict(np.load(f)) for f in fn]
                        cases[key] = shift_vs_shift(n, k)
                        fw = [grid_file(OUT, geom, a.world, sid, h, t) for t in tags]
                        if all(f.exists() for f in fw):
                            effs[key] = effect_noise(n, [dict(np.load(f)) for f in fw], k)
                if not cases:
                    continue
                tr = times.get((geom, "N", st), [])
                m = _mean(list(cases.values()))
                e = _mean(list(effs.values())) if effs and len(effs) == len(cases) else {}
                # ratio of the case means (a per-case ratio is dominated by a near-zero-noise case)
                ratio = (e["world_changed"] / e["noise_changed"]) if e and e.get("noise_changed") else None
                row = {"geom": geom, "setting": st, "samples": s, "depth": d, "K": k, "cases": len(cases),
                       "seconds": round(float(np.mean([r["seconds"] for r in tr])), 1) if tr else None,
                       "peak_vram_mb": max((r["peak_vram_mb_trace"] for r in tr), default=None),
                       **m, "type_agree_min": min(c["type_agree"] for c in cases.values()),
                       "az_agree_min": min(c["az_agree_1deg"] for c in cases.values()),
                       "ratio": None if ratio is None else round(ratio, 3),
                       "ratio_min_case": min((x["ratio"] for x in effs.values() if not np.isnan(x["ratio"])),
                                             default=None),
                       **{f"eff_{kk}": v for kk, v in e.items() if kk != "ratio"}}
                row["gate"] = gate(row)
                rows.append(row)
                per_case[f"{geom}/{st}/K{k}"] = {"shift_vs_shift": cases, "effect": effs}
    base = next((r["seconds"] for r in rows if r["geom"] == "full" and r["setting"] == "s1000k_d2" and r["seconds"]),
                None)
    for r in rows:
        r["time_ratio"] = round(r["seconds"] / base, 2) if base and r["seconds"] else None
        r["task5_h"] = round(task5_hours(r["time_ratio"]), 1) if r["time_ratio"] else None
        r["task5_h_fallback3"] = round(task5_hours(r["time_ratio"], patterns=3), 1) if r["time_ratio"] else None
    best = cheapest_passing([r for r in rows if r["seconds"] is not None])
    res = {"rows": rows, "per_case": per_case, "best": best, "gate": GATE, "base_seconds_full_1e6_d2": base}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_grid.json").write_text(json.dumps(res, indent=1, default=str))
    for r in rows:
        print(f"{r['geom']:5s} {r['setting']:12s} K{r['K']:<3d} type {r['type_agree']:.3f} (min {r['type_agree_min']:.3f})"
              f" az {r['az_agree_1deg']:.3f} p95 {r['power_err_db_p95']:.2f} kept {r['kept_energy_p1']:.3f}"
              f" ratio {r['ratio']} t {r['seconds']} s ({r['time_ratio']}x) "
              f"{'PASS' if r['gate']['pass'] else 'fail ' + ','.join(r['gate']['fails'])}")
    print("best:", json.dumps(best, default=str))
    return res


def ensemble(pats: list[dict], k: int = 24) -> dict:
    """The Task 5 fallback estimate of one cell from P shifted lattices: the strongest path over the UNION of the
    patterns' paths (path discovery of P lattices), the mean total power over patterns (first k paths each) and the
    per-cell spread (std of the patterns' total power, dB) stored as the error bar."""
    pw = np.stack([(d["h_re"][:, :k].astype(float) ** 2 + d["h_im"][:, :k].astype(float) ** 2) for d in pats])
    first = pw[:, :, 0]                                    # each pattern's strongest path power [P, cells]
    best = np.argmax(first, 0)
    idx = np.arange(first.shape[1])
    tot = pw.sum(2)
    tdb = 10 * np.log10(np.maximum(tot, 1e-30))
    return {"ptype": np.stack([d["ptype"][:, 0] for d in pats])[best, idx],
            "az_deg": np.stack([d["az_deg"][:, 0] for d in pats])[best, idx],
            "power": tot.mean(0), "spread_db": np.where((tot > 0).all(0), tdb.std(0), np.nan)}


def ensemble_compare(a: dict, b: dict) -> dict:
    """Agreement of two independent P-pattern ensembles (cells where either has a path)."""
    has = (a["ptype"] != 255) | (b["ptype"] != 255)
    daz = np.abs((a["az_deg"] - b["az_deg"] + 180) % 360 - 180)
    daz = np.where(np.isnan(daz), 180.0, daz)[has]
    ok = has & (a["power"] > 0) & (b["power"] > 0)
    err = np.abs(10 * np.log10(np.where(ok, a["power"], 1) / np.where(ok, b["power"], 1)))[ok]
    return {"type_agree": round(float((a["ptype"][has] == b["ptype"][has]).mean()), 4),
            "az_agree_1deg": round(float((daz <= 1.0).mean()), 4),
            "power_err_db_median": round(float(np.median(err)), 3) if len(err) else None,
            "power_err_db_p95": round(float(np.percentile(err, 95)), 3) if len(err) else None}


def _ens_changed(a: dict, b: dict) -> float:
    has = (a["ptype"] != 255) | (b["ptype"] != 255)
    daz = np.nan_to_num(np.abs((a["az_deg"] - b["az_deg"] + 180) % 360 - 180), nan=180.0)
    return float((((a["ptype"] != b["ptype"]) | (daz > 1.0)) & has).mean())


def ensemble_split(n_pats: list[dict], w_pats: list[dict] | None, k: int = 24) -> dict:
    """Fallback check with 2P patterns: ensemble(first P) vs ensemble(last P) for the noise; with W3 traced on the
    same 2P patterns, the world effect ens(N, first P) -> ens(W3, first P) (and the last P) over that noise."""
    h = len(n_pats) // 2
    na, nb = ensemble(n_pats[:h], k), ensemble(n_pats[h:2 * h], k)
    out = {"patterns_per_ensemble": h, **ensemble_compare(na, nb),
           "spread_db_median": round(float(np.nanmedian(na["spread_db"])), 3),
           "spread_db_p95": round(float(np.nanpercentile(na["spread_db"], 95)), 3)}
    noise = _ens_changed(na, nb)
    out["noise_changed"] = round(noise, 4)
    if w_pats is not None and len(w_pats) >= 2 * h:
        wa, wb = ensemble(w_pats[:h], k), ensemble(w_pats[h:2 * h], k)
        world = (_ens_changed(na, wa) + _ens_changed(nb, wb)) / 2
        out["world_changed"] = round(world, 4)
        out["ratio"] = round(world / noise, 3) if noise > 0 else float("inf")
    return out


def report_ensemble(a) -> dict:
    """Task 5 fallback: for every (geom, setting) with 2P patterns traced, ensemble(P) vs ensemble(P)."""
    res = {}
    h = a.patterns
    for geom in a.geoms:
        for s in a.samples:
            for d in a.depths:
                st = setting_tag(s, d)
                tags = [f"{st}_p{i}" for i in range(1, 2 * h + 1)]
                cases = {}
                for sid in a.sensors:
                    for hh in a.heights:
                        fn = [grid_file(OUT, geom, "N", sid, hh, t) for t in tags]
                        if not all(f.exists() for f in fn):
                            continue
                        fw = [grid_file(OUT, geom, a.world, sid, hh, t) for t in tags]
                        w = [dict(np.load(f)) for f in fw] if all(f.exists() for f in fw) else None
                        cases[f"{sid}_h{int(hh)}"] = ensemble_split([dict(np.load(f)) for f in fn], w)
                if cases:
                    m = _mean(list(cases.values()))
                    if all("world_changed" in c for c in cases.values()) and m.get("noise_changed"):
                        m["ratio"] = round(m["world_changed"] / m["noise_changed"], 3)
                    m["type_agree_min"] = min(c["type_agree"] for c in cases.values())
                    res[f"{geom}/{st}/ens{h}"] = {"mean": m, "cases": cases}
                    print(f"{geom:5s} {st:12s} ens{h} vs ens{h}: " + json.dumps(m))
    (OUT / "metrics_ensemble.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ================================================================ corrected metrics (Task 4 audit, 2026-10-26)
# The S7.4 / first Task 4 metrics score every cell where the reference has any path (down to -170 dB) and call a
# near-tie between two paths a disagreement; their world-effect ratio divides a same-lattice (paired) whole-grid
# change rate by a different-lattice noise rate. The corrected metrics (default `report --grid`; `--legacy` keeps
# the old ones) score only detectable cells, accept a strongest path that matches any reference path within TIE_DB,
# and test whether the N -> W3 effect REPRODUCES across lattice patterns. docs/RESULTS/rf_convergence_v3.md.
HW_SPEC = REPO / "configs" / "hardware" / "air7311.yaml"
FLOOR_DB = -115.0          # scored cells: reference strongest-path gain >= this (link_floor_db() - 5 dB)
TIE_DB = 3.0               # strongest-path tolerance: any reference path within 3 dB of the strongest
EFFECT_DP_DB = 0.5         # a cell is affected by the world when its strongest path changes or |dP| > 0.5 dB
MIN_EFFECT_CELLS = 20      # fewer affected cells (union of two patterns) than this: no effect, reproducibility n/a
GATE_V2 = {"type": ("type_agree", ">=", 0.97), "az": ("az_agree_1deg", ">=", 0.95),
           "power": ("power_err_db_p95", "<=", 3.0), "effect_repro": ("eff_jaccard", ">=", 0.5),
           "effect_dp": ("eff_dp_diff_db_median", "<=", 1.0)}
EFFECT_KEYS = ("effect_repro", "effect_dp")    # gated only where the world has an effect (value not None)


def link_floor_db(band: str = BAND) -> float:
    """Path gain (dB) at which a drone burst reaches the AIR7311 detection SNR per sample, from the hardware model
    (configs/hardware/air7311.yaml): noise = -174 + 10 log10(signal BW) + NF; required Rx = noise + detect_snr_db;
    gain = Rx - EIRP + cable loss - antenna gain. 2.44 GHz: -174 + 73.0 + 3 = -98.0 dBm, + 6 dB = -92 dBm,
    - 20 dBm EIRP + 2 dB cable = -110 dB. FLOOR_DB (-115 dB) keeps 5 dB of margin for dwell integration (16 subbands
    x 16 snapshots) and EIRP spread: weaker cells are not detected, so their ray-lattice noise never reaches a DF
    product."""
    hw = yaml.safe_load(HW_SPEC.read_text())
    rx, em = hw["receiver"], hw["emitter"]
    noise = -174.0 + 10 * np.log10(em["signal_bw_mhz"][band] * 1e6) + rx["noise_figure_db"]
    need = noise + hw["processing"]["detect_snr_db"]
    return round(float(need - em["eirp_dbm"] + rx.get("cable_loss_db", 0.0) - hw["antenna"].get("gain_dbi", 0.0)), 2)


def _paths_db(d: dict, k: int) -> np.ndarray:
    p = d["h_re"][:, :k].astype(float) ** 2 + d["h_im"][:, :k].astype(float) ** 2
    return np.where(p > 0, 10 * np.log10(np.maximum(p, 1e-300)), -np.inf)


def _wrap(a, b):
    return np.nan_to_num(np.abs((a - b + 180) % 360 - 180), nan=180.0)


def _tot(d: dict, k: int) -> np.ndarray:
    return (d["h_re"][:, :k].astype(float) ** 2 + d["h_im"][:, :k].astype(float) ** 2).sum(1)


def compare_floor(test: dict, ref: dict, k: int = 24, floor_db: float = FLOOR_DB, tie_db: float = TIE_DB) -> dict:
    """Corrected per-cell agreement of a run with a reference (first k paths each), on cells whose reference strongest
    path gain >= floor_db. The run's strongest path agrees in type / azimuth when it matches the type / azimuth
    (<= 1 deg) of ANY reference path within tie_db of the reference's strongest; `match` needs both on one path.
    A run with no path in a scored cell disagrees. Power error: total power over k paths (cells where both have one).
    *_strict: the same cells without the tie tolerance."""
    pr, pt = _paths_db(ref, k), _paths_db(test, k)
    top = pr.max(1)
    cells = (ref["ptype"][:, 0] != 255) & (top >= floor_db)
    tie = (pr >= top[:, None] - tie_db) & (ref["ptype"][:, :k] != 255)
    j = np.argmax(pt, 1)
    idx = np.arange(len(j))
    has_t = np.isfinite(pt[idx, j]) & (test["ptype"][idx, j] != 255)
    t_type, t_az = test["ptype"][idx, j], test["az_deg"][idx, j]
    same_t = tie & (ref["ptype"][:, :k] == t_type[:, None])
    same_a = tie & (_wrap(ref["az_deg"][:, :k], t_az[:, None]) <= 1.0)
    type_ok, az_ok, match = has_t & same_t.any(1), has_t & same_a.any(1), has_t & (same_t & same_a).any(1)
    strict_t = has_t & (t_type == ref["ptype"][:, 0])
    strict_a = has_t & (_wrap(t_az, ref["az_deg"][:, 0]) <= 1.0)
    p_t, p_r = _tot(test, k), _tot(ref, k)
    ok = cells & (p_t > 0) & (p_r > 0)
    err = np.abs(10 * np.log10(np.where(ok, p_t, 1) / np.where(ok, p_r, 1)))[ok]
    n = int(cells.sum())

    def f(m):
        return round(float(m[cells].mean()), 4) if n else None

    return {"cells": n, "cells_share": round(n / max(len(cells), 1), 4),
            "type_agree": f(type_ok), "az_agree_1deg": f(az_ok), "match": f(match),
            "type_agree_strict": f(strict_t), "az_agree_strict": f(strict_a), "test_missing": f(~has_t),
            "power_err_db_median": round(float(np.median(err)), 3) if len(err) else None,
            "power_err_db_p95": round(float(np.percentile(err, 95)), 3) if len(err) else None}


def shift_vs_shift_floor(pats: list[dict], k: int, floor_db: float = FLOOR_DB, tie_db: float = TIE_DB) -> dict:
    """compare_floor() over every pair of shifted patterns (j vs i, i < j), averaged."""
    ms = [compare_floor(pats[j], pats[i], k, floor_db, tie_db)
          for i in range(len(pats)) for j in range(i + 1, len(pats))]
    return _mean(ms) | {"pairs": len(ms)}


def effect_repro(n_pats: list[dict], w_pats: list[dict], k: int = 24, floor_db: float = FLOOR_DB) -> dict:
    """Does the N -> world effect reproduce across lattice patterns? Per pattern p (N_p and W_p traced on the same
    lattice): change map c_p = strong cells whose strongest path changes (type, or az > 1 deg); affected a_p = c_p or
    a total-power change > EFFECT_DP_DB; dP_p = that power change (dB). Strong cells: mean over patterns of the N
    strongest-path gain >= floor_db (one mask for all patterns). Over pattern pairs: Jaccard of c_p and c_q (the
    structural effect; the power-thresholded map is not used for Jaccard: W3's power effect sits at ~0.5 dB, right at
    any threshold, so its map flickers even when dP itself reproduces), and on the union of a_p, a_q the median |dP|
    (effect size) against the median |dP_p - dP_q| (its lattice noise). None = fewer than MIN_EFFECT_CELLS changed /
    affected cells in every pair: no effect, reproducibility not applicable."""
    top = np.mean([np.max(np.where(np.isfinite(x := _paths_db(n, k)), 10 ** (x / 10), 0.0), 1) for n in n_pats], 0)
    strong = top >= 10 ** (floor_db / 10)
    aff_, dps, chs = [], [], []
    for n, w in zip(n_pats, w_pats):
        has = (n["ptype"][:, 0] != 255) | (w["ptype"][:, 0] != 255)
        ch = ((n["ptype"][:, 0] != w["ptype"][:, 0]) | (_wrap(n["az_deg"][:, 0], w["az_deg"][:, 0]) > 1.0)) & has
        pn, pw_ = _tot(n, k), _tot(w, k)
        dp = np.where((pn > 0) & (pw_ > 0), 10 * np.log10(np.maximum(pw_, 1e-300) / np.maximum(pn, 1e-300)), np.nan)
        aff_.append(strong & (ch | (np.nan_to_num(np.abs(dp), nan=0.0) > EFFECT_DP_DB)))
        dps.append(dp)
        chs.append(ch & strong)
    jac, dpw, dpd = [], [], []
    for i in range(len(aff_)):
        for j in range(i + 1, len(aff_)):
            uc = chs[i] | chs[j]                       # structural effect: the strongest path changes
            if uc.sum() >= MIN_EFFECT_CELLS:
                jac.append((chs[i] & chs[j]).sum() / uc.sum())
            u = aff_[i] | aff_[j]                      # power effect, on every affected cell
            if u.sum() < MIN_EFFECT_CELLS:
                continue
            a, b = dps[i][u], dps[j][u]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.any():
                dpw.append(np.median(np.abs(np.concatenate([a[ok], b[ok]]))))
                dpd.append(np.median(np.abs(a[ok] - b[ok])))
    ns = max(int(strong.sum()), 1)
    return {"strong_cells": int(strong.sum()),
            "world_changed": round(float(np.mean([c.sum() / ns for c in chs])), 4),
            "affected": round(float(np.mean([c.sum() / ns for c in aff_])), 4),
            "jaccard": round(float(np.mean(jac)), 4) if jac else None,
            "dp_world_db_median": round(float(np.mean(dpw)), 3) if dpw else None,
            "dp_diff_db_median": round(float(np.mean(dpd)), 3) if dpd else None}


def gate_v2(row: dict) -> dict:
    """Corrected gate, per sensor/height case: type, az, power on the scored cells; effect reproducibility only where
    the world has an effect (structural Jaccard >= 0.5; world power change reproduced to a median 1 dB; None = no
    such effect in this case, not a failure)."""
    fails = []
    for name, (key, op, thr) in GATE_V2.items():
        v = row.get(key)
        if name in EFFECT_KEYS and v is None:
            continue
        if v is None or not (v >= thr if op == ">=" else v <= thr):
            fails.append(name)
    return {"pass": not fails, "fails": fails}


def report_grid_v2(a) -> dict:
    """Corrected report (default for `report --grid`): per (geom, setting, K) and per sensor/height case the floored,
    tie-tolerant shift-vs-shift agreement, the N -> world effect reproducibility, the per-case gate, the legacy (S7.4)
    numbers of the same case, and the mean over cases. Writes <out>/metrics_grid_v2.json."""
    times: dict = {}
    if (OUT / "trace_log.jsonl").exists():
        for line in (OUT / "trace_log.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "geom" in r:
                times.setdefault((r["geom"], r["variant"], r["setting"]), []).append(r)
    rows = []
    for geom in a.geoms:
        for s in a.samples:
            for d in a.depths:
                st = setting_tag(s, d)
                tags = [f"{st}_p{i}" for i in range(1, a.patterns + 1)]
                for k in a.ks:
                    case_rows = []
                    for sid in a.sensors:
                        for h in a.heights:
                            fn = [grid_file(OUT, geom, "N", sid, h, t) for t in tags]
                            if not all(f.exists() for f in fn):
                                continue
                            n = [dict(np.load(f)) for f in fn]
                            fw = [grid_file(OUT, geom, a.world, sid, h, t) for t in tags]
                            w = [dict(np.load(f)) for f in fw] if all(f.exists() for f in fw) else None
                            old = shift_vs_shift(n, k)
                            row = {"geom": geom, "setting": st, "K": k, "case": f"{sid}_h{int(h)}",
                                   **shift_vs_shift_floor(n, k, a.floor_db, a.tie_db),
                                   "legacy_type_agree": old["type_agree"], "legacy_az_agree_1deg": old["az_agree_1deg"],
                                   "legacy_power_err_db_p95": old["power_err_db_p95"]}
                            if w is not None:
                                row |= {f"eff_{kk}": v for kk, v in effect_repro(n, w, k, a.floor_db).items()}
                                eo = effect_noise(n, w, k)
                                row |= {"legacy_world_changed": eo["world_changed"],
                                        "legacy_noise_changed": eo["noise_changed"], "legacy_ratio": eo["ratio"]}
                            row["gate"] = gate_v2(row)
                            case_rows.append(row)
                    if not case_rows:
                        continue
                    tr = times.get((geom, "N", st), [])
                    sec = round(float(np.mean([r["seconds"] for r in tr])), 1) if tr else None
                    vram = max((r["peak_vram_mb_trace"] for r in tr), default=None)
                    for r in case_rows:
                        r["seconds"], r["peak_vram_mb"] = sec, vram
                    mean = {"geom": geom, "setting": st, "K": k, "case": "mean", "seconds": sec, "peak_vram_mb": vram}
                    for kk in case_rows[0]:
                        if kk in mean or kk in ("gate",):
                            continue
                        v = [r.get(kk) for r in case_rows if isinstance(r.get(kk), (int, float))
                             and np.isfinite(r.get(kk))]
                        mean[kk] = round(float(np.mean(v)), 4) if v else None
                    lw, ln = mean.get("legacy_world_changed"), mean.get("legacy_noise_changed")
                    mean["legacy_ratio"] = round(lw / ln, 3) if lw is not None and ln else None   # ratio of means
                    mean["cases_pass"] = f"{sum(r['gate']['pass'] for r in case_rows)}/{len(case_rows)}"
                    rows += case_rows + [mean]
    res = {"floor_db": a.floor_db, "link_floor_db": link_floor_db(), "tie_db": a.tie_db, "gate": GATE_V2,
           "min_effect_cells": MIN_EFFECT_CELLS, "effect_dp_db": EFFECT_DP_DB, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics_grid_v2.json").write_text(json.dumps(res, indent=1, default=str))

    def fm(v, p=3):
        return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

    print(f"floor {a.floor_db} dB (link floor {link_floor_db()} dB), tie {a.tie_db} dB")
    print("geom  setting      K   case     cells type  az    match p95dB | eff: jacc dP    dPdif | "
          "legacy: type  az    p95   ratio | t s  gate")
    for r in rows:
        g = r.get("gate", {})
        gs = r.get("cases_pass", "") if r["case"] == "mean" else (
            "PASS" if g.get("pass") else "fail " + ",".join(g["fails"]))
        print(f"{r['geom']:5s} {r['setting']:12s} K{r['K']:<3d} {r['case']:8s} {fm(r.get('cells_share'), 2):5s} "
              f"{fm(r.get('type_agree'))} {fm(r.get('az_agree_1deg'))} {fm(r.get('match'))} "
              f"{fm(r.get('power_err_db_p95'), 2):5s} | {fm(r.get('eff_jaccard'), 2):4s} "
              f"{fm(r.get('eff_dp_world_db_median'), 2):5s} {fm(r.get('eff_dp_diff_db_median'), 2):5s} | "
              f"{fm(r.get('legacy_type_agree'))} {fm(r.get('legacy_az_agree_1deg'))} "
              f"{fm(r.get('legacy_power_err_db_p95'), 2):5s} {fm(r.get('legacy_ratio'))} | {fm(r.get('seconds'), 1)} {gs}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trace", "report"])
    ap.add_argument("--sensors", nargs="*", default=["S4", "S1"])
    ap.add_argument("--heights", nargs="*", type=float, default=[5.0, 40.0])
    ap.add_argument("--radius-m", type=float, default=1500.0)
    ap.add_argument("--step-m", type=float, default=20.0)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--chunk-heavy", type=int, default=6000)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None, help="output dir (default runs/convergence)")
    ap.add_argument("--scene-dir", default=None, help="trace another v2 scene (e.g. a world variant)")
    ap.add_argument("--prefix", default="", help="file prefix for runs on --scene-dir (e.g. W1t_)")
    g = ap.add_argument_group("v3 grid (Task 4): --grid replaces the S7.4 RUNS")
    g.add_argument("--grid", action="store_true")
    g.add_argument("--geoms", nargs="*", default=list(GEOMS), choices=GEOMS)
    g.add_argument("--samples", nargs="*", type=float, default=[1e6, 3e6, 1e7])
    g.add_argument("--depths", nargs="*", type=int, default=[2])
    g.add_argument("--patterns", type=int, default=3, help="shifted lattices per setting (>= 3)")
    g.add_argument("--variants", nargs="*", default=["N", "W3"], help="trace: scenes (N first)")
    g.add_argument("--world", default="W3", help="report: the world variant of the effect / noise ratio")
    g.add_argument("--ks", nargs="*", type=int, default=list(GRID_KS))
    g.add_argument("--status-task", default=None, help="scripts/pipeline_status.py task to update per trace")
    g.add_argument("--ensemble", action="store_true",
                   help="report: Task 5 fallback check, ensemble of --patterns P vs the next P (needs 2P traced)")
    g.add_argument("--legacy", action="store_true",
                   help="report: the S7.4 / first Task 4 metrics (all cells, strict strongest path, effect/noise ratio)")
    g.add_argument("--floor-db", type=float, default=FLOOR_DB,
                   help=f"report: score cells whose reference strongest-path gain >= this (default {FLOOR_DB} dB)")
    g.add_argument("--tie-db", type=float, default=TIE_DB, help=f"report: strongest-path tolerance (default {TIE_DB} dB)")
    a = ap.parse_args()
    a.samples = [int(x) for x in a.samples]
    global OUT
    OUT = Path(a.out) if a.out else OUT
    if a.grid:
        if a.patterns < 3:
            raise SystemExit("--patterns must be >= 3 (shift-vs-shift noise floor)")
        if a.cmd == "trace":
            trace_grid(a)
        elif a.ensemble:
            report_ensemble(a)
        elif a.legacy:
            report_grid(a)
        else:
            report_grid_v2(a)
    elif a.cmd == "trace":
        trace(a)
    else:
        report(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
