"""S7.3 model-mismatch sweep with a positive control, at the DF / localisation level (no LLM; audit G Q47).

Only the BELIEF side is corrupted; the stored observations (v2 test scenarios, seeds 13-16 _X) and the truth stay as
built. For every (track, epoch) group with bearings from >= 2 sensors and a real emitter behind the track, the expert's
locate pipeline runs on the agent's tools (tools.df.fit_candidates rt_match against the belief Sionna cache via
predict_aoa's AoAPredictor, then the per-node direct bearings triangulated), and is scored against truth:
  fix error   distance (2-D) of the fix to the emitter's true position at the epoch
  ghost acc   per sensor with >= 2 bearings and a true direct (LOS) one: the bearing the pipeline calls direct is it

Perturbations (belief side):
  heading    the belief's array heading is off by +-d per sensor (fixed random signs): reported bearings are rotated
             before the tools see them. d = 1, 3, 10, 20 deg.
  phase_cal  the belief assumes a calibrated array but each element carries a phase error N(0, s) per (sensor, band):
             the az bias this causes in MUSIC (order 1, 20 dB, AIR7311 square at the band centre, DFEngine with
             nominal steering) is applied to the reported bearings. s = 1, 5, 15 deg. (Re-running DF on the IQ is not
             possible from stored observations, so the bias map of the noiseless MUSIC estimate is used.)
  height     every structure in the belief geometry is dh too tall / too short. A Sionna re-trace of the belief cache
             is not possible on CPU, so the effect is applied to predict_aoa's peaks: the 2.5-D line-of-sight margin
             from the sensor to the candidate (tools.df._los_detail over the site objects) is shifted by -dh; a direct
             peak whose margin goes negative is removed (believed blocked), and a blocked candidate whose margin
             becomes >= 0 gains a direct peak at the geometric bearing. Specular/diffracted azimuths are unchanged
             (to first order a height error moves them in elevation, which rt_match does not use).
             dh = -10, -5, -2, +2, +5, +10 m.
  control    positive control: heading +20 deg on EVERY sensor (a rotation of the whole picture). It must produce a
             significant degradation (CI excluding 0); if it did not, the sweep could not see anything.

Statistics: paired per group (same observations, nominal vs perturbed), two-level cluster bootstrap (scenario seed,
then group), 95 % percentile CI, on (a) the change in mean fix error (errors capped at 2 km), (b) the change in the
share of fixes within 100 m, (c) the change in ghost accuracy.

    python scripts/mismatch_sweep.py run --max-groups 250 --workers 8     # -> runs/s73/mismatch.json
    python scripts/mismatch_sweep.py report

Scenario sets come from the data version's root (SCENESCOUT_DATA_VERSION, envbuild.variant_roots.data_roots); the
default output is runs/s73/mismatch.json for v2 and runs/s73/mismatch_<version>.json otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import data_roots, data_version  # noqa: E402


def set_dir(seed: int) -> Path:
    """The v2 nominal observation set of a seed under the data version's scenario root."""
    return data_roots()["scenarios"] / f"kandla_s{seed}_X"

SEEDS = [13, 14, 15, 16]
CAP_M = 2000.0
NEAR_M = 100.0
MAX_HYPS = 15
F_BAND = {"sub_ghz_868_915": 0.9e9, "ism_2400": 2.44e9, "ism_5800": 5.8e9}
MATCHED = {"ism_5800": "A_5g8", "ism_2400": "B_2g4", "sub_ghz_868_915": "C_sub1g"}

CONDITIONS = [("nominal", 0.0)] + [("heading", d) for d in (1.0, 3.0, 10.0, 20.0)] \
    + [("phase_cal", s) for s in (1.0, 5.0, 15.0)] + [("height", h) for h in (-10.0, -5.0, -2.0, 2.0, 5.0, 10.0)] \
    + [("control_heading_all", 20.0)]


# --- perturbation models (pure functions, unit-tested) ---------------------------------------------------------
def heading_offsets(sensors: list[str], mag_deg: float, seed: int = 0) -> dict[str, float]:
    """Belief heading error per sensor: +-mag with a fixed random sign (same signs for every magnitude)."""
    rng = random.Random(seed)
    return {s: mag_deg * (1 if rng.random() < 0.5 else -1) for s in sorted(sensors)}


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def phase_bias_table(sigma_deg: float, side_m: float, f_hz: float, rng: np.random.Generator,
                     hw: dict | None = None, snr_db: float = 20.0) -> np.ndarray:
    """az bias (deg) of the order-1 MUSIC peak at each true azimuth 0..359 when the 4 elements carry phase errors
    N(0, sigma) that the steering vectors do not know about (noiseless covariance + white noise at snr_db)."""
    import torch

    from sim.df.air7311 import DFEngine, square_positions

    hw = hw or _hw()
    eng = DFEngine(hw, "cpu")
    pos = square_positions(side_m)
    A = eng.scan_steering(pos, f_hz)                                          # [n_el, n_az, 4]
    e = np.exp(1j * np.radians(rng.normal(0.0, sigma_deg, 4)))
    az_true = np.arange(360.0)
    from sim.rf.channel import steering

    a = steering(az_true, np.zeros(360), pos, f_hz) * e[None, :]              # [360, 4]
    R = np.einsum("bi,bj->bij", a, a.conj()) + 10 ** (-snr_db / 10) * np.eye(4)[None]
    Rt = torch.as_tensor(R, dtype=torch.complex64)
    ev, V = torch.linalg.eigh(Rt)
    En = V[:, :, :3]                                                          # order 1: 3 noise eigenvectors
    Af = A.reshape(-1, 4)
    proj = (Af.conj()[None] @ En).abs().pow(2).sum(-1).clamp_min(1e-12)       # [B, D]
    music = (1.0 / proj).reshape(360, A.shape[0], A.shape[1]).amax(1).numpy()
    est = eng.az[np.argmax(music, axis=1)]
    return _wrap(est - az_true)


def apply_bias(bearing: float, table: np.ndarray) -> float:
    """Reported bearing under a bias map indexed by the true azimuth (evaluated at the measured bearing)."""
    i = int(round(bearing)) % 360
    return float((bearing + table[i]) % 360.0)


def perturb_peaks(peaks: list[dict], margin_m: float | None, dh_m: float, geo_bearing: float) -> list[dict]:
    """Height-error proxy on predict_aoa peaks (see module doc). margin_m: 2.5-D LOS clearance (None = no
    structure on the line; m > 0 clear, < 0 blocked). dh_m > 0: the belief's structures are too tall."""
    if peaks is None or dh_m == 0 or margin_m is None:
        return peaks
    new = margin_m - dh_m
    has_direct = any(p["kind"] == "direct" for p in peaks)
    if has_direct and margin_m >= 0 > new:
        out = [p for p in peaks if p["kind"] != "direct"]
        if out:
            top = max(p["rel_power_db"] for p in out)
            out = [{**p, "rel_power_db": round(p["rel_power_db"] - top, 1)} for p in out]
        return out
    if not has_direct and margin_m < 0 <= new:
        return [{"bearing_deg": round(geo_bearing % 360.0, 1), "el_deg": 0.0, "rel_power_db": 0.0, "kind": "direct",
                 "via": None}] + [{**p} for p in peaks]
    return peaks


def _hw() -> dict:
    import yaml

    return yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())


# --- groups and the expert locate pipeline ---------------------------------------------------------------------
def load_groups(seed: int, max_groups: int, rng_seed: int = 0) -> list[dict]:
    """(track, epoch) groups with >= 2 sensors whose track is a real emitter, with truth position and hyp labels."""
    import duckdb

    from tasks.truth_access import TruthStore

    d = set_dir(seed)
    con = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    rows = con.execute("SELECT hyp_id, track_id, sensor_id, t_s, bearing_deg, snr_db, power_db FROM bearings "
                       "WHERE track_id IS NOT NULL").fetchall()
    con.close()
    ts = TruthStore(d)
    lab = ts.hyp_labels()
    trk = ts.track_of()
    by: dict = {}
    for h, tr, s, t, b, snr, p in rows:
        by.setdefault((tr, float(t)), []).append({"hyp": h, "sensor": s, "bearing": b, "snr": snr, "power": p,
                                                  "los": lab.get(h) == "LOS"})
    groups = []
    for (tr, t), hs in sorted(by.items()):
        em, kind = trk.get(tr, ("none", "spurious"))
        if em == "none" or kind == "spurious" or len({h["sensor"] for h in hs}) < 2:
            continue
        pos = ts.position(em, t)
        if pos is None:
            continue
        hs = sorted(hs, key=lambda h: -h["power"])[:MAX_HYPS]
        groups.append({"seed": seed, "track": tr, "t": t, "emitter": em, "kind": kind, "truth": pos, "hyps": hs})
    ts.close()
    random.Random(rng_seed + seed).shuffle(groups)
    return groups[:max_groups] if max_groups else groups


_W: dict = {}


def _worker_init():
    import torch

    torch.set_num_threads(1)


def _ctx(seed: int):
    if seed not in _W:
        from sim.belief.site_view import SiteView
        from tools.registry import ToolContext, load_all

        load_all()
        _W[seed] = ToolContext(SiteView(set_dir(seed)))
    return _W[seed]


def locate(ctx, g: dict, bearing_of, peaks_wrap=None) -> dict:
    """The expert's locate pipeline (tasks/expert_kandla.py, df families) on perturbed bearings / predictions."""
    import tools.df as TD
    from tools.errors import ToolError
    from tools.registry import REGISTRY

    orig_hyps = TD._hyps
    pred = TD.predictor(ctx)
    orig_peaks = pred.peaks
    hmap = {h["hyp"]: h for h in g["hyps"]}

    def hyps(ctx_, refs):
        out = orig_hyps(ctx_, refs)
        for h in out:
            h["bearing"] = bearing_of(h)
        return out
    TD._hyps = hyps
    if peaks_wrap is not None:
        pred.peaks = peaks_wrap(orig_peaks)
    try:
        spec = REGISTRY["fit_candidates"]
        try:
            fit = spec.fn(ctx, spec.args(hyp_refs=[f"hyp:{h}" for h in hmap], method="rt_match")).data
        except ToolError as e:
            return {"ok": False, "err": e.code if hasattr(e, "code") else str(e)}
        top = fit["candidates"][0]
        best: dict = {}
        for x in top["hyps"]:
            hid = x["ref"].split(":", 1)[1].rsplit("@", 1)[0]
            s = hmap[hid]["sensor"]
            if x["explained_as"] == "direct" and (s not in best or x["residual_deg"] < best[s][1]):
                best[s] = (hid, x["residual_deg"])
        pos = (top["position"]["x_m"], top["position"]["y_m"])
        if len(best) >= 2:
            try:
                tri = spec.fn(ctx, spec.args(hyp_refs=[f"hyp:{h}" for h, _ in best.values()],
                                             method="triangulate")).data
                c = tri["candidates"][0]["position"]
                pos = (c["x_m"], c["y_m"])
            except ToolError:
                pass
        return {"ok": True, "pos": pos, "direct": {s: h for s, (h, _) in best.items()}}
    finally:
        TD._hyps = orig_hyps
        pred.peaks = orig_peaks


def score(g: dict, r: dict) -> dict:
    if not r.get("ok"):
        return {"err": CAP_M, "near": 0.0, "ghost": _ghost(g, {})}
    e = math.hypot(r["pos"][0] - g["truth"][0], r["pos"][1] - g["truth"][1])
    return {"err": min(e, CAP_M), "near": float(e <= NEAR_M), "ghost": _ghost(g, r["direct"])}


def _ghost(g: dict, direct: dict) -> float | None:
    per: dict = {}
    for h in g["hyps"]:
        per.setdefault(h["sensor"], []).append(h)
    ok = []
    for s, hs in per.items():
        if len(hs) >= 2 and any(h["los"] for h in hs):
            ok.append(float(any(h["hyp"] == direct.get(s) and h["los"] for h in hs)))
    return float(np.mean(ok)) if ok else None


def run_group(job) -> dict:
    g, tables = job
    ctx = _ctx(g["seed"])
    import tools.df as TD
    from sim.geometry import bearing_deg

    sens = ctx.site.sensors
    band = TD._band(ctx, g["track"], g["t"])
    out = {"seed": g["seed"], "track": g["track"], "t": g["t"], "kind": g["kind"], "band": band, "res": {}}
    for name, mag in CONDITIONS:
        key = f"{name}:{mag:g}"
        wrap = None
        if name == "heading":
            off = heading_offsets(list(sens), mag)
            bf = (lambda off_: (lambda h: (h["bearing"] + off_[h["sensor"]]) % 360.0))(off)
        elif name == "control_heading_all":
            bf = (lambda m: (lambda h: (h["bearing"] + m) % 360.0))(mag)
        elif name == "phase_cal":
            tb = tables[f"{mag:g}"]
            bf = (lambda tb_: (lambda h: apply_bias(h["bearing"], np.asarray(tb_[f"{h['sensor']}|{band}"]))))(tb)
        else:
            bf = (lambda h: h["bearing"])
        if name == "height":
            memo: dict = {}

            def wrap(orig, dh=mag, memo=memo):
                def peaks(x, y, z, sensor, band_):
                    pk = orig(x, y, z, sensor, band_)
                    if not pk and pk != []:
                        return pk
                    k = (round(x, 1), round(y, 1), round(z, 1), sensor)
                    if k not in memo:
                        p = sens[sensor]["pose"]
                        memo[k] = TD._los_detail(ctx, (p["x_m"], p["y_m"], p["z_m"]), (x, y, z), set())[2]
                    geo = bearing_deg(sens[sensor]["pose"]["x_m"], sens[sensor]["pose"]["y_m"], x, y)
                    return perturb_peaks(pk, memo[k], dh, geo)
                return peaks
        out["res"][key] = score(g, locate(ctx, g, bf, wrap))
    return out


def phase_tables(sensors: list[str], seed: int = 1) -> dict:
    hw = _hw()
    out = {}
    for s_deg in (1.0, 5.0, 15.0):
        rng = np.random.default_rng(seed)
        t = {}
        for s in sorted(sensors):
            for band, f in F_BAND.items():
                t[f"{s}|{band}"] = phase_bias_table(s_deg, hw["antenna"]["candidate_sides_m"][MATCHED[band]], f, rng,
                                                    hw).round(3).tolist()
        out[f"{s_deg:g}"] = t
    return out


# --- statistics ---------------------------------------------------------------------------------------------------
def cluster_ci(clusters: dict, n: int = 2000, seed: int = 0) -> dict:
    from eval.run import cluster_bootstrap

    cl = {k: v for k, v in clusters.items() if v}
    if not cl:
        return {"mean": None, "ci95": [None, None], "n": 0, "n_clusters": 0}
    r = cluster_bootstrap(cl, n, seed)
    return {"mean": round(r["mean"], 4), "ci95": r["ci95"], "n": r["n"], "n_clusters": r["n_clusters"]}


def summarise(rows: list[dict]) -> dict:
    out = {}
    nom = "nominal:0"
    for name, mag in CONDITIONS:
        key = f"{name}:{mag:g}"
        errs = [r["res"][key]["err"] for r in rows]
        d_err, d_near, d_gh = {}, {}, {}
        for r in rows:
            a, b = r["res"][nom], r["res"][key]
            sc = str(r["seed"])
            d_err.setdefault(sc, []).append(b["err"] - a["err"])
            d_near.setdefault(sc, []).append(b["near"] - a["near"])
            if a["ghost"] is not None and b["ghost"] is not None:
                d_gh.setdefault(sc, []).append(b["ghost"] - a["ghost"])
        gh = [r["res"][key]["ghost"] for r in rows if r["res"][key]["ghost"] is not None]
        out[key] = {"n": len(rows), "fix_err_median_m": round(float(np.median(errs)), 1),
                    "fix_err_p90_m": round(float(np.percentile(errs, 90)), 1),
                    "within_100m": round(float(np.mean([r["res"][key]["near"] for r in rows])), 3),
                    "ghost_acc": round(float(np.mean(gh)), 3) if gh else None, "n_ghost": len(gh),
                    "d_mean_err_m": cluster_ci(d_err), "d_within_100m": cluster_ci(d_near),
                    "d_ghost_acc": cluster_ci(d_gh)}
        ci = out[key]["d_mean_err_m"]["ci95"]
        cg = out[key]["d_ghost_acc"]["ci95"]
        out[key]["significant_degradation"] = bool((ci[0] is not None and ci[0] > 0)
                                                   or (cg[0] is not None and cg[1] < 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--max-groups", type=int, default=250, help="per seed (0 = all)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    v = data_version()
    ap.add_argument("--out", default=str(REPO / "runs" / "s73" / ("mismatch.json" if v == "v2" else f"mismatch_{v}.json")))
    a = ap.parse_args()
    out = Path(a.out)
    if a.what == "run":
        groups = [g for s in SEEDS for g in load_groups(s, a.max_groups)]
        sens = sorted(_ctx(SEEDS[0]).site.sensors)
        tables = phase_tables(sens)
        rms = {k: round(float(np.sqrt(np.mean(np.square([v for t in tb.values() for v in t])))), 2)
               for k, tb in tables.items()}
        print(f"{len(groups)} groups; phase-error az bias rms per sigma: {rms}", flush=True)
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.workers, initializer=_worker_init) as ex:
            for i, r in enumerate(ex.map(run_group, [(g, tables) for g in groups], chunksize=2)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(groups)}  {time.time() - t0:.0f} s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "phase_bias_rms_deg": rms, "conditions": CONDITIONS}, indent=1))
    d = json.loads(out.read_text())
    summ = summarise(d["rows"])
    d["summary"] = summ
    out.write_text(json.dumps(d, indent=1))
    print(f"phase-error az bias rms: {d.get('phase_bias_rms_deg')}")
    print("condition              n   med m   p90 m  <100m  ghost   d_mean_err m [CI]         d_ghost [CI]          sig")
    for k, v in summ.items():
        e, g = v["d_mean_err_m"], v["d_ghost_acc"]
        print(f"{k:22s}{v['n']:4d} {v['fix_err_median_m']:7.1f} {v['fix_err_p90_m']:7.1f} {v['within_100m']:.3f} "
              f"{v['ghost_acc']}  {e['mean']:+8.1f} {e['ci95']}   {g['mean']} {g['ci95']}  {v['significant_degradation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
