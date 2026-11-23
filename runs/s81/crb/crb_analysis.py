"""Per-burst DF bearing error of the v3 data against the Cramer-Rao bound (offline, world-side, CPU only).

Error definition: identical to runs/s77/d_q32_v3.py (the source of the "0.9 deg median per-burst bearing error" in
runs/s77/v3_report.md): bursts.bearing_deg (top MUSIC peak ranked by Bartlett power, + heading-datum bias) minus the
horizontal bearing from the sensor pose to the truth position of the emitter of the burst's track (track_labels;
emitter_points sample nearest the epoch, within 2 s); bursts without track, of tracks labelled none / clutter /
spurious, are skipped.
Bound per burst: crb_lib.crb_az_deg at the burst's carrier (bursts.f_mhz), the band's array side
(configs/hardware/air7311.yaml track_b_custom), the truth elevation, the per-element in-band SNR (bursts.snr_db mapped
to true SNR with the detector calibration measured in mc_check.py part B) and N = bw x dur independent snapshots
(bursts.bw_khz x bursts.dur_us; the covariance is the band-passed burst, sim.rf.bursts.burst_cov).
Truth fields read (offline only): truth/truth.duckdb track_labels (track_id, emitter_id, emitter_kind),
emitter_points (emitter_id, t_s, x_m, y_m, z_m); LOS/NLOS from data/site_kandla/rf_cache_v3 ptype (as d_q32_v3).
Heading-datum bias per (seed, sensor) recomputed from the build's deterministic rule (sim.obs_model.observe_v2.hw_errors).

    python runs/s81/crb/crb_analysis.py            # writes crb_summary.json, crb_bursts.csv.gz, crb_tables.md
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("SCENESCOUT_DATA_VERSION", "v3")
os.environ.setdefault("SCENESCOUT_RF_GEOM", "rfr")
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from crb_lib import SIDES, band_of, crb_az_deg  # noqa: E402
from sim.geometry import stable_seed  # noqa: E402

SC_ROOT = Path("data/scenarios_v3")
TEST = [15, 16] + list(range(51, 61))
REPRO = [13, 14, 15, 16] + list(range(51, 61))          # the seed pool of runs/s77/v3_report.md
RFC = Path("data/site_kandla/rf_cache_v3")
HW = yaml.safe_load(Path("configs/hardware/air7311.yaml").read_text())
assert HW["array_profile"] if "array_profile" in HW else True
BIAS_SD = HW["heading_datum"]["bias_deg_1sigma"]
assert not HW["heading_datum"].get("correction_deg"), "nominal config expected (no heading correction)"
assert HW["array_profiles"]["track_b_custom"]["sides_m"] == SIDES
AZ_STEP = HW["processing"]["az_step_deg"]
GRID_RMS = AZ_STEP / math.sqrt(12)
RNG = np.random.default_rng(2026_11_06)
NBOOT = 2000


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


_pc = {}


def los_status(sid, band, x, y, h):
    k = (sid, band)
    if k not in _pc:
        z = np.load(RFC / f"air7311_{sid}" / "spec_diff" / f"{band}.npz")
        _pc[k] = {n: z[n] for n in ("xs", "ys", "heights", "ptype")}
    z = _pc[k]
    xs, ys, hs = z["xs"], z["ys"], z["heights"]
    if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
        return "out"
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    p = z["ptype"][int(np.argmin(np.abs(hs - h))), iy, ix]
    return "LOS" if (p == 0).any() else ("NLOS" if (p != 255).any() else "nopath")


def snr_map():
    """reported burst snr_db -> true per-element in-band SNR (dB), from mc_check.py part B (median reported per
    true SNR, detected rows). Monotone interpolation; clamped at the table ends."""
    mc = json.loads((HERE / "mc_check.json").read_text())
    rows = [r for r in mc["part_b"] if r.get("trials")]
    rep = np.array([r["reported_snr_db_median"] for r in rows])
    tru = np.array([r["snr_true_db"] for r in rows], float)
    o = np.argsort(rep)
    rep, tru = rep[o], tru[o]
    assert np.all(np.diff(tru) > 0), "reported SNR not monotone in true SNR"
    return lambda s: np.interp(s, rep, tru), list(zip(rep.tolist(), tru.tolist()))


def load(seeds):
    out = []
    for s in seeds:
        d = SC_ROOT / f"kandla_s{s}_X"
        sens = {q["id"]: q["pose"] for q in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]
                if q["type"] == "fixed_df"}
        o = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
        tr = duckdb.connect(str(d / "truth" / "truth.duckdb"), read_only=True)
        lab = dict(tr.execute("select track_id, emitter_id from track_labels").fetchall())
        kind = dict(tr.execute("select track_id, emitter_kind from track_labels").fetchall())
        pts = defaultdict(list)
        for e, t, x, y, z in tr.execute("select emitter_id, t_s, x_m, y_m, z_m from emitter_points").fetchall():
            pts[e].append((t, x, y, z))
        for e in pts:
            pts[e].sort()
        tarr = {e: np.array([p[0] for p in v]) for e, v in pts.items()}

        def pos(e, t):
            if e not in pts:
                return None
            i = int(np.argmin(np.abs(tarr[e] - t)))
            return pts[e][i][1:] if abs(tarr[e][i] - t) <= 2.0 else None
        bias = {sid: float(np.random.default_rng(stable_seed(s, "heading", sid)).normal(0, BIAS_SD)) for sid in sens}
        rows = o.execute("select sensor_id, t_s, f_mhz, bw_khz, dur_us, bearing_deg, snr_db, track_id from bursts"
                         ).fetchall()
        for sid, t, f, bw, dur, brg, snr, tid in rows:
            if brg is None or tid is None or lab.get(tid) in (None, "none") or kind.get(tid) in ("clutter", "spurious"):
                continue
            p = pos(lab[tid], t)
            if p is None:
                continue
            sp = sens[sid]
            true_az = math.degrees(math.atan2(p[0] - sp["x_m"], p[1] - sp["y_m"])) % 360
            el = math.degrees(math.atan2(p[2] - sp["z_m"], math.hypot(p[0] - sp["x_m"], p[1] - sp["y_m"])))
            band = band_of(f)
            out.append(dict(seed=s, sensor=sid, t=t, f_mhz=f, band=band, bw_khz=bw, dur_us=dur, snr_db=snr,
                            kind=kind[tid], err=float(wrap(brg - true_az)), bias=bias[sid], el=el, true_az=true_az,
                            los=los_status(sid, band, *p)))
    return out


def boot_ci(rows):
    """95 % percentile bootstrap over scenarios (seeds resampled with replacement), all metrics per resample."""
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r)
    arr = {k: (np.array([abs(r["err"]) for r in v]), np.array([abs(r["err_db"]) for r in v]),
               np.array([r["crb"] for r in v])) for k, v in by.items()}
    ks = list(arr)
    vals = defaultdict(list)
    for _ in range(NBOOT):
        pick = RNG.choice(len(ks), len(ks), replace=True)
        ae = np.concatenate([arr[ks[i]][0] for i in pick])
        aed = np.concatenate([arr[ks[i]][1] for i in pick])
        c = np.concatenate([arr[ks[i]][2] for i in pick])
        vals["err_med_deg_ci95"].append(np.median(ae))
        vals["crb_med_deg_ci95"].append(np.median(c))
        vals["ratio_med_ci95"].append(np.median(ae / c))
        vals["ratio_debiased_med_ci95"].append(np.median(aed / c))
        vals["frac_within_2crb_ci95"].append(np.mean(ae <= 2 * c))
        vals["frac_within_5crb_ci95"].append(np.mean(ae <= 5 * c))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def rsig(e):                       # robust sigma (Gaussian-consistent MAD)
    e = np.asarray(e)
    return float(1.4826 * np.median(np.abs(e - np.median(e))))


def group_stats(rows, ci=True):
    e = np.array([r["err"] for r in rows])
    ed = np.array([r["err_db"] for r in rows])
    c = np.array([r["crb"] for r in rows])
    ae = np.abs(e)
    d = dict(n=len(rows),
             err_med_deg=float(np.median(ae)), err_p90_deg=float(np.percentile(ae, 90)),
             err_rms_deg=float(np.sqrt(np.mean(e ** 2))), err_rsigma_deg=rsig(e),
             err_debiased_med_deg=float(np.median(np.abs(ed))), err_debiased_rsigma_deg=rsig(ed),
             crb_med_deg=float(np.median(c)), crb_p90_deg=float(np.percentile(c, 90)),
             crb_rms_deg=float(np.sqrt(np.mean(c ** 2))),
             crb_det_med_deg=float(np.median([r["crb_det"] for r in rows])),
             ratio_med=float(np.median(ae / c)), ratio_p90=float(np.percentile(ae / c, 90)),
             ratio_rsigma_to_crbrms=rsig(e) / float(np.sqrt(np.mean(c ** 2))),
             ratio_debiased_med=float(np.median(np.abs(ed) / c)),
             frac_within_2crb=float(np.mean(ae <= 2 * c)), frac_within_5crb=float(np.mean(ae <= 5 * c)),
             frac_debiased_within_2crb=float(np.mean(np.abs(ed) <= 2 * c)),
             frac_debiased_within_5crb=float(np.mean(np.abs(ed) <= 5 * c)),
             frac_crb_gt_grid=float(np.mean(c > GRID_RMS)),
             snr_med_db=float(np.median([r["snr_db"] for r in rows])),
             snr_true_med_db=float(np.median([r["snr_true_db"] for r in rows])),
             nsnap_med=float(np.median([r["nsnap"] for r in rows])),
             el_med_deg=float(np.median([r["el"] for r in rows])))
    if ci and len({r["seed"] for r in rows}) >= 3:
        d.update(boot_ci(rows))
    return d


def budget(rows):
    """Robust error budget (variances add for independent terms): total robust sigma^2 = heading bias^2 + grid^2 +
    CRB^2 + residual (multipath / coupling / calibration / LO phase noise / association)."""
    e = np.array([r["err"] for r in rows])
    ed = np.array([r["err_db"] for r in rows])
    b = np.array([r["bias"] for r in rows])
    c = np.array([r["crb"] for r in rows])
    tot, deb = rsig(e), rsig(ed)
    crb2 = float(np.mean(c ** 2))
    resid2 = deb ** 2 - GRID_RMS ** 2 - crb2
    return dict(total_rsigma_deg=tot, debiased_rsigma_deg=deb, bias_rms_deg=float(np.sqrt(np.mean(b ** 2))),
                grid_rms_deg=GRID_RMS, crb_rms_deg=math.sqrt(crb2),
                residual_model_deg=math.sqrt(max(resid2, 0.0)),
                thermal_share_of_debiased_var=crb2 / deb ** 2 if deb > 0 else float("nan"))


def enrich(rows, fmap):
    s_true = fmap(np.array([r["snr_db"] for r in rows]))
    rho = 10 ** (s_true / 10)
    nsnap = np.array([max(r["bw_khz"] * 1e3 * r["dur_us"] * 1e-6, 1.0) for r in rows])
    side = np.array([SIDES[r["band"]] for r in rows])
    f = np.array([r["f_mhz"] * 1e6 for r in rows])
    az = np.array([r["true_az"] for r in rows])
    el = np.array([r["el"] for r in rows])
    crb_s = crb_az_deg(az, el, side, f, rho, nsnap, stochastic=True)
    crb_d = crb_az_deg(az, el, side, f, rho, nsnap)
    rho_naive = 10 ** (np.array([r["snr_db"] for r in rows]) / 10)
    crb_naive = crb_az_deg(az, el, side, f, rho_naive, nsnap, stochastic=True)
    for i, r in enumerate(rows):
        r.update(snr_true_db=float(s_true[i]), nsnap=float(nsnap[i]), crb=float(crb_s[i]), crb_det=float(crb_d[i]),
                 crb_naive=float(crb_naive[i]), err_db=float(wrap(r["err"] - r["bias"])))
    return rows


def snr_bin(s):
    return "lt10" if s < 10 else "10to20" if s < 20 else "20to30" if s < 30 else "ge30"


def main():
    fmap, table = snr_map()
    out = {"meta": dict(seeds_test=len(TEST), az_grid_step_deg=AZ_STEP, grid_rms_deg=GRID_RMS,
                        heading_bias_sd_deg=BIAS_SD, nboot=NBOOT,
                        side_sub_ghz_m=SIDES["sub_ghz_868_915"], side_2g4_m=SIDES["ism_2400"],
                        side_5g8_m=SIDES["ism_5800"])}
    md = []
    for name, seeds in (("test", TEST), ("repro", REPRO)):
        rows = enrich(load(seeds), fmap)
        if name == "test":
            import gzip
            import csv
            with gzip.open(HERE / "crb_bursts.csv.gz", "wt", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        groups = {"all": rows}
        for key, fn in (("band", lambda r: r["band"]), ("snr", lambda r: snr_bin(r["snr_db"])),
                        ("sensor", lambda r: r["sensor"]), ("los", lambda r: r["los"])):
            for r in rows:
                groups.setdefault(f"{key}_{fn(r)}", []).append(r)
        res = {}
        for g, rr in groups.items():
            if len(rr) >= 30:
                res[g] = group_stats(rr, ci=(name == "test"))
        res["budget"] = budget(rows)
        for b in SIDES:
            res[f"budget_band_{b}"] = budget([r for r in rows if r["band"] == b])
        # sensitivity: naive SNR mapping (reported snr_db taken as the true per-element SNR)
        res["sens_naive_snr_crb_med_deg"] = float(np.median([r["crb_naive"] for r in rows]))
        res["sens_naive_snr_ratio_med"] = float(np.median([abs(r["err"]) / r["crb_naive"] for r in rows]))
        # sensitivity: N halved / doubled (OBW vs detector box width) -> CRB x sqrt(2)^-+1
        res["sens_n_half_ratio_med"] = float(np.median([abs(r["err"]) / (r["crb"] * math.sqrt(2)) for r in rows]))
        res["sens_n_double_ratio_med"] = float(np.median([abs(r["err"]) / (r["crb"] / math.sqrt(2)) for r in rows]))
        out[name] = res
        if name == "test":
            md.append("| group | n | med SNR dB | med N | med abs err deg [CI] | p90 abs err | med CRB deg [CI] | "
                      "med err/CRB [CI] | med debiased err/CRB | within 2x / 5x CRB |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for g, d in res.items():
                if not isinstance(d, dict) or "n" not in d:
                    continue
                md.append(f"| {g} | {d['n']} | {d['snr_med_db']:.1f} | {d['nsnap_med']:.0f} | {d['err_med_deg']:.2f} "
                          f"[{d['err_med_deg_ci95'][0]:.2f}, {d['err_med_deg_ci95'][1]:.2f}] | {d['err_p90_deg']:.1f} | "
                          f"{d['crb_med_deg']:.3f} [{d['crb_med_deg_ci95'][0]:.3f}, {d['crb_med_deg_ci95'][1]:.3f}] | "
                          f"{d['ratio_med']:.1f} [{d['ratio_med_ci95'][0]:.1f}, {d['ratio_med_ci95'][1]:.1f}] | "
                          f"{d['ratio_debiased_med']:.1f} | {d['frac_within_2crb']:.3f} / {d['frac_within_5crb']:.3f} |"
                          if "err_med_deg_ci95" in d else f"| {g} | {d['n']} | ... |")
    out["snr_map_reported_to_true_db"] = {f"{a:.2f}": b for a, b in table}
    mc = json.loads((HERE / "mc_check.json").read_text())
    out["mc"] = {"kappa_db": mc["kappa_db"]}
    for r in mc["part_a"]:
        out["mc"][f"a/{r['band']}/snr{r['snr_db']}/ratio_rmse_crb"] = r["ratio_rmse_crb_sto"]
        out["mc"][f"a/{r['band']}/snr{r['snr_db']}/ratio_median_crb"] = r["median_abs_deg"] / (0.6745 * r["crb_sto_deg"])
    for r in mc["part_b"]:
        if r.get("trials"):
            out["mc"][f"b/snr{r['snr_true_db']}/rmse_deg"] = r["rmse_deg"]
            out["mc"][f"b/snr{r['snr_true_db']}/crb_plus_grid_deg"] = r["crb_plus_grid_deg"]
    if (HERE / "hw_floor.json").exists():                 # written by hw_floor.py
        out["hwfloor"] = json.loads((HERE / "hw_floor.json").read_text())
    flat = {}

    def fl(k, v):
        if isinstance(v, dict):
            for kk, vv in v.items():
                fl(f"{k}/{kk}" if k else kk, vv)
        else:
            flat[k] = v
    fl("", out)
    (HERE / "crb_summary.json").write_text(json.dumps(flat, indent=1, sort_keys=True))
    (HERE / "crb_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for k in ("test/budget", "test/budget_band_sub_ghz_868_915", "test/budget_band_ism_2400",
              "test/budget_band_ism_5800", "repro/all/err_med_deg", "repro/all/n", "test/sens_naive_snr_crb_med_deg",
              "test/sens_naive_snr_ratio_med", "test/sens_n_half_ratio_med", "test/sens_n_double_ratio_med"):
        print(k, {kk: v for kk, v in flat.items() if kk.startswith(k)})


if __name__ == "__main__":
    main()
