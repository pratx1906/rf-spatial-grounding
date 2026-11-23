"""Monte-Carlo sanity check of the CRB against the project's own MUSIC code (CPU only).

Part A  covariance domain, ideal square array, AWGN, stochastic CN(0, rho) source, N snapshots, project
        DFEngine.covariance (forward-backward) + DFEngine.spectra (MDL order, MUSIC) + select_peaks, on a FINE
        scan grid (az 0.05 deg, el 1 deg) so the grid does not hide the bound. Shows RMSE -> CRB at high SNR.
Part B  the full per-burst path of the v3 build on a synthetic dwell (125 MS/s, 4 channels): band-limited burst on
        an ideal array + AWGN -> sim.rf.bursts.detect -> burst_cov -> observe_v2.burst_df (production grid az 0.5 deg,
        el 4 deg, steering at the burst centre, cal None, heading bias 0). Gives (i) the map from the reported burst
        snr_db to the true per-element in-band SNR, (ii) the implementation floor (grid) at high SNR.
Output: runs/s81/crb/mc_check.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from crb_lib import SIDES, crb_az_deg, square_positions, steer  # noqa: E402
from sim.df.air7311 import DFEngine  # noqa: E402
from sim.obs_model.observe_air7311 import HW_FILE, select_peaks  # noqa: E402
from sim.obs_model.observe_v2 import burst_df  # noqa: E402
from sim.rf import bursts as B  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
OUT = Path(__file__).resolve().parent / "mc_check.json"
hw0 = yaml.safe_load(HW_FILE.read_text())
hw0 = {**hw0, "processing": {**hw0["processing"], "model_error_floor_db": -20.0}}     # as observe_v2
FAST = os.environ.get("CRB_MC_FAST") == "1"


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def part_a():
    hw = {**hw0, "processing": {**hw0["processing"], "az_step_deg": 0.05, "el_grid_deg": [0, 80, 1]}}
    eng = DFEngine(hw, "cpu")
    rng = np.random.default_rng(81)
    res = []
    bands = {"sub_ghz_868_915": 895e6, "ism_2400": 2441.75e6, "ism_5800": 5800e6}
    T = 40 if FAST else 200
    for band, f in bands.items():
        pos = square_positions(SIDES[band])
        A = eng.scan_steering(pos, f)
        for N in (64,):
            for snr_db in (-10, -5, 0, 5, 10, 20, 30):
                rho = 10 ** (snr_db / 10)
                errs = []
                for t0 in range(0, T, 8):
                    nb = min(8, T - t0)
                    az_t = rng.uniform(0, 360, nb)
                    el_t = 10.0
                    Xs = []
                    for az in az_t:
                        a = steer(math.radians(az), math.radians(el_t), pos, f)
                        s = (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(rho / 2)
                        n = (rng.normal(size=(4, N)) + 1j * rng.normal(size=(4, N))) / math.sqrt(2)
                        Xs.append(a[:, None] * s[None] + n)
                    X = torch.as_tensor(np.stack(Xs), dtype=torch.complex64)
                    R = DFEngine.covariance(X, fb=True)
                    bart, music = eng.spectra(R, A, N)
                    pk, _ = select_peaks(eng.az, music.numpy(), bart.numpy(), hw["processing"]["peak_floor_db"], 3)
                    errs += [wrap(pk[i, 0] - az_t[i]) for i in range(nb)]
                e = np.asarray(errs)
                crb_d = crb_az_deg(0.0, 10.0, SIDES[band], f, rho, N)
                crb_s = crb_az_deg(0.0, 10.0, SIDES[band], f, rho, N, stochastic=True)
                rmse = float(np.sqrt(np.mean(e ** 2)))
                rmse_trim = float(np.sqrt(np.mean(np.minimum(np.abs(e), 30) ** 2)))
                row = dict(band=band, N=N, snr_db=snr_db, trials=len(e), rmse_deg=rmse, rmse_capped30_deg=rmse_trim,
                           median_abs_deg=float(np.median(np.abs(e))), outlier_frac_gt10deg=float(np.mean(np.abs(e) > 10)),
                           crb_det_deg=float(crb_d), crb_sto_deg=float(crb_s), ratio_rmse_crb_sto=rmse / float(crb_s))
                res.append(row)
                print("A", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    # az-independence and az/el decoupling of the bound (square array) + el dependence
    chk = {f"crb_rel_az{a}_el{e}": float(crb_az_deg(a, e, SIDES['ism_2400'], 2441.75e6, 1.0, 1000)
                                         / crb_az_deg(0, 0, SIDES['ism_2400'], 2441.75e6, 1.0, 1000))
           for a in (0, 17, 45) for e in (0, 10, 30, 60)}
    chk["el_known_vs_unknown_el30"] = float(crb_az_deg(17, 30, SIDES['ism_2400'], 2441.75e6, 1.0, 1000, el_known=True)
                                            / crb_az_deg(17, 30, SIDES['ism_2400'], 2441.75e6, 1.0, 1000))
    # closed form at el = 0: var = 1 / (2 N rho k^2 side^2)  (sum over the square of (x cos az - y sin az)^2 = side^2)
    k = 2 * math.pi * 2441.75e6 / 299792458.0
    chk["closed_form_el0_ratio"] = float(crb_az_deg(0, 0, SIDES['ism_2400'], 2441.75e6, 1.0, 1000)
                                         / math.degrees(1 / math.sqrt(2 * 1000 * k ** 2 * SIDES['ism_2400'] ** 2)))
    print("A checks", chk, flush=True)
    return res, chk


def part_b():
    eng = DFEngine(hw0, "cpu")
    fs = 125e6
    n = 1 << 19                                   # 4.2 ms dwell (production: 8 ms)
    band, f0 = "ism_2400", 2441.75e6
    pos = square_positions(SIDES[band])
    rng = np.random.default_rng(1811)
    off, bw, dur, t0 = 5e6, 2e6, 1.0e-3, 1.5e-3
    i0, i1 = int(t0 * fs), int((t0 + dur) * fs)
    fr = np.fft.fftfreq(i1 - i0, 1 / fs)
    keep = np.abs(fr - off) <= bw / 2
    T = 15 if FAST else 60
    res = []
    for snr_db in (0, 3, 6, 10, 15, 20, 30, 40):
        rho = 10 ** (snr_db / 10)
        rows = []
        for _ in range(T):
            az, el = rng.uniform(0, 360), rng.uniform(2, 20)
            S = np.zeros(i1 - i0, complex)
            S[keep] = rng.normal(size=keep.sum()) + 1j * rng.normal(size=keep.sum())
            s = np.fft.ifft(S)
            s *= math.sqrt(rho * bw / fs / np.mean(np.abs(s) ** 2))       # in-band PSD = rho x noise PSD
            a = steer(math.radians(az), math.radians(el), pos, f0 + off)
            x = (rng.normal(size=(4, n)) + 1j * rng.normal(size=(4, n))) / math.sqrt(2)
            x[:, i0:i1] += a[:, None] * s[None]
            xt = torch.as_tensor(x, dtype=torch.complex64)
            bs = [b for b in B.detect(xt, fs) if b.dur >= 150e-6]
            bs = [b for b in bs if b.t0 < t0 + dur and b.t0 + b.dur > t0 and b.f_lo < off + bw and b.f_hi > off - bw]
            if not bs:
                rows.append((np.nan,) * 6)
                continue
            b = max(bs, key=lambda q: q.dur * q.bw)
            R = B.burst_cov(xt, fs, b)
            pk = burst_df(eng, R, b, pos, f0, hw0, 0.0, cal=None)
            err = wrap(pk[0][0] - az) if pk else np.nan
            rows.append((err, b.snr_db, b.bw, b.dur, el, b.bw * b.dur))
        r = np.asarray(rows, float)
        ok = np.isfinite(r[:, 0])
        e = r[ok, 0]
        if not ok.any():
            res.append(dict(snr_true_db=snr_db, trials=0, detected_frac=0.0))
            print("B", res[-1], flush=True)
            continue
        n_eff_box = float(np.median(r[ok, 5]))
        crb_true = float(np.sqrt(np.mean(crb_az_deg(np.zeros(ok.sum()), r[ok, 4], SIDES[band], f0 + off, rho,
                                                   bw * dur, stochastic=True) ** 2)))
        row = dict(snr_true_db=snr_db, trials=int(ok.sum()), detected_frac=float(ok.mean()),
                   reported_snr_db_median=float(np.median(r[ok, 1])), reported_snr_db_p10=float(np.percentile(r[ok, 1], 10)),
                   reported_snr_db_p90=float(np.percentile(r[ok, 1], 90)),
                   box_bw_mhz_median=float(np.median(r[ok, 2]) / 1e6), box_dur_ms_median=float(np.median(r[ok, 3]) * 1e3),
                   n_true=bw * dur, n_box_median=n_eff_box,
                   rmse_deg=float(np.sqrt(np.mean(e ** 2))), median_abs_deg=float(np.median(np.abs(e))),
                   crb_sto_true_deg=crb_true,
                   grid_floor_deg=0.5 / math.sqrt(12),
                   crb_plus_grid_deg=math.sqrt(crb_true ** 2 + 0.5 ** 2 / 12))
        res.append(row)
        print("B", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return res


def noise_floor_ratio():
    """kappa = mean noise cell power / detector floor (20th percentile of 3x3-smoothed 4-channel STFT power)."""
    fs, n = 125e6, 1 << 19
    rng = np.random.default_rng(7)
    ks = []
    for _ in range(3):
        x = torch.as_tensor((rng.normal(size=(4, n)) + 1j * rng.normal(size=(4, n))) / math.sqrt(2), dtype=torch.complex64)
        win = torch.hann_window(1024, dtype=torch.float32)
        Sx = torch.stft(x - x.mean(1, keepdim=True), 1024, 512, window=win, return_complex=True, center=False)
        P = Sx.abs().pow(2).sum(0).numpy()
        Ps = sum(np.roll(np.roll(P, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9
        ks.append(float(Ps.mean() / np.percentile(Ps, 20)))
    return float(np.mean(ks))


if __name__ == "__main__":
    t = time.time()
    kappa = noise_floor_ratio()
    print("kappa (mean noise / floor) =", kappa, f"= {10 * math.log10(kappa):.3f} dB", flush=True)
    a, chk = part_a()
    b = part_b()
    OUT.write_text(json.dumps({"kappa": kappa, "kappa_db": 10 * math.log10(kappa), "part_a": a, "checks": chk,
                               "part_b": b, "seconds": time.time() - t}, indent=1))
    print("wrote", OUT, f"{time.time() - t:.0f} s")
