"""S7.3 detection characterisation of the v2 (S7.2) receive chain: Pd vs per-element SNR, azimuth and waveform; Pfa.

Chain (the components observe_v2 uses, world side, CPU):
    waveform (sim.rf.waveforms, the observe_v2 generators) -> resample to 125 MSPS at +10 MHz in the window
    (channel.resample_shift, ibw 100 MHz) -> single plane-wave path into the 4-element square
    (channel.propagate, 2.44 GHz, side 0.054 m) -> front end (sim.rf.frontend, no blockers) -> receiver
    (sim.rf.receiver.receive: thermal noise, residual cal, LO phase noise, IQ imbalance, DC, AGC, 14 bit)
    -> detector (sim.rf.bursts.detect, non-coherent channel sum) -> observe_v2's 150 us minimum duration.

SNR = per-element in-band SNR: signal power / (k T B F) with B the waveform's 99 % occupied bandwidth (one hop
channel). Pd is per truth burst: a burst counts as detected when a detection overlaps it in time and frequency.
Pfa: noise-only dwells (detections per 8 ms dwell, Clopper-Pearson 95 % bound) and dwells with a strong CW carrier
(detections that do not overlap the carrier +-1 MHz: IQ image, DC, spurs).

    python scripts/characterise_detection.py pd  --trials 2 --workers 8
    python scripts/characterise_detection.py pfa --dwells 2000 --workers 8
    python scripts/characterise_detection.py report          # figures (needs matplotlib: Windows venv)
Outputs: runs/s73/det_pd.json, runs/s73/det_pfa.json, docs/RESULTS/figures/s73_pd_*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FS = 125e6                       # observe_v2.FS
DWELL_S = 8e-3                   # observe_v2.DWELL_S
F0 = 2.44e9                      # window centre (ism_2400)
F_OFF = 10e6                     # carrier offset inside the window
MIN_DUR_S = 150e-6               # observe_v2 burst filter
SIDE_M = 0.054                   # B_2g4 square side
SNRS = list(range(-6, 21, 2))
AZS = [0.0, 30.0, 45.0, 100.0, 190.0, 260.0]

# name -> (generator kind, label used by the classifier truth)
WAVEFORMS = {
    "ocusync10": "ocusync_like",
    "wifi": "wifi",
    "lora125": "lora",
    "lora500": "lora",
    "gfsk_rc": "gfsk",
    "gfsk_sik": "gfsk",
    "fm_video": "fm_video",
}


def hw() -> dict:
    import yaml

    return yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())


def make_waveform(name: str, rng: np.random.Generator, dur_s: float):
    """One realisation with the observe_v2 parameters (hop waveforms on one channel so B is the channel width)."""
    from sim.rf import waveforms as W

    if name == "ocusync10":
        return W.ofdm_ocusync_like(10.0, dur_s, rng)
    if name == "wifi":
        return W.wifi_ofdm(dur_s, rng, packet_us=rng.uniform(200, 1200), period_us=rng.uniform(1500, 4000))
    if name == "lora125":                      # tank telemetry parameters (SF7), packet rate raised so the dwell
        return W.lora_elrs(125e3, 7, 150.0, [0.0], dur_s, rng, data_symbols=8)   # holds one (20.7 ms packet)
    if name == "lora500":                      # ELRS sub-GHz parameters (SF6, 200 Hz)
        return W.lora_elrs(500e3, 6, 200.0, [0.0], dur_s, rng, data_symbols=8)
    if name == "gfsk_rc":                      # RC FHSS uplink: 1 Msym/s, 0.5 ms bursts every 1.4 ms
        return W.gfsk_hop(1e6, 0.5e-3, 1.4e-3, [0.0], dur_s, rng)
    if name == "gfsk_sik":                     # SiK telemetry: 64 kbit/s, 2 ms bursts every 10 ms
        return W.gfsk_hop(64e3, 2e-3, 3e-3, [0.0], dur_s, rng)
    if name == "fm_video":
        return W.fm_fpv(dur_s, rng)
    raise KeyError(name)


_BW: dict[str, float] = {}


def occupied_bw(name: str) -> float:
    """99 % occupied bandwidth of the waveform (fixed per waveform, from a reference realisation)."""
    from sim.rf import waveforms as W

    if name not in _BW:
        w = make_waveform(name, np.random.default_rng(12345), 4 * DWELL_S)
        act = int((np.abs(w.iq) > 0).sum())
        _BW[name] = W.occupied_bw_hz(w.iq, w.fs, nfft=min(4096, 2 ** int(math.log2(max(act // 8, 64)))))
    return _BW[name]


def noise_w(bw_hz: float, nf_db: float) -> float:
    return 10 ** ((-174 + 10 * math.log10(bw_hz) + nf_db - 30) / 10)


def dwell(sig, h, tau_s, az, el, gen_seed: int, H: dict, fe: bool = True):
    """Antenna-port multipath signal -> front end -> receiver. sig: torch [n + pre] at FS (or None: noise only).
    Returns ADC samples [4, n]."""
    import torch

    from sim.df.air7311 import square_positions
    from sim.rf import channel as ch
    from sim.rf import frontend as FE
    from sim.rf import receiver as rx

    n = int(FS * DWELL_S)
    pos = square_positions(SIDE_M)
    if sig is None:
        y = torch.zeros((4, n), dtype=torch.complex64)
    else:
        y = ch.propagate(sig, FS, np.asarray(h, complex), np.asarray(tau_s, float), np.asarray(az, float),
                         np.asarray(el, float), pos, F0 + F_OFF, F0, 0.0, f_lo_hz=F0, n_pre=sig.shape[0] - n)
    if fe:
        y, _ = FE.front_end(y, H["front_end"], [])
    x, _ = rx.receive(y, FS, H, torch.Generator().manual_seed(gen_seed), interchip_deg=2.0)
    return x


def source(name: str, seed: int, snr_db: float, H: dict):
    """Waveform realisation at the dwell rate (pre-roll + dwell), scaled to the per-element in-band SNR.
    Returns (signal [n + pre], truth bursts [(t0, t1, f_lo, f_hi)] in dwell time / window frequency)."""
    from sim.rf import channel as ch

    n = int(FS * DWELL_S)
    pre = ch.pre_roll(n)
    rng = np.random.default_rng(seed)
    w = make_waveform(name, rng, (n + pre) / FS)
    s = ch.resample_shift(w.iq, w.fs, FS, F_OFF, n + pre, 0.0, ibw_hz=H["receiver"].get("ibw_mhz", 100.0) * 1e6)
    bw = occupied_bw(name)
    amp = math.sqrt(noise_w(bw, H["receiver"]["noise_figure_db"]) * 10 ** (snr_db / 10))
    t_pre = pre / FS
    truth, edge = [], []
    for t0, d, f_off, b in w.bursts:
        a, z = t0 - t_pre, t0 + d - t_pre
        box = (max(a, 0.0), min(z, DWELL_S), F_OFF + f_off - b / 2, F_OFF + f_off + b / 2)
        if z <= 0 or a >= DWELL_S:
            continue
        # bursts with < 0.2 ms inside the dwell are not scored for Pd, but detecting them is not a false alarm
        (edge if (z <= 0.2e-3 or a >= DWELL_S - 0.2e-3) else truth).append(box)
    return s * amp, truth, edge


def match(dets, truth, guard_hz: float = 0.5e6, edge=()):
    """Per truth burst: index of the first overlapping detection (time and frequency) or -1; and the detections
    overlapping no truth burst, scored or edge (false)."""
    hit, used = [], set()
    for t0, t1, f_lo, f_hi in truth:
        k = next((i for i, b in enumerate(dets) if b.t0 < t1 and b.t0 + b.dur > t0
                  and b.f_lo < f_hi + guard_hz and b.f_hi > f_lo - guard_hz), -1)
        hit.append(k)
        if k >= 0:
            used.add(k)
    false = [i for i, b in enumerate(dets) if i not in used and not any(
        b.t0 < t1 and b.t0 + b.dur > t0 and b.f_lo < f_hi + guard_hz and b.f_hi > f_lo - guard_hz
        for t0, t1, f_lo, f_hi in list(truth) + list(edge))]
    return hit, false


def detect(x):
    from sim.rf import bursts as B

    return [b for b in B.detect(x, FS) if b.dur >= MIN_DUR_S]


def _init():
    import torch

    torch.set_num_threads(1)


def pd_job(args) -> dict:
    """One dwell: waveform, SNR, azimuth, trial -> per-burst detection (+ classifier output, multipath option)."""
    name, snr, az, trial, classify, mp = args
    from sim.rf import classify as C

    H = hw()
    seed = int(np.frombuffer(name.encode(), np.uint8).sum()) * 1000 + trial   # same realisation at every SNR / az
    sig, truth, edge = source(name, seed, snr, H)
    if mp is None:
        h, tau, azs, els = [1.0 + 0j], [2e-6], [az], [0.0]
    else:
        h, tau, azs, els = mp
    x = dwell(sig, h, tau, azs, els, seed * 7 + int(snr * 10) + int(az), H)
    dets = detect(x)
    hit, false = match(dets, truth, edge=edge)
    out = {"name": name, "snr": snr, "az": az, "trial": trial, "n_truth": len(truth),
           "hits": int(sum(k >= 0 for k in hit)), "false": len(false), "n_det": len(dets)}
    if classify:
        lab = []
        for k in hit:
            if k < 0:
                lab.append("missed")
                continue
            p, conf = C.classify(C.features(x, FS, dets[k]))
            lab.append(p)
        out["labels"] = lab
    return out


def pfa_job(args) -> dict:
    kind, k, p_dbm = args
    import torch

    H = hw()
    n = int(FS * DWELL_S)
    if kind == "noise":
        x = dwell(None, None, None, None, None, 900_000 + k, H)
        dets = detect(x)
        return {"kind": kind, "k": k, "n_det": len(dets), "det": [[b.fc / 1e6, b.bw / 1e6, b.dur * 1e6, b.snr_db]
                                                                  for b in dets]}
    from sim.rf import channel as ch

    pre = ch.pre_roll(n)
    t = torch.arange(n + pre, dtype=torch.float64) / FS
    rng = np.random.default_rng(k)
    s = torch.polar(torch.ones_like(t), 2 * math.pi * F_OFF * t + rng.uniform(0, 2 * math.pi)).to(torch.complex64)
    s = s * math.sqrt(10 ** ((p_dbm - 30) / 10))
    x = dwell(s, [1.0 + 0j], [2e-6], [float(rng.uniform(0, 360))], [0.0], 800_000 + k, H)
    dets = detect(x)
    far = [b for b in dets if not (b.f_lo < F_OFF + 1e6 and b.f_hi > F_OFF - 1e6)]
    return {"kind": kind, "k": k, "p_dbm": p_dbm, "n_det": len(dets), "n_false": len(far),
            "false": [[b.fc / 1e6, b.bw / 1e6, b.dur * 1e6, b.snr_db] for b in far]}


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p), summed in logs (no scipy in the WSL venv)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(0, k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial CI by bisection on the binomial tails."""
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    hi = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p), alpha / 2)
    lo = 0.0 if k == 0 else solve(lambda p: _binom_cdf(k - 1, n, p), 1 - alpha / 2)
    return lo, hi


def snr_at(snrs, pd, level: float) -> float | None:
    """Lowest SNR at which the (monotone-ised) Pd curve crosses `level`, linear interpolation."""
    s = np.asarray(snrs, float)
    p = np.maximum.accumulate(np.asarray(pd, float))
    if p[-1] < level:
        return None
    i = int(np.argmax(p >= level))
    if i == 0:
        return float(s[0]) if p[0] >= level else None
    return float(s[i - 1] + (level - p[i - 1]) / max(p[i] - p[i - 1], 1e-9) * (s[i] - s[i - 1]))


def run_pool(fn, jobs, workers):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(fn, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    return out


def summarise_pd(rows: list[dict]) -> dict:
    out = {}
    for name in WAVEFORMS:
        rs = [r for r in rows if r["name"] == name]
        if not rs:
            continue
        snrs = sorted({r["snr"] for r in rs})
        azs = sorted({r["az"] for r in rs})

        def pd(sel):
            n = sum(r["n_truth"] for r in sel)
            return sum(r["hits"] for r in sel) / n if n else float("nan")
        curve = [pd([r for r in rs if r["snr"] == s]) for s in snrs]
        per_az = {str(a): [pd([r for r in rs if r["snr"] == s and r["az"] == a]) for s in snrs] for a in azs}
        s50 = {a: snr_at(snrs, c, 0.5) for a, c in per_az.items()}
        v = [x for x in s50.values() if x is not None]
        out[name] = {"bw_hz": occupied_bw(name), "snr_db": snrs, "pd": [round(c, 3) for c in curve],
                     "n_bursts_per_snr": sum(r["n_truth"] for r in rs if r["snr"] == snrs[0]),
                     "snr50_db": snr_at(snrs, curve, 0.5), "snr90_db": snr_at(snrs, curve, 0.9),
                     "snr50_per_az": s50, "az_spread_snr50_db": (max(v) - min(v)) if len(v) > 1 else None,
                     "false_per_dwell": round(sum(r["false"] for r in rs) / len(rs), 3),
                     "pd_per_az": {a: [round(x, 3) for x in c] for a, c in per_az.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pd", "pfa", "report"])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--dwells", type=int, default=1000)
    ap.add_argument("--carrier-dwells", type=int, default=24)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(REPO / "runs" / "s73"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "pd":
        names = a.only or list(WAVEFORMS)
        jobs = [(n, s, az, k, True, None) for n in names for s in SNRS for az in AZS for k in range(a.trials)]
        print(f"pd: {len(jobs)} dwells", flush=True)
        rows = run_pool(pd_job, jobs, a.workers)
        f = out / "det_pd.json"
        old = json.loads(f.read_text())["rows"] if f.exists() and a.only else []
        rows = [r for r in old if r["name"] not in names] + rows
        f.write_text(json.dumps({"rows": rows, "summary": summarise_pd(rows),
                                 "config": {"fs": FS, "dwell_s": DWELL_S, "f0": F0, "f_off": F_OFF, "side_m": SIDE_M,
                                            "azs": AZS, "snrs": SNRS, "trials": a.trials}}, indent=1))
        print(json.dumps({k: {q: v[q] for q in ("snr50_db", "snr90_db", "az_spread_snr50_db", "false_per_dwell")}
                          for k, v in summarise_pd(rows).items()}, indent=1))
    elif a.what == "pfa":
        jobs = [("noise", k, None) for k in range(a.dwells)]
        jobs += [("carrier", k, p) for p in (-80.0, -70.0, -60.0, -50.0, -40.0, -30.0) for k in range(a.carrier_dwells)]
        print(f"pfa: {len(jobs)} dwells", flush=True)
        rows = run_pool(pfa_job, jobs, a.workers)
        (out / "det_pfa.json").write_text(json.dumps({"rows": rows}))
        noise = [r for r in rows if r["kind"] == "noise"]
        k = sum(r["n_det"] > 0 for r in noise)
        n_det = sum(r["n_det"] for r in noise)
        lo, hi = clopper_pearson(k, len(noise))
        # STFT cells per dwell (the detector's test statistic): 1024 bins x frames
        frames = (int(FS * DWELL_S) - 1024) // 512 + 1
        summ = {"noise_dwells": len(noise), "dwells_with_detection": k, "detections": n_det,
                "pfa_dwell": k / len(noise), "pfa_dwell_ci95": [lo, hi],
                "pfa_cell_upper95": hi / (1024 * frames), "cells_per_dwell": 1024 * frames,
                "false_alarm_rate_per_s_upper95": hi / DWELL_S, "carrier": {}}
        for p in sorted({r["p_dbm"] for r in rows if r["kind"] == "carrier"}):
            rs = [r for r in rows if r["kind"] == "carrier" and r["p_dbm"] == p]
            kk = sum(r["n_false"] > 0 for r in rs)
            fr = [f for r in rs for f in r["false"]]
            summ["carrier"][str(p)] = {"dwells": len(rs), "dwells_with_false": kk,
                                       "false_per_dwell": sum(r["n_false"] for r in rs) / len(rs),
                                       "ci95": clopper_pearson(kk, len(rs)),
                                       "false_fc_mhz": sorted({round(f[0], 1) for f in fr})[:12],
                                       "carrier_snr_per_bin_db": p - (-174 + 10 * math.log10(FS / 1024) + 3.0)}
        (out / "det_pfa.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
        print(json.dumps(summ, indent=1))
    else:
        report(out)
    return 0


def report(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = REPO / "docs" / "RESULTS" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    d = json.loads((out / "det_pd.json").read_text())["summary"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, v in d.items():
        ax.plot(v["snr_db"], v["pd"], marker="o", ms=3, label=f"{name} ({v['bw_hz'] / 1e6:.2f} MHz)")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.axhline(0.9, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("per-element in-band SNR (dB)")
    ax.set_ylabel("Pd (per burst, 6 azimuths)")
    ax.set_title("v2 chain detection probability (S7.2 non-coherent detector)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "s73_pd_vs_snr.png", dpi=130)
    fig, axs = plt.subplots(1, len(d), figsize=(2.3 * len(d), 2.8), sharey=True)
    for ax, (name, v) in zip(np.atleast_1d(axs), d.items()):
        for a, c in v["pd_per_az"].items():
            ax.plot(v["snr_db"], c, lw=0.9, label=f"{float(a):.0f}")
        ax.set_title(name, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("SNR (dB)", fontsize=7)
    np.atleast_1d(axs)[0].set_ylabel("Pd")
    np.atleast_1d(axs)[0].legend(fontsize=6, title="az", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "s73_pd_per_az.png", dpi=130)
    print("figures:", figs / "s73_pd_vs_snr.png", figs / "s73_pd_per_az.png")


if __name__ == "__main__":
    sys.exit(main())
