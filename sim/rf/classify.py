"""Protocol classification of a detected burst from transparent signal features (RF phase 2, Task 6).

Features (computed on the band-passed, frequency-centred burst, channel 0):
  bw_hz        occupied width of the detection
  dur_s        burst duration
  papr_db      peak-to-average power ratio (constant envelope ~0-1 dB for FM/GFSK/LoRa; OFDM ~8-11 dB)
  cp_corr      cyclic-prefix autocorrelation at lag 1/15 kHz (OcuSync-like; at most CP/(N+CP) = 0.066 for normal
               CP, against ~0.002 at other lags) and 3.2 us (Wi-Fi, at most 0.2)
  stf_corr     autocorrelation at 0.8 us (Wi-Fi L-STF)
  chirp_peak   dechirp peak-to-mean with the down-chirp over that with the up-chirp, over the standard LoRa
               bandwidths (125/250/500/812.5 kHz) and SF6-8
  ce_frac      constant-modulus fraction from the 2nd/4th envelope moments (M2M4):
                   ce = sqrt(max(2 M2^2 - M4, 0)) / M2,  M2 = E|x|^2, M4 = E|x|^4
               For a constant-envelope signal of power S in complex Gaussian noise N, 2 M2^2 - M4 = S^2, so
               ce = S / (S + N) (the in-box SNR fraction); for Gaussian-like OFDM (and noise) M4 = 2 M2^2 and
               ce ~ 0 at any SNR. Measured (S7.2b): OFDM <= 0.27 (narrow noise-peak fragments of OFDM
               bursts at <= 2 dB up to 0.57), GFSK/LoRa/FM >= 0.6 from 4 dB in-band SNR; threshold CE_MIN 0.6.
               Unlike the PAPR it does not rise with noise in a wide decimation band, so it keeps narrowband GFSK
               (SiK, 64 kbit/s) and low-SNR GFSK out of the OFDM branch (S7.3: 37-39 % unknown).
  line_ratio   analogue video line structure: the lag-1 product d[n] = x[n+1] conj(x[n]) carries the instantaneous
               frequency; composite video repeats every 64 us line (sync + picture), so the AC power of d sits on
               the 15.625 kHz harmonic comb. line_ratio = (share of d's AC power within one FFT bin of k x 15.625
               kHz) / (share of bins there), ~1 for anything aperiodic at 64 us. Measured: FM video 5-60 from 0 dB
               (including the separate sync-tip sub-burst), every other class <= 1.5. Bursts >= 1 ms only.
               This replaces the bandwidth-only FM rule, whose 3-5 MHz gap took FM video (in-box occupied width
               3.9-4.4 MHz: the detection box keeps only the dense part of the deviation) to unknown 80-88 %.
Rules map features to {ocusync_like, wifi, lora, fm_video, gfsk, unknown} with a confidence; no learned model, so
every decision can be read from the features. Low SNR degrades the features and the confusion is reported.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sim.rf.bursts import Burst, burst_samples


_DEV = {"device": "cpu"}


def _acorr(x, lag: int) -> float:
    """Normalised autocorrelation magnitude at `lag`, on the GPU when the dwell is on the GPU."""
    if lag <= 0 or lag >= len(x) - 8:
        return 0.0
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=_DEV["device"])
    a, b = t[:-lag], t[lag:]
    num = torch.abs(torch.sum(a.conj() * b))
    den = torch.sqrt(torch.sum(a.abs() ** 2) * torch.sum(b.abs() ** 2)).clamp_min(1e-30)
    return float(num / den)


def features(x: torch.Tensor, fs: float, b: Burst) -> dict:
    _DEV["device"] = x.device
    seg = burst_samples(x[:1], fs, b)[0].cpu().numpy()
    t = np.arange(len(seg)) / fs
    # 99 % occupied bandwidth inside the detection box (the box itself includes spectral skirts at high SNR)
    nf = 1024
    m = (len(seg) // nf) * nf
    fc, obw = b.fc, b.bw
    if m >= nf:
        psd = np.fft.fftshift((np.abs(np.fft.fft(seg[:m].reshape(-1, nf) * np.hanning(nf), axis=1)) ** 2).mean(0))
        c = np.cumsum(psd) / max(psd.sum(), 1e-30)
        lo, hi = np.searchsorted(c, 0.005), np.searchsorted(c, 0.995)
        f = (np.arange(nf) - nf / 2) * fs / nf
        obw, fc = float(f[min(hi, nf - 1)] - f[lo]), float(0.5 * (f[min(hi, nf - 1)] + f[lo]))
    seg = seg * np.exp(-2j * np.pi * fc * t)
    fs_eff = fs
    if 8 * max(obw, 50e3) < fs and len(seg) > 4096:          # narrowband: decimate (FFT crop) to ~8 x bandwidth
        keep = max(2048, int(len(seg) * 8 * max(obw, 50e3) / fs) // 2 * 2)
        X = np.fft.fftshift(np.fft.fft(seg))
        c = len(X) // 2
        seg = np.fft.ifft(np.fft.ifftshift(X[c - keep // 2:c + keep // 2])) * (keep / len(X))
        fs_eff = fs * keep / len(X)
    p = np.abs(seg) ** 2
    papr = 10 * math.log10(p.max() / max(p.mean(), 1e-30)) if len(p) else 0.0
    m2, m4 = (float(p.mean()), float((p * p).mean())) if len(p) else (0.0, 0.0)
    ce = math.sqrt(max(2 * m2 * m2 - m4, 0.0)) / max(m2, 1e-30)
    line = _line_ratio(seg[: 1 << 18], fs_eff) if b.dur >= 1e-3 else 0.0
    fs = fs_eff
    L = int(round(fs / 15e3))
    seg_c = torch.as_tensor(seg[: 65_536 + L + 64], device=x.device)   # 64k-sample window, on the dwell's device
    cp15 = max((_acorr(seg_c, L + d) for d in (-2, -1, 0, 1, 2)), default=0.0)
    base = np.mean([_acorr(seg_c, L + d) for d in (-53, -41, 41, 53)]) + 1e-4
    cp_ratio = cp15 / base                         # CP lag against neighbouring lags (robust to SNR, multipath)
    cp_wifi = _acorr(seg_c, int(round(3.2e-6 * fs)))
    stf = _acorr(seg[: int(8e-6 * fs)], int(round(0.8e-6 * fs)))
    chirp = 0.0
    for sf, bw in ((sf_, bw_) for sf_ in (6, 7, 8) for bw_ in (125e3, 250e3, 500e3, 812.5e3)
                   if bw_ < 1.5 * obw and obw < 1.5e6):           # LoRa test only for narrowband bursts
        nsym = 2 ** sf
        n = int(round(nsym / bw * fs))
        if 16 < n < len(seg):
            k = (np.arange(n) / n * nsym) % nsym
            down = np.exp(-1j * math.pi * (k ** 2 / nsym - k))
            pk = lambda ref: (lambda sp: sp.max() / max(sp.mean(), 1e-30))(np.abs(np.fft.fft(seg[:n] * ref)) ** 2)  # noqa: E731
            # an up-chirp collapses to a tone only with the matching down-chirp; GFSK/tones respond to both alike
            chirp = max(chirp, float(pk(down) / max(pk(np.conj(down)), 1e-30)))
    return {"bw_hz": obw, "fc_hz": fc, "dur_s": b.dur, "papr_db": papr, "cp_corr": cp15, "cp_ratio": cp_ratio, "cp_wifi": cp_wifi,
            "stf_corr": stf, "chirp_peak": chirp, "ce_frac": ce, "line_ratio": line, "snr_db": b.snr_db}


LINE_HZ = 15625.0            # 625-line / 50 Hz analogue video (64 us lines; 525/60: 15.734 kHz, not modelled)


def _line_ratio(seg: np.ndarray, fs: float) -> float:
    """Concentration of the lag-1 product's AC power on the video line-rate comb (see module docstring)."""
    if len(seg) < 64:
        return 0.0
    d = seg[1:] * np.conj(seg[:-1])
    d = d - d.mean()
    S = np.abs(np.fft.fft(d)) ** 2
    fr = np.abs(np.fft.fftfreq(len(d), 1 / fs))
    pos = fr > 0
    r = fr / LINE_HZ
    on = pos & (np.abs(r - np.round(r)) * LINE_HZ <= 1.01 * fs / len(d)) & (r >= 0.5)
    tot = float(S[pos].sum())
    if not on.any() or tot <= 0:
        return 0.0
    return float(S[on].sum() / tot) / (on.sum() / pos.sum())


CE_MIN = 0.6                 # constant-envelope family: OFDM <= 0.27 (narrow low-SNR fragments of OFDM bursts
                             # up to 0.57), GFSK >= 0.62 from 2 dB, SiK >= 0.8 at 14 dB
LINE_MIN = 3.0               # video line comb (FM video >= 5, others <= 1.5)


def classify(f: dict) -> tuple[str, float]:
    if f["chirp_peak"] > 5 and f["bw_hz"] < 1.5e6:
        return "lora", min(0.95, 0.5 + f["chirp_peak"] / 50)
    if f.get("line_ratio", 0.0) > LINE_MIN:                     # 64 us line structure: analogue video
        return "fm_video", min(0.95, 0.7 + f["line_ratio"] / 200)
    if f["papr_db"] > 5.5 and f.get("ce_frac", 0.0) < CE_MIN:
        if f["stf_corr"] > 0.6 and 14e6 < f["bw_hz"] < 22e6:
            return "wifi", 0.6 + 0.3 * f["stf_corr"]
        if f.get("cp_ratio", 0) > 4 and f["bw_hz"] > 7e6:     # CP lag stands out of its neighbours
            return "ocusync_like", 0.5 + min(0.45, f["cp_corr"])
        if f["cp_wifi"] > 0.15 and 14e6 < f["bw_hz"] < 22e6:
            return "wifi", 0.6
        return "unknown", 0.3
    if f["bw_hz"] > 5e6:
        return "fm_video", 0.7
    if f["bw_hz"] < 3e6:
        return "gfsk", 0.6
    return "unknown", 0.3


# ---- batched GPU features (Task 2b). features() above is the reference; features_batch() computes the same
# features for all bursts of a dwell from their band-passed channel-0 segments on the device, in complex128 like the
# numpy reference, with two host syncs per dwell instead of ~15 per burst (tests/unit/test_t2b_batched.py).

_HANN: dict = {}


def _line_mask(n: int, fs: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pos, on, on.sum()/pos.sum(), on.any()) of _line_ratio for a lag-product length n at rate fs, on the
    device with numpy's fftfreq arithmetic (integer bins x 1/(n d) in float64), so the masks are identical."""
    k = torch.arange(n, device=device, dtype=torch.int64)
    k = torch.where(k < (n - 1) // 2 + 1, k, k - n)
    fr = (k.to(torch.float64) * (1.0 / (n * (1 / fs)))).abs()
    pos = fr > 0
    r = fr / LINE_HZ
    on = pos & ((r - torch.round(r)).abs() * LINE_HZ <= 1.01 * fs / n) & (r >= 0.5)
    n_on = on.sum()
    return pos, on, n_on.to(torch.float64) / pos.sum().to(torch.float64), n_on > 0


def _acorr_t(t: torch.Tensor, lag: int) -> torch.Tensor:
    """_acorr() as a device scalar (no sync)."""
    if lag <= 0 or lag >= len(t) - 8:
        return torch.zeros((), dtype=torch.float64, device=t.device)
    a, b = t[:-lag], t[lag:]
    num = torch.abs(torch.sum(a.conj() * b))
    den = torch.sqrt(torch.sum(a.abs() ** 2) * torch.sum(b.abs() ** 2)).clamp_min(1e-30)
    return (num / den).to(torch.float64)


def _acorr_lags(t: torch.Tensor, lags: list[int]) -> list[torch.Tensor]:
    """_acorr_t() at several lags in one pass: row k of a [K, n] stack of shifted copies (zero beyond the end) holds
    t[i + lag_k], so the sums over i < n - lag_k are full-row sums."""
    n = len(t)
    ok = [0 < lg < n - 8 for lg in lags]
    use = [lg for lg, o in zip(lags, ok) if o]
    zero = torch.zeros((), dtype=torch.float64, device=t.device)
    if not use:
        return [zero for _ in lags]
    pad = torch.nn.functional.pad(t, (0, max(use)))
    idx = torch.arange(n, device=t.device)
    sh = torch.stack([pad[lg:lg + n] for lg in use])                                   # [K, n]: t[i + lag]
    valid = idx[None, :] < (n - torch.tensor(use, device=t.device))[:, None]
    a2 = t.abs() ** 2
    num = (t.conj()[None, :] * sh).sum(1).abs()
    ea = (a2[None, :] * valid).sum(1)
    eb = (sh.abs() ** 2).sum(1)
    r = (num / torch.sqrt(ea * eb).clamp_min(1e-30)).to(torch.float64)
    out, k = [], 0
    for o in ok:
        out.append(r[k] if o else zero)
        k += o
    return out


def features_batch(segs0: list[torch.Tensor], fs: float, bursts: list[Burst]) -> list[dict]:
    """features() for each (band-passed channel-0 segment, burst) pair: segs0[i] = burst_samples(x[:1], fs, b)[0]."""
    from sim.rf.fastfft import fft_any, ifft_any

    if not segs0:
        return []
    dev = segs0[0].device
    nf = 1024
    if (nf, str(dev)) not in _HANN:
        _HANN[(nf, str(dev))] = (torch.as_tensor(np.hanning(nf), device=dev),
                                 torch.tensor([0.005, 0.995], dtype=torch.float64, device=dev))
    hann, qs = _HANN[(nf, str(dev))]
    fgrid = (np.arange(nf) - nf / 2) * fs / nf
    # stage 1: 99 % occupied bandwidth (needs a sync: the decimation length depends on it)
    lohi = []
    for seg in segs0:
        m = (len(seg) // nf) * nf
        if m >= nf:
            fr = seg[:m].reshape(-1, nf).to(torch.complex128) * hann
            psd = torch.fft.fftshift((torch.fft.fft(fr, dim=1).abs() ** 2).mean(0))
            c = torch.cumsum(psd, 0) / psd.sum().clamp_min(1e-30)
            lohi.append(torch.searchsorted(c, qs))
        else:
            lohi.append(torch.full((2,), -1, dtype=torch.int64, device=dev))
    lohi = torch.stack(lohi).cpu().numpy()
    # stage 2: everything else on the device; chirp test on the host from a short prefix
    rows, prefixes, meta = [], [], []
    for seg, b, (lo, hi) in zip(segs0, bursts, lohi):
        fc, obw = b.fc, b.bw
        if lo >= 0:
            obw, fc = float(fgrid[min(hi, nf - 1)] - fgrid[lo]), float(0.5 * (fgrid[min(hi, nf - 1)] + fgrid[lo]))
        n_s = len(seg)
        t = torch.arange(n_s, device=dev, dtype=torch.float64) / fs
        s = seg.to(torch.complex128) * torch.exp(-2j * np.pi * fc * t)
        fs_eff = fs
        if 8 * max(obw, 50e3) < fs and n_s > 4096:
            keep = max(2048, int(n_s * 8 * max(obw, 50e3) / fs) // 2 * 2)
            X = torch.fft.fftshift(fft_any(s))
            c = len(X) // 2
            s = ifft_any(torch.fft.ifftshift(X[c - keep // 2:c + keep // 2])) * (keep / len(X))
            fs_eff = fs * keep / len(X)
        p = s.abs() ** 2
        z = torch.zeros((), dtype=torch.float64, device=dev)
        if len(p):
            pm = p.mean()
            papr = 10 * torch.log10(p.max() / pm.clamp_min(1e-30))
            m2, m4 = pm, (p * p).mean()
        else:
            papr = m2 = m4 = z
        line = z
        if b.dur >= 1e-3:
            sl = s[: 1 << 18]
            if len(sl) >= 64:
                d = sl[1:] * sl[:-1].conj()
                d = d - d.mean()
                S = fft_any(d).abs() ** 2
                pos, on, share, any_on = _line_mask(len(d), fs_eff, dev)
                tot = (S * pos).sum()
                on_sum = (S * on).sum()
                line = torch.where(any_on & (tot > 0), on_sum / tot.clamp_min(1e-300) / share.clamp_min(1e-300), z)
        L = int(round(fs_eff / 15e3))
        seg_c = s[: 65_536 + L + 64]
        ac = _acorr_lags(seg_c, [L + dd for dd in (-2, -1, 0, 1, 2, -53, -41, 41, 53)]
                         + [int(round(3.2e-6 * fs_eff))])
        ac.append(_acorr_t(s[: int(8e-6 * fs_eff)], int(round(0.8e-6 * fs_eff))))
        rows.append(torch.stack([papr.to(torch.float64), m2.to(torch.float64), m4.to(torch.float64),
                                 line.to(torch.float64)] + ac))
        combos = [(sf_, bw_) for sf_ in (6, 7, 8) for bw_ in (125e3, 250e3, 500e3, 812.5e3)
                  if bw_ < 1.5 * obw and obw < 1.5e6]
        ns = [int(round(2 ** sf_ / bw_ * fs_eff)) for sf_, bw_ in combos]
        nmax = max([n for n in ns if 16 < n < len(s)], default=0)
        prefixes.append(s[:nmax])
        meta.append((fc, obw, fs_eff, combos, len(s)))
    vals = torch.stack(rows).cpu().numpy()
    pre_h = [q.cpu().numpy() for q in prefixes] if any(len(q) for q in prefixes) else [np.zeros(0)] * len(prefixes)
    out = []
    for b, v, pre, (fc, obw, fs_eff, combos, n_s) in zip(bursts, vals, pre_h, meta):
        papr, m2, m4, line = float(v[0]), float(v[1]), float(v[2]), float(v[3])
        ce = math.sqrt(max(2 * m2 * m2 - m4, 0.0)) / max(m2, 1e-30)
        cp15 = max(float(a) for a in v[4:9])
        base = np.mean(v[9:13]) + 1e-4
        chirp = 0.0
        for sf, bw in combos:
            nsym = 2 ** sf
            n = int(round(nsym / bw * fs_eff))
            if 16 < n < n_s:
                k = (np.arange(n) / n * nsym) % nsym
                down = np.exp(-1j * math.pi * (k ** 2 / nsym - k))
                pk = lambda ref: (lambda sp: sp.max() / max(sp.mean(), 1e-30))(np.abs(np.fft.fft(pre[:n] * ref)) ** 2)  # noqa: E731
                chirp = max(chirp, float(pk(down) / max(pk(np.conj(down)), 1e-30)))
        out.append({"bw_hz": obw, "fc_hz": fc, "dur_s": b.dur, "papr_db": papr, "cp_corr": cp15,
                    "cp_ratio": cp15 / base, "cp_wifi": float(v[13]), "stf_corr": float(v[14]), "chirp_peak": chirp,
                    "ce_frac": ce, "line_ratio": line, "snr_db": b.snr_db})
    return out
