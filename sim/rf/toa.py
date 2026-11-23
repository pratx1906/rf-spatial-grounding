"""Time of arrival of a burst at a node (RF phase 2, Task 6).

Leading edge of the burst envelope (channel-summed power, smoothed over 1/bw), refined to the first sample above
`frac` of the burst's median power, plus the node's GNSS timing error (configs/hardware gnss_timing_ns_1sigma).
The leading edge follows the FIRST arriving path (not the strongest), which is what TDOA needs. Resolution is
about 1/bandwidth: 50 ns for 20 MHz, about 2 us for 500 kHz LoRa (reported, not hidden).

A burst cut by the dwell (starting within `guard_s` of the dwell start, or still on air within `guard_s` of its
end) has no measurable leading edge (it would be the dwell start at every node): its TOA is NaN (audit D Q30).
"""

from __future__ import annotations

import numpy as np
import torch

from sim.rf.bursts import Burst, burst_samples


def toa_s(x: torch.Tensor, fs: float, b: Burst, rng: np.random.Generator, gnss_ns: float, frac: float = 0.3,
          guard_s: float = 20e-6) -> float:
    if b.t0 < guard_s or b.t0 + b.dur > x.shape[1] / fs - guard_s:
        return float("nan")
    pre = Burst(max(0.0, b.t0 - guard_s), b.dur + guard_s, b.f_lo, b.f_hi, b.snr_db)
    seg = burst_samples(x, fs, pre)
    p = seg.abs().pow(2).sum(0).cpu().numpy()
    w = max(1, int(fs / max(b.bw, 1.0)))
    ps = np.convolve(p, np.ones(w) / w, "same")
    lvl = frac * np.median(ps[ps > 0.5 * ps.max()]) if (ps > 0).any() else 0.0
    i = int(np.argmax(ps > lvl))
    return pre.t0 + (i - w / 2) / fs + rng.normal(0, gnss_ns * 1e-9)


def toa_batch(x: torch.Tensor, fs: float, bursts: list[Burst], rng: np.random.Generator, gnss_ns: float,
              frac: float = 0.3, guard_s: float = 20e-6) -> list[float]:
    """toa_s() for the bursts of one dwell, in order (same rng draws): the band-passes run on the device
    (sim.rf.bursts.band_pass) and one transfer brings all envelopes to the host for the leading-edge search."""
    from sim.rf.bursts import band_pass

    cut = [b.t0 < guard_s or b.t0 + b.dur > x.shape[1] / fs - guard_s for b in bursts]
    pres = [None if c else Burst(max(0.0, b.t0 - guard_s), b.dur + guard_s, b.f_lo, b.f_hi, b.snr_db)
            for b, c in zip(bursts, cut)]
    env = [band_pass(x, fs, p).abs().pow(2).sum(0) for p in pres if p is not None]
    host = torch.cat(env).cpu().numpy() if env else np.zeros(0, np.float32)
    out, k, o = [], 0, 0
    for b, p in zip(bursts, pres):
        if p is None:
            out.append(float("nan"))
            continue
        n = len(env[k])
        pw = host[o:o + n]
        k, o = k + 1, o + n
        w = max(1, int(fs / max(b.bw, 1.0)))
        ps = np.convolve(pw, np.ones(w) / w, "same")
        lvl = frac * np.median(ps[ps > 0.5 * ps.max()]) if (ps > 0).any() else 0.0
        i = int(np.argmax(ps > lvl))
        out.append(p.t0 + (i - w / 2) / fs + rng.normal(0, gnss_ns * 1e-9))
    return out
