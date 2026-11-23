"""Drone and controller waveforms as complex baseband (RF phase 2, Task 6; world side).

Structure-faithful, not bit-exact: each generator reproduces the parameters an RF analyst measures (occupied
bandwidth, subcarrier spacing and cyclic prefix, preambles, chirp rate, hop pattern, burst timing, envelope), with
random payloads. Proprietary details (e.g. OcuSync scrambling, DroneID field layout) are not modelled.

Every generator returns a Waveform: complex64 samples at its native rate `fs`, centred on the emitter's carrier
(0 Hz), unit average power over the active bursts, plus the ground-truth burst list (t0_s, dur_s, f_off_hz, bw_hz)
used only for validation. Reference parameters:
  ocusync_like  OFDM, 15 kHz subcarrier spacing, FFT 1024/2048/4096 for 10/20/40 MHz (fs 15.36/30.72/61.44 MSPS),
                ~90 % occupied, normal CP (144/2048 of the FFT length, first symbol +16/2048), QPSK data, BPSK pilots
                on every 6th subcarrier; video downlink as 1-3 ms bursts at about 60 % duty
  wifi_ofdm     IEEE 802.11a/g/n legacy preamble (L-STF 8 us, L-LTF 8 us), 64-FFT at 20 MSPS, 52 used subcarriers,
                4 us symbols (0.8 us GI); also carries Remote ID beacons
  lora_elrs     LoRa CSS, BW 500 kHz (868/915) or 812.5 kHz (2.4 GHz SX1280), SF6/7, 8 preamble up-chirps + 2 sync +
                2.25 down-chirps, then data; FHSS: one channel per packet, packet period 1/rate
  fm_fpv        analogue FM video, peak deviation 4 MHz, composite baseband with 15.625 kHz lines (64 us) and sync
                pulses; constant envelope
  gfsk_hop      GFSK (BT 0.5, h 0.5) bursts hopping over a channel list: the RC uplink (about 1 Msym/s, 0.5 ms
                bursts) and SiK telemetry (64 kbit/s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Waveform:
    iq: np.ndarray                     # complex64
    fs: float
    bursts: list[tuple[float, float, float, float]] = field(default_factory=list)   # t0, dur, f_off, bw
    meta: dict = field(default_factory=dict)


def _norm(x: np.ndarray) -> np.ndarray:
    on = np.abs(x) > 0
    p = np.mean(np.abs(x[on]) ** 2) if on.any() else 1.0
    return (x / math.sqrt(p)).astype(np.complex64)


def _qpsk(rng, n):
    return ((rng.integers(0, 2, n) * 2 - 1) + 1j * (rng.integers(0, 2, n) * 2 - 1)) / math.sqrt(2)


def ofdm_ocusync_like(bw_mhz: float, dur_s: float, rng: np.random.Generator, duty: float = 0.6) -> Waveform:
    nfft = {10.0: 1024, 20.0: 2048, 40.0: 4096}[float(bw_mhz)]
    fs = nfft * 15e3
    used = int(round(0.9 * bw_mhz * 1e6 / 15e3)) // 2 * 2
    cp = nfft * 144 // 2048
    sym = nfft + cp
    k = np.r_[np.arange(-used // 2, 0), np.arange(1, used // 2 + 1)]                 # DC unused
    n = int(dur_s * fs)
    x = np.zeros(n, np.complex64)
    bursts, t = [], rng.uniform(0, 1e-3)
    while t < dur_s:
        blen = rng.uniform(1e-3, 3e-3)
        n0, n1 = int(t * fs), min(n, int((t + blen) * fs))
        i = n0
        while i + sym <= n1:
            X = np.zeros(nfft, complex)
            d = _qpsk(rng, used)
            d[::6] = (rng.integers(0, 2, len(d[::6])) * 2 - 1)                          # BPSK pilots
            X[k % nfft] = d
            s = np.fft.ifft(X) * math.sqrt(nfft)
            x[i:i + sym] = np.r_[s[-cp:], s]
            i += sym
        if i > n0:
            bursts.append((n0 / fs, (i - n0) / fs, 0.0, used * 15e3))
        t += blen / duty
    return Waveform(_norm(x), fs, bursts, {"kind": "ocusync_like", "nfft": nfft, "cp": cp, "scs_hz": 15e3,
                                           "used": used})


def wifi_ofdm(dur_s: float, rng: np.random.Generator, packet_us: float = 400.0, period_us: float = 2000.0) -> Waveform:
    fs, nfft = 20e6, 64
    used = np.r_[np.arange(-26, 0), np.arange(1, 27)]
    stf = np.zeros(nfft, complex)
    stf_k = np.array([-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24])
    stf[stf_k % nfft] = math.sqrt(13 / 6) * (1 + 1j) * np.array([1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1])
    s16 = np.fft.ifft(stf)[:16]
    l_stf = np.tile(s16, 10)                                                          # 10 x 0.8 us
    ltf = np.zeros(nfft, complex)
    ltf[used % nfft] = rng.choice([-1, 1], len(used))
    lt = np.fft.ifft(ltf)
    l_ltf = np.r_[lt[-32:], lt, lt]                                                  # 1.6 us GI + 2 x 3.2 us
    n = int(dur_s * fs)
    x = np.zeros(n, np.complex64)
    bursts, t = [], rng.uniform(0, period_us * 1e-6)
    while t < dur_s:
        i = int(t * fs)
        pkt = [l_stf, l_ltf]
        for _ in range(int((packet_us - 16) / 4)):
            X = np.zeros(nfft, complex)
            X[used % nfft] = _qpsk(rng, len(used))
            s = np.fft.ifft(X)
            pkt.append(np.r_[s[-16:], s])
        p = np.concatenate(pkt)
        p = p[: max(0, n - i)]
        x[i:i + len(p)] = p
        if len(p):
            bursts.append((t, len(p) / fs, 0.0, 52 * 312.5e3))
        t += period_us * 1e-6 * rng.uniform(0.7, 1.3)
    return Waveform(_norm(x), fs, bursts, {"kind": "wifi_ofdm", "stf_period_s": 0.8e-6})


def _chirp(n: int, up: bool, start: int = 0, over: int = 1) -> np.ndarray:
    """LoRa chirp of n chips, sampled `over` times per chip with continuous phase (instantaneous frequency sweeps
    -BW/2 .. +BW/2 and wraps at chip `start`): a held (zero-order) chirp would add spectral images."""
    k = (np.arange(n * over) / over + start) % n
    ph = math.pi * (k ** 2 / n - k)
    return np.exp(1j * (ph if up else -ph))


def lora_elrs(bw_hz: float, sf: int, rate_hz: float, channels_hz: list[float], dur_s: float,
              rng: np.random.Generator, data_symbols: int = 10) -> Waveform:
    """Hops are offsets from the band centre (channels_hz); fs covers the hop span."""
    span = (max(channels_hz) - min(channels_hz)) + 2 * bw_hz
    fs = float(2 ** math.ceil(math.log2(span / bw_hz))) * bw_hz
    over = int(fs / bw_hz)
    nsym = 2 ** sf
    up = _chirp(nsym, True, over=over)
    down = _chirp(nsym, False, over=over)
    n = int(dur_s * fs)
    x = np.zeros(n, np.complex64)
    # FHSS: a pseudo-random channel order with a random starting point (the old sequential index from channel 0
    # gave the same 3-4 lowest channels in every realisation, audit B Q12)
    order = rng.permutation(len(channels_hz))
    bursts, t, hop = [], rng.uniform(0, 1 / rate_hz), int(rng.integers(0, len(channels_hz)))
    tt = np.arange(n) / fs
    while t < dur_s:
        f0 = channels_hz[int(order[hop % len(channels_hz)])]
        hop += 1
        pkt = [up] * 8 + [up, up] + [down, down, down[: len(down) // 4]]
        pkt += [_chirp(nsym, True, int(rng.integers(0, nsym)), over) for _ in range(data_symbols)]
        p = np.concatenate(pkt)
        i = int(t * fs)
        p = p[: max(0, n - i)]
        x[i:i + len(p)] = p * np.exp(2j * np.pi * f0 * tt[i:i + len(p)])
        if len(p):
            bursts.append((t, len(p) / fs, f0, bw_hz))
        t += 1 / rate_hz
    return Waveform(_norm(x), fs, bursts, {"kind": "lora", "sf": sf, "bw_hz": bw_hz, "rate_hz": rate_hz,
                                           "chirp_rate_hz_per_s": bw_hz * bw_hz / nsym})


def fm_fpv(dur_s: float, rng: np.random.Generator, dev_hz: float = 4e6, fs: float = 61.44e6,
           audio_hz: float = 6.5e6) -> Waveform:
    """Composite video (sync tip -1, black 0, picture to +1, 64 us lines) plus the 6.5 MHz FM audio subcarrier
    most analogue VTXs carry (level 0.15), frequency-modulating the carrier with peak deviation dev_hz."""
    n = int(dur_s * fs)
    tt = np.arange(n) / fs
    line = 64e-6
    ph_line = (tt % line) / line
    pic = 0.5 + 0.5 * np.sin(2 * np.pi * (rng.uniform(2, 40) * ph_line + rng.random()))
    video = np.where(ph_line < 4.7e-6 / line, -1.0, np.where(ph_line < 10.5e-6 / line, 0.0, pic))
    aud = 0.15 * np.cos(2 * np.pi * audio_hz * tt + 2 * np.pi * 50e3 / 1e3 * np.sin(2 * np.pi * 1e3 * tt))
    m = (video + aud) / np.abs(video + aud).max()
    phase = 2 * np.pi * dev_hz * np.cumsum(m) / fs
    x = np.exp(1j * phase)
    return Waveform(x.astype(np.complex64), fs, [(0.0, dur_s, 0.0, 2 * (dev_hz + audio_hz))],
                    {"kind": "fm_fpv", "line_rate_hz": 1 / line, "dev_hz": dev_hz, "audio_hz": audio_hz})


def _gaussian_taps(bt: float, sps: int, span: int = 4) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / sps
    a = math.sqrt(math.log(2) / 2) / bt
    g = math.sqrt(math.pi) / a * np.exp(-(math.pi * t / a) ** 2)
    return g / g.sum()


def gfsk_hop(sym_rate: float, burst_s: float, period_s: float, channels_hz: list[float], dur_s: float,
             rng: np.random.Generator, bt: float = 0.5, h: float = 0.5, fs: float | None = None) -> Waveform:
    span = max(channels_hz) - min(channels_hz) + 4 * sym_rate
    fs = fs or float(2 ** math.ceil(math.log2(max(span, 8 * sym_rate) / sym_rate)) * sym_rate)
    sps = int(round(fs / sym_rate))
    g = _gaussian_taps(bt, sps)
    n = int(dur_s * fs)
    x = np.zeros(n, np.complex64)
    tt = np.arange(n) / fs
    bursts, t, hop = [], rng.uniform(0, period_s), 0
    while t < dur_s:
        nsym = int(burst_s * sym_rate)
        bits = np.repeat(rng.integers(0, 2, nsym) * 2 - 1, sps).astype(float)
        f = np.convolve(bits, g, "same") * h * sym_rate / 2
        ph = 2 * np.pi * np.cumsum(f) / fs
        f0 = channels_hz[int(rng.integers(0, len(channels_hz)))]
        i = int(t * fs)
        s = np.exp(1j * ph)[: max(0, n - i)]
        x[i:i + len(s)] = s * np.exp(2j * np.pi * f0 * tt[i:i + len(s)])
        if len(s):
            bursts.append((t, len(s) / fs, f0, sym_rate * (1 + h)))
        t += period_s
        hop += 1
    return Waveform(x.astype(np.complex64), fs, bursts, {"kind": "gfsk", "sym_rate": sym_rate, "bt": bt, "h": h})


def occupied_bw_hz(iq: np.ndarray, fs: float, frac: float = 0.99, nfft: int = 4096) -> float:
    """Occupied bandwidth containing `frac` of the power (Welch PSD over the active samples)."""
    x = iq[np.abs(iq) > 0]
    n = (len(x) // nfft) * nfft
    if n == 0:
        return 0.0
    seg = x[:n].reshape(-1, nfft) * np.hanning(nfft)
    psd = np.fft.fftshift((np.abs(np.fft.fft(seg, axis=1)) ** 2).mean(0))
    c = np.cumsum(psd) / psd.sum()
    lo, hi = np.searchsorted(c, (1 - frac) / 2), np.searchsorted(c, 1 - (1 - frac) / 2)
    return float((hi - lo) * fs / nfft)
