"""Spectrum environment at the AIR7311 nodes (RF phase 2, Task 5; WORLD side, never imported by tools).

Per node, band, signal channel and 10 s epoch:
  * background emitters (world_cache_v2/spectrum_env.yaml) are active with probability `duty` (seeded per emitter
    and epoch);
  * received power = EIRP - cable loss + path gain from the node's path cache by reciprocity (incoherent sum of
    the cached paths at the emitter's cell); outside the cache, free space + CLUTTER_DB (estimated);
  * co-channel: the part of the emitter's spectrum inside the signal channel adds to the noise;
  * in-window, off-channel: the rest of an in-window emitter adds after the channeliser's ACS;
  * blocking: total in-window power above the input P1dB, or any out-of-window signal above the LNA P1dB (no
    preselector assumed), flags the node as blocked for the epoch (its observations are dropped);
  * pulsed sources (S-band radar) are always present: `duty` is the pulse duty, so they blank that fraction of the
    samples when strong (reported as blanked_fraction) instead of blocking whole epochs.
Values in configs/hardware/air7311.yaml front_end are assumptions [HUMAN].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sim.geometry import stable_seed

CLUTTER_DB = 15.0
BAND_OF = (("sub_ghz_868_915", 840.0, 960.0), ("ism_2400", 2300.0, 2550.0), ("ism_5800", 5600.0, 5950.0))


def band_of(f_mhz: float) -> str | None:
    return next((b for b, lo, hi in BAND_OF if lo <= f_mhz <= hi), None)


def dbm_sum(vals_dbm) -> float:
    s = sum(10 ** (v / 10) for v in vals_dbm)
    return 10 * math.log10(s) if s > 0 else -math.inf


def fspl_db(d_m: float, f_mhz: float) -> float:
    return 20 * math.log10(max(d_m, 1.0)) + 20 * math.log10(f_mhz) - 27.55


@dataclass
class Bg:
    id: str
    kind: str
    f_mhz: float
    bw_mhz: float
    eirp_dbm: float
    duty: float
    pulsed: bool
    x: float
    y: float
    z: float
    known: bool


class SpectrumWorld:
    def __init__(self, env_file: Path, cache_root: Path, hw: dict, sensors: dict[str, dict]):
        raw = yaml.safe_load(Path(env_file).read_text())["emitters"]
        self.em = [Bg(e["id"], e["kind"], float(e["f_mhz"]), float(e["bw_mhz"]), float(e["eirp_dbm"]),
                      float(e["duty"]), e["kind"] == "sband_radar", float(e["x_m"]), float(e["y_m"]), float(e["z_m"]),
                      bool(e.get("known_to_operator"))) for e in raw]
        self.hw, self.fe, self.sensors = hw, hw["front_end"], sensors
        lo, hi = hw["receiver"]["freq_range_mhz"]
        self.in_range = {e.id: lo <= e.f_mhz <= hi for e in self.em}
        self._zc: dict = {}
        self.gain_db = {(e.id, s): self._gain(Path(cache_root), e, s) for e in self.em for s in sensors
                        if self.in_range[e.id]}

    def _gain(self, root: Path, e: Bg, sid: str) -> float:
        p = self.sensors[sid]
        d = math.dist((e.x, e.y, e.z), (p["x_m"], p["y_m"], p["z_m"]))
        band = band_of(e.f_mhz)
        f = root / f"air7311_{sid}" / "spec_diff" / f"{band}.npz" if band else None
        if f is not None and f.exists():
            if f not in self._zc:                              # load each node cache once (was per emitter)
                from sim.rf import pathcache                   # shared with the epoch loop (Task 2b)

                self._zc[f] = pathcache.load(f)
            z = self._zc[f]
            xs, ys, hs = z["xs"], z["ys"], z["heights"]
            if xs[0] - 10 <= e.x <= xs[-1] + 10 and ys[0] - 10 <= e.y <= ys[-1] + 10 and d > 30:
                ix = int(np.clip(round((e.x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
                iy = int(np.clip(round((e.y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
                ih = int(np.argmin(np.abs(hs - e.z)))
                g = float((z["h_re"][ih, iy, ix].astype("f8") ** 2 + z["h_im"][ih, iy, ix].astype("f8") ** 2).sum())
                if g > 0:
                    return 10 * math.log10(g)
        return -(fspl_db(d, e.f_mhz) + (CLUTTER_DB if d > 30 else 0.0))       # no cache / same mast: free space

    def window(self, band: str, f_sig_mhz: float) -> tuple[float, float]:
        ws = self.fe["windows_mhz"][band]
        c = min(ws, key=lambda w: abs(w - f_sig_mhz))
        half = self.hw["receiver"]["ibw_mhz"] / 2
        return c - half, c + half

    def active(self, e: Bg, t_s: float, seed: int) -> bool:
        if e.pulsed:
            return True
        r = np.random.default_rng(stable_seed(seed, e.id, int(t_s // 10)))
        return bool(r.random() < e.duty)

    def epoch(self, sid: str, band: str, f_sig_mhz: float, bw_sig_mhz: float, t_s: float, seed: int,
              extra: list[tuple[str, float, float, float]] | None = None) -> dict:
        """Interference at node sid for a signal channel. extra: (id, f_mhz, bw_mhz, p_rx_dbm) always-on sources
        (e.g. a jammer, received power already computed). Returns co-channel and adjacent interference (dBm),
        in-window total, blocked flag and the active contributors."""
        w0, w1 = self.window(band, f_sig_mhz)
        s0, s1 = f_sig_mhz - bw_sig_mhz / 2, f_sig_mhz + bw_sig_mhz / 2
        cable = self.hw["receiver"]["cable_loss_db"]
        co, adj, win, reasons, act = [], [], [], [], []
        blank = 0.0
        pulse = {e.id: e.duty for e in self.em if e.pulsed}
        srcs = [(e.id, e.f_mhz, e.bw_mhz, e.eirp_dbm - cable + self.gain_db[(e.id, sid)])
                for e in self.em if self.in_range[e.id] and self.active(e, t_s, seed)]
        for eid, f, bw, p in srcs + list(extra or []):
            if eid in pulse:                               # pulsed: blanks its duty fraction when strong
                if p > self.fe["p1db_lna_dbm"] + self.fe["out_of_window_rejection_db"]:
                    blank += pulse[eid]
                continue
            e0, e1 = f - bw / 2, f + bw / 2
            in_win = max(0.0, min(e1, w1) - max(e0, w0)) / bw
            if in_win <= 0:
                if p > self.fe["p1db_lna_dbm"] + self.fe["out_of_window_rejection_db"]:
                    reasons.append(f"lna_compression:{eid}")
                continue
            act.append(eid)
            win.append(p + 10 * math.log10(in_win))
            ov = max(0.0, min(e1, s1) - max(e0, s0)) / bw
            if ov > 0:
                co.append(p + 10 * math.log10(ov))
            if in_win - ov > 1e-9:
                adj.append(p + 10 * math.log10(in_win - ov) - self.fe["acs_db"])
        total = dbm_sum(win)
        if total > self.fe["p1db_in_dbm"]:
            reasons.append(f"in_window_compression:{total:.1f}dBm")
        return {"i_cochannel_dbm": dbm_sum(co), "i_adjacent_dbm": dbm_sum(adj), "window_total_dbm": total,
                "blocked": bool(reasons), "reasons": reasons, "active": act, "blanked_fraction": round(blank, 4)}

    def noise_dbm(self, sid: str, band: str, f_sig_mhz: float, bw_sig_mhz: float, t_s: float, seed: int,
                  extra=None) -> tuple[float, dict]:
        """Effective noise in the signal channel: kTB F + co-channel + ACS-filtered adjacent interference."""
        nf = self.hw["receiver"]["noise_figure_db"]
        kt = -174.0 + 10 * math.log10(bw_sig_mhz * 1e6) + nf
        ep = self.epoch(sid, band, f_sig_mhz, bw_sig_mhz, t_s, seed, extra)
        return dbm_sum([kt, ep["i_cochannel_dbm"], ep["i_adjacent_dbm"]]), ep
