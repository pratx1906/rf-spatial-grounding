"""Observation model with the Deepwave AIR7311 DF receiver (WORLD side; main spec 4A.1, DECISIONS D-022).

Per (emitter, 10 s sample, fixed DF node): the node's AIR7311 Sionna path cache (`rf_cache/air7311_<S>/spec_diff`,
specular + diffraction, depth 2, top-24 paths with complex gain, delay and arrival direction, 20 m cells within
1.5 km of the node) gives the multipath at the emitter's cell. The paths drive the 4-channel engine
(`sim.df.air7311.DFEngine`: wideband snapshots, residual calibration errors, 14-bit ADC, forward-backward
covariance, MUSIC located / Bartlett-ranked peaks). The engine's peaks become the observed bearings; their SNR
and power come from the received power over kTB + NF. Each peak is labelled for truth only by the nearest cached
path (LOS / reflect:<obj> / diffract:<obj>, else spurious). The tracker is the same as `observe_rt`: each node's
strongest peak -> weighted least squares with leave-one-out gating.

Coverage: a node only observes cells inside its cache (1.5 km radius); outside, it reports nothing.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from sim.belief.rf_cache import ellipse, triangulate
from sim.df.air7311 import DFEngine, square_positions
from sim.geometry import angle_diff_deg, bearing_deg, stable_seed
from sim.obs_model.observe import _klass_probs, _sigma_deg

REPO = Path(__file__).resolve().parents[2]
HW_FILE = REPO / "configs" / "hardware" / "air7311.yaml"
SETUP = "spec_diff"
BEARING_DT_S = 10.0
MATCH_DEG = 5.0
MUSIC_WINDOW_DB = 15.0      # measured 2026-10-18: path peaks -0..-9 dB, ripples -20..-31 dB
MIN_SEP_DEG = 8.0           # peaks closer than this are one direction (4-element array, broad beams)
INTERCHIP_DEG = 2.0           # residual AD9371 A/B phase offset after calibration (study case)
MATCHED = {"ism_5800": "A_5g8", "ism_2400": "B_2g4", "sub_ghz_868_915": "C_sub1g"}
_ENGINES: dict[str, DFEngine] = {}


def _engine(hw: dict, device: str) -> DFEngine:
    """One engine per (device, processing config): a cache keyed by device alone silently reused the first
    caller's processing settings for every later hardware profile."""
    import json

    key = f"{device}|{json.dumps(hw['processing'], sort_keys=True)}"
    if key not in _ENGINES:
        _ENGINES[key] = DFEngine(hw, device)
    return _ENGINES[key]


def noise_w(hw: dict, band: str) -> float:
    bw = hw["emitter"]["signal_bw_mhz"][band] * 1e6
    return 10 ** ((-174 + 10 * math.log10(bw) + hw["receiver"]["noise_figure_db"] - 30) / 10)


def air7311_peaks(hw: dict, band: str, h, tau_ns, az, el, valid, eirp_dbm, seed: int = 0, device: str = "cuda",
                  f_hz: float | None = None, max_peaks: int = 3) -> list[list[dict]]:
    """Run the AIR7311 engine on B cells of paths ([B, K] arrays). Returns per cell a list of peaks, strongest
    first: {az_deg, el_deg, snr_db, power_dbm}. Cells below the detection SNR return []."""
    eng = _engine(hw, device)
    f = f_hz or {"ism_2400": 2.44e9, "ism_5800": 5.8e9, "sub_ghz_868_915": 0.9e9}[band]
    bw = hw["emitter"]["signal_bw_mhz"][band] * 1e6
    nw = noise_w(hw, band)
    pos = square_positions(hw["antenna"]["candidate_sides_m"][MATCHED[band]])
    p_w = 10 ** ((np.asarray(eirp_dbm, float) - hw["receiver"]["cable_loss_db"] - 30) / 10)
    tot = (np.abs(h) ** 2 * valid).sum(-1)                                     # path gain, linear
    p_rx_dbm = 10 * np.log10(np.maximum(p_w * tot, 1e-30)) + 30
    snr = 10 * np.log10(np.maximum(p_w * tot / nw, 1e-30))
    det = hw["processing"]["detect_snr_db"]
    out: list[list[dict]] = [[] for _ in range(len(h))]
    idx = np.flatnonzero(snr >= det)
    if not len(idx):
        return out
    gen = torch.Generator(device=eng.dev).manual_seed(int(seed))
    imp = hw["impairments"]
    g_err = torch.randn(4, generator=gen, device=eng.dev) * imp["channel_gain_sigma_db"]
    p_err = torch.randn(4, generator=gen, device=eng.dev) * imp["channel_phase_sigma_deg"]
    A = eng.scan_steering(pos, f)
    n_snap = hw["processing"]["subbands"] * hw["processing"]["snapshots_per_subband"]
    X = eng.snapshots(h[idx], tau_ns[idx], az[idx], el[idx], valid[idx], pos, f, bw, p_w[idx][:, None, None], nw,
                      gen, INTERCHIP_DEG, g_err, p_err)
    R = eng.covariance(X)
    bart, music = eng.spectra(R, A, n_snap)
    pk_az, pk_db = select_peaks(eng.az, music.cpu().numpy(), bart.cpu().numpy(), hw["processing"]["peak_floor_db"],
                                max_peaks)
    # elevation of each peak: argmax of the Bartlett spectrum over the elevation scan at the peak azimuth
    Af = A.reshape(-1, 4)
    full = torch.einsum("dn,bnm,dm->bd", Af.conj(), R, Af).real.reshape(len(idx), *A.shape[:2]).cpu().numpy()
    step = hw["processing"]["az_step_deg"]
    for j, cell in enumerate(idx):
        top = pk_db[j, 0]
        for k in range(max_peaks):
            a = pk_az[j, k]
            if np.isnan(a):
                continue
            ia = int(round(a / step)) % full.shape[2]
            e = float(eng.el[int(np.argmax(full[j, :, ia]))])
            rel = float(pk_db[j, k] - top)
            out[cell].append({"az_deg": float(a) % 360.0, "el_deg": e, "snr_db": float(snr[cell] + rel),
                              "power_dbm": float(p_rx_dbm[cell] + rel)})
    return out


def select_peaks(az_grid: np.ndarray, music: np.ndarray, bart: np.ndarray, floor_db: float, max_peaks: int = 3,
                 min_sep_deg: float = MIN_SEP_DEG) -> tuple[np.ndarray, np.ndarray]:
    """MUSIC local maxima ranked by Bartlett power (pseudo-spectrum height is not power), with greedy non-maximum
    suppression: a real DF processor reports distinct directions, not several ripples of one broad peak.
    Returns az [B, max_peaks] and Bartlett level dB [B, max_peaks], nan-padded."""
    B = music.shape[0]
    left, right = np.roll(music, 1, -1), np.roll(music, -1, -1)
    # MUSIC peaks of real paths stand within a few dB of the top; noise ripples on a 4-element array sit ~20 dB
    # down but can carry nearly the full (very broad) Bartlett power, so they must be cut before the power ranking
    is_pk = (music >= left) & (music > right) & (music >= music.max(-1, keepdims=True) - MUSIC_WINDOW_DB)
    out_az = np.full((B, max_peaks), np.nan)
    out_db = np.full((B, max_peaks), np.nan)
    for b in range(B):
        cand = np.flatnonzero(is_pk[b])
        cand = cand[np.argsort(-bart[b, cand])]
        kept: list[int] = []
        for i in cand:
            if kept and bart[b, i] < bart[b, kept[0]] - floor_db:
                break
            if all(abs((az_grid[i] - az_grid[k] + 180.0) % 360.0 - 180.0) >= min_sep_deg for k in kept):
                kept.append(i)
            if len(kept) == max_peaks:
                break
        out_az[b, :len(kept)] = az_grid[kept]
        out_db[b, :len(kept)] = bart[b, kept]
    return out_az, out_db


def _label(az: float, paz, ptype, via, names, pw) -> tuple[str, str | None]:
    """Truth label: the strongest cached path within MATCH_DEG of the observed peak."""
    best, best_pw = None, -1.0
    for k in range(len(paz)):
        if ptype[k] == 255 or abs(angle_diff_deg(az, float(paz[k]))) > MATCH_DEG:
            continue
        if pw[k] > best_pw:
            best, best_pw = k, pw[k]
    if best is None:
        return "spurious", None
    v = str(names[via[best]]) if via[best] >= 0 else None
    if ptype[best] == 0:
        return "LOS", None
    return (f"diffract:{v}", v) if ptype[best] == 8 else (f"reflect:{v}", v)


class _Cache:
    def __init__(self, rf_root: Path, sensor: str, band: str):
        f = rf_root / f"air7311_{sensor}" / SETUP / f"{band}.npz"
        self.ok = f.exists()
        if self.ok:
            z = np.load(f)
            self.z = {k: z[k] for k in z.files}

    def cell(self, x: float, y: float, h: float) -> tuple[int, int, int] | None:
        z = self.z
        xs, ys, hs = z["xs"], z["ys"], z["heights"]
        if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
            return None
        ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
        iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
        ih = int(np.argmin(np.abs(hs - h)))
        if not (z["ptype"][ih, iy, ix] != 255).any():
            return None
        return ih, iy, ix


def observe_air7311(seed: int, emitters, site_build: dict, rf_root: str | Path, device: str = "cuda",
                    hw: dict | None = None, weather: dict | None = None) -> dict:
    """weather (world variants, D-031): {rain_mm_h, rh_pct} -> ITU gas + rain loss on every path amplitude."""
    hw = hw or yaml.safe_load(HW_FILE.read_text())
    rf_root = Path(rf_root)
    rng = random.Random(seed * 104729 + 13)
    sensors = [s for s in site_build["sensors"] if s.type == "fixed_df"]
    ids = rng.sample(range(10, 9999), len(emitters))
    track_of = {e.id: f"{ids[i]:04d}" for i, e in enumerate(emitters)}
    caches: dict[tuple[str, str], _Cache] = {}

    # 1. gather every (emitter, t, sensor) observation request, grouped by (sensor, band) for batching
    req: dict[tuple[str, str], list] = {}
    for e in emitters:
        for t, ex, ey, ez in e.points:
            if t % BEARING_DT_S:
                continue
            band = e.band_at(t)
            for s in sensors:
                if band not in s.bands:
                    continue
                key = (s.id, band)
                if key not in caches:
                    caches[key] = _Cache(rf_root, s.id, band)
                c = caches[key]
                if not c.ok:
                    continue
                cell = c.cell(ex, ey, ez)
                if cell is None or math.hypot(ex - s.x, ey - s.y) < 30.0:
                    continue
                req.setdefault(key, []).append((e, t, ex, ey, ez, cell))

    # 2. run the engine per (sensor, band)
    results: dict[tuple[str, str, float], tuple] = {}      # (emitter, sensor, t) -> (peaks, labels)
    sens = {s.id: s for s in sensors}
    for (sid, band), rows in sorted(req.items()):
        s, z = sens[sid], caches[(sid, band)].z
        ih, iy, ix = (np.array([r[5][k] for r in rows]) for k in range(3))
        h = (z["h_re"][ih, iy, ix] + 1j * z["h_im"][ih, iy, ix]).astype(np.complex64)
        az = z["az_deg"][ih, iy, ix].copy()
        el = z["el_deg"][ih, iy, ix].copy()
        ptype = z["ptype"][ih, iy, ix]
        valid = ptype != 255
        # exact geometry for the direct path (the cell centre is up to 14 m off), distance-corrected gains
        for j, (e, t, ex, ey, ez, _) in enumerate(rows):
            cx, cy, ch = float(z["xs"][ix[j]]), float(z["ys"][iy[j]]), float(z["heights"][ih[j]])
            d_true = max(math.dist((s.x, s.y, s.z), (ex, ey, ez)), 1.0)
            d_cell = max(math.dist((s.x, s.y, s.z), (cx, cy, ch)), 1.0)
            h[j] *= d_cell / d_true
            los = ptype[j] == 0
            az[j][los] = bearing_deg(s.x, s.y, ex, ey)
            el[j][los] = math.degrees(math.atan2(ez - s.z, math.hypot(ex - s.x, ey - s.y)))
        if weather:
            from sim.obs_model.atmosphere import amp_factor

            h *= amp_factor(z["tau_ns"][ih, iy, ix], band, weather).astype(np.float32)
        eirp = np.array([r[0].eirp_dbm for r in rows])
        pk = air7311_peaks(hw, band, h, z["tau_ns"][ih, iy, ix], az, el, valid, eirp,
                           seed=seed * 31 + stable_seed(sid, band) % 1000, device=device)
        pw = np.abs(h) ** 2
        for j, (e, t, *_r) in enumerate(rows):
            labs = [_label(p["az_deg"], az[j], ptype[j], z["via"][ih[j], iy[j], ix[j]], z["names"], pw[j])
                    for p in pk[j]]
            if pk[j]:
                results[(e.id, sid, t)] = (pk[j], labs)

    # 3. bearings, labels, tracker
    bearings, labels, fixes = [], [], {}
    hyp_n = 0
    for e in emitters:
        for t, ex, ey, ez in e.points:
            if t % BEARING_DT_S:
                continue
            band = e.band_at(t)
            per_sensor = []
            for s in sensors:
                r = results.get((e.id, s.id, t))
                if r is None:
                    continue
                kept = []
                for p, (lab, via) in zip(*r, strict=True):
                    sig = max(0.7, _sigma_deg(p["snr_db"]))
                    kept.append((round(p["az_deg"], 1), round(p["el_deg"], 1), round(p["snr_db"], 1),
                                 round(p["power_dbm"], 1), lab, via, sig))
                per_sensor.append((s, kept))
                for brg, _el, snr, pwr, lab, via, _ in sorted(kept, key=lambda k: k[0]):
                    hyp_n += 1
                    hid = f"{s.id}_{track_of[e.id]}_{int(t):04d}_{hyp_n % 1000:03d}"
                    bearings.append((hid, track_of[e.id], s.id, t, brg, snr, pwr))
                    labels.append((hid, lab, via))
            if not per_sensor:
                continue
            obs = [(s.x, s.y, max(kept, key=lambda q: q[2])[0], max(kept, key=lambda q: q[2])[6])
                   for s, kept in per_sensor]
            fix = triangulate(obs)
            if fix is not None and len(obs) >= 3:
                best = fix
                for i in range(len(obs)):
                    alt = triangulate(obs[:i] + obs[i + 1:])
                    if alt is not None and alt[2] < 0.5 * best[2]:
                        best = alt
                fix = best
            if fix is None:
                continue
            xy, cov, _res = fix
            near_s, near_k = min(per_sensor, key=lambda sk: math.hypot(xy[0] - sk[0].x, xy[1] - sk[0].y))
            elv = max(near_k, key=lambda q: q[2])[1]
            zz = near_s.z + math.hypot(xy[0] - near_s.x, xy[1] - near_s.y) * math.tan(math.radians(elv))
            a, b_, th = ellipse(cov)
            fixes.setdefault(e.id, []).append((t, float(xy[0]), float(xy[1]), float(np.clip(zz, 0.0, 400.0)), a, b_,
                                               th, len(per_sensor), band))
    return _tracks(emitters, fixes, bearings, labels, track_of, rng)


def _tracks(emitters, fixes, bearings, labels, track_of, rng) -> dict:
    tracks, points, track_labels = [], [], []
    for e in emitters:
        fx = fixes.get(e.id, [])
        if len(fx) < 2:
            continue
        tid = track_of[e.id]
        rows = []
        for (t0, x0, y0, z0, a0, b0, th0, n0, bd0), (t1, x1, y1, z1, a1, b1, _th1, _n1, _bd1) in zip(fx, fx[1:]):
            if t1 - t0 > 3 * BEARING_DT_S:
                continue                                         # coverage gap: never bridge
            for t in np.arange(t0, t1, 2.0):
                f = (t - t0) / (t1 - t0)
                rows.append((tid, float(t), round(x0 + f * (x1 - x0), 1), round(y0 + f * (y1 - y0), 1),
                             round(z0 + f * (z1 - z0), 1), round(min(a0 + f * (a1 - a0), 2000.0), 1),
                             round(min(b0 + f * (b1 - b0), 2000.0), 1), round(th0, 1), bd0, n0))
        t, x, y, z, a, b_, th, n, bd = fx[-1]
        rows.append((tid, float(t), round(x, 1), round(y, 1), round(z, 1), round(min(a, 2000.0), 1),
                     round(min(b_, 2000.0), 1), round(th, 1), bd, n))
        rows.sort(key=lambda r: r[1])
        if len(rows) < 3:
            continue
        points += rows
        probs = _klass_probs(rng, e.kind)
        conf = round(min(0.99, probs["drone"] * (0.7 + 0.1 * max(r[9] for r in rows))), 3)
        tb = [bb for bb in bearings if bb[1] == tid]
        power = sorted(bb[6] for bb in tb)[len(tb) // 2] if tb else None
        tracks.append((tid, rows[0][8], e.freq_mhz[rows[0][8]], probs, e.protocol, e.bandwidth_khz, power,
                       rows[0][1], rows[-1][1], max(r[9] for r in rows), conf))
        track_labels.append((tid, e.id, e.kind))
    kept_tracks = {t[0] for t in tracks}
    bearings = [b for b in bearings if b[1] in kept_tracks]
    kept_h = {b[0] for b in bearings}
    labels = [lb for lb in labels if lb[0] in kept_h]
    return {"tracks": tracks, "points": points, "bearings": bearings, "bearing_labels": labels,
            "track_labels": track_labels}
