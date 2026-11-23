"""v2 observation model: detection-driven AIR7311 network over simulated IQ (RF phase 2, Task 7; WORLD side).

Per 10 s epoch, each DF node takes one dwell (DWELL_S at FS) per tuning window. The dwell holds every active
transmitter in that window: drones and controllers (sim.world.scenario_v2), background emitters
(world_cache_v2/spectrum_env.yaml). Each transmission is generated once per epoch (sim.rf.waveforms) and reaches
each node through that node's Sionna paths at the transmitter's position (world cache; dual polarisation with the
drone antenna attitude: bank from the turn rate, pitch from the climb; node element pattern; per-path Doppler), then
residual mutual coupling, the strong-signal front end (sim.rf.frontend: preselector, Rapp compression, IM3,
desensitisation by out-of-window blockers, blocked dwells) and the receive chain (sim.rf.receiver).
The node detects bursts, classifies them, measures DF per burst (steering at the burst frequency, MDL with N = B x T)
and TOA. Detections feed tracker_v2, which never
sees truth. Truth labels for scoring are attached afterwards by matching tracks to emitters (world side).

GPU paths (Task 2b, build speed): each transmitter's resampled slice is transformed once per tuning window and
shared by every node; arbitrary-length FFTs run on power-of-two plans (sim.rf.fastfft); detection stays on the
device (bursts.detect_gpu: identical bursts); the bursts of a dwell are processed together (one band-pass each shared
by covariance, power and features; batched DF; batched TOA); the path caches load once per process in background
threads (sim.rf.pathcache). SCENESCOUT_V2_REFERENCE=1 runs the per-item reference functions instead. Equivalence:
docs/RESULTS/t2b_build_speed.md and tests/unit/test_t2b_batched.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from sim.df.air7311 import DFEngine, square_positions
from sim.df.calibration import calibrated_steering, coupling_cal_enabled
from sim.geometry import bearing_deg, stable_seed
from sim.obs_model.observe_air7311 import HW_FILE, MATCHED, select_peaks
from sim.rf import antenna as ant
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import classify as C
from sim.rf import receiver as rx
from sim.rf import toa as T
from sim.rf import waveforms as W
from sim.rf import frontend as FE
from sim.rf import pathcache
from sim.rf.spectrum import CLUTTER_DB, SpectrumWorld, band_of, fspl_db

FS = 125e6
DWELL_S = 8e-3
BANK = 3                # waveform realisations per transmitter per scenario
EPOCH_S = 10.0
F_BAND = {"sub_ghz_868_915": 0.9e9, "ism_2400": 2.44e9, "ism_5800": 5.8e9}
ELRS_SUB = [i * 0.6e6 - 7.5e6 for i in range(26)]          # hop offsets around the carrier (approx. 868/915 tables)
ELRS_24 = [i * 1.0e6 - 38e6 for i in range(77)]
RC_HOPS = [i * 1.0e6 - 35e6 for i in range(71)]


@dataclass
class Detection:
    t: float
    sensor: str
    band: str
    f_mhz: float
    bw_khz: float
    t0_s: float
    dur_s: float
    protocol: str
    conf: float
    peaks: list = field(default_factory=list)      # (az, el, snr_db)
    toa_s: float = 0.0
    snr_db: float = 0.0
    power_dbm: float = 0.0


ISM_2400_HZ = (2400.0e6, 2483.5e6)


def _in_band(hops: list[float], fc_hz: float | None, half_bw_hz: float, band=ISM_2400_HZ) -> list[float]:
    """Hop offsets whose whole channel stays inside the band (audit B Q16: hops around a carrier near the band edge
    reached 2366-2517.5 MHz). Falls back to the nearest in-band offset."""
    if fc_hz is None:
        return hops
    lo, hi = band[0] + half_bw_hz - fc_hz, band[1] - half_bw_hz - fc_hz
    keep = [o for o in hops if lo <= o <= hi]
    return keep or [float(np.clip(0.0, lo, hi))]


def waveform(proto: str, rng: np.random.Generator, dur: float, bw_khz: float,
             fc_hz: float | None = None) -> W.Waveform:
    """One transmitter waveform; fc_hz (the carrier) keeps 2.4 GHz hop tables inside 2400-2483.5 MHz."""
    if proto == "ocusync_like":
        return W.ofdm_ocusync_like(10.0 if bw_khz <= 12000 else 20.0, dur, rng)
    if proto in ("wifi", "wifi_ap", "port_wifi"):
        return W.wifi_ofdm(dur, rng, packet_us=rng.uniform(200, 1200), period_us=rng.uniform(1500, 4000))
    if proto in ("lte_site", "cctv_link"):
        return W.ofdm_ocusync_like(20.0, dur, rng, duty=1.0)
    if proto == "fm_video":
        return W.fm_fpv(dur, rng)
    if proto == "elrs":
        return W.lora_elrs(500e3, 6, 200.0, ELRS_SUB, dur, rng, data_symbols=8)
    if proto == "elrs24":
        return W.lora_elrs(812.5e3, 6, 250.0, _in_band(ELRS_24, fc_hz, 812.5e3 / 2), dur, rng, data_symbols=8)
    if proto in ("tank_telemetry",):
        return W.lora_elrs(125e3, 7, 20.0, [0.0], dur, rng, data_symbols=8)
    if proto == "rc_fhss":
        return W.gfsk_hop(1e6, 0.5e-3, 1.4e-3, _in_band(RC_HOPS, fc_hz, 1.5e6 / 2), dur, rng)
    if proto in ("sik", "crane_remote", "vhf_base"):
        return W.gfsk_hop(64e3, 2e-3, 10e-3, [0.0], dur, rng)
    return W.wifi_ofdm(dur, rng)


class _Paths:
    """Node path cache (v2 fields) with nearest-cell lookup."""

    def __init__(self, root: Path, sid: str, band: str):
        f = root / f"air7311_{sid}" / "spec_diff" / f"{band}.npz"
        if not f.exists():
            alt = root / f"air7311_{sid}" / "spec_diff_scat" / f"{band}.npz"
            f = alt if alt.exists() else f
        self.f = f
        self.ok = f.exists()
        if self.ok:
            self.z = pathcache.load(f)          # loaded once per process (shared, read-only; Task 2b)

    @staticmethod
    def file(root: Path, sid: str, band: str) -> Path:
        f = root / f"air7311_{sid}" / "spec_diff" / f"{band}.npz"
        alt = root / f"air7311_{sid}" / "spec_diff_scat" / f"{band}.npz"
        return f if f.exists() or not alt.exists() else alt

    def at(self, x, y, h):
        z = self.z
        xs, ys, hs = z["xs"], z["ys"], z["heights"]
        if not (xs[0] - 10 <= x <= xs[-1] + 10 and ys[0] - 10 <= y <= ys[-1] + 10):
            return None
        ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
        iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
        ih = int(np.argmin(np.abs(hs - h)))
        return ih, iy, ix


def _pos_at(e, t):
    best = min(e.points, key=lambda p: abs(p[0] - t))
    return best if abs(best[0] - t) <= 2.0 else None


def band_at(e, t):
    """The emitter's band at time t (its band_schedule: (from_t, band), sorted by from_t)."""
    band = e.band_schedule[0][1]
    for t0, b in e.band_schedule:
        if t0 <= t:
            band = b
    return band


def match_emitter(emitters, fx, max_frac=0.5):
    """World side: the emitter behind a track = lowest median distance to the track's fixes, counting only fixes
    where the emitter transmits in the fix's band (v3 audit: position alone gave CCTV-link tracks a nearby sub-GHz
    drone's label). Returns (emitter or None, median distance m)."""
    best, bd = None, 1e9
    for e in emitters:
        ds = []
        for f in fx:
            if band_at(e, f["t"]) not in {g["band"] for g in f["grp"]}:
                continue
            p = _pos_at(e, f["t"])
            if p is not None:
                ds.append(math.hypot(p[1] - f["x"], p[2] - f["y"]))
        if len(ds) >= max(2, int(len(fx) * max_frac)) and np.median(ds) < bd:
            best, bd = e, float(np.median(ds))
    return best, bd


def burst_key(d) -> tuple:
    """Key of one received burst for track tagging: (sensor, epoch, start, carrier). The carrier matters because
    continuous carriers all start at t0 ~ 0 (v3 audit: without it, co-timed CCTV / Wi-Fi bursts took drone tracks)."""
    return (d.sensor, d.t, d.t0_s, round(float(d.f_mhz), 4))


def group_keys(g: dict, t: float) -> set:
    """burst_key of every member burst of a tracker group (fix group); groups without per-member carriers (older
    pickles) key on the start time only (carrier None)."""
    if "fs" in g:
        return {(g["sensor"], t, t0, round(float(f), 4)) for t0, f in zip(g["t0s"], g["fs"], strict=True)}
    return {(g["sensor"], t, t0, None) for t0 in g.get("t0s", [g["t0"]])}


def _vel_at(e, t):
    pts = sorted(e.points, key=lambda p: abs(p[0] - t))[:2]
    if len(pts) < 2 or pts[0][0] == pts[1][0]:
        return np.zeros(3)
    a, b = sorted(pts)
    return (np.array(b[1:]) - np.array(a[1:])) / (b[0] - a[0])


def cal_residuals(hw: dict, seed: int, sid: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel gain (dB) and phase (deg) residuals after calibration, from `impairments` sigmas. Fixed per node
    for a scenario (one calibration per deployment), independent between nodes and scenarios."""
    imp = hw.get("impairments", {})
    r = np.random.default_rng(stable_seed(seed, "cal", sid))
    g = r.normal(0, imp.get("channel_gain_sigma_db", 0.0), 4)
    p = r.normal(0, imp.get("channel_phase_sigma_deg", 0.0), 4)
    u = imp.get("use_phase_sigma_deg", 0.0)
    if u:       # use-time-only phase error (mismatch world M3, audit B1): the cal tones never see it; own stream
        p = p + np.random.default_rng(stable_seed(seed, "use_phase", sid)).normal(0, u, 4)
    return torch.tensor(g, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)


def deep_merge(base: dict, override: dict | None) -> dict:
    """A copy of `base` with `override` merged in key by key (nested dicts merged, everything else replaced)."""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_hw(override: dict | None = None) -> dict:
    """The hardware config (configs/hardware/air7311.yaml) with an optional mismatch-world override deep-merged
    (configs/hardware/mismatch/M*.yaml, plan 2026-10-28 T6)."""
    return deep_merge(yaml.safe_load(HW_FILE.read_text()), override)


def realised_hw_errors(hw: dict, seed: int, sids) -> dict:
    """World-side record of what a node's hardware really does in a scenario (truth/world_variant.yaml): the heading
    bias each node gets from hw_errors() and the use-time phase residuals from cal_residuals(), the same draws
    observe_v2 makes."""
    out = {}
    for sid in sids:
        _, ph = cal_residuals(hw, seed, sid)
        out[sid] = {"heading_bias_deg": round(hw_errors(hw, seed, sid)[0], 6),
                    "channel_phase_deg": [round(float(x), 6) for x in ph.numpy()]}
    return out


def hw_errors(hw: dict, seed: int, sid: str) -> tuple[float, np.ndarray]:
    """Heading-datum bias (deg) and ovality-ripple phases (rad, per element) of a node: fixed for a deployment
    (scenario), independent between nodes AND scenarios (they were keyed by node only, so the same 'unknown' bias
    recurred in every seed; audit C Q21)."""
    sd = hw.get("heading_datum", {}).get("bias_deg_1sigma", 0.0)
    bias = float(np.random.default_rng(stable_seed(seed, "heading", sid)).normal(0, sd))
    # T-FIX T3 (docs/paper/TWIN_FIX_PREREG.md): a heading-datum correction entered into the receiver, estimated
    # truth-free by eval/paper/self_calib.py; it subtracts from the reported azimuth (absent = 0, nominal unchanged)
    bias -= float((hw.get("heading_datum", {}).get("correction_deg") or {}).get(sid, 0.0))
    rip = np.random.default_rng(stable_seed(seed, "ovality", sid)).uniform(0, 2 * np.pi, 4)
    return bias, rip


def in_window(fc_hz: float, f0_hz: float, band: str, ibw_hz: float) -> bool:
    """A carrier is sampled in the dwell IQ when it is in the window's band and within ibw/2 + 10 MHz of the window
    centre (its in-window part is kept by resample_shift's IBW filter). Everything else reaches the receiver only
    through the preselector, as a blocker power (sim.rf.frontend), instead of being dropped (S7.2, audit B Q16)."""
    return abs(fc_hz - f0_hz) <= ibw_hz / 2 + 10e6 and band_of(fc_hz / 1e6) == band


def link_gain_db(pc, xyz, node_xyz, f_hz: float) -> float:
    """Power gain (dB) of a link for budget purposes: incoherent sum of the cached paths at the transmitter's cell
    (reciprocity), or free space (+ CLUTTER_DB beyond 30 m) without a cell, as SpectrumWorld._gain does."""
    d = math.dist(xyz, node_xyz)
    cell = pc.at(*xyz) if (pc is not None and pc.ok and d >= 30) else None
    if cell is not None:
        g = float((pc.z["h_re"][cell].astype("f8") ** 2 + pc.z["h_im"][cell].astype("f8") ** 2).sum())
        if g > 0:
            return 10 * math.log10(g)
    return -(fspl_db(d, f_hz / 1e6) + (CLUTTER_DB if d > 30 else 0.0))


def coupling(seed: int, sid: str) -> np.ndarray:
    """Residual mutual coupling of a node's 4 elements (identity + RESIDUAL_DB random-phase off-diagonals, applied
    as y <- C y at the antenna ports): fixed per node for a scenario, independent between nodes and scenarios
    (S7.2, audit C Q21: implemented in sim.rf.antenna but never applied)."""
    return ant.residual_coupling(np.random.default_rng(stable_seed(seed, "coupling", sid)))


def coupling_cal_tones(c_true: np.ndarray, pos: np.ndarray, f_hz: float, cfg: dict, rng: np.random.Generator,
                       hw: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WORLD side of the DF coupling calibration (sim.df.calibration): what the calibration procedure measures.

    K = n_tones CW tones at surveyed bearings (evenly spaced, random start) and tone_el_deg. Each tone arrives from
    the surveyed bearing + N(0, bearing_sigma_deg) and carries one range reflection (multipath_rel_db, random
    direction and phase). The coupling at calibration time is the true one plus a drift (drift_rel_db, complex
    Gaussian off-diagonals: what changes between calibration and use). The receive channels are in a cal-time
    gain/phase state (independent draw of the impairment sigmas + inter-chip offset) and each tone vector is the mean
    of tone_snapshots snapshots at tone_snr_db. Returns (Y [4, K], surveyed az [K], el [K])."""
    from sim.df.calibration import nominal_steering

    K = int(cfg["n_tones"])
    az_s = (rng.uniform(0, 360.0 / K) + np.arange(K) * 360.0 / K) % 360
    el = np.full(K, float(cfg["tone_el_deg"]))
    az_t = az_s + rng.normal(0, cfg["bearing_sigma_deg"], K)
    d = 10 ** (cfg["drift_rel_db"] / 20) / math.sqrt(2)
    drift = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) * d
    np.fill_diagonal(drift, 0)
    c_cal = c_true + drift
    a = nominal_steering(az_t, el, pos, f_hz)                                         # [K, 4]
    m_az, m_el = rng.uniform(0, 360, K), rng.uniform(-10, 30, K)
    mp = 10 ** (cfg["multipath_rel_db"] / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi, K))[:, None] \
        * nominal_steering(m_az, m_el, pos, f_hz)
    imp = (hw or {}).get("impairments", {})
    g = 10 ** (rng.normal(0, imp.get("channel_gain_sigma_db", 0.0), 4) / 20)
    ph = np.radians(rng.normal(0, imp.get("channel_phase_sigma_deg", 0.0), 4) + np.array([0, 0, 2.0, 2.0]))
    D = np.diag(g * np.exp(1j * ph))
    Y = D @ c_cal @ (a + mp).T                                                        # [4, K]
    sig = math.sqrt(10 ** (-cfg["tone_snr_db"] / 10) / max(int(cfg["tone_snapshots"]), 1) / 2)
    Y = Y + sig * (rng.normal(size=Y.shape) + 1j * rng.normal(size=Y.shape))
    return Y, az_s, el


def node_coupling_cal(hw: dict, seed: int, sid: str, band: str, pos: np.ndarray, f_hz: float) -> np.ndarray:
    """The node's calibrated coupling estimate C_hat for one band (one calibration per deployment and band):
    the world simulates the procedure's measurements; only those go to the belief-side estimator."""
    from sim.df import calibration as dcal

    rng = np.random.default_rng(stable_seed(seed, "coupling_cal", sid, band))
    Y, az, el = coupling_cal_tones(coupling(seed, sid), pos, f_hz, dcal.cal_config(hw), rng, hw)
    return dcal.estimate_coupling(Y, az, el, pos, f_hz)


def drone_axis(e, t: float, half_span_s: float = 6.0) -> np.ndarray:
    """Dipole axis (ENU) of a drone's antenna at t: heading from the velocity, bank from the turn rate and pitch
    from the climb and forward acceleration (sim.rf.antenna.attitude) on the trajectory samples within
    +-half_span_s of t (S7.2, audit A Q9: bank and pitch were fixed at 0, so the cross-pol term was always 0)."""
    pts = sorted(p for p in e.points if abs(p[0] - t) <= half_span_s)
    if len(pts) < 3:
        return ant.dipole_axis(0.0, 0.0, 0.0)
    ts = np.array([p[0] for p in pts], float)
    xyz = np.array([p[1:4] for p in pts], float)
    i = int(np.argmin(np.abs(ts - t)))
    v = np.gradient(xyz, ts, axis=0)[i]
    hd = math.atan2(v[0], v[1]) if math.hypot(v[0], v[1]) > 0.5 else 0.0
    bank, pitch = ant.attitude(xyz, ts)
    return ant.dipole_axis(hd, float(bank[i]), float(pitch[i]))


def burst_df(eng: DFEngine, R: torch.Tensor, b, pos_m: np.ndarray, f0: float, hw: dict, bias_deg: float,
             steer_at_burst: bool = True, cal: torch.Tensor | None = None) -> list[tuple[float, float, float]]:
    """Per-burst DF: MUSIC peaks ranked by Bartlett power -> [(az + heading bias, el, snr_db)].

    Steering at the burst centre frequency f0 + b.fc (S7.2: it was the window centre f0, up to 50 MHz off, which
    biases the elevation) and MDL with the burst's effective snapshot count B x T (was a fixed 256).
    cal: the node's calibrated coupling estimate C_hat (sim.df.calibration; None = nominal manifold)."""
    A = calibrated_steering(eng.steering_at(pos_m, f0 + (b.fc if steer_at_burst else 0.0)), cal)
    R = R.to(device=A.device, dtype=torch.complex64)
    bart, music = eng.spectra(R[None], A, DFEngine.mdl_snapshots(b.bw, b.dur, FS))
    pk, db = select_peaks(eng.az, music.cpu().numpy(), bart.cpu().numpy(), hw["processing"]["peak_floor_db"], 3)
    full = torch.einsum("dn,nm,dm->d", A.reshape(-1, 4).conj(), R, A.reshape(-1, 4)).real \
        .reshape(A.shape[:2]).cpu().numpy()                                                     # [n_el, n_az] Bartlett
    step = hw["processing"]["az_step_deg"]
    return [((float(a) + bias_deg) % 360, float(eng.el[int(np.argmax(full[:, int(round(a / step)) % full.shape[1]]))]),
             float(b.snr_db + (d - db[0, 0]))) for a, d in zip(pk[0], db[0]) if np.isfinite(a)]


def burst_df_batch(eng: DFEngine, R: torch.Tensor, bs: list, pos_m: np.ndarray, f0: float, hw: dict,
                   bias_deg: float, steer_at_burst: bool = True, cal: torch.Tensor | None = None
                   ) -> list[list[tuple[float, float, float]]]:
    """burst_df() for all bursts of a dwell at once (R [B, 4, 4]): one batched eigendecomposition and one transfer
    (Task 2b). Same steering, MDL snapshot counts, peak selection and elevation pick per burst."""
    if not bs:
        return []
    A = calibrated_steering(torch.stack([eng.steering_at(pos_m, f0 + (b.fc if steer_at_burst else 0.0)) for b in bs]),
                            cal)
    R = R.to(device=A.device, dtype=torch.complex64)
    N = torch.tensor([DFEngine.mdl_snapshots(b.bw, b.dur, FS) for b in bs], dtype=torch.float64)
    bart, music, full = eng.spectra_batch(R, A, N)
    elix = full.argmax(1)                                                                   # [B, n_az]
    host = torch.cat([bart, music], 1).cpu().numpy()
    elix = elix.cpu().numpy()
    na = full.shape[2]
    pk, db = select_peaks(eng.az, host[:, na:], host[:, :na], hw["processing"]["peak_floor_db"], 3)
    step = hw["processing"]["az_step_deg"]
    return [[((float(a) + bias_deg) % 360, float(eng.el[int(elix[i, int(round(a / step)) % na])]),
              float(b.snr_db + (d - db[i, 0]))) for a, d in zip(pk[i], db[i]) if np.isfinite(a)]
            for i, b in enumerate(bs)]


def observe_v2(seed: int, emitters, sensors, world_root: Path, env_file: Path | None, device: str = "cuda",
               hw: dict | None = None, epochs: list[float] | None = None, duration_s: float = 600.0,
               jammers: list[dict] | None = None, links: dict | None = None) -> list[Detection]:
    hw = hw or yaml.safe_load(HW_FILE.read_text())
    hw = {**hw, "processing": {**hw["processing"], "model_error_floor_db": -20.0}}
    world_root = Path(world_root)
    nodes = {s.id: s for s in sensors if s.type == "fixed_df"}
    poses = {sid: {"x_m": s.x, "y_m": s.y, "z_m": s.z} for sid, s in nodes.items()}
    import os

    # Task 2b: the GPU paths (batched per dwell, see module notes) unless SCENESCOUT_V2_REFERENCE=1 selects the
    # per-item reference functions (kept for equivalence checks)
    fast = os.environ.get("SCENESCOUT_V2_REFERENCE") != "1"
    pathcache.prefetch(_Paths.file(world_root, sid, band) for sid, s in nodes.items()
                       for band in hw["front_end"]["windows_mhz"] if band in s.bands)
    sw = SpectrumWorld(env_file, world_root, hw, poses) if env_file else None
    eng = DFEngine(hw, device)
    if str(device).startswith("cuda"):
        # every distinct FFT length (waveform resampling) caches a cuFFT plan + workspace; the default cache (4096
        # plans) held ~15 GB in a build: keep it small and release cached blocks every epoch. The fast path uses
        # power-of-two plans only (sim.rf.fastfft), a few dozen, so they can all stay cached.
        torch.backends.cuda.cufft_plan_cache[torch.device(device).index or 0].max_size = 64 if fast else 16
    prof = hw.get("array_profiles", {}).get(hw.get("array_profile", "track_b_custom"))
    espec = (prof or {}).get("element", {})
    gam = (espec.get("vswr", 1.0) - 1) / (espec.get("vswr", 1.0) + 1)
    elem_amp = 10 ** (espec.get("gain_dbi", 0.0) / 20) * math.sqrt(1 - gam ** 2)     # gain x mismatch loss
    ripple_db = 0.15 * espec.get("ovality_db", 0.0)                                     # residual after per-element cal
    hwe = {sid: hw_errors(hw, seed, sid) for sid in nodes}
    bias = {sid: v[0] for sid, v in hwe.items()}
    rip_ph = {sid: v[1] for sid, v in hwe.items()}
    cal = {sid: cal_residuals(hw, seed, sid) for sid in nodes}
    cpl = {sid: torch.as_tensor(coupling(seed, sid), dtype=torch.complex64, device=device) for sid in nodes}
    # belief-side calibrated manifold (sim.df.calibration): C_hat per node and band from a simulated calibration
    # (df_calibration in the hardware config, SCENESCOUT_DF_COUPLING_CAL overrides); None = nominal steering
    use_cal = coupling_cal_enabled(hw)
    cal_m: dict = {}
    fe = hw["front_end"]
    cable = hw["receiver"]["cable_loss_db"]
    paths = {}
    dets: list[Detection] = []
    bank: dict = {}
    dwells: list[tuple] = []          # (t, sensor, band, window_mhz, noise_dbm_per_mhz, clipped_frac, n_bursts)
    blocked: list[tuple] = []         # (t, sensor, band, window_mhz, reasons): front end in compression, dropped

    def _pc(sid, band):
        if band is None:
            return None
        if (sid, band) not in paths:
            paths[(sid, band)] = _Paths(world_root, sid, band)
        return paths[(sid, band)]
    n = int(FS * DWELL_S)
    pre = ch.pre_roll(n)          # waveform history before each dwell: path delays are linear, never wrapped (Q30)
    ibw = hw["receiver"].get("ibw_mhz", 100.0) * 1e6
    gnss = hw["receiver"]["gnss_timing_ns_1sigma"]
    resample = ch.resample_shift_fast if fast else ch.resample_shift
    detect = B.detect_gpu if fast else B.detect

    duration_s = float(os.environ.get("SCENESCOUT_V2_MAX_T", duration_s))     # smoke tests only
    for t in (epochs or [float(k) for k in np.arange(0, duration_s, EPOCH_S)]):
        # transmissions this epoch: (id, protocol, carrier_hz, eirp_dbm, xyz, vel, is_drone, waveform)
        tx = []
        for e in emitters:
            p = _pos_at(e, t)
            if p is None:
                continue
            band = e.band_at(t)
            r = np.random.default_rng(stable_seed(seed, e.id, int(t)))
            tx.append((e.id, e.protocol, e.freq_mhz[band] * 1e6, e.eirp_dbm, np.array(p[1:]), _vel_at(e, t),
                       e.kind != "controller", r, e))
        if sw is not None:
            for b in sw.em:
                if sw.in_range[b.id] and not b.pulsed and sw.active(b, t, seed):
                    r = np.random.default_rng(stable_seed(seed, b.id, int(t)))
                    tx.append((b.id, b.kind, b.f_mhz * 1e6, b.eirp_dbm, np.array([b.x, b.y, b.z]), np.zeros(3),
                               False, r, None))
        axes = {q[0]: drone_axis(q[8], t) for q in tx if q[6] and q[8] is not None}   # attitude per drone (Q9)
        p_rx: dict = {}               # (tx, node) -> received power (dBm) at the antenna port, for the blocker budget

        def _p_rx(q, sid):
            if (q[0], sid) not in p_rx:
                s = nodes[sid]
                gdb = sw.gain_db.get((q[0], sid)) if (q[8] is None and sw is not None) else None
                if gdb is None:
                    gdb = link_gain_db(_pc(sid, band_of(q[2] / 1e6)), q[4], (s.x, s.y, s.z), q[2])
                p_rx[(q[0], sid)] = q[3] - cable + gdb
            return p_rx[(q[0], sid)]
        wf = {}
        spec = {}                     # (tx, window) -> FFT of the scaled, resampled slice: the same at every node
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        for sid, s in nodes.items():
            for band, wins in hw["front_end"]["windows_mhz"].items():
                if band not in s.bands:
                    continue
                pc = _pc(sid, band)
                if not pc.ok:
                    continue
                pos_m = square_positions(prof["sides_m"][band] if prof else hw["antenna"]["candidate_sides_m"][MATCHED[band]])
                if use_cal and (sid, band) not in cal_m:          # cal tone at the band's mean window centre
                    f_cal = float(np.mean(wins)) * 1e6
                    cal_m[(sid, band)] = torch.as_tensor(node_coupling_cal(hw, seed, sid, band, pos_m, f_cal),
                                                         dtype=torch.complex64, device=device)
                ccal = cal_m.get((sid, band))
                for wc in wins:
                    f0 = wc * 1e6
                    jam = [j for j in (jammers or []) if j["band"] == band and j["t0_s"] <= t < j["t1_s"]
                           and abs(j["f_mhz"] - wc) < 60]
                    if any(j["sensor"] == sid for j in jam):
                        dwells.append((t, sid, band, wc, None, None, -1))          # own TX: receiver blanked
                        continue
                    y = torch.zeros((4, n), dtype=torch.complex64, device=device)
                    # outside the window: through the preselector as blocker powers (were dropped; audit B Q16)
                    blk = [(q[0], _p_rx(q, sid)) for q in tx if not in_window(q[2], f0, band, ibw)]
                    for j in (jammers or []):
                        if (j["t0_s"] <= t < j["t1_s"] and j["sensor"] != sid
                                and not (j["band"] == band and abs(j["f_mhz"] - wc) < 60)):
                            lk = (links or {}).get(j["band"], {}).get(f"{j['sensor']}->{sid}")
                            if lk is not None:
                                blk.append((j["id"], j["eirp_dbm"] - cable + lk["gain_db"]))
                    for j in jam:                                                  # other node's TX: noise in
                        lk = (links or {}).get(band, {}).get(f"{j['sensor']}->{sid}")
                        if lk is None:
                            continue
                        js = nodes[j["sensor"]]
                        rj = np.random.default_rng(stable_seed(seed, j["id"], sid, int(t)))
                        nz = (rj.normal(size=n) + 1j * rj.normal(size=n)) / math.sqrt(2)
                        fr = np.fft.fftfreq(n, 1 / FS)
                        cut = np.abs(fr - (j["f_mhz"] - wc) * 1e6) > j["bw_mhz"] * 1e6 / 2
                        if fast:                                       # same numpy draws, FFTs on the device
                            Zt = torch.fft.fft(torch.as_tensor(nz, device=device))
                            Zt[torch.as_tensor(cut, device=device)] = 0
                            zt = torch.fft.ifft(Zt)
                            nz = (zt / math.sqrt(max(float(zt.abs().pow(2).mean()), 1e-30))).cpu().numpy()
                        else:
                            Z = np.fft.fft(nz)
                            Z[cut] = 0
                            nz = np.fft.ifft(Z) / math.sqrt(max(np.mean(np.abs(np.fft.ifft(Z)) ** 2), 1e-30))
                        p_w = 10 ** ((j["eirp_dbm"] - hw["receiver"]["cable_loss_db"] + lk["gain_db"] - 30) / 10)
                        d = math.dist((js.x, js.y, js.z), (s.x, s.y, s.z))
                        y += ch.propagate(torch.as_tensor(nz.astype(np.complex64), device=device) * math.sqrt(p_w), FS,
                                          np.array([1.0 + 0j]), np.array([d / ch.C]),
                                          np.array([bearing_deg(s.x, s.y, js.x, js.y)]), np.array([0.0]), pos_m, f0, f0)
                    for tid, proto, fc, eirp, xyz, vel, is_drone, r, src in tx:
                        if not in_window(fc, f0, band, ibw):
                            continue
                        d_node = math.dist(xyz, (s.x, s.y, s.z))
                        cell = pc.at(*xyz) if d_node >= 30 else None
                        if cell is None and d_node >= 30:
                            continue                                       # outside the node's traced area
                        az_los = bearing_deg(s.x, s.y, xyz[0], xyz[1])
                        el_los = math.degrees(math.atan2(xyz[2] - s.z, math.hypot(xyz[0] - s.x, xyz[1] - s.y)))
                        if cell is None:
                            # same mast (< 30 m, not resolved by the cache): one free-space LOS path instead of
                            # dropping the source (co-located emitters are the strongest blockers; S7.2)
                            tau_p = np.array([d_node / ch.C])
                            h = 10 ** (link_gain_db(None, xyz, (s.x, s.y, s.z), fc) / 20) \
                                * np.exp(-2j * math.pi * F_BAND[band] * tau_p)
                            valid = np.ones(1, bool)
                            az, el = np.array([az_los]), np.array([el_los])
                            az_r, el_r, hvh = np.array([(az_los + 180) % 360]), np.array([-el_los]), None
                        else:
                            z = pc.z
                            h = z["h_re"][cell] + 1j * z["h_im"][cell]
                            valid = z["ptype"][cell] != 255
                            if not valid.any():
                                continue
                            tau_p = np.nan_to_num(z["tau_ns"][cell]) * 1e-9
                            az, el = z["az_deg"][cell].copy(), z["el_deg"][cell].copy()
                            los = z["ptype"][cell] == 0
                            az[los], el[los] = az_los, el_los
                            if "az_r_deg" in z:                            # drone-end directions (toward the node)
                                az_r, el_r = z["az_r_deg"][cell].astype(float), z["el_r_deg"][cell].astype(float)
                                az_r[los], el_r[los] = (az_los + 180) % 360, -el_los
                            else:
                                az_r = el_r = None
                            hvh = (z["h_vh_re"][cell] + 1j * z["h_vh_im"][cell]) if "h_vh_re" in z else None
                        if is_drone and az_r is not None and (hvh is not None or cell is None):
                            # dipole along the airframe axis with bank (turn rate) and pitch (climb), Q9
                            cv, chh = ant.drone_pol_coeffs(axes.get(tid, ant.dipole_axis(0.0, 0.0, 0.0)),
                                                           np.nan_to_num(az_r), np.nan_to_num(el_r))
                            h = cv * h + (chh * hvh if hvh is not None else 0)   # node is V-pol: LOS H is lost
                        h = h * ant.node_element_gain(el) * elem_amp
                        h = np.where(valid, h, 0)
                        amp = math.sqrt(10 ** ((eirp - hw["receiver"]["cable_loss_db"] - 30) / 10))
                        if tid not in wf:
                            # waveform bank: BANK realisations per transmitter per scenario, picked per epoch, shifted
                            # by a random start (same structure and statistics; generating per epoch cost ~15 s/epoch)
                            key = (tid, int(r.integers(0, BANK)), int(round(fc / 1e5)))
                            if key not in bank:
                                bwk = next((e.bandwidth_khz for e in emitters if e.id == tid), 20000.0)
                                bank[key] = waveform(proto, np.random.default_rng(stable_seed(seed, *key)), 2 * DWELL_S,
                                                     bwk, fc_hz=fc)
                            w = bank[key]
                            need = int(round((n + pre) / FS * w.fs))
                            off = int(r.integers(0, max(1, len(w.iq) - need)))
                            wf[tid] = (w, off)
                        w, off = wf[tid]
                        # the same slice (pre-roll + dwell) at every node: one transmission, per-node delays
                        if fast:
                            if (tid, wc) not in spec:              # resampled and transformed once per window
                                iq = w.iq[off:off + int(round((n + pre) / FS * w.fs))]
                                spec[(tid, wc)] = torch.fft.fft(
                                    resample(iq, w.fs, FS, fc - f0, n + pre, 0.0, device=device, ibw_hz=ibw) * amp)
                        else:
                            iq = w.iq[off:off + int(round((n + pre) / FS * w.fs))]
                            s_ = resample(iq, w.fs, FS, fc - f0, n + pre, 0.0, device=device, ibw_hz=ibw)
                        dop = 0.0
                        if is_drone and az_r is not None:                  # per-path Doppler (Q14)
                            ok = np.isfinite(az_r) & np.isfinite(el_r)
                            dop = np.where(ok, ch.doppler_hz(vel, np.nan_to_num(az_r), np.nan_to_num(el_r), fc), 0.0)
                        rip = 10 ** (ripple_db * np.sin(2 * np.radians(np.nan_to_num(az))[:, None] + rip_ph[sid][None, :]) / 20)
                        if fast:
                            S_ = spec[(tid, wc)]
                            y += ch.propagate_fast(S_, FS, h, tau_p, np.nan_to_num(az), np.nan_to_num(el), pos_m,
                                                   fc, F_BAND[band], dop, elem_gain=rip, f_lo_hz=f0, n_pre=pre, S=S_)
                        else:
                            y += ch.propagate(s_ * amp, FS, h, tau_p, np.nan_to_num(az),
                                              np.nan_to_num(el), pos_m, fc, F_BAND[band], dop, elem_gain=rip,
                                              f_lo_hz=f0, n_pre=pre)
                    y = cpl[sid] @ y                                       # residual mutual coupling (Q21)
                    y, fi = FE.front_end(y, fe, blk)                       # preselector, LNA / IF compression, IM3
                    g = torch.Generator(device=device).manual_seed(stable_seed(seed, sid, int(t), int(wc)))
                    ge, pe = cal[sid]
                    x, info = rx.receive(y, FS, hw, g, interchip_deg=2.0, gain_err_db=ge.to(device),
                                         phase_err_deg=pe.to(device))
                    p20 = float(torch.quantile(x[:, :: 64].abs().pow(2).flatten().float(), 0.2)) / 0.2231
                    # referred to the antenna with the nominal gain: blocker desense shows as a floor rise
                    floor = 10 * math.log10(max(p20 * info["scale"] ** 2, 1e-30) / (FS / 1e6)) + 30 - fi["gain_db"]
                    if fi["blocked"]:                                      # in compression: observations dropped
                        blocked.append((t, sid, band, wc, fi["reasons"]))
                        dwells.append((t, sid, band, wc, round(floor, 1), round(info["clipped_frac"], 5), 0))
                        continue
                    bs = [b for b in detect(x, FS) if b.dur >= 150e-6]      # shortest real burst: Wi-Fi ~200 us
                    dwells.append((t, sid, band, wc, round(floor, 1), round(info["clipped_frac"], 5), len(bs)))
                    if not bs:
                        continue
                    rng_t = np.random.default_rng(stable_seed(seed, sid, int(t), "toa"))
                    if fast:
                        # every burst of the dwell together: one band-pass per burst shared by the covariance, the
                        # power and the features; batched DF; one transfer per stage (Task 2b)
                        segs = [B.band_pass(x, FS, b) for b in bs]
                        feats = C.features_batch([s_[0] for s_ in segs], FS, bs)
                        peaks_all = burst_df_batch(eng, torch.stack([B.cov_from(s_) for s_ in segs]), bs, pos_m, f0,
                                                   hw, bias[sid], cal=ccal)
                        pw = torch.stack([s_.abs().pow(2).mean() for s_ in segs]).cpu().numpy()
                        toas = T.toa_batch(x, FS, bs, rng_t, gnss)
                        for b, f, peaks, pb, toa in zip(bs, feats, peaks_all, pw, toas):
                            proto, conf = C.classify(f)
                            if os.environ.get("SCENESCOUT_V2_DEBUG"):
                                print("FEAT", sid, band, {k: round(float(v), 3) for k, v in f.items()}, proto, flush=True)
                            p_dbm = 10 * math.log10(max(float(pb) * info["scale"] ** 2, 1e-30)) + 30
                            dets.append(Detection(t, sid, band, (f0 + f["fc_hz"]) / 1e6, f["bw_hz"] / 1e3, b.t0,
                                                  b.dur, proto, conf, peaks, t + toa, b.snr_db, p_dbm))
                        continue
                    for b in bs:
                        f = C.features(x, FS, b)
                        proto, conf = C.classify(f)
                        if os.environ.get("SCENESCOUT_V2_DEBUG"):
                            print("FEAT", sid, band, {k: round(float(v), 3) for k, v in f.items()}, proto, flush=True)
                        R = B.burst_cov(x, FS, b)
                        peaks = burst_df(eng, R, b, pos_m, f0, hw, bias[sid], cal=ccal)
                        p_dbm = 10 * math.log10(max(float(B.burst_samples(x, FS, b).abs().pow(2).mean())
                                                    * info["scale"] ** 2, 1e-30)) + 30
                        dets.append(Detection(t, sid, band, (f0 + f["fc_hz"]) / 1e6, f["bw_hz"] / 1e3, b.t0, b.dur,
                                              proto, conf, peaks, t + T.toa_s(x, FS, b, rng_t, gnss), b.snr_db, p_dbm))
    observe_v2.dwells = dwells
    observe_v2.coupling_cal = bool(use_cal)
    observe_v2.blocked = blocked
    return dets


def products_v2(seed: int, emitters, sensors, dets: list[Detection], world_root: Path, gnss_ns: float,
                bg_positions: dict | None = None, jammers: list[dict] | None = None) -> dict:
    """Detections -> belief tables (tracks, track_points, bearings, bursts) with v1 column layouts (track_points adds
    nlos_suspect, S7.1; points are causal filtered states), plus world-side
    labels (track_labels, hyp_labels) from matching tracks to emitters AFTER tracking (never fed back)."""
    import random

    from sim.obs_model.observe_air7311 import _label
    from sim.obs_model.tracker_v2 import build_tracks, causal_points, klass_probs

    nodes = {s.id: s for s in sensors if s.type == "fixed_df"}
    pose = {sid: (s.x, s.y, s.z) for sid, s in nodes.items()}
    tr = build_tracks(dets, pose, gnss_ns, seed)
    rng = random.Random(seed * 104729 + 17)
    ids = rng.sample(range(10, 9999), max(len(tr["tracks"]), 1))
    tracks, points, bearings, labels, tlabels, burst_rows = [], [], [], [], [], []
    paths = {}
    hyp_n = 0
    det_track = {}
    for k, t in enumerate(tr["tracks"]):
        fx = t["fixes"]
        if len(fx) < 2:
            continue
        tid = f"{ids[k]:04d}"
        # causal filtered states (S7.1): each point uses only fixes up to its time; last column nlos_suspect
        rows = causal_points(t, tid, t["band"])
        fl = fx[-1]
        if len(rows) < 3:
            continue
        points += rows
        probs = klass_probs(t)
        protos = [g["protocol"] for f in fx for g in f["grp"]]
        proto = max(set(protos), key=protos.count)
        pw = sorted(g["power"] for f in fx for g in f["grp"])
        conf = round(min(0.99, probs["drone"] * (0.7 + 0.1 * max(len(f["grp"]) for f in fx))), 3)
        tracks.append((tid, t["band"], round(float(np.median([g["f_mhz"] for f in fx for g in f["grp"]])), 3), probs,
                       proto, round(float(np.median([g["bw_khz"] for f in fx for g in f["grp"]])), 1),
                       round(pw[len(pw) // 2], 1), fx[0]["t"], fx[-1]["t"], max(len(f["grp"]) for f in fx), conf))
        # world side: which emitter is this track? (median distance over the in-band fix times)
        best, bd = match_emitter(emitters, fx)
        if best is not None and bd < 250:
            tlabels.append((tid, best.id, best.kind))
        else:
            near = min(((bid, math.hypot(x - fl["x"], y - fl["y"])) for bid, (x, y) in (bg_positions or {}).items()),
                       key=lambda q: q[1], default=(None, 1e9))
            tlabels.append((tid, near[0] if near[1] < 300 else "none", "clutter" if near[1] < 300 else "spurious"))
        for f in fx:
            for g in f["grp"]:
                for az, el, snr in g["peaks"][:3]:
                    hyp_n += 1
                    hid = f"{g['sensor']}_{tid}_{int(f['t']):04d}_{hyp_n % 1000:03d}"
                    bearings.append((hid, tid, g["sensor"], f["t"], round(az, 1), round(snr, 1), round(g["power"], 1)))
                    lab, via = "spurious", None
                    if best is not None and bd < 250:
                        p = _pos_at(best, f["t"])
                        key = (g["sensor"], g["band"])          # per fix: a track may hop band (S7.1)
                        if key not in paths:
                            paths[key] = _Paths(Path(world_root), g["sensor"], g["band"])
                        pc = paths[key]
                        cell = pc.at(*p[1:]) if (p is not None and pc.ok) else None
                        if cell is not None:
                            z = pc.z
                            h = z["h_re"][cell] + 1j * z["h_im"][cell]
                            s = nodes[g["sensor"]]
                            paz = z["az_deg"][cell].copy()
                            paz[z["ptype"][cell] == 0] = bearing_deg(s.x, s.y, p[1], p[2])
                            lab, via = _label(az, paz, z["ptype"][cell], z["via"][cell], z["names"], np.abs(h) ** 2)
                    labels.append((hid, lab, via))
                for k in group_keys(g, f["t"]):             # every burst of the emission (TDOA tools need them)
                    det_track[k] = tid
            stack = list(f.get("dups", []))                  # duplicate fixes merged into this one (S7.1)
            while stack:
                dup = stack.pop()
                stack += dup.get("dups", [])
                for g in dup["grp"]:
                    for k in group_keys(g, f["t"]):
                        det_track.setdefault(k, tid)
    for i, d in enumerate(dets):
        az = d.peaks[0][0] if d.peaks else None
        burst_rows.append((f"b{i:06d}", d.sensor, d.t, round(d.f_mhz, 4), round(d.bw_khz, 1), round(d.t0_s * 1e6, 1),
                           round(d.dur_s * 1e6, 1), d.protocol, round(d.conf, 2), None if az is None else round(az, 1),
                           round(d.snr_db, 1), round(d.toa_s, 9) if math.isfinite(d.toa_s) else None,   # NULL: cut burst
                           det_track.get(burst_key(d), det_track.get((d.sensor, d.t, d.t0_s, None)))))
    return {"tracks": tracks, "points": points, "bearings": bearings, "bearing_labels": labels,
            "track_labels": tlabels, "bursts": burst_rows, "dwells": list(getattr(observe_v2, "dwells", [])),
            "effector_log": [(j["id"], j["sensor"], j["band"], j["f_mhz"], j["bw_mhz"], j["eirp_dbm"], j["t0_s"], j["t1_s"])
                             for j in (jammers or [])],
            "fix_modes": {m: sum(f["mode"] == m for f in tr["fixes"]) for m in ("aoa", "aoa+tdoa")}}


def ellipse_of(f: dict) -> tuple[float, float, float]:
    from sim.belief.rf_cache import ellipse

    return ellipse(f["cov"])
