"""Truth-free tracker for v2 detections (RF phase 2, Task 7; Kalman / GNN tracker since S7.1).

1. per node and epoch: bursts -> emissions. Fixed carriers (OFDM, Wi-Fi, FM, unknown) group by frequency
   (|df| < bw/2); hoppers (LoRa, GFSK) group by protocol and strongest bearing (within HOP_DEG); the emission
   bearing is the SNR-weighted circular mean of its bursts (hop fusion).
2. per epoch: emissions from different nodes are associated greedily (strongest first) when band and protocol
   agree and, for fixed carriers, frequency agrees; the set is kept only if the bearings intersect consistently
   (weighted least squares with leave-one-out gating). With >= 3 nodes and >= 10 MHz bandwidth, the same complete
   burst is matched across nodes (sim.belief.tdoa.match_toas: every pair physically possible, >= 3 nodes, one
   emission time at the AoA fix within TDOA_TOL) and the fix is refined with TDOA on 3-D ranges; the
   refinement is kept only if its residuals are consistent, and then carries its own (AoA + TDOA) covariance.
   Each fix gets:
   - a measurement covariance R = R_SCALE^2 x max(1, chi2/dof) x cov + R_FLOOR_M^2 I (x K2_INFL for 2-node fixes,
     x NLOS_INFL when flagged);
   - a height from every node's elevation, weighted by its elevation sigma (range x sigma_el / cos^2 el), with a
     variance; a wide prior when no node has elevation; clipped to 0-400 m;
   - an NLOS / inconsistency flag: a large normalised bearing residual (>= 3 nodes), one node's strongest peak
     contradicting the others (leave-one-out on an inconsistent set), or a 2-node fix with a poor crossing angle.
3. across epochs: constant-velocity Kalman filter (x, y, vx, vy; white-acceleration process noise; height filtered
   separately as a random walk). Same-epoch duplicate fixes of one carrier are merged (dedupe). Per epoch, fixes are
   assigned to tracks one-to-one by global nearest neighbour (Hungarian on Mahalanobis distance + log|S| from the
   PREDICTED state, gated at GATE_D2). Hoppers are compatible across frequency, so only that kinematic gate links
   them. Tracks are born tentative and confirmed with CONF_M fixes within CONF_N epochs (CONF_GOOD_M if one fix is
   a consistent >= GOOD_NODES-node fix); a confirmed track ends after DELETE_S without a fix. Afterwards, duplicate
   tracks of one source are merged (track-to-track), and fragment repair re-links a track that ended to a later one
   (gap <= REPAIR_S, also across bands for band-hopping drones) when the predicted state agrees; the filter is then
   re-run forward over the linked fixes.
4. track points (causal_points): the filtered state at each 2 s step uses only fixes up to that time (prediction
   from the last fix, so ellipses grow while coasting); no interpolation toward future fixes. The reported 2-sigma
   ellipse is OUT_SCALE x the filter covariance: the filter is tuned for accuracy on the error core, the reported
   ellipse is calibrated on the heavy-tailed errors (2-sigma containment 0.86 on training seeds; the old WLS
   covariance gave 0.4-0.6). Constants fitted on training seeds 1-12, 17-28 _X with scripts/track_metrics.py.
5. class probabilities from observables only: speed, height, protocol class, stationarity.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from sim.belief.assign import linear_sum_assignment
from sim.belief.rf_cache import bearing_chi2, ellipse, triangulate
from sim.belief.tdoa import match_toas, tdoa_cov, tdoa_fix
from sim.rf.bursts import fuse_bearings

HOP_DEG = 6.0
RES_GATE_M = 60.0          # rms perpendicular residual (m) for >= 3 bearings; 2 bearings always intersect
MAX_SPEED = 40.0
HOPPERS = {"lora", "gfsk"}
EPOCH_S = 10.0

# measurement model (all constants fitted on training seeds 1-12, 17-28 _X; scripts/track_metrics.py)
R_SCALE = 1.0
R_FLOOR_M = 12.0
K2_INFL = 1.5
NLOS_INFL = 4.0
NLOS_CHI2 = 9.0            # chi2 / dof of the bearings above this: inconsistent
CROSS_MIN_DEG = 20.0       # 2-node crossing angle below this: poorly conditioned

# Kalman filter and track management
Q_ACC = 0.8                # white-acceleration spectral density, m^2/s^3
V0_SIG = 10.0              # initial velocity sigma, m/s
QZ = 10.0                  # height random walk, m^2/s
Z_PRIOR, Z_PRIOR_SIG = 60.0, 100.0
EL_SIG_FLOOR_DEG = 2.0     # elevation is read on a 4 deg grid
GATE_D2 = 9.2              # chi2(2) 99 %
CONF_M, CONF_N = 4, 5
CONF_GOOD_M = 2            # fixes enough when one of them is a consistent fix from >= GOOD_NODES nodes
GOOD_NODES = 3
BIRTH_NLOS = True           # may a flagged fix start a track
DELETE_S = 30.0
REPAIR_S = 60.0
REPAIR_MAX_M = 300.0       # and the later track starts within this distance of the predicted state
REPAIR_D2 = 9.2
XBAND_PEN = 3.0            # repair across bands / carrier classes (band-hopping drones)
MERGE_MAX_M = 60.0         # duplicate tracks of one source: median separation below this
MERGE_FRAC = 0.8
MAX_COAST_S = 30.0         # track points are written up to this long after the last fix
OUT_SCALE = 2.3            # reported ellipse = OUT_SCALE x filter covariance (empirical containment calibration)
DUP_D2 = 6.0              # same-epoch duplicate fixes (chi2(2) 95 %)
BIG = 1e9

# TDOA association (Task 2c): emission-time tolerance against the AoA fix = TDOA_TOL_M + TDOA_TOL_SIG x the AoA
# fix's major-axis sigma, at most TDOA_TOL_MAX_M (sim.belief.tdoa.match_toas)
TDOA_TOL_M, TDOA_TOL_SIG, TDOA_TOL_MAX_M = 150.0, 6.0, 1000.0


def _hop(p: str) -> bool:
    return p in HOPPERS


def _compatible(a: dict, b: dict) -> bool:
    """Same band; both hoppers, or both fixed carriers on the same frequency. The protocol LABEL is not required to
    match: classification varies with SNR between nodes (Wi-Fi vs OcuSync-like at low SNR)."""
    if a["band"] != b["band"] or _hop(a["protocol"]) != _hop(b["protocol"]):
        return False
    return _hop(a["protocol"]) or abs(a["f_mhz"] - b["f_mhz"]) < max(a["bw_khz"], b["bw_khz"]) / 2000
C0 = 299_792_458.0


def _sigma(snr_db: float) -> float:
    return max(0.7, 8.0 * 10 ** (-max(snr_db, 0) / 20))


def emissions(dets) -> list[dict]:
    out = []
    by = defaultdict(list)
    for d in dets:
        if d.peaks:
            by[(d.t, d.sensor, d.band)].append(d)
    for (t, s, band), ds in by.items():
        used = [False] * len(ds)
        for i, d in sorted(enumerate(ds), key=lambda k: -k[1].snr_db):
            if used[i]:
                continue
            grp = [d]
            used[i] = True
            for j, o in enumerate(ds):
                if used[j] or _hop(o.protocol) != _hop(d.protocol):
                    continue
                if _hop(d.protocol):
                    ok = abs((o.peaks[0][0] - d.peaks[0][0] + 180) % 360 - 180) < HOP_DEG
                else:
                    ok = abs(o.f_mhz - d.f_mhz) < max(d.bw_khz, o.bw_khz) / 2000
                if ok:
                    grp.append(o)
                    used[j] = True
            brg = fuse_bearings([g.peaks[0][0] for g in grp], [g.snr_db for g in grp])
            labels = [g.protocol for g in grp if g.protocol != "unknown"] or [d.protocol]
            els = [g.peaks[0][1] for g in grp if math.isfinite(g.peaks[0][1])]
            out.append({"t": t, "sensor": s, "band": band, "protocol": max(set(labels), key=labels.count),
                        "f_mhz": float(np.median([g.f_mhz for g in grp])),
                        "bw_khz": float(np.median([g.bw_khz for g in grp])), "brg": brg,
                        "el": float(np.median(els)) if els else math.nan,
                        "snr": max(g.snr_db for g in grp), "power": max(g.power_dbm for g in grp),
                        "toas": sorted(g.toa_s for g in grp if math.isfinite(g.toa_s)),   # complete bursts only
                        "toa": min((g.toa_s for g in grp if math.isfinite(g.toa_s)), default=math.nan),
                        "t0": min(g.t0_s for g in grp), "t0s": [g.t0_s for g in grp], "fs": [g.f_mhz for g in grp],
                        "n": len(grp),
                        "peaks": [p for g in grp for p in g.peaks], "conf": float(np.mean([g.conf for g in grp]))})
    return out


def _consistent(xy, group, pose) -> bool:
    """Every bearing points at the fix (rejects intersections behind a sensor or far off one bearing)."""
    for g in group:
        sx, sy = pose[g["sensor"]][:2]
        pred = math.degrees(math.atan2(xy[0] - sx, xy[1] - sy)) % 360
        if abs((g["brg"] - pred + 180) % 360 - 180) > 3 * _sigma(g["snr"]) + 2.0:
            return False
    return math.hypot(xy[0] - sx, xy[1] - sy) < 4000


def _obs(group, pose):
    return [(pose[g["sensor"]][0], pose[g["sensor"]][1], g["brg"], _sigma(g["snr"])) for g in group]


def _fix(group, pose):
    """WLS fix with leave-one-out: (xy, cov, rms residual, index of the dropped bearing or None)."""
    obs = _obs(group, pose)
    fix = triangulate(obs)
    drop = None
    if fix is not None and len(obs) >= 3:
        best = fix
        for i in range(len(obs)):
            alt = triangulate(obs[:i] + obs[i + 1:])
            if alt is not None and alt[2] < 0.5 * best[2]:
                best, drop = alt, i
        fix = best
    return None if fix is None else (*fix, drop)


def height3(xy, grp, pose) -> tuple[float, float]:
    """Height (m) and its variance from every node's elevation, weighted by 1 / sigma_z^2 with
    sigma_z = range x sigma_el / cos^2(el); inflated by the spread between nodes; prior when no elevation."""
    zs, ws = [], []
    for g in grp:
        el = g.get("el")
        if el is None or not math.isfinite(el):
            continue
        sx, sy, sz = pose[g["sensor"]]
        d = max(math.hypot(xy[0] - sx, xy[1] - sy), 1.0)
        e = math.radians(min(max(el, -80.0), 80.0))
        sig_el = math.radians(max(EL_SIG_FLOOR_DEG, 2.0 * _sigma(g.get("snr", 0.0))))
        s = d * sig_el / math.cos(e) ** 2 + 3.0
        zs.append(sz + d * math.tan(e))
        ws.append(1.0 / s ** 2)
    if not zs:
        return Z_PRIOR, Z_PRIOR_SIG ** 2
    zs, ws = np.array(zs), np.array(ws)
    z = float(np.sum(ws * zs) / np.sum(ws))
    var = 1.0 / float(np.sum(ws))
    if len(zs) >= 2:
        var *= max(1.0, float(np.sum(ws * (zs - z) ** 2)) / (len(zs) - 1))
    return float(np.clip(z, 0.0, 400.0)), var


def _crossing_deg(group) -> float:
    a = abs(group[0]["brg"] - group[1]["brg"]) % 180.0
    return min(a, 180.0 - a)


def fix_from_group(grp, pose, gnss_ns: float = 5.0) -> dict | None:
    """One fix from a cross-node group of emissions (>= 2 nodes): position, calibrated covariance R, height with
    variance, NLOS / inconsistency flag. None if the geometry is degenerate."""
    f = _fix(grp, pose)
    if f is None:
        return None
    xy, cov, _res, drop = f
    used = [g for i, g in enumerate(grp) if i != drop]
    chi2, _zs = bearing_chi2(xy, _obs(used, pose))
    dof = len(used) - 2
    contradicted = False
    if drop is not None:                       # one node's strongest peak against the others (full-set residual)
        full = triangulate(_obs(grp, pose))
        contradicted = full is not None and bearing_chi2(full[0], _obs(grp, pose))[0] / (len(grp) - 2) > NLOS_CHI2
    poor = len(used) == 2 and _crossing_deg(used) < CROSS_MIN_DEG
    nlos = contradicted or (dof >= 1 and chi2 / dof > NLOS_CHI2) or poor
    xy_aoa = (float(xy[0]), float(xy[1]))
    z, var_z = height3(xy, used, pose)
    mode, toas, tried = "aoa", {}, False
    if len(used) >= 3 and min(g["bw_khz"] for g in used) >= 10000:
        # same burst at every node, physically possible pair by pair and consistent with the AoA fix on one emission
        # time (Task 2c); refinement kept only if its residuals are consistent (audit D Q30)
        tol = min(TDOA_TOL_MAX_M, TDOA_TOL_M + TDOA_TOL_SIG * math.sqrt(max(float(np.linalg.eigvalsh(cov)[-1]), 0.0)))
        toas = match_toas({g["sensor"]: g.get("toas", []) for g in used}, pose, xy=xy_aoa, z_m=z, tol_m=tol)
        tried = len(toas) >= 3
        if tried:
            gt = sorted(({**g, "toa": toas.get(g["sensor"], math.nan)} for g in used),
                        key=lambda g: not math.isfinite(g["toa"]))
            xy2, ok, _info = tdoa_fix(xy, gt, pose, gnss_ns, z_m=z)
            if ok:
                xy, mode = xy2, "aoa+tdoa"
                cov = tdoa_cov(xy, gt, pose, gnss_ns, z_m=z)
                z, var_z = height3(xy, used, pose)
    f = {"x": float(xy[0]), "y": float(xy[1]), "z": z, "var_z": var_z, "cov": np.asarray(cov), "nlos": bool(nlos),
         "mode": mode, "xy_aoa": xy_aoa, "toas": toas, "tdoa_tried": tried, "grp": grp, "chi2": chi2,
         "n_used": len(used)}
    f["R"] = calib_R(f)
    return f


def calib_R(f: dict) -> np.ndarray:
    """Calibrated measurement covariance of a fix (see module docstring, step 2)."""
    dof = f["n_used"] - 2
    infl = max(1.0, f["chi2"] / dof) if dof >= 1 else K2_INFL
    R = R_SCALE ** 2 * infl * f["cov"] + R_FLOOR_M ** 2 * np.eye(2)
    return R * NLOS_INFL if f["nlos"] else R


# ---------------------------------------------------------------------------------------------------- Kalman
def _F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def _Q(dt: float) -> np.ndarray:
    q = np.zeros((4, 4))
    for i in (0, 1):
        q[i, i] = dt ** 3 / 3
        q[i, i + 2] = q[i + 2, i] = dt ** 2 / 2
        q[i + 2, i + 2] = dt
    return Q_ACC * q


def predict(x, P, dt: float):
    F = _F(dt)
    return F @ x, F @ P @ F.T + _Q(dt)


def _init(f) -> dict:
    P = np.zeros((4, 4))
    P[:2, :2] = f["R"]
    P[2, 2] = P[3, 3] = V0_SIG ** 2
    return {"t": f["t"], "x": np.array([f["x"], f["y"], 0.0, 0.0]), "P": P, "z": f["z"], "vz": f["var_z"],
            "nlos": f["nlos"], "n": len(f["grp"]), "band": f["grp"][0]["band"]}


def _update(st: dict, f) -> dict:
    dt = f["t"] - st["t"]
    x, P = predict(st["x"], st["P"], dt)
    S = P[:2, :2] + f["R"]
    K = P[:, :2] @ np.linalg.inv(S)
    x = x + K @ (np.array([f["x"], f["y"]]) - x[:2])
    P = (np.eye(4) - K[:, :2] @ np.eye(2, 4)) @ P
    P = 0.5 * (P + P.T)
    vz = st["vz"] + QZ * dt
    kz = vz / (vz + f["var_z"])
    return {"t": f["t"], "x": x, "P": P, "z": float(st["z"] + kz * (f["z"] - st["z"])), "vz": (1 - kz) * vz,
            "nlos": f["nlos"], "n": len(f["grp"]), "band": f["grp"][0]["band"]}


def run_filter(fixes: list[dict]) -> list[dict]:
    """Filtered state after each fix (time order; causal)."""
    hist = []
    for f in sorted(fixes, key=lambda q: q["t"]):
        hist.append(_init(f) if not hist else _update(hist[-1], f))
    return hist


def _d2(st, f) -> tuple[float, float]:
    x, P = predict(st["x"], st["P"], f["t"] - st["t"])
    S = P[:2, :2] + f["R"]
    v = np.array([f["x"], f["y"]]) - x[:2]
    return float(v @ np.linalg.solve(S, v)), float(np.log(max(np.linalg.det(S), 1e-9)))


def _desc(tr) -> dict:
    a0 = tr["fixes"][-1]["grp"][0]
    return {"band": a0["band"], "protocol": a0["protocol"], "f_mhz": a0["f_mhz"], "bw_khz": a0["bw_khz"]}


def dedupe(fs: list[dict]) -> list[dict]:
    """One fix per source and epoch: compatible fixes of one epoch that agree within DUP_D2 (Mahalanobis on
    R1 + R2) are duplicates (the same carrier grouped twice across nodes); the best one (most nodes, then the
    smallest covariance) is kept and the others ride along in its `dups` (their bearings stay in the products)."""
    out = []
    for f in sorted(fs, key=lambda q: (-q.get("n_used", len(q["grp"])), float(np.trace(q["R"])))):
        for k in out:
            if not _compatible(_desc_fix(k), f["grp"][0]):
                continue
            v = np.array([f["x"] - k["x"], f["y"] - k["y"]])
            if float(v @ np.linalg.solve(k["R"] + f["R"], v)) < DUP_D2:
                k.setdefault("dups", []).append(f)
                break
        else:
            out.append(f)
    return out


def _desc_fix(f) -> dict:
    a0 = f["grp"][0]
    return {"band": a0["band"], "protocol": a0["protocol"], "f_mhz": a0["f_mhz"], "bw_khz": a0["bw_khz"]}


def track_fixes(fixes: list[dict]) -> list[dict]:
    """GNN tracking of fixes -> confirmed tracks {band, protocol, f_mhz, bw_khz, fixes, hist}, by first fix time."""
    by_t = defaultdict(list)
    for f in fixes:
        by_t[f["t"]].append(f)
    live, done = [], []
    for t in sorted(by_t):
        fs = dedupe(by_t[t])
        keep = []
        for tr in live:
            age = t - tr["fixes"][0]["t"]
            if tr["conf"] and t - tr["st"]["t"] > DELETE_S:
                done.append(tr)
            elif not tr["conf"] and age > (CONF_N - 1) * EPOCH_S:
                pass                                               # tentative track never confirmed: dropped
            else:
                keep.append(tr)
        live = keep
        C = np.full((len(live), len(fs)), BIG)
        for i, tr in enumerate(live):
            d = _desc(tr)
            for j, f in enumerate(fs):
                if not _compatible(d, f["grp"][0]):
                    continue
                d2, ld = _d2(tr["st"], f)
                if d2 < GATE_D2:
                    C[i, j] = d2 + ld
        taken = set()
        if live and fs:
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < BIG:
                    tr = live[i]
                    tr["st"] = _update(tr["st"], fs[j])
                    tr["fixes"].append(fs[j])
                    if not tr["conf"] and _confirmed(tr["fixes"]):
                        tr["conf"] = True
                    taken.add(j)
        for j, f in enumerate(fs):
            if j not in taken and (BIRTH_NLOS or not f["nlos"]):
                live.append({"fixes": [f], "st": _init(f), "conf": _confirmed([f])})
    done += [tr for tr in live if tr["conf"]]
    chains = _merge_overlaps([sorted(tr["fixes"], key=lambda f: f["t"]) for tr in done])
    chains = _repair(chains)
    return [{**_desc_first(c), "fixes": c, "hist": run_filter(c)} for c in sorted(chains, key=lambda c: c[0]["t"])]


def _desc_first(fixes) -> dict:
    bands = [f["grp"][0]["band"] for f in fixes]
    a0 = next(f["grp"][0] for f in fixes if f["grp"][0]["band"] == max(set(bands), key=bands.count))
    return {"band": a0["band"], "protocol": a0["protocol"], "f_mhz": a0["f_mhz"], "bw_khz": a0["bw_khz"]}


def _repair(chains: list[list[dict]]) -> list[list[dict]]:
    """Fragment repair: link a track that ended to a later-born one (gap <= REPAIR_S) when the predicted end state
    agrees with the later track's first fix (Mahalanobis below REPAIR_D2, and within REPAIR_MAX_M). Across bands or
    carrier classes (a drone hopping band) only with XBAND_PEN added to its d2. One successor per track, smallest
    first; the filter is re-run over the linked fixes (positions stay causal)."""
    ends = [run_filter(c)[-1] for c in chains]
    cands = []
    for a, A in enumerate(chains):
        end = ends[a]
        da = _desc_fix(A[-1])
        for b, B in enumerate(chains):
            f0 = B[0]
            gap = f0["t"] - end["t"]
            if b == a or not (0 < gap <= REPAIR_S):
                continue
            d2, _ = _d2(end, f0)
            xp = predict(end["x"], end["P"], gap)[0]
            if d2 < REPAIR_D2 and math.hypot(f0["x"] - xp[0], f0["y"] - xp[1]) < REPAIR_MAX_M:
                cands.append((d2 + (0.0 if _compatible(da, f0["grp"][0]) else XBAND_PEN), a, b))
    succ, pred = {}, {}
    for _d, a, b in sorted(cands):
        if a in succ or b in pred:
            continue
        k, loop = b, False                       # no cycles
        while k in succ:
            k = succ[k]
            loop |= k == a
        if loop:
            continue
        succ[a], pred[b] = b, a
    out = []
    for a in range(len(chains)):
        if a in pred:
            continue
        fx, k = list(chains[a]), a
        while k in succ:
            k = succ[k]
            fx += chains[k]
        out.append(sorted(fx, key=lambda f: f["t"]))
    return out


def _grid(hist: list[dict], step: float = EPOCH_S) -> dict:
    """Causal positions on the epoch grid (as causal_points, without coasting beyond MAX_COAST_S)."""
    out, k = {}, 0
    for tt in np.arange(hist[0]["t"], hist[-1]["t"] + 1e-6, step):
        while k + 1 < len(hist) and hist[k + 1]["t"] <= tt + 1e-9:
            k += 1
        dt = float(tt - hist[k]["t"])
        if dt <= MAX_COAST_S:
            out[round(float(tt), 3)] = predict(hist[k]["x"], hist[k]["P"], dt)[0][:2]
    return out


def _merge_overlaps(chains: list[list[dict]]) -> list[list[dict]]:
    """Track-to-track merge of duplicates: two compatible tracks that coexist for >= 2 epochs with a median
    separation below MERGE_MAX_M (and >= MERGE_FRAC of the common epochs closer than that) are one source tracked
    twice. Their fixes are united; where both have a fix the smaller-covariance one is kept (the other goes to its
    `dups`). Repeats until no pair merges."""
    while True:
        grids = [_grid(run_filter(c)) for c in chains]
        best = None
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                if not _compatible(_desc_fix(chains[i][-1]), chains[j][0]["grp"][0]) and \
                        not _compatible(_desc_fix(chains[j][-1]), chains[i][0]["grp"][0]):
                    continue
                common = sorted(set(grids[i]) & set(grids[j]))
                if len(common) < 2:
                    continue
                d = np.array([float(np.hypot(*(grids[i][t] - grids[j][t]))) for t in common])
                med = float(np.median(d))
                if med < MERGE_MAX_M and np.mean(d < MERGE_MAX_M) >= MERGE_FRAC and (best is None or med < best[0]):
                    best = (med, i, j)
        if best is None:
            return chains
        _m, i, j = best
        by = {}
        for f in chains[i] + chains[j]:
            k = f["t"]
            if k not in by:
                by[k] = f
            else:
                keep, drop = sorted((by[k], f), key=lambda q: float(np.trace(q["R"])))
                keep.setdefault("dups", []).append(drop)
                by[k] = keep
        merged = sorted(by.values(), key=lambda f: f["t"])
        chains = [c for k, c in enumerate(chains) if k not in (i, j)] + [merged]


def causal_points(tr: dict, tid: str, band: str, step: float = 2.0) -> list[tuple]:
    """Track points every `step` s from the first to the last fix: the filtered state of the last fix at or before
    each time, predicted to that time (no future fixes). Rows: track_id, t, x, y, z, a, b, theta, band, n_sensors,
    nlos_suspect. Steps more than MAX_COAST_S after the last fix are skipped."""
    hist = tr["hist"]
    ts = [h["t"] for h in hist]
    rows = []
    k = 0
    for tt in np.arange(ts[0], ts[-1] + 1e-6, step):
        while k + 1 < len(hist) and hist[k + 1]["t"] <= tt + 1e-9:
            k += 1
        h = hist[k]
        dt = float(tt - h["t"])
        if dt > MAX_COAST_S:
            continue
        x, P = predict(h["x"], h["P"], dt)
        a, b, th = ellipse(OUT_SCALE ** 2 * P[:2, :2])
        rows.append((tid, float(tt), round(float(x[0]), 1), round(float(x[1]), 1), round(float(h["z"]), 1),
                     round(min(a, 2000.0), 1), round(min(b, 2000.0), 1), round(th, 1), h.get("band", band), int(h["n"]),
                     bool(h["nlos"])))
    return rows


def build_tracks(dets, pose: dict, gnss_ns: float = 5.0, seed: int = 0) -> dict:
    em = emissions(dets)
    fixes, single = [], []
    by_t = defaultdict(list)
    for x in em:
        by_t[x["t"]].append(x)
    for t, es in sorted(by_t.items()):
        free = sorted(es, key=lambda q: -q["snr"])
        while free:
            a = free.pop(0)
            grp = [a]
            for s in sorted({q["sensor"] for q in free} - {a["sensor"]}):
                cands = [q for q in free if q["sensor"] == s and _compatible(a, q)]
                if not cands:
                    continue
                best, best_res = None, math.inf
                for c in cands:
                    f = _fix(grp + [c], pose)
                    if f is not None and _consistent(f[0], grp + [c], pose) and f[2] < best_res:
                        best, best_res = c, f[2]
                if best is not None and best_res < RES_GATE_M:
                    grp.append(best)
                    free.remove(best)
            if len(grp) < 2:
                single.append(a)
                continue
            f = fix_from_group(grp, pose, gnss_ns)
            if f is None:
                single.append(a)
                continue
            fixes.append({"t": t, **f})
    return {"tracks": track_fixes(fixes), "emissions": em, "single": single, "fixes": fixes}


def klass_probs(tr: dict) -> dict:
    fx = tr["fixes"]
    zs = [f["z"] for f in fx]
    sp = 0.0
    if len(fx) >= 2:
        sp = float(np.median([math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / max(b["t"] - a["t"], 1)
                              for a, b in zip(fx, fx[1:])]))
    low, still = np.median(zs) < 8, sp < 1.5
    if low and still:
        p = {"drone": 0.15, "controller": 0.55, "clutter": 0.25, "decoy": 0.05}
    elif still:
        p = {"drone": 0.25, "controller": 0.05, "clutter": 0.6, "decoy": 0.1}
    else:
        p = {"drone": 0.8, "controller": 0.02, "clutter": 0.05, "decoy": 0.13}
    return p


def _confirmed(fx: list[dict]) -> bool:
    """M-of-N confirmation: CONF_M fixes, or CONF_GOOD_M when one is a consistent fix from >= GOOD_NODES nodes."""
    good = any(f.get("n_used", len(f["grp"])) >= GOOD_NODES and not f["nlos"] for f in fx)
    return len(fx) >= (min(CONF_M, CONF_GOOD_M) if good else CONF_M)
