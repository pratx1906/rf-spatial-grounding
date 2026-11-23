"""Hybrid AoA + TDOA position refinement (pure geometry; used by the v2 tracker and by the agent's tdoa_fix tool).

match_toas():  the SAME burst across nodes: per node a list of TOAs of complete bursts (truncated bursts have no
               TOA); a reference burst is matched to the burst at each other node within that pair's baseline / c
               plus a margin (bursts of one emitter are >= 100s of us apart; baselines are a few us). Every pair of
               the set must be physically possible, >= 3 nodes must agree, and with the AoA fix given every node
               must imply the same emission time (Task 2c; the v21 audit had pairs at up to 2.2 x baseline / c).
tdoa_refine(): Gauss-Newton on bearing residuals (sigma from SNR) and range-difference residuals against a
               reference node (sigma = sqrt(2) x GNSS timing sigma x c + c / bandwidth), starting from the AoA fix.
               Ranges are 3-D when the emitter height z_m is given (2-D ranges miss dz^2 / 2r: tens of metres at
               200-300 m for a 100 m drone). Members without a finite TOA count for bearings only.
tdoa_fix():    tdoa_refine plus a residual check: the refinement is accepted only if every range difference is
               within GATE sigma and every bearing still points at the fix; otherwise the AoA fix is kept (audit D Q30).
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299_792_458.0
GATE = 3.0


def _sigma(snr_db: float) -> float:
    return max(0.7, 8.0 * 10 ** (-max(snr_db, 0) / 20))


def _finite(v) -> bool:
    return v is not None and np.isfinite(v)


def _merge(v, dup_s: float = 1e-6) -> list:
    """One TOA per burst: detections of one burst split in frequency (e.g. an OFDM burst cut at its DC null) give
    leading edges within ~1/B of each other; they are averaged."""
    out: list = []
    for t in sorted(t for t in v if _finite(t)):
        if out and t - out[-1][-1] < dup_s:
            out[-1].append(t)
        else:
            out.append([t])
    return [float(np.mean(c)) for c in out]


def _bound(pose, a, b, margin_s) -> float:
    """Largest physically possible |TOA_a - TOA_b| of one burst: the a-b baseline / c plus the timing margin."""
    return math.dist(pose[a], pose[b]) / C0 + margin_s


def _prune_pairs(m: dict, pose: dict, margin_s: float) -> dict:
    """Drop nodes until every pair of matched TOAs is physically possible (|dt| <= baseline / c + margin); the node
    with the most violations (then the largest total excess) goes first."""
    m = dict(m)
    while len(m) >= 2:
        bad = {s: [0, 0.0] for s in m}
        for a in m:
            for b in m:
                ex = abs(m[a] - m[b]) - _bound(pose, a, b, margin_s)
                if a < b and ex > 0:
                    for s in (a, b):
                        bad[s][0] += 1
                        bad[s][1] += ex
        worst = max(m, key=lambda s: tuple(bad[s]))
        if bad[worst][0] == 0:
            break
        del m[worst]
    return m


def _prune_emission_time(m: dict, pose: dict, xy, z_m, tol_m: float) -> dict:
    """Keep the nodes whose implied emission time TOA - range(xy) / c agrees with the median of the set within
    tol_m / c (a TOA inside every pair bound can still contradict the AoA fix by kilometres)."""
    def rng(s):
        p = pose[s]
        return math.sqrt((xy[0] - p[0]) ** 2 + (xy[1] - p[1]) ** 2 + ((z_m - p[2]) ** 2 if z_m is not None else 0.0))
    m = dict(m)
    while len(m) >= 2:
        te = {s: m[s] - rng(s) / C0 for s in m}
        med = float(np.median(list(te.values())))
        worst = max(m, key=lambda s: abs(te[s] - med))
        if abs(te[worst] - med) * C0 <= tol_m:
            break
        del m[worst]
    return m


def match_toas(toas: dict, pose: dict, margin_s: float = 0.5e-6, min_nodes: int = 3, xy=None,
               z_m: float | None = None, tol_m: float = 300.0) -> dict:
    """toas: {sensor: [toa_s of complete bursts]}. Returns {sensor: toa_s} of one burst seen at as many nodes as
    possible (ties: the earliest reference burst), or {} when fewer than min_nodes agree. Per reference burst, each
    other node contributes its unique burst within the PAIR's physical bound (baseline / c + margin); the set is then
    pruned until every pair is physically possible, and, when the AoA fix `xy` (height z_m) is given, until every
    node implies the same emission time within tol_m / c (Task 2c: no cross-node pairs of different bursts or of a
    corrupted leading edge)."""
    toas = {s: _merge(v) for s, v in toas.items()}
    ids = [s for s in toas if toas[s]]
    if len(ids) < max(min_nodes, 2):
        return {}
    best: dict = {}
    for ref in ids:
        for t0 in sorted(t for t in toas[ref] if _finite(t)):
            m = {ref: t0}
            for s in ids:
                if s == ref:
                    continue
                c = [t for t in toas[s] if _finite(t) and abs(t - t0) <= _bound(pose, ref, s, margin_s)]
                if len(c) == 1:
                    m[s] = c[0]
            m = _prune_pairs(m, pose, margin_s)
            if xy is not None:
                m = _prune_emission_time(m, pose, xy, z_m, tol_m)
            if len(m) > len(best) or (len(m) == len(best) and m and min(m.values()) < min(best.values())):
                best = m
    return best if len(best) >= max(min_nodes, 2) else {}


def _rng(x, s, z_m):
    if z_m is None:
        return math.hypot(x[0] - s[0], x[1] - s[1])
    return math.sqrt((x[0] - s[0]) ** 2 + (x[1] - s[1]) ** 2 + (z_m - s[2]) ** 2)


def _residuals(x, group, pose, gnss_ns, z_m):
    """Bearing residuals (rad) with sigmas, range-difference residuals (m) with sigma, and their Jacobians."""
    J, r, sig = [], [], []
    for g in group:
        sx, sy = pose[g["sensor"]][:2]
        dx, dy = x[0] - sx, x[1] - sy
        rr = max(math.hypot(dx, dy), 1.0)
        pred = math.degrees(math.atan2(dx, dy)) % 360
        r.append(((g["brg"] - pred + 180) % 360 - 180) * math.pi / 180)
        J.append([dy / rr ** 2, -dx / rr ** 2])
        sig.append(math.radians(_sigma(g["snr"])))
    nb = len(r)
    tg = [g for g in group if _finite(g.get("toa"))]
    if len(tg) >= 2:
        ref = tg[0]
        s0 = pose[ref["sensor"]]
        r0 = max(_rng(x, s0, z_m), 1.0)
        sig_t = math.sqrt(2) * gnss_ns * 1e-9 * C0 + C0 / (min(g["bw_khz"] for g in tg) * 1e3)
        for g in tg[1:]:
            s = pose[g["sensor"]]
            ri = max(_rng(x, s, z_m), 1.0)
            r.append((g["toa"] - ref["toa"]) * C0 - (ri - r0))
            J.append([(x[0] - s[0]) / ri - (x[0] - s0[0]) / r0, (x[1] - s[1]) / ri - (x[1] - s0[1]) / r0])
            sig.append(sig_t)
    return np.array(J), np.array(r), np.array(sig), nb


def tdoa_refine(xy, group, pose, gnss_ns, z_m: float | None = None):
    """Gauss-Newton on bearing + TDOA residuals from the AoA fix (2-D position; 3-D ranges if z_m is given)."""
    x = np.array(xy, float)
    for _ in range(10):
        J, r, sig, _nb = _residuals(x, group, pose, gnss_ns, z_m)
        w = 1 / sig ** 2
        H = J.T @ (J * w[:, None])
        try:
            step = np.linalg.solve(H, J.T @ (w * r))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        x += step
        if np.hypot(*step) < 0.1:
            break
    return x


def tdoa_fix(xy, group, pose, gnss_ns, z_m: float | None = None, gate: float = GATE):
    """Residual-gated refinement. Returns (xy, accepted, info); xy is the AoA fix (unchanged) when rejected."""
    xy0 = tuple(float(v) for v in xy)
    n_t = sum(_finite(g.get("toa")) for g in group)
    if n_t < 3:
        return xy0, False, {"reason": f"{n_t} node(s) with a matched TOA"}
    x = tdoa_refine(xy0, group, pose, gnss_ns, z_m)
    if not np.all(np.isfinite(x)):
        return xy0, False, {"reason": "diverged"}
    J, r, sig, nb = _residuals(x, group, pose, gnss_ns, z_m)
    zr = np.abs(r[nb:]) / sig[nb:]
    bad_brg = np.abs(np.degrees(r[:nb])) > gate * np.degrees(sig[:nb]) + 2.0        # same test as the AoA gate
    info = {"max_z_tdoa": float(zr.max()), "n_bad_bearings": int(bad_brg.sum()),
            "shift_m": float(math.hypot(x[0] - xy0[0], x[1] - xy0[1]))}
    if zr.max() > gate or bad_brg.any():
        info["reason"] = "inconsistent residuals"
        return xy0, False, info
    return (float(x[0]), float(x[1])), True, info


def tdoa_cov(xy, group, pose, gnss_ns, z_m: float | None = None) -> np.ndarray:
    """2x2 covariance (m^2) of a hybrid AoA + TDOA fix: inverse Fisher information (J^T W J)^-1 of the bearing and
    range-difference residuals at xy (the refined covariance kept by the tracker; bearings are range-scaled through J)."""
    J, _r, sig, _nb = _residuals(np.array(xy, float), group, pose, gnss_ns, z_m)
    H = J.T @ (J / sig[:, None] ** 2)
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.eye(2) * 1e6
