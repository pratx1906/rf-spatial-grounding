"""Track class probabilities from observables only (RF phase 2, v2 sets).

The tracker's rule (speed / height thresholds) called most background emitters "drone": v2 fixes are 38 m noisy and
mixed associations move clutter tracks by hundreds of metres, so speed alone does not separate them. This module
replaces it with a small multinomial logistic model over observable track features (geometry, bearing stability per
node, persistence of the carrier at that bearing over the whole scenario, protocol, band, bandwidth, power).

The model is FITTED OFFLINE on training seeds against world labels (like any classifier trained on simulated data;
scripts/track_class_v2.py fit) and stored as plain coefficients in configs/track_class_v2.json. At product time only
observed tables are read: no truth, no world cache.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ("drone", "controller", "clutter", "decoy")
PROTOS = ("ocusync_like", "wifi", "lora", "gfsk", "fm_video", "unknown")
BANDS = ("ism_2400", "ism_5800", "sub_ghz_868_915")
FEATURES = ("n_sensors", "log_dur", "log_npts", "log_r90", "log_net_speed", "log_step_speed", "z_med", "log_z_std",
            "log_a_med", "log_az_std_med", "log_az_std_min", "presence_max", "presence_med", "log_bw", "power",
            "frac_multi") + tuple(f"p_{p}" for p in PROTOS) + tuple(f"b_{b}" for b in BANDS)
MODEL = Path(__file__).resolve().parents[2] / "configs" / "track_class_v2.json"


def _circ_std(a) -> float:
    r = float(np.abs(np.mean(np.exp(1j * np.radians(a)))))
    return math.degrees(math.sqrt(-2 * math.log(min(max(r, 1e-9), 1.0))))


def track_features(con) -> dict[str, np.ndarray]:
    """track_id -> feature vector (FEATURES order) from an observed.duckdb connection."""
    epochs = [r[0] for r in con.execute("select distinct t_s from dwells").fetchall()]
    ne = max(len(epochs), 1)
    bys = defaultdict(list)
    for s, ts, f, az in con.execute("select sensor_id, t_s, f_mhz, bearing_deg from bursts "
                                    "where bearing_deg is not null").fetchall():
        bys[s].append((ts, f, az))
    pts = defaultdict(list)
    for r in con.execute("select track_id, t_s, x_m, y_m, z_m, a_m, n_sensors from track_points order by t_s").fetchall():
        pts[r[0]].append(r[1:])
    brg = defaultdict(lambda: defaultdict(dict))
    for tid, s, ts, az, snr in con.execute("select track_id, sensor_id, t_s, bearing_deg, snr_db from bearings").fetchall():
        m = brg[tid][s]
        if ts not in m or snr > m[ts][1]:
            m[ts] = (az, snr)
    out = {}
    for tid, proto, band, f0, bw0, pw, t0, t1, ns in con.execute(
            "select track_id, protocol_guess, band, freq_mhz, bandwidth_khz, power_dbm, first_seen, last_seen, "
            "n_sensors from tracks").fetchall():
        p = np.array(pts.get(tid) or [[t0, 0, 0, 0, 0, 1]], dtype=float)
        ts, x, y, z, a, nsp = p.T
        dur = float(ts[-1] - ts[0])
        rad = np.hypot(x - np.median(x), y - np.median(y))
        net = math.hypot(x[-1] - x[0], y[-1] - y[0]) / max(dur, 1.0)
        dt = np.diff(ts)
        step = np.hypot(np.diff(x), np.diff(y))[dt > 0] / dt[dt > 0] if len(ts) > 1 else np.zeros(1)
        stds, pres = [], []
        tol = max(bw0 or 0.0, 500.0) / 1000.0
        for s, m in brg.get(tid, {}).items():
            azs = np.array([v[0] for v in m.values()])
            if len(azs) >= 2:
                stds.append(_circ_std(azs))
            mz = math.degrees(math.atan2(np.mean(np.sin(np.radians(azs))), np.mean(np.cos(np.radians(azs)))))
            hits = {q[0] for q in bys.get(s, []) if abs(q[1] - f0) < tol and abs((q[2] - mz + 180) % 360 - 180) < 4}
            pres.append(len(hits) / ne)
        v = [ns or 0, math.log1p(dur), math.log1p(len(ts)), math.log1p(float(np.percentile(rad, 90))),
             math.log1p(net), math.log1p(float(np.median(step)) if len(step) else 0.0), float(np.median(z)) / 100.0,
             math.log1p(float(np.std(z))), math.log1p(float(np.median(a))),
             math.log1p(float(np.median(stds))) if stds else math.log1p(30.0),
             math.log1p(float(np.min(stds))) if stds else math.log1p(30.0),
             max(pres) if pres else 0.0, float(np.median(pres)) if pres else 0.0,
             math.log1p(bw0 or 0.0), (pw if pw is not None else -100.0) / 10.0, float(np.mean(nsp >= 3))]
        v += [1.0 if proto == q else 0.0 for q in PROTOS]
        v += [1.0 if band == b else 0.0 for b in BANDS]
        out[tid] = np.array(v, dtype=float)
    return out


def label_class(kind: str) -> str:
    """World kind -> class (offline fitting only)."""
    return {"drone": "drone", "friendly_drone": "drone", "controller": "controller", "decoy": "decoy"}.get(kind, "clutter")


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 3000, lr: float = 0.2) -> dict:
    """Class-balanced multinomial logistic regression (full-batch gradient descent). y: class indices."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    k = len(CLASSES)
    cnt = np.bincount(y, minlength=k).astype(float)
    w_s = (len(y) / (k * np.maximum(cnt, 1)))[y]
    W, b = np.zeros((Z.shape[1], k)), np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(iters):
        P = _softmax(Z @ W + b)
        G = (P - Y) * w_s[:, None] / w_s.sum()
        W -= lr * (Z.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return {"features": list(FEATURES), "classes": list(CLASSES), "mu": mu.tolist(), "sd": sd.tolist(),
            "W": W.tolist(), "b": b.tolist(), "l2": l2}


def _softmax(L: np.ndarray) -> np.ndarray:
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def load_model(path: Path = MODEL) -> dict:
    m = json.loads(Path(path).read_text())
    if m["features"] != list(FEATURES) or m["classes"] != list(CLASSES):
        raise ValueError("track_class model does not match this feature set; refit")
    return m


def predict(m: dict, X: np.ndarray) -> np.ndarray:
    Z = (np.atleast_2d(X) - np.array(m["mu"])) / np.array(m["sd"])
    return _softmax(Z @ np.array(m["W"]) + np.array(m["b"]))


def confidence(p_drone: float, n_sensors: int) -> float:
    """Same form as the tracker's product confidence."""
    return round(min(0.99, p_drone * (0.7 + 0.1 * n_sensors)), 3)
