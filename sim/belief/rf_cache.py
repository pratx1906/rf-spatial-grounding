"""BELIEF side: the nominal Sionna RT cache and DF localisation maths (what the agent's tools may use).

The cache stores, per (band, sensor, height, grid cell), the top-K propagation paths computed by Sionna RT with
the transmitter at the sensor (reciprocity): compass bearing and elevation at the sensor, gain, path type and the
first object hit. It is the nominal model; the world side perturbs it (main spec 4A.1).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LOS, SPECULAR, DIFFUSE, DIFFRACTION = 0, 1, 2, 8
PTYPE_NAME = {LOS: "LOS", SPECULAR: "reflect", DIFFUSE: "scatter", DIFFRACTION: "diffract"}


class RFCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self._d: dict[tuple[str, str], dict | None] = {}

    def available(self) -> bool:
        return (self.dir / "manifest.json").exists()

    def get(self, band: str, sensor: str) -> dict | None:
        key = (band, sensor)
        if key not in self._d:
            f = self.dir / band / f"aoa_{sensor}.npz"
            if not f.exists():
                self._d[key] = None
            else:
                z = np.load(f)
                self._d[key] = {k: z[k] for k in z.files}
        return self._d[key]

    def paths(self, band: str, sensor: str, x: float, y: float, z: float) -> list[dict] | None:
        """Paths at the nearest cached cell, strongest first. None if outside the RF scene or no cache."""
        c = self.get(band, sensor)
        if c is None:
            return None
        xs, ys, hs = c["xs"], c["ys"], c["heights"]
        if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
            return None
        ix = int(round((x - xs[0]) / (xs[1] - xs[0])))
        iy = int(round((y - ys[0]) / (ys[1] - ys[0])))
        ih = int(np.argmin(np.abs(hs - z)))
        out = []
        for k in range(c["az_deg"].shape[-1]):
            pt = int(c["ptype"][ih, iy, ix, k])
            if pt == 255:
                continue
            v = int(c["via"][ih, iy, ix, k])
            out.append({"az": float(c["az_deg"][ih, iy, ix, k]), "el": float(c["el_deg"][ih, iy, ix, k]),
                        "gain_db": float(c["gain_db"][ih, iy, ix, k]), "ptype": pt,
                        "via": str(c["names"][v]) if v >= 0 else None,
                        "cell": (float(xs[ix]), float(ys[iy]), float(hs[ih]))})
        out.sort(key=lambda p: -p["gain_db"])
        return out


def triangulate(obs: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """obs: (sensor_x, sensor_y, bearing_deg, sigma_deg). Weighted least-squares intersection of bearing lines.
    Returns (xy, 2x2 covariance in m^2, rms perpendicular residual in m) or None if the geometry is degenerate."""
    if len(obs) < 2:
        return None
    A, b, W = [], [], []
    for sx, sy, brg, sig in obs:
        th = math.radians(brg)
        n = np.array([math.cos(th), -math.sin(th)])          # normal to a line with this compass bearing
        A.append(n)
        b.append(n @ np.array([sx, sy]))
        W.append(1.0 / max(math.radians(sig), 1e-3) ** 2)
    A, b, W = np.array(A), np.array(b), np.array(W)
    M = A.T @ (A * W[:, None])
    if np.linalg.cond(M) > 1e8:
        return None
    xy = np.linalg.solve(M, A.T @ (W * b))
    rng = [math.hypot(xy[0] - sx, xy[1] - sy) for sx, sy, _, _ in obs]
    Wr = np.array([1.0 / max(r * math.radians(s[3]), 1.0) ** 2 for r, s in zip(rng, obs, strict=True)])
    cov = np.linalg.inv(A.T @ (A * Wr[:, None]))
    res = A @ xy - b
    return xy, cov, float(np.sqrt(np.mean(res ** 2)))


def ellipse(cov: np.ndarray) -> tuple[float, float, float]:
    """~2-sigma ellipse: (a_m, b_m, theta_deg = compass bearing of the major axis, 0..180)."""
    w, v = np.linalg.eigh(cov)
    major = v[:, 1]
    return (2 * math.sqrt(max(w[1], 1.0)), 2 * math.sqrt(max(w[0], 1.0)),
            math.degrees(math.atan2(major[0], major[1])) % 180.0)


def bearing_chi2(xy, obs: list[tuple[float, float, float, float]]) -> tuple[float, list[float]]:
    """Normalised bearing residuals at a fix: sum of ((bearing - predicted) / sigma)^2 and the per-bearing |z|.
    obs as in triangulate: (sensor_x, sensor_y, bearing_deg, sigma_deg)."""
    zs = []
    for sx, sy, brg, sig in obs:
        pred = math.degrees(math.atan2(xy[0] - sx, xy[1] - sy)) % 360
        zs.append(abs((brg - pred + 180) % 360 - 180) / max(sig, 1e-3))
    return float(sum(z * z for z in zs)), zs
