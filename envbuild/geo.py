"""WGS-84 <-> local ENU (metres) with exact ellipsoid formulas (ENV spec 2.1, 4). No external dependencies.

ENU axes: x = east, y = north, z = up, origin at (lat0, lon0, h0). Bearings from ENU are relative to TRUE north
(the local tangent plane is aligned with the meridian), so no UTM grid-convergence correction is needed here.
"""

from __future__ import annotations

import numpy as np

A = 6378137.0                 # WGS-84 semi-major axis
F = 1 / 298.257223563
E2 = F * (2 - F)


def geodetic_to_ecef(lat, lon, h=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - E2) + h) * np.sin(lat)
    return x, y, z


def ecef_to_geodetic(x, y, z):
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(6):                      # converges to < 1e-9 rad in a few iterations
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


class ENU:
    def __init__(self, lat0: float, lon0: float, h0: float = 0.0):
        self.lat0, self.lon0, self.h0 = lat0, lon0, h0
        self.x0, self.y0, self.z0 = geodetic_to_ecef(lat0, lon0, h0)
        la, lo = np.radians(lat0), np.radians(lon0)
        self.R = np.array([[-np.sin(lo), np.cos(lo), 0.0],
                           [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                           [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])

    def to_enu(self, lat, lon, h=0.0):
        x, y, z = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float))
        d = np.stack([x - self.x0, y - self.y0, z - self.z0])
        e, n, u = np.tensordot(self.R, d, axes=1)
        return e, n, u

    def to_geodetic(self, e, n, u=0.0):
        e, n, u = (np.asarray(v, float) for v in (e, n, u))
        d = np.tensordot(self.R.T, np.stack([e, n, np.broadcast_to(u, np.shape(e))]), axes=1)
        return ecef_to_geodetic(d[0] + self.x0, d[1] + self.y0, d[2] + self.z0)

    def meta(self) -> dict:
        return {"lat": self.lat0, "lon": self.lon0, "alt_m": self.h0, "datum": "WGS-84 ellipsoid",
                "frame": "local ENU tangent plane at origin; bearings relative to true north"}
