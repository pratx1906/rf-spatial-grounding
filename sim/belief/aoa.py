"""BELIEF side: nominal predicted angles of arrival from the site-wide Sionna RT cache (what predict_aoa reads).

The cache (`<rf_dir>/sionna_rt/<band>/aoa_<sensor>.npz`, 40 m cells over +-3.2 km, heights 5-120 m, top-24 paths
per cell with gain, delay, arrival direction, interaction type and first object hit) is unpacked once into
memory-mapped arrays (`<band>/<sensor>/<key>.npy`) so a lookup touches only one cell (tool latency, RAM).

Paths are grouped the way the AIR7311 resolves them: arrivals within RESOLVE_DEG merge into one predicted peak
(power-summed, labelled by its strongest path). Peaks more than FLOOR_DB below the strongest are dropped.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np

from sim.geometry import angle_diff_deg, bearing_deg

RESOLVE_DEG = 8.0
FLOOR_DB = 15.0
MAX_PEAKS = 4
KIND = {0: "direct", 1: "reflection", 2: "scatter", 8: "diffraction"}
KEYS = ("az_deg", "el_deg", "gain_db", "tau_ns", "ptype", "via", "h_re", "h_im")
_LOCK = threading.Lock()


def unpack(npz: Path, out: Path) -> None:
    """One-time conversion of a cache .npz into per-key .npy files (memory-mappable)."""
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(npz)
    for k in z.files:
        np.save(out / f"{k}.npy", z[k])


class AoAPredictor:
    def __init__(self, site, backend: str = "sionna_rt"):
        self.root = Path(site.rf_dir) / backend
        self.sensors = site.sensors
        self._c: dict[tuple[str, str], dict | None] = {}

    def available(self) -> bool:
        return (self.root / "manifest.json").exists()

    def cache(self, band: str, sensor: str) -> dict | None:
        key = (band, sensor)
        if key in self._c:
            return self._c[key]
        with _LOCK:
            if key in self._c:
                return self._c[key]
            npz = self.root / band / f"aoa_{sensor}.npz"
            d = self.root / band / sensor
            if not npz.exists():
                self._c[key] = None
                return None
            if not (d / "gain_db.npy").exists():
                unpack(npz, d)
            c = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in KEYS}
            for k in ("xs", "ys", "heights", "names"):
                c[k] = np.load(d / f"{k}.npy")
            self._c[key] = c
            return c

    def cell(self, c: dict, x: float, y: float, z: float) -> tuple[int, int, int] | None:
        xs, ys, hs = c["xs"], c["ys"], c["heights"]
        step = xs[1] - xs[0]
        if not (xs[0] - step / 2 <= x <= xs[-1] + step / 2 and ys[0] - step / 2 <= y <= ys[-1] + step / 2):
            return None
        ix = int(np.clip(round((x - xs[0]) / step), 0, len(xs) - 1))
        iy = int(np.clip(round((y - ys[0]) / step), 0, len(ys) - 1))
        ih = int(np.argmin(np.abs(hs - z)))
        return ih, iy, ix

    def paths(self, x: float, y: float, z: float, sensor: str, band: str) -> list[dict] | None:
        """Raw cached paths at the cell, strongest first (None outside the cache)."""
        c = self.cache(band, sensor)
        if c is None:
            return None
        cell = self.cell(c, x, y, z)
        if cell is None:
            return None
        ih, iy, ix = cell
        pt = np.asarray(c["ptype"][ih, iy, ix])
        out = []
        for k in np.flatnonzero(pt != 255):
            v = int(c["via"][ih, iy, ix, k])
            out.append({"az": float(c["az_deg"][ih, iy, ix, k]), "el": float(c["el_deg"][ih, iy, ix, k]),
                        "gain_db": float(c["gain_db"][ih, iy, ix, k]), "tau_ns": float(c["tau_ns"][ih, iy, ix, k]),
                        "kind": KIND.get(int(pt[k]), "scatter"), "via": str(c["names"][v]) if v >= 0 else None})
        out.sort(key=lambda p: -p["gain_db"])
        return out

    def peaks(self, x: float, y: float, z: float, sensor: str, band: str) -> list[dict] | None:
        """Predicted resolvable peaks at the sensor for an emitter at (x, y, z): strongest first,
        [{bearing_deg, el_deg, rel_power_db, kind, via}]. None outside the cache; [] if no path reaches it."""
        ps = self.paths(x, y, z, sensor, band)
        if ps is None:
            return None
        pose = self.sensors[sensor]["pose"]
        groups: list[dict] = []
        for p in ps:
            az = p["az"]
            if p["kind"] == "direct":           # the cell centre is up to 28 m off: use exact geometry
                az = bearing_deg(pose["x_m"], pose["y_m"], x, y)
            lin = 10 ** (p["gain_db"] / 10)
            for g in groups:
                if abs(angle_diff_deg(az, g["az"])) < RESOLVE_DEG:
                    g["lin"] += lin
                    break
            else:
                groups.append({"az": az, "el": p["el"], "lin": lin, "kind": p["kind"], "via": p["via"]})
        if not groups:
            return []
        top = max(g["lin"] for g in groups)
        out = []
        for g in sorted(groups, key=lambda g: -g["lin"]):
            rel = 10 * math.log10(g["lin"] / top)
            if rel < -FLOOR_DB or len(out) == MAX_PEAKS:
                break
            out.append({"bearing_deg": round(g["az"] % 360.0, 1), "el_deg": round(g["el"], 1),
                        "rel_power_db": round(rel, 1), "kind": g["kind"], "via": g["via"]})
        return out
