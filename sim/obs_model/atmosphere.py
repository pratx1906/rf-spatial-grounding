"""World-side atmospheric attenuation for the RF living world (D-031).

Specific attenuation (dB/km) = ITU-R P.676 gaseous + ITU-R P.838 rain, from configs/atmos_table.json (built with
the open-source ITU-Rpy: scripts/atmos_table.py), interpolated linearly in rain rate and relative humidity.
Over this site's <= 1.5 km node caches the effect is small (about 0.17 dB at 5.8 GHz in 25 mm/h rain) and nearly
equal across paths, so it shifts SNR slightly and does not move DF peaks; the belief side assumes clear air.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE = Path(__file__).resolve().parents[2] / "configs" / "atmos_table.json"
C_M_PER_NS = 0.299792458


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(TABLE.read_text())


@lru_cache(maxsize=4096)
def specific_atten_db_per_km(band: str, rain_mm_h: float, rh_pct: float) -> float:
    t = _table()
    grid = np.asarray(t["bands"][band], float)                  # [rain, rh]
    rain, rh = np.asarray(t["rain_mm_h"], float), np.asarray(t["rh_pct"], float)
    by_rh = np.array([np.interp(rh_pct, rh, row) for row in grid])
    return float(np.interp(rain_mm_h, rain, by_rh))


def amp_factor(tau_ns: np.ndarray, band: str, weather: dict | None) -> np.ndarray:
    """Multiplicative factor on the complex path amplitude: 10^(-L/20), L = gamma * path length (from delay).
    Empty path slots (NaN delay) and no weather give 1."""
    tau = np.asarray(tau_ns, float)
    if not weather:
        return np.ones_like(tau)
    g = specific_atten_db_per_km(band, float(weather.get("rain_mm_h", 0.0)), float(weather.get("rh_pct", 60.0)))
    km = np.where(np.isfinite(tau), tau * C_M_PER_NS / 1000.0, 0.0)
    return 10.0 ** (-g * km / 20.0)
