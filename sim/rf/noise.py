"""Receiver noise for the v2 observation model (RF phase 2, Task 5; world side).

thermal:   kTB with the receiver noise figure.
man-made:  ITU-R P.372-16 median man-made noise, Fam = c - d log10(f_MHz) dB above kT0, environments city,
           residential, rural, quiet rural. The Recommendation specifies it for 0.3-250 MHz only; at the AIR7311 ISM
           bands (868 MHz-5.8 GHz) it returns None and the floor is thermal + in-band interference (spectrum.py).
"""

from __future__ import annotations

import math

FAM = {"city": (76.8, 27.7), "residential": (72.5, 27.7), "rural": (67.2, 27.7), "quiet_rural": (53.6, 28.6)}
F_MIN_HZ, F_MAX_HZ = 0.3e6, 250e6


def man_made_fam_db(f_hz: float, env: str = "city") -> float | None:
    if not (F_MIN_HZ <= f_hz <= F_MAX_HZ):
        return None
    c, d = FAM[env]
    return c - d * math.log10(f_hz / 1e6)


def thermal_dbm(bw_hz: float, nf_db: float) -> float:
    return -174.0 + 10 * math.log10(bw_hz) + nf_db


def total_noise_dbm(bw_hz: float, nf_db: float, f_hz: float, env: str = "city") -> float:
    """Receiver noise referred to the antenna: kT0B F + kT0B 10^(Fam/10) (external noise adds as a temperature)."""
    fam = man_made_fam_db(f_hz, env)
    f_lin = 10 ** (nf_db / 10) + (10 ** (fam / 10) if fam is not None else 0.0)
    if fam is None:
        return thermal_dbm(bw_hz, nf_db)
    return -174.0 + 10 * math.log10(bw_hz) + 10 * math.log10(f_lin)
