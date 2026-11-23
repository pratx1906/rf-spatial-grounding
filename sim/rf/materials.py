"""Frequency-dependent material models for the v2 RF scenes (world and belief share them).

sea_water: Klein & Swift (1977), "An improved model for the dielectric constant of sea water at microwave
frequencies", IEEE Trans. Antennas Propag. 25(1): single-Debye relaxation with temperature- and salinity-dependent
static permittivity, relaxation time and ionic conductivity; eps_inf = 4.9. Valid about 1-10 GHz (used at 0.9 GHz
with the ionic term dominating). Returns Sionna's (relative permittivity, effective conductivity) pair:
eps' and sigma_eff = omega eps0 eps'' (dipolar + ionic loss).
"""

from __future__ import annotations

import math

EPS0 = 8.8541878128e-12


def sea_water(f_hz: float, t_c: float = 25.0, s_psu: float = 35.0) -> tuple[float, float]:
    T, S = t_c, s_psu
    eps_s_t = 87.134 - 1.949e-1 * T - 1.276e-2 * T ** 2 + 2.491e-4 * T ** 3
    a = 1 + 1.613e-5 * T * S - 3.656e-3 * S + 3.210e-5 * S ** 2 - 4.232e-7 * S ** 3
    tau_t = 1.768e-11 - 6.086e-13 * T + 1.104e-14 * T ** 2 - 8.111e-17 * T ** 3
    b = 1 + 2.282e-5 * T * S - 7.638e-4 * S - 7.760e-6 * S ** 2 + 1.105e-8 * S ** 3
    sig25 = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S ** 2 - 1.28205e-7 * S ** 3)
    d = 25.0 - T
    phi = d * (2.033e-2 + 1.266e-4 * d + 2.464e-6 * d ** 2 - S * (1.849e-5 - 2.551e-7 * d + 2.551e-8 * d ** 2))
    sig = sig25 * math.exp(-phi)
    eps_s, tau, eps_inf = eps_s_t * a, tau_t * b, 4.9
    w = 2 * math.pi * f_hz
    deb = (eps_s - eps_inf) / complex(1.0, w * tau)
    eps_r = eps_inf + deb.real
    eps_im = -deb.imag + sig / (w * EPS0)
    return float(eps_r), float(eps_im * w * EPS0)


# ---- Sea roughness (demo-v2 S7.4, audit A Q7) ----
# Sea state (WMO / Douglas code 0-4) -> representative significant wave height H_1/3 (mid of the WMO band; sea
# state 1 = 0-0.1 m, 2 = 0.1-0.5 m, 3 = 0.5-1.25 m, 4 = 1.25-2.5 m) and rms surface height sigma_h = H_1/3 / 4
# (Gaussian sea, H_1/3 = 4 sigma_h). Values are ESTIMATED for a sheltered creek; no Kandla wave measurements.
SEA_STATE_HS_M = {0: 0.0, 1: 0.05, 2: 0.3, 3: 0.875, 4: 1.875}
C0 = 299792458.0


def sea_state_sigma_h(sea_state: int) -> float:
    """rms wave height sigma_h [m] for a sea state 0-4 (SEA_STATE_HS_M / 4)."""
    if int(sea_state) not in SEA_STATE_HS_M:
        raise ValueError(f"sea state must be 0-4, got {sea_state}")
    return SEA_STATE_HS_M[int(sea_state)] / 4.0


def ament_rho(sin_psi, sigma_h_m: float, f_hz: float):
    """Ament specular-reduction (field) factor of a rough surface with Gaussian heights:
    rho = exp(-2 (2 pi sigma_h sin psi / lambda)^2) = exp(-g / 2), g = (4 pi sigma_h sin psi / lambda)^2 the Rayleigh
    roughness parameter (Ament 1953; Beckmann & Spizzichino 1963). psi is the grazing angle. Multiplies the smooth
    (Fresnel / slab) reflection coefficient; the scattered remainder is diffuse and not modelled here."""
    import numpy as np

    lam = C0 / f_hz
    x = 2 * math.pi * sigma_h_m * np.asarray(sin_psi, float) / lam
    return np.exp(-2.0 * x ** 2)
