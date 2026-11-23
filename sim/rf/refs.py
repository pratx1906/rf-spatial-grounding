"""Closed-form RF references for the validation suite (docs/RESULTS/rf_validation.md).

Conventions: angles of incidence are measured from the surface normal; bearings from +y clockwise (sim.geometry);
losses are positive dB; complex permittivity eps_c = eps_r - j sigma / (omega eps0) (ITU-R P.2040 form).
numpy only (the locked WSL venv has no scipy).
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12


def fspl_db(d_m, f_hz):
    """Free-space path loss 20 log10(4 pi d f / c) (Friis, isotropic antennas)."""
    return 20.0 * np.log10(4.0 * math.pi * np.asarray(d_m, dtype=float) * f_hz / C0)


def fresnel_cs(v):
    """Fresnel integrals C(v) = int_0^v cos(pi t^2/2) dt, S(v) = int_0^v sin(pi t^2/2) dt.
    Simpson's rule for |v| <= 5 (error < 1e-7), the leading asymptotic terms beyond."""
    v = np.atleast_1d(np.asarray(v, dtype=float))
    c, s = np.empty_like(v), np.empty_like(v)
    for i, x in enumerate(v):
        a = abs(x)
        if a <= 5.0:
            n = 20_000
            t = np.linspace(0.0, a, n + 1)
            w = np.ones(n + 1)
            w[1:-1:2], w[2:-1:2] = 4.0, 2.0
            h = a / n / 3.0
            ci, si = h * (w * np.cos(math.pi * t * t / 2)).sum(), h * (w * np.sin(math.pi * t * t / 2)).sum()
        else:
            ph = math.pi * a * a / 2
            ci, si = 0.5 + math.sin(ph) / (math.pi * a), 0.5 - math.cos(ph) / (math.pi * a)
        c[i], s[i] = math.copysign(ci, x), math.copysign(si, x)
    return c, s


def knife_edge_db(v):
    """Single knife-edge diffraction loss J(v), ITU-R P.526-15 eq. (26) with exact Fresnel integrals."""
    c, s = fresnel_cs(v)
    j = -20.0 * np.log10(np.sqrt((1 - c - s) ** 2 + (c - s) ** 2) / 2.0)
    return float(j[0]) if np.ndim(v) == 0 else j


def fresnel_gamma(eps_r: float, sigma: float, f_hz: float, theta_deg: float) -> tuple[complex, complex]:
    """Plane-wave reflection coefficients of a half-space: (TE / perpendicular, TM / parallel).
    Over horizontal ground TE is horizontal polarisation and TM vertical."""
    ec = complex(eps_r, -sigma / (2 * math.pi * f_hz * EPS0))
    th = math.radians(theta_deg)
    ct, st2 = math.cos(th), math.sin(th) ** 2
    root = np.sqrt(ec - st2)
    g_te = (ct - root) / (ct + root)
    g_tm = (ec * ct - root) / (ec * ct + root)
    return complex(g_te), complex(g_tm)


def two_ray_db(d_m, ht_m: float, hr_m: float, f_hz: float, eps_r: float, sigma: float, pol: str = "V"):
    """Flat-earth two-ray path loss (direct + ground-reflected, Fresnel coefficient), isotropic antennas."""
    d = np.asarray(d_m, dtype=float)
    lam = C0 / f_hz
    k = 2 * math.pi / lam
    r1, r2 = np.hypot(d, ht_m - hr_m), np.hypot(d, ht_m + hr_m)
    theta = 90.0 - np.degrees(np.arctan2(ht_m + hr_m, d))
    g = np.array([fresnel_gamma(eps_r, sigma, f_hz, t)[1 if pol == "V" else 0] for t in np.atleast_1d(theta)])
    e = lam / (4 * math.pi) * (np.exp(-1j * k * r1) / r1 + g.reshape(d.shape) * np.exp(-1j * k * r2) / r2)
    return -20.0 * np.log10(np.abs(e))


def crb_bearing_deg(pos_xy, lam: float, az_deg: float, snr_db: float, n_snap: int) -> float:
    """Deterministic Cramer-Rao bound (std, deg) on the azimuth of one source for a planar array in the
    horizontal plane: var = 1 / (2 N SNR sum_m (dphi_m/daz - mean)^2), phi_m = k (x sin az + y cos az)."""
    p = np.asarray(pos_xy, dtype=float)
    k = 2 * math.pi / lam
    a = math.radians(az_deg)
    dphi = k * (p[:, 0] * math.cos(a) - p[:, 1] * math.sin(a))
    dphi -= dphi.mean()
    var = 1.0 / (2.0 * n_snap * 10 ** (snr_db / 10) * float((dphi ** 2).sum()))
    return math.degrees(math.sqrt(var))


def slab_gamma(eps_r: float, sigma: float, f_hz: float, theta_deg: float, d_m: float) -> tuple[complex, complex]:
    """Reflection coefficients (TE, TM) of a homogeneous slab of thickness d in free space, ITU-R P.2040-3
    eq. (37)-(38) form: Gamma = G (1 - e^{-j2q}) / (1 - G^2 e^{-j2q}), q = k0 d sqrt(eps_c - sin^2 theta).
    This is the model Sionna RT applies to a radio material with a thickness."""
    ec = complex(eps_r, -sigma / (2 * math.pi * f_hz * EPS0))
    th = math.radians(theta_deg)
    q = 2 * math.pi * f_hz / C0 * d_m * np.sqrt(ec - math.sin(th) ** 2)
    ph = np.exp(-2j * q)
    return tuple(complex(g * (1 - ph) / (1 - g * g * ph)) for g in fresnel_gamma(eps_r, sigma, f_hz, theta_deg))


def utd_transition(x: float) -> complex:
    """UTD transition function F(x) = 2j sqrt(x) e^{jx} int_{sqrt x}^inf e^{-j tau^2} d tau (Kouyoumjian & Pathak
    1974), via Fresnel integrals: int_a^inf e^{-j tau^2} = sqrt(pi/2) [(1/2 - C(u)) - j (1/2 - S(u))], u = a sqrt(2/pi)."""
    if x <= 0:
        return 0j
    a = math.sqrt(x)
    u = a * math.sqrt(2 / math.pi)
    c, s = fresnel_cs(u)
    tail = math.sqrt(math.pi / 2) * complex(0.5 - c[0], -(0.5 - s[0]))
    return 2j * a * complex(math.cos(x), math.sin(x)) * tail


def _utd_a(beta: float, n: float, sign: int) -> float:
    N = round((beta + sign * math.pi) / (2 * n * math.pi))
    return 2 * math.cos((2 * n * math.pi * N - beta) / 2) ** 2


def utd_wedge_d(n: float, phi: float, phi_p: float, L: float, k: float, hard: bool) -> complex:
    """UTD diffraction coefficient of a perfectly conducting wedge with exterior angle n*pi (normal incidence on
    the edge, beta0 = 90 deg). phi, phi' measured from the 0-face; soft (E parallel to the edge) or hard."""
    pre = -complex(math.cos(-math.pi / 4), math.sin(-math.pi / 4)) / (2 * n * math.sqrt(2 * math.pi * k))

    def term(beta: float) -> complex:
        t = 0j
        for sgn in (+1, -1):
            arg = (math.pi + sgn * beta) / (2 * n)
            cot = math.cos(arg) / math.sin(arg) if abs(math.sin(arg)) > 1e-12 else 0.0
            t += cot * utd_transition(k * L * _utd_a(beta, n, sgn))
        return t

    return pre * (term(phi - phi_p) + (1 if hard else -1) * term(phi + phi_p))


def utd_point_source_field(n: float, phi: float, phi_p: float, sp: float, s: float, k: float, hard: bool) -> complex:
    """Diffracted field of a unit isotropic point source (incident e^{-jks'}/s') at distance s' from the edge,
    observed at s: E_d = E_i(Q) D sqrt(s' / (s (s + s'))) e^{-jks}, L = s s' / (s + s')."""
    L = s * sp / (s + sp)
    e_q = complex(math.cos(-k * sp), math.sin(-k * sp)) / sp
    return e_q * utd_wedge_d(n, phi, phi_p, L, k, hard) * math.sqrt(sp / (s * (s + sp))) \
        * complex(math.cos(-k * s), math.sin(-k * s))


def luebbers_d(n: float, phi: float, phi_p: float, L: float, k: float, r0: complex, rn: complex) -> complex:
    """Heuristic UTD coefficient of a lossy wedge (Luebbers 1984): the two reflection-boundary terms are weighted by
    the face reflection coefficients R_0 (face 0, grazing angle phi') and R_n (face n, grazing angle n pi - phi).
    Soft polarisation uses R_TE, hard R_TM; for a perfect conductor (R = -1 / +1) this is Kouyoumjian-Pathak."""
    pre = -complex(math.cos(-math.pi / 4), math.sin(-math.pi / 4)) / (2 * n * math.sqrt(2 * math.pi * k))

    def ct(arg: float, beta: float, sgn: int) -> complex:
        c = math.cos(arg) / math.sin(arg) if abs(math.sin(arg)) > 1e-12 else 0.0
        return c * utd_transition(k * L * _utd_a(beta, n, sgn))

    bm, bp = phi - phi_p, phi + phi_p
    return pre * (ct((math.pi + bm) / (2 * n), bm, +1) + ct((math.pi - bm) / (2 * n), bm, -1)
                  + rn * ct((math.pi + bp) / (2 * n), bp, +1) + r0 * ct((math.pi - bp) / (2 * n), bp, -1))
