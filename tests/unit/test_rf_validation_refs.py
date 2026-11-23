"""Closed-form RF references used by the validation suite (Task 1, docs/RESULTS/rf_validation.md).
Each value below is a textbook number, so a wrong formula fails here before it can 'validate' the simulator."""

import math

import numpy as np
import pytest

from sim.rf import refs


def test_friis_free_space_loss_at_1km_2g44():
    assert refs.fspl_db(1000.0, 2.44e9) == pytest.approx(100.19, abs=0.02)
    assert refs.fspl_db(2000.0, 2.44e9) - refs.fspl_db(1000.0, 2.44e9) == pytest.approx(6.02, abs=0.01)


def test_fresnel_integrals_limits_and_known_point():
    c, s = refs.fresnel_cs(np.array([0.0, 1.0, 50.0]))
    assert c[0] == pytest.approx(0.0) and s[0] == pytest.approx(0.0)
    assert c[1] == pytest.approx(0.7798934, abs=1e-5) and s[1] == pytest.approx(0.4382591, abs=1e-5)
    assert c[2] == pytest.approx(0.5, abs=0.01) and s[2] == pytest.approx(0.5, abs=0.01)


def test_knife_edge_loss_itu_p526():
    assert refs.knife_edge_db(0.0) == pytest.approx(6.02, abs=0.02)              # grazing: half the field
    assert refs.knife_edge_db(-3.0) == pytest.approx(0.0, abs=0.6)               # well clear: ~free space
    for v in (0.5, 1.0, 2.0, 3.0):                                              # P.526 eq. 31 approximation
        approx = 6.9 + 20 * math.log10(math.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
        assert refs.knife_edge_db(v) == pytest.approx(approx, abs=0.6)


def test_fresnel_reflection_lossless_dielectric():
    eps = 4.0
    g_te, g_tm = refs.fresnel_gamma(eps, 0.0, 1e9, 0.0)                         # normal incidence
    assert g_te == pytest.approx(-1 / 3, abs=1e-9) and abs(g_tm) == pytest.approx(1 / 3, abs=1e-9)
    th_b = math.degrees(math.atan(math.sqrt(eps)))                               # Brewster, from the normal
    _, g = refs.fresnel_gamma(eps, 0.0, 1e9, th_b)
    assert abs(g) < 1e-9
    g_te, g_tm = refs.fresnel_gamma(eps, 0.0, 1e9, 89.99)                         # grazing -> -1 for both
    assert abs(g_te) == pytest.approx(1.0, abs=1e-3) and abs(g_tm) == pytest.approx(1.0, abs=1e-3)


def test_good_conductor_reflects_almost_everything():
    g_te, g_tm = refs.fresnel_gamma(1.0, 1e7, 2.44e9, 30.0)
    assert abs(g_te) > 0.999 and abs(g_tm) > 0.999


def test_crb_square_array_is_isotropic_and_matches_closed_form():
    lam = 3e8 / 2.44e9
    d = 0.45 * lam
    pos = np.array([[-d / 2, -d / 2], [d / 2, -d / 2], [d / 2, d / 2], [-d / 2, d / 2]])
    stds = [refs.crb_bearing_deg(pos, lam, az, snr_db=10.0, n_snap=256) for az in (0.0, 33.0, 90.0, 217.0)]
    assert max(stds) - min(stds) < 1e-9
    expect = math.degrees(1 / math.sqrt(2 * 256 * 10.0) / (2 * math.pi / lam * d))
    assert stds[0] == pytest.approx(expect, rel=1e-6)                            # 0.283 deg


def test_two_ray_over_sea_deep_nulls_in_h_pol_shallow_in_v_pol():
    """Over sea water (eps 70, sigma 4 S/m) |Gamma_H| ~ 1 at every grazing angle -> deep nulls; |Gamma_V| has a
    pseudo-Brewster minimum near 7 deg grazing, so V-pol nulls are shallow at these geometries."""
    f = 2.44e9
    d = np.linspace(50, 3000, 4000)
    fs = refs.fspl_db(np.hypot(d, 15.0), f)
    pl_h = refs.two_ray_db(d, 25.0, 40.0, f, eps_r=70.0, sigma=4.0, pol="H")
    pl_v = refs.two_ray_db(d, 25.0, 40.0, f, eps_r=70.0, sigma=4.0, pol="V")
    for pl in (pl_h, pl_v):
        assert np.all(pl > fs - 6.1)                                             # at most +6 dB constructive
    assert (pl_h - fs).max() > 15
    assert (pl_v - fs).max() < (pl_h - fs).max() - 5
    g = [abs(refs.fresnel_gamma(70.0, 4.0, f, 90 - psi)[1]) for psi in np.linspace(1, 30, 300)]
    assert 4 < np.linspace(1, 30, 300)[int(np.argmin(g))] < 10                   # pseudo-Brewster ~ 7 deg grazing


def test_slab_reflection_limits():
    f = 2.44e9
    half = refs.fresnel_gamma(70.0, 5.0, f, 60.0)
    thick = refs.slab_gamma(70.0, 5.0, f, 60.0, 10.0)                           # lossy and thick -> half-space
    assert abs(thick[0] - half[0]) < 1e-6 and abs(thick[1] - half[1]) < 1e-6
    thin = refs.slab_gamma(5.0, 0.0, f, 30.0, 1e-9)                              # vanishing slab reflects nothing
    assert abs(thin[0]) < 1e-6 and abs(thin[1]) < 1e-6
    lam_in = refs.C0 / f / math.sqrt(4.0)                                           # half-wave lossless slab, normal
    hw = refs.slab_gamma(4.0, 0.0, f, 0.0, lam_in / 2)
    assert abs(hw[0]) < 1e-3                                                     # transparent (Fabry-Perot)


def test_utd_transition_function_limits():
    assert abs(refs.utd_transition(1e4) - 1.0) < 1e-3                          # far from boundaries: F -> 1
    x = 1e-4                                       # small argument: (sqrt(pi x) - 2x e^{j pi/4}) e^{j(pi/4 + x)}
    e4 = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    expect = (math.sqrt(math.pi * x) - 2 * x * e4) * complex(math.cos(math.pi / 4 + x), math.sin(math.pi / 4 + x))
    assert abs(refs.utd_transition(x) - expect) < 0.01 * abs(expect)


def test_utd_reduces_to_keller_gtd_away_from_boundaries():
    n, k, L = 1.5, 2 * math.pi / 0.123, 250.0
    phi_p, phi = math.radians(30.0), math.radians(230.0)                         # deep shadow, far from SSB/RSB
    for hard in (False, True):
        d = refs.utd_wedge_d(n, phi, phi_p, L, k, hard)
        s = -1 if not hard else 1
        bm, bp = phi - phi_p, phi + phi_p
        keller = (complex(math.cos(-math.pi / 4), math.sin(-math.pi / 4)) * math.sin(math.pi / n)
                  / (n * math.sqrt(2 * math.pi * k))
                  * (1 / (math.cos(math.pi / n) - math.cos(bm / n)) + s / (math.cos(math.pi / n) - math.cos(bp / n))))
        assert abs(d - keller) < 0.02 * abs(keller)


def test_utd_half_plane_total_field_is_continuous_across_the_shadow_boundary():
    """n = 2 (knife edge): incident + diffracted just inside the lit side equals diffracted just inside the shadow."""
    lam = 0.123
    k = 2 * math.pi / lam
    sp, s = 500.0, 500.0
    phi_p = math.radians(20.0)
    eps = math.radians(0.0005)       # the O(eps) correction of the singular term flips sign across the SSB
    tot = []
    for phi, lit in ((math.pi + phi_p - eps, True), (math.pi + phi_p + eps, False)):
        e_d = refs.utd_point_source_field(2.0, phi, phi_p, sp, s, k, hard=False)
        e_i = complex(math.cos(-k * (sp + s)), math.sin(-k * (sp + s))) / (sp + s) if lit else 0
        tot.append(e_i + e_d)
    assert abs(abs(tot[0]) - abs(tot[1])) < 0.02 * abs(tot[0])
    assert abs(tot[1]) * (sp + s) == pytest.approx(0.5, abs=0.02)               # -6 dB at the boundary


def test_luebbers_reduces_to_kouyoumjian_pathak_for_a_perfect_conductor():
    n, k = 1.5, 2 * math.pi / 0.123
    phi_p, phi, L = math.radians(5.0), math.radians(200.0), 250.0
    soft = refs.luebbers_d(n, phi, phi_p, L, k, r0=-1.0, rn=-1.0)
    hard = refs.luebbers_d(n, phi, phi_p, L, k, r0=1.0, rn=1.0)
    assert abs(soft - refs.utd_wedge_d(n, phi, phi_p, L, k, hard=False)) < 1e-12
    assert abs(hard - refs.utd_wedge_d(n, phi, phi_p, L, k, hard=True)) < 1e-12
