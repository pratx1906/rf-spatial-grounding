"""Standard-UTD edge diffraction for Sionna RT 2.2 (measured defect + replacement; docs/RESULTS/rf_validation.md).

Measured with scripts/rf_validate.py (cases D-F) on canonical 90-degree wedges:
  * perfectly conducting (steel) wedges: Sionna's field matches the Kouyoumjian-Pathak (KP) *hard* coefficient
    where KP requires *soft* (E parallel to the edge) and vice versa, on horizontal and vertical edges (0.24-0.58 dB
    from the exchanged coefficient);
  * lossy (0.3 m concrete) wedges: Sionna gives the conductor-hard level for both polarisations, 15-20 dB less loss
    deep in the shadow than Luebbers' lossy-wedge UTD.
Cause (sionna/rt/radio_materials/radio_material.py, _diffraction_matrix): the coefficients are assembled as
R_0 (d1 - d4) + R_n (d2 - d3) in rotated face bases, and R_n is forced to -I for exterior angles > pi + phi'
(every 90-degree corner), instead of the standard KP/Luebbers form.

Replacement (applied by `apply()`): Sionna's own geometry, angles, distances, transition terms d1..d4 and slab
reflection coefficients are reused unchanged (Apache-2.0 code, copied below with this attribution); only the
assembly is replaced by the textbook form, in KP's edge-fixed bases (phi', beta0') -> (phi, beta0):
    D_s = d1 + d2 + R_n^TE d3 + R_0^TE d4,   D_h = d1 + d2 + R_n^TM d3 + R_0^TM d4      (Luebbers 1984)
    E_phi = -D_h E_phi',  E_beta = -D_s E_beta'                                          (KP dyadic)
with R_0 at the incident ray's angle on face 0 and R_n at the diffracted ray's angle on face n. For a perfect
conductor (R^TE = -1, R^TM = +1) this is exactly KP. Reflection, refraction, LOS and diffuse paths are untouched.
"""

from __future__ import annotations

PATCH_ID = "sionna-2.2-utd-luebbers-v2"
_applied = False


def _fixed_diffraction_matrix(self, to_world, ki_local, ko_local, si, eta, wavenumber):
    import drjit as dr
    import mitsuba as mi
    from sionna.rt.utils import (cot_times_f_utd, implicit_basis_vector, itu_coefficients_single_layer_slab,
                                 jones_matrix_from_real_imag, jones_matrix_rotator, wedge_interior_angle)

    # ---- geometry and transition terms: verbatim from Sionna RT 2.2 RadioMaterial._diffraction_matrix ----
    e_hat_local = si.dn_du
    nn_local = si.dn_dv
    n0_local = mi.Vector3f(0, 0, 1)
    wedge_angle = wedge_interior_angle(n0_local, nn_local)
    beta0 = dr.safe_acos(dr.abs(dr.dot(ki_local, e_hat_local)))
    exterior_angle = dr.two_pi - wedge_angle
    s = si.dp_du.x
    s_prime = si.dp_du.y
    to_hat_local = dr.normalize(dr.cross(n0_local, e_hat_local))
    ki_local_proj = dr.normalize(ki_local - dr.dot(ki_local, e_hat_local) * e_hat_local)
    ko_local_proj = dr.normalize(ko_local - dr.dot(ko_local, e_hat_local) * e_hat_local)
    phi_prime = dr.pi - dr.safe_acos(-dr.dot(ki_local_proj, to_hat_local))
    phi_prime *= -dr.sign(-dr.dot(ki_local_proj, n0_local))
    phi_prime += dr.pi
    phi = dr.pi - dr.safe_acos(dr.dot(ko_local_proj, to_hat_local))
    phi *= -dr.sign(dr.dot(ko_local_proj, n0_local))
    phi += dr.pi
    l = s * s_prime * dr.rcp(s + s_prime) * dr.sin(beta0) ** 2
    n = exterior_angle * dr.rcp(dr.pi)
    dif_phi = phi - phi_prime
    sum_phi = phi + phi_prime

    def a_p_m(beta):
        n_p = dr.round((beta + dr.pi) * dr.rcp(2 * exterior_angle))
        n_m = dr.round((beta - dr.pi) * dr.rcp(2 * exterior_angle))
        return 2 * dr.cos(exterior_angle * n_p - beta / 2) ** 2, 2 * dr.cos(exterior_angle * n_m - beta / 2) ** 2

    a1, a2 = a_p_m(dif_phi)
    a3, a4 = a_p_m(sum_phi)
    factor = -dr.exp(mi.Complex2f(0, -dr.pi / 4))
    factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * wavenumber) * dr.sin(beta0))
    kl = wavenumber * l
    d1 = factor * cot_times_f_utd((dr.pi + dif_phi) * dr.rcp(2 * n), kl * a1, n, kl)
    d2 = factor * cot_times_f_utd((dr.pi - dif_phi) * dr.rcp(2 * n), kl * a2, n, kl)
    d3 = factor * cot_times_f_utd((dr.pi + sum_phi) * dr.rcp(2 * n), kl * a3, n, kl)
    d4 = factor * cot_times_f_utd((dr.pi - sum_phi) * dr.rcp(2 * n), kl * a4, n, kl)
    # ---- end of verbatim part ----

    wavelength = dr.two_pi / wavenumber
    r_te_0, r_tm_0, _, _ = itu_coefficients_single_layer_slab(dr.abs(dr.dot(ki_local, n0_local)), eta, self._d,
                                                              wavelength)
    r_te_n, r_tm_n, _, _ = itu_coefficients_single_layer_slab(dr.abs(dr.dot(ko_local, nn_local)), eta, self._d,
                                                              wavelength)
    red = dr.sqrt(1. - dr.square(self._s))          # rough-surface reduction, as Sionna
    d_s = d1 + d2 + (r_te_n * red) * d3 + (r_te_0 * red) * d4
    d_h = d1 + d2 + (r_tm_n * red) * d3 + (r_tm_0 * red) * d4

    # KP edge-fixed bases: phi_hat' = (ki x e)/|.|, phi_hat = (e x ko)/|.| (as Sionna); second vectors k x first
    phi_hat_prime = dr.normalize(dr.cross(ki_local, e_hat_local))
    phi_hat = -dr.normalize(dr.cross(ko_local, e_hat_local))
    real = mi.Matrix2f(-d_h.real, 0., 0., -d_s.real)
    imag = mi.Matrix2f(-d_h.imag, 0., 0., -d_s.imag)

    ki_world = to_world @ ki_local
    ko_world = to_world @ ko_local
    theta_i_local = to_world.T @ implicit_basis_vector(ki_world)
    theta_o_local = to_world.T @ implicit_basis_vector(ko_world)
    w_in = jones_matrix_rotator(ki_local, theta_i_local, phi_hat_prime)
    w_out = jones_matrix_rotator(ko_local, phi_hat, theta_o_local)
    return jones_matrix_from_real_imag(w_out @ real @ w_in, w_out @ imag @ w_in)


def apply() -> str:
    """Patch once per process; returns the patch id that every cache manifest records."""
    global _applied
    if not _applied:
        from sionna.rt.radio_materials.radio_material import RadioMaterial

        RadioMaterial._diffraction_matrix = _fixed_diffraction_matrix
        _applied = True
    return PATCH_ID
