"""Klein & Swift (1977) sea-water permittivity (single Debye; standard 1-10 GHz) used for the v2 RF scenes."""

import pytest

from sim.rf.materials import sea_water


def test_static_conductivity_and_pure_water_static_permittivity():
    assert sea_water(1e6, 25.0, 35.0)[1] == pytest.approx(5.30, abs=0.05)       # S/m, 25 C, 35 psu
    eps_pure, _ = sea_water(1e3, 25.0, 0.0)
    assert eps_pure == pytest.approx(78.2, abs=0.4)                              # pure water, 25 C


@pytest.mark.parametrize("f_hz,eps_lo,eps_hi,sig_lo,sig_hi", [
    (0.9e9, 69.5, 72.0, 5.2, 5.8),
    (2.44e9, 68.5, 71.0, 5.8, 7.0),
    (5.8e9, 63.5, 67.5, 10.0, 12.5),
])
def test_band_values_include_dipolar_loss(f_hz, eps_lo, eps_hi, sig_lo, sig_hi):
    eps, sig = sea_water(f_hz, 25.0, 35.0)
    assert eps_lo < eps < eps_hi and sig_lo < sig < sig_hi
