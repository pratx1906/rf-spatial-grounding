"""World-side atmospheric attenuation (ITU-R P.676 gas + P.838 rain, table built with ITU-Rpy, D-031)."""

import numpy as np
import pytest

from sim.obs_model.atmosphere import amp_factor, specific_atten_db_per_km

BANDS = ("sub_ghz_868_915", "ism_2400", "ism_5800")


def test_clear_air_is_negligible():
    for b in BANDS:
        assert 0.0 <= specific_atten_db_per_km(b, 0.0, 85.0) < 0.05


def test_rain_orders_by_frequency_and_has_itu_magnitude():
    g = {b: specific_atten_db_per_km(b, 25.0, 85.0) for b in BANDS}
    assert g["ism_5800"] > g["ism_2400"] > g["sub_ghz_868_915"]
    assert 0.05 <= g["ism_5800"] <= 0.3                    # P.838-3: k_H~7e-4, alpha_H~1.6 near 6 GHz


def test_interpolates_between_table_rows():
    lo, mid, hi = (specific_atten_db_per_km("ism_5800", r, 85.0) for r in (5.0, 15.0, 25.0))
    assert lo < mid < hi


def test_amplitude_factor_of_a_1_km_path_matches_the_specific_attenuation():
    w = {"rain_mm_h": 25.0, "rh_pct": 85.0}
    tau_ns = np.array([1000.0 / 0.299792458])              # 1 km
    g = specific_atten_db_per_km("ism_5800", 25.0, 85.0)
    assert amp_factor(tau_ns, "ism_5800", w)[0] == pytest.approx(10 ** (-g / 20), rel=1e-6)
    assert amp_factor(np.array([np.nan]), "ism_5800", w)[0] == 1.0   # empty path slots untouched


def test_no_weather_means_no_change():
    assert np.all(amp_factor(np.array([3000.0, 6000.0]), "ism_2400", None) == 1.0)
