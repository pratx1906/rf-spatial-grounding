"""ITU-R P.372 man-made noise (RF phase 2, Task 5): Fam = c - d log10(f_MHz), specified 0.3-250 MHz only."""

import pytest

from sim.rf import noise


@pytest.mark.parametrize("env,f_mhz,expect", [("city", 100.0, 21.4), ("residential", 100.0, 17.1),
                                               ("rural", 200.0, 3.47), ("quiet_rural", 100.0, -3.6)])
def test_fam_curve_values(env, f_mhz, expect):
    assert noise.man_made_fam_db(f_mhz * 1e6, env) == pytest.approx(expect, abs=0.05)


def test_outside_the_specified_range_returns_none():
    assert noise.man_made_fam_db(2.44e9, "city") is None        # P.372 does not specify it at ISM bands
    assert noise.man_made_fam_db(0.1e6, "city") is None


def test_noise_power_adds_external_noise_to_the_receiver_figure():
    kt = noise.thermal_dbm(1e6, 3.0)                            # -174 + 60 + 3
    assert kt == pytest.approx(-111.0, abs=0.01)
    assert noise.total_noise_dbm(1e6, 3.0, 2.44e9, "city") == pytest.approx(kt, abs=1e-9)   # ISM: thermal only
    hi = noise.total_noise_dbm(1e6, 3.0, 100e6, "city")          # 21.4 dB external dominates the 3 dB NF
    assert hi > kt + 15
