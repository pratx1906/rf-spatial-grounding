"""Spectrum environment at the AIR7311 nodes (RF phase 2, Task 5)."""

import math
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
ENV = SITE / "world_cache_v2" / "spectrum_env.yaml"
CACHE = SITE / "rf_cache_v2"
pytestmark = pytest.mark.skipif(not (ENV.exists() and (CACHE / "air7311_S1" / "spec_diff" / "ism_5800.npz").exists()),
                                reason="v2 spectrum env / caches not built")


@pytest.fixture(scope="module")
def sw():
    from sim.rf.spectrum import SpectrumWorld

    hw = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
    sens = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]
            if s["type"] == "fixed_df"}
    return SpectrumWorld(ENV, CACHE, hw, sens)


def _always_on(sw, monkeypatch):
    monkeypatch.setattr(type(sw), "active", lambda self, e, t, seed: True)


def test_cctv_on_the_same_mast_desensitises_its_own_channel_not_the_next(sw, monkeypatch):
    _always_on(sw, monkeypatch)
    kt = -174 + 10 * math.log10(20e6) + 3.0
    n_same, ep = sw.noise_dbm("S1", "ism_5800", 5745.0, 20.0, 0.0, 1)
    n_other, _ = sw.noise_dbm("S1", "ism_5800", 5825.0, 20.0, 0.0, 1)
    assert "fe_cctv_1" in ep["active"]
    assert n_same - kt > 30                        # co-channel link on the same mast: tens of dB of desense
    assert n_other - kt < n_same - kt - 20         # another channel: only the ACS-filtered leakage


def test_lte_b40_carrier_is_in_window_but_off_the_ism_channels(sw, monkeypatch):
    _always_on(sw, monkeypatch)
    ep = sw.epoch("S2", "ism_2400", 2462.0, 20.0, 0.0, 1)
    assert "bg_lte_b40_2" in ep["active"]          # inside the 2391.75-2491.75 MHz tuning window
    co_only = sw.epoch("S2", "ism_2400", 2412.0, 20.0, 0.0, 1)
    assert co_only["i_adjacent_dbm"] > -math.inf


def test_out_of_range_emitters_are_ignored_and_duty_cycles_are_respected(sw):
    from sim.rf.spectrum import band_of

    assert band_of(156.8) is None and not sw.in_range["fe_vhf_1"]
    on = sum(sw.active(next(e for e in sw.em if e.id == "fe_lora_1"), t * 10.0, 3) for t in range(2000))
    assert 0.005 < on / 2000 < 0.05                 # duty 0.02


def test_a_strong_co_located_source_blocks_the_node(sw):
    ep = sw.epoch("S3", "ism_2400", 2437.0, 20.0, 0.0, 1, extra=[("jammer", 2437.0, 40.0, 10.0)])
    assert ep["blocked"] and any("compression" in r for r in ep["reasons"])


def test_pulsed_radar_blanks_samples_instead_of_blocking(sw):
    ep = sw.epoch("S1", "ism_2400", 2437.0, 20.0, 0.0, 1)
    radar = next(e for e in sw.em if e.kind == "sband_radar")
    p = radar.eirp_dbm - 2.0 + sw.gain_db[(radar.id, "S1")]
    if p > sw.fe["p1db_lna_dbm"] + sw.fe["out_of_window_rejection_db"]:     # preselector first
        assert ep["blanked_fraction"] == pytest.approx(radar.duty)
    assert not any("sband" in r or radar.id in r for r in ep["reasons"])
