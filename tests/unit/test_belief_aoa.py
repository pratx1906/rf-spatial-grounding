"""Belief-side AoA predictions from the nominal Sionna cache, and the Kandla site view extensions."""

from pathlib import Path

import pytest

from sim.belief.aoa import AoAPredictor
from sim.belief.site_view import SiteView
from sim.geometry import angle_diff_deg, bearing_deg

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"
SEED1 = REPO / "data" / "scenarios" / "kandla_s1"
pytestmark = pytest.mark.skipif(not (KANDLA / "rf_cache" / "sionna_rt").exists(), reason="Kandla caches missing")


@pytest.fixture(scope="module")
def site():
    return SiteView(KANDLA)


def test_scenario_seed_reads_the_shared_rf_cache():
    s = SiteView(SEED1)
    assert s.rf_dir.resolve() == (KANDLA / "rf_cache").resolve()


def test_direct_peak_matches_geometry_in_open_water(site):
    pred = AoAPredictor(site)
    s = site.sensors["S1"]["pose"]
    x, y = s["x_m"] + 400.0, s["y_m"] + 600.0          # over the creek, north-east of the jetty head
    peaks = pred.peaks(x, y, 60.0, "S1", "ism_2400")
    assert peaks, "a point 720 m from S1 over water must have paths"
    direct = [p for p in peaks if p["kind"] == "direct"]
    assert direct, peaks
    assert abs(angle_diff_deg(direct[0]["bearing_deg"], bearing_deg(s["x_m"], s["y_m"], x, y))) < 1.0
    assert peaks[0]["rel_power_db"] == 0.0
    assert all(p["kind"] in ("direct", "reflection", "diffraction", "scatter") for p in peaks)


def test_outside_the_cache_is_none(site):
    assert AoAPredictor(site).peaks(9000.0, 9000.0, 60.0, "S1", "ism_2400") is None


def test_groups_and_events_are_loaded(site):
    assert any(g for g in site.groups.values())
    assert site.events and {"evt_id", "type", "track_id", "t_s"} <= set(site.events[0])
