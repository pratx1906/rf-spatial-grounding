"""AIR7311 observation model: Sionna paths -> 4-channel DF engine -> observed peaks (world side)."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.obs_model.observe_air7311 import air7311_peaks, observe_air7311
from sim.world.scenario import Emitter

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
KANDLA = REPO / "data" / "site_kandla"


def _paths(az_list, gains_db, el=10.0):
    k = len(az_list)
    h = np.array([10 ** (g / 20) * np.exp(1j * 0.7 * i) for i, g in enumerate(gains_db)], np.complex64)[None]
    return dict(h=h, tau_ns=np.array([[100.0 + 30 * i for i in range(k)]], np.float32),
                az=np.array([az_list], np.float32), el=np.full((1, k), el, np.float32),
                valid=np.ones((1, k), bool))


def test_single_path_gives_one_peak_at_its_bearing():
    p = _paths([60.0], [-80.0])
    out = air7311_peaks(HW, "ism_2400", **p, eirp_dbm=np.array([20.0]), seed=1, device="cpu")
    peaks = out[0]
    assert len(peaks) >= 1
    assert abs((peaks[0]["az_deg"] - 60.0 + 180) % 360 - 180) < 1.0
    assert peaks[0]["snr_db"] > 20


def test_direct_path_ranked_first_over_weaker_reflection():
    p = _paths([100.0, 140.0], [-80.0, -86.0])
    peaks = air7311_peaks(HW, "ism_2400", **p, eirp_dbm=np.array([20.0]), seed=2, device="cpu")[0]
    assert len(peaks) == 2
    assert abs((peaks[0]["az_deg"] - 100.0 + 180) % 360 - 180) < 2.0
    assert abs((peaks[1]["az_deg"] - 140.0 + 180) % 360 - 180) < 3.0
    assert peaks[0]["power_dbm"] > peaks[1]["power_dbm"]


def test_below_detection_threshold_reports_nothing():
    p = _paths([60.0], [-160.0])
    assert air7311_peaks(HW, "ism_2400", **p, eirp_dbm=np.array([20.0]), seed=3, device="cpu")[0] == []


@pytest.mark.skipif(not (KANDLA / "rf_cache" / "air7311_S1").exists(), reason="Kandla AIR7311 caches not built")
def test_emitter_outside_every_cache_radius_is_not_observed():
    from envbuild.site_kandla import build

    b = build(0)
    far = Emitter("emt_far", "drone", [(0.0, "ism_2400")], {"ism_2400": 2440.0}, 20.0, "ocusync-like", 10000.0,
                  0.0, 40.0, [(9000.0, 9000.0, 60.0)], 0.0)
    far.points = [(t, 9000.0, 9000.0, 60.0) for t in (0.0, 10.0, 20.0, 30.0, 40.0)]
    obs = observe_air7311(0, [far], b, KANDLA / "rf_cache", device="cpu")
    assert obs["bearings"] == [] and obs["tracks"] == []


@pytest.mark.skipif(not (KANDLA / "rf_cache" / "air7311_S1").exists(), reason="Kandla AIR7311 caches not built")
def test_hovering_emitter_near_s1_is_tracked_close_to_truth():
    from envbuild.site_kandla import build

    b = build(0)
    s1 = next(s for s in b["sensors"] if s.id == "S1")
    x, y = s1.x + 300.0, s1.y - 250.0
    e = Emitter("emt_h", "drone", [(0.0, "ism_2400")], {"ism_2400": 2440.0}, 20.0, "ocusync-like", 10000.0,
                0.0, 60.0, [(x, y, 60.0)], 0.0)
    e.points = [(float(t), x, y, 60.0) for t in range(0, 62, 2)]
    obs = observe_air7311(0, [e], b, KANDLA / "rf_cache", device="cpu")
    assert obs["bearings"], "a drone 400 m from S1 must be seen"
    pts = obs["points"]
    assert pts, "and tracked"
    err = np.median([math.hypot(p[2] - x, p[3] - y) for p in pts])
    assert err < 150.0
