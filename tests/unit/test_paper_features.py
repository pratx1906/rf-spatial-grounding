"""P9 difficulty features: pure functions (eval/paper/features.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from eval.paper.features import (GDOP_CAP, angdiff, azimuth_deg, gdop_bearing, ghost_fraction, grid_index,
                                 los_strata, nearest_sensor_distance, task_track_ids, task_window)


def test_angdiff_wraps():
    assert angdiff(359.0, 1.0) == pytest.approx(2.0)
    assert angdiff(10.0, 190.0) == pytest.approx(180.0)
    assert np.allclose(angdiff([0, 90], [350, 100]), [10, 10])


def test_azimuth_enu_clockwise_from_north():
    assert azimuth_deg(0, 0, 0, 10) == pytest.approx(0.0)
    assert azimuth_deg(0, 0, 10, 0) == pytest.approx(90.0)
    assert azimuth_deg(0, 0, -10, 0) == pytest.approx(270.0)


def test_gdop_orthogonal_two_sensors():
    # target at origin, sensors at 1000 m east and north: J = diag(1/r^2, 1/r^2) -> sqrt(2) r per rad
    g = gdop_bearing([(1000, 0), (0, 1000)], (0, 0))
    assert g == pytest.approx(math.sqrt(2) * 1000 * math.pi / 180, rel=1e-6)


def test_gdop_caps_single_and_collinear():
    assert gdop_bearing([(1000, 0)], (0, 0)) == GDOP_CAP
    assert gdop_bearing([(1000, 0), (2000, 0)], (0, 0)) == GDOP_CAP


def test_gdop_grows_with_range_and_shrinks_with_sensors():
    near = gdop_bearing([(500, 0), (0, 500)], (0, 0))
    far = gdop_bearing([(2000, 0), (0, 2000)], (0, 0))
    more = gdop_bearing([(500, 0), (0, 500), (-500, 0), (0, -500)], (0, 0))
    assert far > near > more


def test_nearest_sensor_and_ghost_fraction():
    assert nearest_sensor_distance([(0, 0), (300, 400)], (300, 400)) == 0.0
    assert nearest_sensor_distance([(0, 0)], (3, 4)) == pytest.approx(5.0)
    assert ghost_fraction([10, 12, 40, 200], 11.0) == pytest.approx(0.5)
    assert math.isnan(ghost_fraction([], 0.0))


def test_grid_index_nearest_and_outside():
    xs, ys, hs = np.arange(0, 100, 10.0), np.arange(0, 50, 10.0), np.array([2.0, 40.0, 80.0])
    assert grid_index(xs, ys, hs, 21, 39, 50) == (4, 2, 1)
    assert grid_index(xs, ys, hs, 500, 0, 2) is None


def test_task_track_ids_order_and_dedupe():
    t = {"params": {"track": "trk:12"}, "answer_key": {"track_refs": ["trk:12@40", "trk:7"]},
         "question": "Where is trk:9 relative to trk:7?"}
    assert task_track_ids(t) == ["12", "7", "9"]
    assert task_track_ids({"question": "no refs"}) == []


def test_task_window():
    assert task_window({"params": {"t": 100.0}}, (0, 600)) == (95.0, 105.0)
    assert task_window({"answer_key": {"t_s": 50}}, None) == (45.0, 55.0)
    assert task_window({"params": {"t0": 10, "t1": 90}}, None) == (10.0, 90.0)
    assert task_window({"params": {}}, (3.0, 9.0)) == (3.0, 9.0)


def test_los_strata():
    df = pd.DataFrame({"sensor_id": ["S1", "S1", "S2", "S2"], "t_s": [0, 0, 0, 0],
                       "power_db": [-60, -70, -50, -65], "label": ["reflect:x", "LOS", "diffract:y", "reflect:z"]})
    out = los_strata(df)
    assert out["los_look_frac"] == pytest.approx(0.5)       # S1 has a LOS bearing, S2 does not
    assert out["strongest_los_frac"] == pytest.approx(0.0)  # both strongest bearings are ghosts
    assert out["los_stratum"] == "LOS"
    assert los_strata(df.assign(label=None))["los_stratum"] is None
