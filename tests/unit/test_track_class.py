"""v2 track classifier: observables only, stored model matches the feature set, sane probabilities."""

import inspect
import json

import duckdb
import numpy as np

from sim.obs_model import track_class as tc


def _db(path, moving: bool):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE tracks (track_id VARCHAR, band VARCHAR, freq_mhz DOUBLE, klass_probs VARCHAR, "
                "protocol_guess VARCHAR, bandwidth_khz DOUBLE, power_dbm DOUBLE, first_seen DOUBLE, last_seen DOUBLE, "
                "n_sensors INTEGER, confidence DOUBLE, group_id VARCHAR)")
    con.execute("CREATE TABLE track_points (track_id VARCHAR, t_s DOUBLE, x_m DOUBLE, y_m DOUBLE, z_m DOUBLE, a_m DOUBLE, "
                "b_m DOUBLE, theta_deg DOUBLE, band VARCHAR, n_sensors INTEGER)")
    con.execute("CREATE TABLE bearings (hyp_id VARCHAR, track_id VARCHAR, sensor_id VARCHAR, t_s DOUBLE, "
                "bearing_deg DOUBLE, snr_db DOUBLE, power_db DOUBLE)")
    con.execute("CREATE TABLE bursts (burst_id VARCHAR, sensor_id VARCHAR, t_s DOUBLE, f_mhz DOUBLE, bw_khz DOUBLE, "
                "t0_us DOUBLE, dur_us DOUBLE, protocol_est VARCHAR, protocol_conf DOUBLE, bearing_deg DOUBLE, "
                "snr_db DOUBLE, toa_s DOUBLE, track_id VARCHAR)")
    con.execute("CREATE TABLE dwells (t_s DOUBLE, sensor_id VARCHAR, band VARCHAR, window_mhz DOUBLE, "
                "noise_dbm_per_mhz DOUBLE, clipped_frac DOUBLE, n_bursts INTEGER)")
    con.execute("INSERT INTO tracks VALUES ('0001','ism_2400',2437.0,'{}','wifi',20000,-70,0,200,3,0.5,NULL)")
    for k, t in enumerate(range(0, 600, 10)):
        con.execute("INSERT INTO dwells VALUES (?, 'S1', 'ism_2400', 2441.75, -100, 0, 1)", [float(t)])
        az = 40.0 + (0.5 * k if moving else 0.0)
        con.execute("INSERT INTO bursts VALUES (?, 'S1', ?, 2437.0, 18000, 0, 100, 'wifi', 0.9, ?, 20, 0, NULL)",
                    [f"b{k}", float(t), az])
        if t <= 200:
            x = 100.0 + (15.0 * t if moving else 0.0)
            con.execute("INSERT INTO track_points VALUES ('0001', ?, ?, 500, 20, 40, 20, 0, 'ism_2400', 3)", [float(t), x])
            con.execute("INSERT INTO bearings VALUES (?, '0001', 'S1', ?, ?, 20, -70)", [f"h{k}", float(t), az])
    return con


def test_features_and_predict(tmp_path):
    con = _db(tmp_path / "a.duckdb", moving=False)
    f = tc.track_features(con)
    assert set(f) == {"0001"} and f["0001"].shape == (len(tc.FEATURES),)
    i = tc.FEATURES.index("presence_max")
    assert f["0001"][i] > 0.95                     # same carrier at the same bearing in every epoch
    m = tc.load_model()
    p = tc.predict(m, f["0001"])[0]
    assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


def test_moving_scores_more_drone_than_persistent(tmp_path):
    m = tc.load_model()
    d = tc.CLASSES.index("drone")
    still = tc.predict(m, tc.track_features(_db(tmp_path / "s.duckdb", moving=False))["0001"])[0][d]
    move = tc.predict(m, tc.track_features(_db(tmp_path / "m.duckdb", moving=True))["0001"])[0][d]
    assert move > still


def test_model_file_matches_features():
    m = json.loads(tc.MODEL.read_text())
    assert m["features"] == list(tc.FEATURES) and m["classes"] == list(tc.CLASSES)
    assert np.array(m["W"]).shape == (len(tc.FEATURES), len(tc.CLASSES))
    assert all(not d.startswith(("kandla_s13", "kandla_s14", "kandla_s15", "kandla_s16")) for d in m["train_dirs"])


def test_no_truth_access():
    src = inspect.getsource(tc.track_features) + inspect.getsource(tc.predict)
    assert "truth" not in src and "world_cache" not in src
