"""World-side track labels must match the emitter's band, not only its position (v3 audit 2026-10-26:
a 5.8 GHz CCTV-link track hovering near a sub-GHz drone was labelled as that drone)."""

from sim.obs_model.observe_v2 import band_at, match_emitter
from sim.world.scenario import Emitter


def _emitter(eid, band_schedule, xy=(0.0, 0.0)):
    return Emitter(eid, "drone", band_schedule, {b: 900.0 for _, b in band_schedule}, 20.0, "elrs", 500.0, 0.0, 100.0,
                   [], 10.0, points=[(float(t), xy[0], xy[1], 30.0) for t in range(0, 101)])


def _fixes(band, ts=(10, 11, 12, 13), xy=(20.0, 0.0)):
    return [{"t": float(t), "x": xy[0], "y": xy[1], "grp": [{"band": band}]} for t in ts]


def test_band_at_follows_the_schedule():
    e = _emitter("d1", [(0.0, "sub_ghz_868_915"), (50.0, "ism_2400")])
    assert band_at(e, 10.0) == "sub_ghz_868_915"
    assert band_at(e, 60.0) == "ism_2400"


def test_track_in_another_band_is_not_labelled_as_the_nearby_drone():
    e = _emitter("d1", [(0.0, "sub_ghz_868_915")])
    best, _ = match_emitter([e], _fixes("ism_5800"))
    assert best is None


def test_same_band_nearby_drone_is_matched():
    e = _emitter("d1", [(0.0, "ism_5800")])
    best, d = match_emitter([e], _fixes("ism_5800"))
    assert best is e and d < 250


def test_band_hopping_emitter_matches_only_while_in_the_track_band():
    e = _emitter("d1", [(0.0, "sub_ghz_868_915"), (12.0, "ism_5800")])
    best, _ = match_emitter([e], _fixes("ism_5800", ts=(12, 13, 14, 15)))
    assert best is e
    best, _ = match_emitter([e], _fixes("ism_5800", ts=(2, 3, 4, 5)))
    assert best is None


def test_nearest_band_matched_emitter_wins_over_closer_wrong_band():
    near_wrong = _emitter("d1", [(0.0, "sub_ghz_868_915")], xy=(20.0, 0.0))
    far_right = _emitter("d2", [(0.0, "ism_5800")], xy=(150.0, 0.0))
    best, d = match_emitter([near_wrong, far_right], _fixes("ism_5800"))
    assert best is far_right and abs(d - 130.0) < 1e-6


def test_retrack_emitters_carry_the_band_schedule(tmp_path):
    import duckdb

    from scripts.retrack_v2 import _emitters

    (tmp_path / "truth").mkdir()
    c = duckdb.connect(str(tmp_path / "truth" / "truth.duckdb"))
    c.execute("create table emitters (emitter_id varchar, kind varchar, bands varchar, eirp_dbm double, swarm_id varchar)")
    c.execute("insert into emitters values ('d1', 'drone', '[[0.0, \"ism_5800\"], [50.0, \"ism_2400\"]]', 20, null)")
    c.execute("create table emitter_points (emitter_id varchar, t_s double, x_m double, y_m double, z_m double)")
    c.execute("insert into emitter_points values ('d1', 10, 0, 0, 30), ('d1', 60, 0, 0, 30)")
    c.close()
    (e,) = _emitters(tmp_path)
    assert band_at(e, 10.0) == "ism_5800" and band_at(e, 60.0) == "ism_2400"
