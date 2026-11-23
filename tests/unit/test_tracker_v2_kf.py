"""S7.1 tracker: constant-velocity Kalman filter, GNN assignment, M-of-N confirmation, fragment repair, causal
track points, 3-D height and the per-fix NLOS / inconsistency flag (synthetic, closed-form checks)."""

import math

import numpy as np

from sim.obs_model import tracker_v2 as T

POSE = {"S1": (0.0, 0.0, 10.0), "S2": (2000.0, 0.0, 10.0), "S3": (1000.0, 2000.0, 10.0)}


def _em(sensor, brg, el=math.nan, snr=20.0, band="ism_2400", proto="ocusync_like", f=2440.0, bw=10000.0):
    return {"sensor": sensor, "brg": brg, "el": el, "snr": snr, "band": band, "protocol": proto, "f_mhz": f,
            "bw_khz": bw, "power": -60.0, "peaks": [(brg, 0.0, snr)], "toas": [], "t0": 0.0, "t0s": [0.0]}


def _fix(t, x, y, sig=15.0, band="ism_2400", proto="ocusync_like", f=2440.0, nlos=False):
    return {"t": float(t), "x": float(x), "y": float(y), "z": 50.0, "var_z": 30.0 ** 2,
            "R": np.eye(2) * sig ** 2, "cov": np.eye(2) * sig ** 2, "nlos": nlos, "mode": "aoa",
            "grp": [_em("S1", 0.0, band=band, proto=proto, f=f), _em("S2", 0.0, band=band, proto=proto, f=f)]}


def _brg(sx, sy, x, y):
    return math.degrees(math.atan2(x - sx, y - sy)) % 360


def test_single_isolated_fix_never_becomes_a_track():
    assert T.track_fixes([_fix(0, 500, 500)]) == []


def test_straight_track_is_one_track_and_velocity_is_estimated():
    fx = [_fix(t, 100 + 10 * t, 500) for t in range(0, 200, 10)]
    trs = T.track_fixes(fx)
    assert len(trs) == 1 and len(trs[0]["fixes"]) == 20
    x, P = trs[0]["hist"][-1]["x"], trs[0]["hist"][-1]["P"]
    assert abs(x[2] - 10.0) < 1.5 and abs(x[3]) < 1.5


def test_crossing_hoppers_keep_identity():
    """Two hoppers on different frequencies cross at 90 deg: the predicted state (not the last fix) decides."""
    fa = [_fix(t, -600 + 12 * t, 0, proto="lora", f=868.0, band="sub_ghz_868_915") for t in range(0, 110, 10)]
    fb = [_fix(t, 0, -600 + 12 * t, proto="lora", f=915.0, band="sub_ghz_868_915") for t in range(0, 110, 10)]
    trs = T.track_fixes(fa + fb)
    assert len(trs) == 2
    for tr in trs:
        xs = np.array([f["x"] for f in tr["fixes"]])
        ys = np.array([f["y"] for f in tr["fixes"]])
        assert np.ptp(xs) < 1 or np.ptp(ys) < 1          # each track stays on its own straight line


def test_one_to_one_per_epoch():
    fx = [_fix(t, 10 * t, 0) for t in range(0, 60, 10)] + [_fix(50, 520, 30)]
    trs = T.track_fixes(fx)
    for tr in trs:
        ts = [f["t"] for f in tr["fixes"]]
        assert len(ts) == len(set(ts))


def test_far_hopper_fix_does_not_join_across_frequency():
    fa = [_fix(t, 8 * t, 0, proto="lora", f=868.0, band="sub_ghz_868_915") for t in range(0, 60, 10)]
    fb = [_fix(t, 1500, 1500 + 5 * t, proto="lora", f=915.0, band="sub_ghz_868_915") for t in range(0, 60, 10)]
    trs = T.track_fixes(fa + fb)
    assert len(trs) == 2


def test_fragment_repair_across_a_45_s_gap():
    fx = [_fix(t, 10 * t, 0) for t in range(0, 100, 10)] + [_fix(t, 10 * t, 0) for t in range(140, 240, 10)]
    trs = T.track_fixes(fx)
    assert len(trs) == 1 and len(trs[0]["fixes"]) == 20


def test_no_repair_when_state_disagrees():
    fx = [_fix(t, 10 * t, 0) for t in range(0, 100, 10)] + [_fix(t, 3000 - 10 * t, 2000) for t in range(140, 240, 10)]
    assert len(T.track_fixes(fx)) == 2


def test_causal_points_ignore_future_fixes_and_ellipse_grows_while_coasting():
    fx = [_fix(t, 10 * t, 0) for t in range(0, 100, 10)] + [_fix(t, 10 * t + 200, 100) for t in range(120, 200, 10)]
    full = T.causal_points(T.track_fixes(fx)[0], "0001", "ism_2400")
    cut = T.causal_points(T.track_fixes([f for f in fx if f["t"] <= 90])[0], "0001", "ism_2400")
    a = {r[1]: r for r in full}
    for r in cut:
        assert a[r[1]][2:5] == r[2:5]                     # same position up to t = 90 with or without later fixes
    coast = [a[t][5] for t in (90.0, 94.0, 98.0, 102.0, 106.0, 110.0)]
    assert all(b > c for b, c in zip(coast[1:], coast[:-1]))
    assert all(len(r) == 11 and isinstance(r[10], bool) for r in full)


def test_height_from_two_elevations_weighted_by_sigma():
    x, y, z = 1000.0, 500.0, 120.0
    grp = []
    for s in ("S1", "S2"):
        sx, sy, sz = POSE[s]
        d = math.hypot(x - sx, y - sy)
        grp.append(_em(s, _brg(sx, sy, x, y), el=math.degrees(math.atan2(z - sz, d))))
    zh, var = T.height3((x, y), grp, POSE)
    assert abs(zh - z) < 2.0 and var > 0
    zn, varn = T.height3((x, y), [_em("S1", 0.0), _em("S2", 0.0)], POSE)     # no elevation: wide prior
    assert varn > 50.0 ** 2
    zc, _ = T.height3((x, y), [_em("S1", 0.0, el=80.0)], POSE)
    assert 0.0 <= zc <= 400.0


def test_nlos_flags():
    x, y = 1000.0, 700.0
    good = [_em(s, _brg(POSE[s][0], POSE[s][1], x, y)) for s in ("S1", "S2", "S3")]
    f = T.fix_from_group(good, POSE)
    assert f is not None and not f["nlos"] and math.hypot(f["x"] - x, f["y"] - y) < 5
    bad = good[:2] + [_em("S3", (_brg(1000.0, 2000.0, x, y) + 25.0) % 360)]      # one node's peak points elsewhere
    fb = T.fix_from_group(bad, POSE)
    assert fb is None or fb["nlos"]
    # two nodes almost on one line through the target: poor crossing angle
    far = (1000.0, 60.0)
    thin = [_em("S1", _brg(0, 0, *far)), _em("S2", _brg(2000.0, 0.0, *far))]
    ft = T.fix_from_group(thin, POSE)
    assert ft is not None and ft["nlos"]


def test_inflated_covariance_for_flagged_fix():
    x, y = 1000.0, 700.0
    good = [_em(s, _brg(POSE[s][0], POSE[s][1], x, y)) for s in ("S1", "S2")]
    f = T.fix_from_group(good, POSE)
    assert np.trace(f["R"]) >= np.trace(f["cov"])            # calibrated, never below the WLS covariance


def test_duplicate_fixes_of_one_source_in_one_epoch_give_one_track():
    fx = []
    for t in range(0, 100, 10):
        fx.append(_fix(t, 10 * t, 0))
        fx.append(_fix(t, 10 * t + 12, -9, sig=40.0))        # same carrier, same place, worse fix
    trs = T.track_fixes(fx)
    assert len(trs) == 1 and len(trs[0]["fixes"]) == 10
    assert all(f["R"][0, 0] == 15.0 ** 2 for f in trs[0]["fixes"])     # the better fix is kept


def test_band_hop_is_relinked_and_points_carry_the_band_of_each_fix():
    fx = [_fix(t, 10 * t, 0) for t in range(0, 110, 10)]
    fx += [_fix(t, 10 * t, 0, band="ism_5800", f=5800.0) for t in range(120, 220, 10)]
    trs = T.track_fixes(fx)
    assert len(trs) == 1
    rows = T.causal_points(trs[0], "0001", trs[0]["band"])
    bands = {r[1]: r[8] for r in rows}
    assert bands[100.0] == "ism_2400" and bands[200.0] == "ism_5800"


def test_overlapping_duplicate_tracks_merge_but_separate_drones_do_not():
    a = [_fix(t, 10 * t, 0, sig=8.0) for t in range(0, 210, 10)]
    b = [_fix(t, 10 * t + 30, 25, sig=8.0) for t in range(50, 160, 10)]        # same source, offset fixes
    trs = T.track_fixes(a + b)
    assert len(trs) == 1
    assert len({f["t"] for f in trs[0]["fixes"]}) == len(trs[0]["fixes"])        # one fix per epoch
    c = [_fix(t, 10 * t, 600, sig=8.0) for t in range(0, 210, 10)]               # a second drone 600 m away
    assert len(T.track_fixes(a + c)) == 2
