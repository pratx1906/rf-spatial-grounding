"""RF convergence Task 2c: (1) the v21 2-node fix-error rise is the input bearing error, not the tracker: a 2-node
fix is exactly the crossing of the measured bearings, height never moves it, and the causal Kalman points carry no
lag. (2) TDOA association: a matched TOA set must be physically possible pair by pair (|dt| <= baseline / c +
margin), span >= 3 nodes, and agree with the AoA fix on one emission time."""

import math

import numpy as np

from sim.belief import tdoa as TD
from sim.obs_model import tracker_v2 as T
from sim.obs_model.observe_v2 import Detection

C0 = 299_792_458.0
POSE = {"S1": (0.0, 0.0, 10.0), "S2": (1500.0, 0.0, 10.0), "S3": (750.0, 1500.0, 10.0), "S4": (-300.0, 1200.0, 10.0)}


def _brg(s, x, y):
    return math.degrees(math.atan2(x - s[0], y - s[1])) % 360


def _two_node_dets(bias_deg=0.0, el=math.nan, noise=0.0, seed=0, path=None):
    """One emitter seen by S1 and S2 only (2-node fixes), S2 bearings + bias_deg; path(t) -> (x, y)."""
    rng = np.random.default_rng(seed)
    path = path or (lambda t: (750.0 + 150.0 * math.cos(2 * math.pi * t / 600), 800.0 + 150.0 * math.sin(2 * math.pi * t / 600)))
    dets, truth = [], {}
    for t in np.arange(0.0, 600.0, 10.0):
        x, y = path(t)
        truth[float(t)] = (x, y)
        for sid in ("S1", "S2"):
            b = _brg(POSE[sid], x, y) + rng.normal(0, noise) + (bias_deg if sid == "S2" else 0.0)
            dets.append(Detection(float(t), sid, "ism_5800", 5785.0, 10000.0, 1e-3, 1e-3, "ocusync_like", 0.9,
                                  [(b % 360, el, 25.0)], math.nan, 25.0, -60.0))
    return dets, truth


def _intersect(b1, b2):
    """Closed-form crossing of the S1 and S2 bearing lines."""
    (x1, y1, _), (x2, y2, _) = POSE["S1"], POSE["S2"]
    d1 = np.array([math.sin(math.radians(b1)), math.cos(math.radians(b1))])
    d2 = np.array([math.sin(math.radians(b2)), math.cos(math.radians(b2))])
    s = np.linalg.solve(np.column_stack([d1, -d2]), np.array([x2 - x1, y2 - y1]))
    return x1 + s[0] * d1[0], y1 + s[0] * d1[1]


def test_two_node_fix_error_is_the_input_bearing_error():
    """v21 2-node case: S2 off by 1.3 deg (the v21 2-node member median, demo v2 0.9) at ~900 m. The tracker's fix is
    exactly the crossing of the two measured bearings (no tracker bias), so the fix error is the geometric error of
    the input bearings: tens of metres."""
    dets, truth = _two_node_dets(bias_deg=1.3)
    out = T.build_tracks(dets, POSE)
    assert len(out["fixes"]) == 60
    for f in out["fixes"]:
        b1, b2 = (g["brg"] for g in sorted(f["grp"], key=lambda g: g["sensor"]))
        assert math.dist((f["x"], f["y"]), _intersect(b1, b2)) < 0.5
    e = [math.dist((f["x"], f["y"]), truth[f["t"]]) for f in out["fixes"]]
    assert 15.0 < np.median(e) < 40.0


def test_height_does_not_move_a_two_node_fix():
    """The S7.1 3-D height (node elevations) enters only the TDOA ranges (>= 3 nodes), never a 2-node AoA fix."""
    a = T.build_tracks(_two_node_dets(bias_deg=1.3)[0], POSE)
    b = T.build_tracks(_two_node_dets(bias_deg=1.3, el=12.0)[0], POSE)
    assert [round(f["z"]) for f in a["fixes"]] != [round(f["z"]) for f in b["fixes"]]       # height did change
    for fa, fb in zip(a["fixes"], b["fixes"]):
        assert (fa["x"], fa["y"]) == (fb["x"], fb["y"])


def test_filtered_points_have_no_lag_on_two_node_fixes():
    """Constant-velocity drone (12 m/s), unbiased noisy 2-node fixes: the causal Kalman points at fix epochs have no
    along-track lag and are no worse than the raw fixes (S7.1 filter is not the 2-node regression)."""
    path = lambda t: (200.0 + 2.0 * t, 900.0 - 0.5 * t)
    dets, truth = _two_node_dets(noise=0.7, path=path, seed=3)
    out = T.build_tracks(dets, POSE)
    tr = max(out["tracks"], key=lambda q: len(q["fixes"]))
    pts = {r[1]: (r[2], r[3]) for r in T.causal_points(tr, "0001", tr["band"])}
    along, ef, er = [], [], []
    u = np.array([2.0, -0.5]) / math.hypot(2.0, -0.5)
    for f in tr["fixes"][3:]:
        p = np.array(pts[f["t"]]) - np.array(truth[f["t"]])
        along.append(float(p @ u))
        ef.append(float(np.hypot(*p)))
        er.append(math.dist((f["x"], f["y"]), truth[f["t"]]))
    assert abs(np.mean(along)) < 5.0
    assert np.median(ef) <= np.median(er) + 1.0


# ------------------------------------------------------------------------------------------------- TDOA association
P5 = {"A": (0.0, 0.0, 10.0), "B": (2600.0, 0.0, 10.0), "C": (1000.0, 1200.0, 10.0), "D": (1500.0, 300.0, 10.0),
      "E": (400.0, 900.0, 10.0)}


def _toas(p, pose, err=None, t_e=1.0e-3):
    return {s: [t_e + math.dist(p, q) / C0 + (err or {}).get(s, 0.0)] for s, q in pose.items()}


def test_match_rejects_a_physically_impossible_pair():
    """D-E baseline is 1,166 m (3.9 us); a D TOA 8 us late is within the old window (largest baseline 2,600 m +
    1 us) of the reference but impossible against E: D must not be matched."""
    p = (900.0, 500.0, 60.0)
    toas = _toas(p, P5, {"D": 8e-6})
    m = TD.match_toas(toas, P5)
    assert "D" not in m and len(m) >= 3
    for a in m:
        for b in m:
            assert abs(m[a] - m[b]) <= math.dist(P5[a], P5[b]) / C0 + 1e-6


def test_match_needs_three_nodes():
    p = (900.0, 500.0, 60.0)
    two = {k: v for k, v in _toas(p, P5).items() if k in ("A", "B")}
    assert TD.match_toas(two, P5) == {}
    assert set(TD.match_toas(two, P5, min_nodes=2)) == {"A", "B"}


def test_match_drops_a_toa_inconsistent_with_the_aoa_fix():
    """A TOA 2 us early (600 m) at C stays inside every pair bound but contradicts the emission time implied by the
    other nodes at the AoA fix: with xy given it is dropped and every kept range difference is right within 100 m."""
    p = (900.0, 500.0, 60.0)
    toas = _toas(p, P5, {"C": -2e-6})
    loose = TD.match_toas(toas, P5)
    assert "C" in loose                                      # pair bounds alone cannot see it
    m = TD.match_toas(toas, P5, xy=(930.0, 480.0), z_m=60.0, tol_m=200.0)
    assert "C" not in m and len(m) >= 3
    ref = min(m, key=m.get)
    for s in m:
        true = math.dist(p, P5[s]) - math.dist(p, P5[ref])
        assert abs((m[s] - m[ref]) * C0 - true) < 100.0


def test_tracker_tdoa_pairs_stay_physical():
    """Tracker end to end: 4 nodes, wideband, one node's TOA 4 us (1.2 km) off, inside the largest-baseline window:
    the fix's matched TOAs exclude it (AoA emission-time check)."""
    p = (700.0, 600.0, 50.0)
    dets = []
    for t in (0.0, 10.0, 20.0, 30.0):
        for sid, s in POSE.items():
            toa = t + 2e-3 + math.dist(p, s) / C0 + (4e-6 if sid == "S3" else 0.0)
            dets.append(Detection(t, sid, "ism_5800", 5785.0, 20000.0, 2e-3, 1e-3, "ocusync_like", 0.9,
                                  [(_brg(s, *p[:2]), math.nan, 25.0)], toa, 25.0, -60.0))
    out = T.build_tracks(dets, POSE)
    assert out["fixes"]
    for f in out["fixes"]:
        assert "S3" not in f["toas"]
        ref = min(f["toas"], key=f["toas"].get) if f["toas"] else None
        for s, v in f["toas"].items():
            true = math.dist(p, POSE[s]) - math.dist(p, POSE[ref])
            assert abs((v - f["toas"][ref]) * C0 - true) < 50.0
