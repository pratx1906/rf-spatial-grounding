"""tdoa_fix tool on the S1 TDOA chain (S1 -> S5 request): the same burst matched across nodes (match_toas), cut
bursts (NULL TOA) ignored, residual-gated refinement on 3-D ranges, AoA fix kept when the residuals disagree."""

import math
from types import SimpleNamespace

import duckdb
import numpy as np

C0 = 299_792_458.0
SENS = {"S1": (0.0, 0.0, 12.0), "S2": (1800.0, 200.0, 15.0), "S3": (600.0, 1700.0, 30.0)}
P = (700.0, 600.0, 150.0)
AOA = (760.0, 540.0)


def _ctx(toa_err=None):
    con = duckdb.connect()
    con.execute("CREATE TABLE bursts (track_id VARCHAR, t_s DOUBLE, sensor_id VARCHAR, bearing_deg DOUBLE, "
                "snr_db DOUBLE, toa_s DOUBLE, bw_khz DOUBLE)")
    rows = []
    for sid, s in SENS.items():
        brg = math.degrees(math.atan2(P[0] - s[0], P[1] - s[1])) % 360
        rng = math.dist(P, s)
        for k, t_emit in enumerate((1.000e-3, 3.000e-3)):
            err = (toa_err or {}).get(sid, 0.0)
            # node S1 hears burst 0 best, the others burst 1: "highest-SNR burst per node" mixes bursts
            snr = 25.0 if (sid == "S1") == (k == 0) else 15.0
            rows.append(("0042", 100.0, sid, brg, snr, t_emit + rng / C0 + err, 20000.0))
        rows.append(("0042", 100.0, sid, brg, 30.0, None, 20000.0))          # a burst cut at the dwell edge
    con.executemany("INSERT INTO bursts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    site = SimpleNamespace(
        cursor=lambda: con, sensors={s: {"pose": {"x_m": v[0], "y_m": v[1], "z_m": v[2]}} for s, v in SENS.items()},
        points={"0042": {"x": np.array([AOA[0]]), "y": np.array([AOA[1]]), "z": np.array([P[2]])}},
        track_index_at=lambda tid, t: 0)
    return SimpleNamespace(site=site, episode={})


def _call(ctx, monkeypatch):
    from tools import signals

    monkeypatch.setattr(signals, "_track", lambda ctx, ref: "0042")
    return signals.tdoa_fix(ctx, signals.TdoaFixArgs(track_ref="trk:0042", at_time_s=100.0)).data


def test_tdoa_fix_matches_the_same_burst_and_refines(monkeypatch):
    d = _call(_ctx(), monkeypatch)
    assert d["accepted"] is True
    assert math.hypot(d["pos"]["x_m"] - P[0], d["pos"]["y_m"] - P[1]) < 5.0


def test_tdoa_fix_keeps_the_aoa_fix_when_residuals_disagree(monkeypatch):
    d = _call(_ctx({"S2": 3e-6}), monkeypatch)
    assert d["accepted"] is False and d["shift_m"] == 0.0
    assert (d["pos"]["x_m"], d["pos"]["y_m"]) == AOA
