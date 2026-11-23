"""SPEC 10.1: one tool call on cached data p95 < 100 ms (Kandla, new tool families)."""

import time
from pathlib import Path

import numpy as np
import pytest

from sim.belief.site_view import SiteView
from tools.registry import REGISTRY, ToolContext, load_all

KANDLA = Path(__file__).resolve().parents[2] / "data" / "site_kandla"
pytestmark = pytest.mark.skipif(not (KANDLA / "rf_cache" / "coverage").exists(), reason="Kandla products missing")


def _calls(site):
    s1 = site.sensors["S1"]["pose"]
    p = {"x_m": s1["x_m"] + 400, "y_m": s1["y_m"] + 600, "z_m": 60}
    hz = {"type": "ref", "ref": "zone:hero_zone"}
    con = site.cursor()
    tid, t, hs = con.execute("SELECT track_id, t_s, list(hyp_id) FROM bearings GROUP BY 1, 2 "
                             "HAVING count(DISTINCT sensor_id) >= 3 ORDER BY 1, 2 LIMIT 1").fetchone()
    hyps = [f"hyp:{h}@{t:g}" for h in hs]
    return [("predict_aoa", {"point": p, "sensor_ref": "sensor:S1", "band": "ism_2400"}),
            ("geometry_check", {"from": "sensor:S4", "to": p, "mode": "los"}),
            ("geometry_check", {"from": "sensor:S4", "to": p, "mode": "reachable"}),
            ("fit_candidates", {"hyp_refs": hyps, "method": "rt_match"}),
            ("prop_query", {"sensor_ref": "sensor:S1", "point": p, "band": "ism_2400"}),
            ("coverage_query", {"band": "ism_2400", "metric": "df_err_p90_deg", "region": hz, "agg": "p90"}),
            ("weak_zones", {"metric": "df_err_p90_deg", "band": "ism_2400", "threshold": 10, "region": hz,
                            "height_m": 15}),
            ("suggest_node", {"region": hz, "band": "ism_2400", "k": 3}),
            ("model_trust", {"target": hz, "band": "ism_2400"}),
            ("effect_footprint", {"effector": "sensor:S1", "band": "ism_2400", "eirp_dbm": 33}),
            ("event_search", {"event_type": "zone_enter"}),
            ("track_history", {"track_ref": f"trk:{tid}", "step_s": 10}),
            ("sql_readonly", {"query": "SELECT band, count(*) FROM tracks GROUP BY band"})]


def test_kandla_tools_p95_under_100ms():
    load_all()
    ctx = ToolContext(SiteView(KANDLA))
    slow = {}
    for name, args in _calls(ctx.site):
        spec = REGISTRY[name]
        a = spec.args.model_validate(args)
        spec.fn(ctx, a)                                      # warm caches (memory maps, rasters)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            spec.fn(ctx, a)
            ts.append((time.perf_counter() - t0) * 1000)
        p95 = float(np.percentile(ts, 95))
        print(f"{name:18s} p95 {p95:7.1f} ms")
        if p95 > 100:
            slow[name] = round(p95, 1)
    assert not slow, slow
