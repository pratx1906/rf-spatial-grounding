"""Family B completion on Kandla: predict_aoa, geometry_check, fit_candidates, swarm groups in track_search."""

import math
from pathlib import Path

import numpy as np
import pytest

from sim.belief.site_view import SiteView
from sim.geometry import angle_diff_deg, bearing_deg
from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"
pytestmark = pytest.mark.skipif(not (KANDLA / "rf_cache" / "sionna_rt").exists(), reason="Kandla caches missing")


@pytest.fixture(scope="module")
def ctx():
    load_all()
    return ToolContext(SiteView(KANDLA))


def call(ctx, name, **args):
    spec = REGISTRY[name]
    return spec.fn(ctx, spec.args.model_validate(args))


def test_predict_aoa_direct_peak_matches_geometry(ctx):
    s = ctx.site.sensors["S1"]["pose"]
    x, y = s["x_m"] + 400.0, s["y_m"] + 600.0
    out = call(ctx, "predict_aoa", point={"x_m": x, "y_m": y, "z_m": 60.0}, sensor_ref="s1", band="ism_2400")
    d = out.data
    assert d["sensor"] == "sensor:S1" and d["model_source"] == "sionna_rt" and 0.0 <= d["trust"] <= 1.0
    direct = [p for p in d["peaks"] if p["kind"] == "direct"]
    assert direct and abs(angle_diff_deg(direct[0]["bearing_deg"], bearing_deg(s["x_m"], s["y_m"], x, y))) < 1.0
    text = str(d)
    assert "reflect:" not in text and "path_type" not in text      # never the truth-label vocabulary


def test_predict_aoa_outside_cache_is_out_of_coverage(ctx):
    with pytest.raises(ToolError) as e:
        call(ctx, "predict_aoa", point={"x_m": 9000.0, "y_m": 9000.0}, sensor_ref="sensor:S1", band="ism_2400")
    assert e.value.code == "OUT_OF_COVERAGE" and e.value.hint


def test_geometry_check_tank_blocks_low_line_of_sight(ctx):
    tank = max((o for o in ctx.site.objects.values() if o.kind == "tank"), key=lambda o: o.height_m)
    c = tank.geom.centroid
    r = math.sqrt(tank.geom.area / math.pi)
    a = {"x_m": c.x - r - 60.0, "y_m": c.y, "z_m": 3.0}
    b = {"x_m": c.x + r + 60.0, "y_m": c.y, "z_m": 3.0}
    low = call(ctx, "geometry_check", **{"from": a, "to": b, "mode": "los"}).data
    assert low["result"] is False and tank.ref in low["blockers"] and low["margin_m"] < 0
    high = call(ctx, "geometry_check", **{"from": {**a, "z_m": 150.0}, "to": {**b, "z_m": 150.0}, "mode": "los"}).data
    assert high["result"] is True and "definition" in str(high) or high["result"] is True


def test_fit_candidates_locates_a_hovering_emitter(ctx):
    """Real AIR7311 bearings, seed 0: pick bearing groups with >= 3 sensors and compare to truth (test-side only)."""
    from tasks.truth_access import TruthStore

    truth = TruthStore(KANDLA)
    tof = truth.track_of()
    con = ctx.site.cursor()
    rows = con.execute("SELECT track_id, t_s, list(hyp_id), list(sensor_id), list(power_db) FROM bearings "
                       "GROUP BY 1, 2 HAVING count(DISTINCT sensor_id) >= 3 ORDER BY 1, 2").fetchall()
    errs_tri, errs_rt = [], []
    for tid, t, hs, ss, pw in rows[::max(1, len(rows) // 25)][:25]:
        best = {}
        for h, s, p in zip(hs, ss, pw, strict=True):
            if s not in best or p > best[s][1]:
                best[s] = (h, p)
        refs = [f"hyp:{h}@{t:g}" for h, _ in best.values()]
        tri = call(ctx, "fit_candidates", hyp_refs=refs, method="triangulate").data["candidates"][0]["position"]
        rt = call(ctx, "fit_candidates", hyp_refs=[f"hyp:{h}@{t:g}" for h in hs], method="rt_match").data
        p = truth.position(tof[tid][0], t)
        errs_tri.append(math.hypot(tri["x_m"] - p[0], tri["y_m"] - p[1]))
        top = rt["candidates"][0]
        errs_rt.append(math.hypot(top["position"]["x_m"] - p[0], top["position"]["y_m"] - p[1]))
        assert all("explained_as" in h for h in top["hyps"])
    truth.close()
    assert np.median(errs_tri) < 80.0, errs_tri
    assert np.median(errs_rt) < 80.0, errs_rt


def test_track_search_filters_by_swarm_group(ctx):
    gid = next(g for g in ctx.site.groups.values() if g)
    out = call(ctx, "track_search", filters={"group_id": gid}, limit=20).data
    assert out["total"] >= 2 and all(it["group_id"] == gid for it in out["items"])


def test_track_detail_reports_zones_and_group(ctx):
    tid = next(t for t, g in ctx.site.groups.items() if g)
    d = call(ctx, "track_detail", track_ref=f"trk:{tid}").data
    assert d["group_id"] == ctx.site.groups[tid]
    p = d["pos"]
    assert set(d["zones"]) == {f"zone:{z}" for z in ctx.site.zones_containing(p["x_m"], p["y_m"])}
