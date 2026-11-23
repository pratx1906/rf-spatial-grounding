"""Family C on Kandla: prop_query, coverage_query, weak_zones, suggest_node, model_trust."""

from pathlib import Path

import pytest

from sim.belief.site_view import SiteView
from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"
pytestmark = pytest.mark.skipif(not (KANDLA / "rf_cache" / "coverage").exists(), reason="coverage not built")


@pytest.fixture(scope="module")
def ctx():
    load_all()
    return ToolContext(SiteView(KANDLA))


def call(ctx, name, **args):
    spec = REGISTRY[name]
    return spec.fn(ctx, spec.args.model_validate(args))


def _circle(ctx, sensor, r=300.0):
    p = ctx.site.sensors[sensor]["pose"]
    return {"type": "circle", "center": {"x_m": p["x_m"], "y_m": p["y_m"]}, "radius_m": r}


def test_prop_query_near_sensor_is_strong_and_has_a_direct_path(ctx):
    p = ctx.site.sensors["S1"]["pose"]
    d = call(ctx, "prop_query", sensor_ref="sensor:S1", point={"x_m": p["x_m"] + 300, "y_m": p["y_m"] + 300,
                                                                 "z_m": 60}, band="ism_2400").data
    assert d["los"] is True and d["n_paths"] >= 1 and 60 < d["path_loss_db"] < 120
    assert d["model_source"] == "sionna_rt" and 0 <= d["trust"] <= 1


def test_detect_prob_near_a_sensor_is_high(ctx):
    d = call(ctx, "coverage_query", band="ism_2400", metric="detect_prob", region=_circle(ctx, "S1"),
             agg="mean", height_m=40).data
    assert d["value"] > 0.9 and d["ras_ref"].startswith("ras:")


def test_weak_zones_shrink_as_the_threshold_relaxes(ctx):
    region = {"type": "ref", "ref": "zone:hero_zone"}
    strict = call(ctx, "weak_zones", metric="df_err_p90_deg", band="sub_ghz_868_915", threshold=3.0,
                  region=region, height_m=15, limit=10).data
    loose = call(ctx, "weak_zones", metric="df_err_p90_deg", band="sub_ghz_868_915", threshold=15.0,
                 region=region, height_m=15, limit=10).data
    assert strict["total_area_m2"] >= loose["total_area_m2"]
    assert all(z["ref"].startswith("ras:") and z["area_m2"] > 0 for z in strict["zones"])


def test_suggest_node_respects_separation_and_reports_gain(ctx):
    d = call(ctx, "suggest_node", region={"type": "ref", "ref": "zone:hero_zone"}, band="ism_2400",
             objective="df_err", constraints={"min_sep_m": 400}, k=3).data
    assert 1 <= len(d["candidates"]) <= 3
    for c in d["candidates"]:
        assert c["ref"].startswith("nodecand:") and c["mount_ref"].startswith("obj:")
        assert c["min_sep_to_existing_m"] >= 400 and c["predicted_gain"] >= 0


def test_model_trust_has_components(ctx):
    p = ctx.site.sensors["S4"]["pose"]
    d = call(ctx, "model_trust", target={"x_m": p["x_m"] + 200, "y_m": p["y_m"] + 150}, sensor_ref="S4",
             band="ism_2400").data
    assert 0 <= d["trust"] <= 1 and "spread_deg" in d["components"]


def test_coverage_query_rejects_unknown_sensor(ctx):
    with pytest.raises(ToolError) as e:
        call(ctx, "coverage_query", sensor_refs=["sensor:S9"], band="ism_2400", metric="snr_db",
             region={"type": "ref", "ref": "zone:hero_zone"}, agg="mean")
    assert e.value.code == "UNKNOWN_REF"


def test_model_trust_per_sensor_names_each_node_and_the_best(ctx):
    p = ctx.site.sensors["S4"]["pose"]
    d = call(ctx, "model_trust", target={"x_m": p["x_m"] + 200, "y_m": p["y_m"] + 150, "z_m": 15.0},
             band="ism_2400", per_sensor=True).data
    rows = d["per_sensor"]
    assert rows and {"sensor", "ghost_risk", "nlos_frac", "spread_deg", "trust"} <= set(rows[0])
    assert all(0 <= r["ghost_risk"] <= 1 for r in rows)
    assert d["best_sensor"] == max(rows, key=lambda r: r["trust"])["sensor"]


def test_model_trust_takes_a_track_at_a_time(ctx):
    tid = next(iter(ctx.site.points))
    t = float(ctx.site.points[tid]["t"][len(ctx.site.points[tid]["t"]) // 2])
    d = call(ctx, "model_trust", track_ref=f"trk:{tid}@{t:g}", band="ism_2400", per_sensor=True)
    assert f"trk:{tid}" in " ".join(d.refs)


def test_model_trust_needs_exactly_one_subject(ctx):
    with pytest.raises(ToolError):
        call(ctx, "model_trust", band="ism_2400")


# --- SFT v1.4 guardrail (PLAN rf-convergence Task 7): implausible weak_zones thresholds get a redirecting hint ------
def test_weak_zones_warns_on_an_implausible_df_threshold(ctx):
    region = {"type": "ref", "ref": "zone:hero_zone"}
    out = call(ctx, "weak_zones", metric="df_err_p90_deg", band="ism_2400", threshold=90.0, region=region, height_m=15)
    assert any("implausible" in w and "10" in w for w in out.warnings), out.warnings
    assert out.data["threshold"] == 90.0                        # the call is still answered as asked


def test_weak_zones_plausible_thresholds_carry_no_warning(ctx):
    region = {"type": "ref", "ref": "zone:hero_zone"}
    for thr in (5.0, 10.0, 20.0, 45.0):
        out = call(ctx, "weak_zones", metric="df_err_p90_deg", band="ism_2400", threshold=thr, region=region)
        assert out.warnings == [], (thr, out.warnings)
    out = call(ctx, "weak_zones", metric="detect_prob", band="ism_2400", threshold=0.5, region=region)
    assert out.warnings == []


def test_weak_zones_warns_on_a_detect_prob_given_in_percent(ctx):
    region = {"type": "ref", "ref": "zone:hero_zone"}
    out = call(ctx, "weak_zones", metric="detect_prob", band="ism_2400", threshold=50.0, region=region)
    assert any("0..1" in w for w in out.warnings), out.warnings
