"""Families D, E, F, G on Kandla: drone search, effects/collateral, replay, read-only SQL."""

from pathlib import Path

import pytest

from sim.belief.site_view import SiteView
from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"
pytestmark = pytest.mark.skipif(not (KANDLA / "rf_cache" / "sionna_rt").exists(), reason="Kandla caches missing")


@pytest.fixture()
def ctx():
    load_all()
    return ToolContext(SiteView(KANDLA))


def call(ctx, name, **args):
    spec = REGISTRY[name]
    return spec.fn(ctx, spec.args.model_validate(args))


# --- F ---------------------------------------------------------------------------------------------------------
def test_event_search_filters_and_paginates(ctx):
    d = call(ctx, "event_search", event_type="zone_enter", limit=3).data
    assert d["total"] > 3 and len(d["items"]) == 3 and d["next_cursor"]
    assert all(it["ref"].startswith("evt:") and it["type"] == "zone_enter" for it in d["items"])
    d2 = call(ctx, "event_search", event_type="zone_enter", limit=3, cursor=d["next_cursor"]).data
    assert {i["ref"] for i in d2["items"]}.isdisjoint({i["ref"] for i in d["items"]})


def test_track_history_downsamples(ctx):
    tid = next(iter(ctx.site.tracks))
    p = ctx.site.points[tid]["t"]
    d = call(ctx, "track_history", track_ref=f"trk:{tid}", time_range_s=[float(p[0]), float(p[-1])], step_s=20).data
    ts = [q["t_s"] for q in d["points"]]
    assert ts == sorted(ts) and all(b - a >= 19.9 for a, b in zip(ts, ts[1:])) and len(ts) <= 60


# --- G ---------------------------------------------------------------------------------------------------------
def test_sql_readonly_runs_a_select(ctx):
    d = call(ctx, "sql_readonly", query="SELECT band, count(*) AS n FROM tracks GROUP BY band ORDER BY n DESC").data
    assert d["columns"] == ["band", "n"] and d["rows"]


@pytest.mark.parametrize("q", ["DROP TABLE tracks", "ATTACH 'x.duckdb' AS x", "SELECT 1; SELECT 2",
                               "SELECT * FROM read_parquet('data/site_kandla/objects.parquet')",
                               "SELECT * FROM emitter_points", "COPY tracks TO 'out.csv'"])
def test_sql_readonly_rejects_anything_but_whitelisted_selects(ctx, q):
    with pytest.raises(ToolError) as e:
        call(ctx, "sql_readonly", query=q)
    assert e.value.code == "NOT_PERMITTED" and e.value.hint


# --- E ---------------------------------------------------------------------------------------------------------
def test_footprint_shrinks_as_threshold_rises_and_collateral_lists_assets(ctx):
    lo = call(ctx, "effect_footprint", effector="sensor:S1", band="ism_2400", eirp_dbm=33, threshold_dbm=-90).data
    hi = call(ctx, "effect_footprint", effector="sensor:S1", band="ism_2400", eirp_dbm=33, threshold_dbm=-60).data
    assert lo["area_m2"] > hi["area_m2"] > 0 and lo["ref"].startswith("fp:")
    c = call(ctx, "collateral_check", fp_ref=lo["ref"]).data
    assert len(c["violations"]) == min(c["n_violations"], 20) and c["n_violations"] >= 1   # page cap 20
    for v in c["violations"]:
        assert v["kind"] in ("protected_asset", "friendly_emitter", "civil_comms", "aviation")
        assert v["ref"].split(":")[0] in ("obj", "fe", "zone") and "predicted_level_dbm" in v


def test_collateral_check_needs_a_footprint_from_this_episode(ctx):
    with pytest.raises(ToolError) as e:
        call(ctx, "collateral_check", fp_ref="fp:deadbeef")
    assert e.value.code == "UNKNOWN_REF"


# --- D ---------------------------------------------------------------------------------------------------------
PAYLOAD = {"target_track": "trk:0000", "t0_s": 100.0, "start": {"x_m": 0.0, "y_m": 0.0, "z_m": 80.0},
           "speed_mps": 12.0, "budget_moves": 3, "sigma_deg": 3.0,
           "prior": {"x_m": 520.0, "y_m": 480.0, "a_m": 240.0, "b_m": 120.0, "theta_deg": 30.0},
           "waypoints": [{"id": "wp:01", "x_m": 500.0, "y_m": 0.0, "z_m": 80.0, "bearing_deg": 0.0, "snr_db": 25.0},
                         {"id": "wp:02", "x_m": 1000.0, "y_m": 500.0, "z_m": 80.0, "bearing_deg": 270.0,
                          "snr_db": 25.0},
                         {"id": "wp:03", "x_m": 0.0, "y_m": 500.0, "z_m": 80.0, "bearing_deg": 90.0, "snr_db": 25.0}]}


def _drone_ctx(ctx):
    from tools.drone import DroneSim

    ctx.episode["drone"] = DroneSim(PAYLOAD)
    return ctx


def test_move_drone_shrinks_the_uncertainty(ctx):
    ctx = _drone_ctx(ctx)
    s0 = call(ctx, "drone_state").data
    plan = call(ctx, "plan_waypoints", target_ref="trk:0000", k=2).data
    assert len(plan["waypoints"]) == 2 and all("expected_a_m" in w for w in plan["waypoints"])
    m1 = call(ctx, "move_drone", wp_ref="wp:01").data
    m2 = call(ctx, "move_drone", wp_ref="wp:03").data
    a0 = s0["estimate"]["uncertainty"]["a_m"] * s0["estimate"]["uncertainty"]["b_m"]
    a2 = m2["estimate"]["uncertainty"]["a_m"] * m2["estimate"]["uncertainty"]["b_m"]
    assert a2 < 0.3 * a0
    est = m2["estimate"]["position"]
    assert abs(est["x_m"] - 500) < 40 and abs(est["y_m"] - 500) < 40
    assert m1["measurement"]["bearing_deg"] == 0.0


def test_move_drone_budget_and_unknown_waypoint(ctx):
    ctx = _drone_ctx(ctx)
    with pytest.raises(ToolError) as e:
        call(ctx, "move_drone", wp_ref="wp:99")
    assert e.value.code == "UNKNOWN_REF"
    for w in ("wp:01", "wp:02", "wp:03"):
        call(ctx, "move_drone", wp_ref=w)
    with pytest.raises(ToolError) as e:
        call(ctx, "move_drone", wp_ref="wp:01")
    assert e.value.code == "BUDGET_EXCEEDED"


def test_drone_tools_without_a_drone_task_say_so(ctx):
    with pytest.raises(ToolError) as e:
        call(ctx, "drone_state")
    assert e.value.code == "NOT_PERMITTED"
