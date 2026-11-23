"""Family S signal tools (RF phase 2, Task 9)."""

from pathlib import Path

import pytest

from sim.belief.site_view import SiteView
from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "data" / "scenarios" / "kandla_s13"
V2 = REPO / "data" / "scenarios" / "kandla_s13_X"


def call(ctx, name, **args):
    spec = REGISTRY[name]
    return spec.fn(ctx, spec.args.model_validate(args))


def test_registered_with_short_descriptions():
    load_all()
    for n in ("signal_detail", "pilot_search", "tdoa_fix", "noise_floor"):
        assert REGISTRY[n].family == "S" and len(REGISTRY[n].description) <= 100


@pytest.mark.skipif(not V1.exists(), reason="v1 scenario missing")
def test_v1_sets_say_signal_data_is_not_available():
    load_all()
    ctx = ToolContext(SiteView(V1))
    tid = next(iter(ctx.site.tracks))
    with pytest.raises(ToolError) as e:
        call(ctx, "signal_detail", track_ref=f"trk:{tid}")
    assert e.value.code == "OUT_OF_COVERAGE" and "v2" in e.value.hint


@pytest.mark.skipif(not V2.exists(), reason="v2 scenario not built")
def test_v2_signal_tools_answer():
    load_all()
    ctx = ToolContext(SiteView(V2))
    tid = next(t for t, tr in ctx.site.tracks.items())
    d = call(ctx, "signal_detail", track_ref=f"trk:{tid}").data
    assert d["n_bursts"] > 0 and d["protocol_estimate"]
    n = call(ctx, "noise_floor", sensor_ref="sensor:S3", band="ism_2400").data
    assert n["noise_dbm_per_mhz"] > -120


@pytest.mark.skipif(not V2.exists(), reason="v2 scenario not built")
def test_pilot_search_ranks_by_controller_probability():
    """Controllers are rarely the argmax class (few are tracked); candidates are tracks with p(controller) >= 0.2."""
    load_all()
    ctx = ToolContext(SiteView(V2))
    best = max(ctx.site.tracks, key=lambda t: ctx.site.tracks[t]["klass_probs"]["controller"])
    if ctx.site.tracks[best]["klass_probs"]["controller"] < 0.2:
        pytest.skip("no controller-like track in this build")
    drone = next(t for t, tr in ctx.site.tracks.items() if t != best and max(tr["klass_probs"], key=tr["klass_probs"].get) == "drone"
                 and min(tr["last_seen"], ctx.site.tracks[best]["last_seen"]) > max(tr["first_seen"], ctx.site.tracks[best]["first_seen"]))
    c = call(ctx, "pilot_search", near_track_ref=f"trk:{drone}", max_range_m=6000).data["candidates"]
    assert c and all(x["p_controller"] >= 0.2 for x in c)
    assert f"trk:{best}" in {x["ref"] for x in c}


V2_JAM = REPO / "data" / "scenarios" / "kandla_s15_X"


@pytest.mark.skipif(not V2_JAM.exists(), reason="v2 scenario not built")
def test_noise_floor_skips_dwells_blanked_by_own_jamming():
    """Own-TX dwells are stored with a NULL noise floor; the tool must ignore them, not crash (S2.5 eval crash)."""
    load_all()
    ctx = ToolContext(SiteView(V2_JAM))
    d = call(ctx, "noise_floor", sensor_ref="sensor:S5", band="ism_5800").data
    assert d["noise_dbm_per_mhz"] > -130 and d["blanked_dwells"] > 0
