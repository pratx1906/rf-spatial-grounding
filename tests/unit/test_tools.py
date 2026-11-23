import pytest

from tests.conftest import call
from tools.site import name_score


@pytest.mark.parametrize("query,expect", [
    ("north fuel depot", "obj:tank_farm_north"),
    ("Tank Farm A", "obj:tank_farm_a"),
    ("oil jetty one", "obj:oil_jetty_1"),
    ("W05", "obj:warehouse_w05"),
    ("substation", "obj:substation_main"),
    ("tank fram b", "obj:tank_farm_b"),          # typo tolerance
])
def test_site_resolve_clear_names(ctx, query, expect):
    st, out = call(ctx, "site_resolve", query=query)
    assert st == "ok", out.to_dict() if st == "err" else None
    assert out.data["candidates"][0]["ref"] == expect


@pytest.mark.parametrize("query,n", [("tank farm", 3), ("oil terminal", 2)])
def test_site_resolve_ambiguous_names_never_silently_pick(ctx, query, n):
    st, err = call(ctx, "site_resolve", query=query)
    assert st == "err" and err.code == "AMBIGUOUS_ENTITY" and len(err.candidates) == n
    assert "ask_user" in err.hint


@pytest.mark.parametrize("query", ["5G tower", "LNG terminal", "Berth 9", "Warehouse W42"])
def test_site_resolve_nonexistent(ctx, query):
    st, err = call(ctx, "site_resolve", query=query)
    assert st == "err" and err.code == "NOT_FOUND" and err.candidates


def test_designator_letters_matter():
    assert name_score("tank farm a", "Tank Farm A") == 1.0
    assert name_score("tank farm a", "Tank Farm B") < 0.8


def test_site_describe_object_and_zone(ctx):
    st, out = call(ctx, "site_describe", ref="obj:tank_farm_a")
    assert st == "ok" and out.data["protected"] is True and "zone:oil_terminal_restricted" in out.data["parents"]
    st, out = call(ctx, "site_describe", ref="zone:oil_terminal_restricted")
    assert "obj:tank_farm_a" in out.data["children"]


def test_unknown_ref_gives_did_you_mean(ctx):
    st, err = call(ctx, "site_describe", ref="obj:tank_farm_c")
    assert st == "err" and err.code == "UNKNOWN_REF" and "obj:tank_farm_a" in err.hint


def test_region_contents_paginates(ctx):
    region = {"type": "circle", "center": {"x_m": 0, "y_m": 0}, "radius_m": 600}
    _, p1 = call(ctx, "region_contents", region=region, limit=3)
    assert p1.data["total"] > 3 and p1.data["next_cursor"] == "c3"
    _, p2 = call(ctx, "region_contents", region=region, limit=3, cursor=p1.data["next_cursor"])
    assert not {i["ref"] for i in p1.data["items"]} & {i["ref"] for i in p2.data["items"]}


def test_spatial_relations_echo_definitions(ctx):
    for rel in ("distance", "bearing", "within_m", "north_of"):
        _, out = call(ctx, "spatial_relation", a_ref="obj:tank_farm_north", b_ref="obj:tank_farm_a", relation=rel)
        assert out.definition, rel
    _, out = call(ctx, "spatial_relation", a_ref="obj:tank_farm_north", b_ref="obj:tank_farm_a", relation="north_of")
    assert out.data["value"] is True
    st, err = call(ctx, "spatial_relation", a_ref="obj:berth_1", b_ref="obj:berth_2", relation="between")
    assert st == "err" and err.code == "INVALID_ARGS"


def test_track_search_counts_and_filters(ctx, site):
    _, all_ = call(ctx, "track_search")
    assert all_.data["total"] == len(site.tracks)
    _, b = call(ctx, "track_search", filters={"band": "ism_2400"})
    assert all(i["band"] == "ism_2400" for i in b.data["items"]) and b.definition["band_mhz"] == [2400, 2483.5]
    _, hi = call(ctx, "track_search", filters={"min_confidence": 0.9})
    assert all(i["confidence"] >= 0.9 for i in hi.data["items"])


def test_track_detail_and_coverage_errors(ctx, site):
    tid = next(iter(site.tracks))
    st, out = call(ctx, "track_detail", track_ref=f"trk:{tid}")
    assert st == "ok" and out.data["supporting_sensors"]
    st, err = call(ctx, "track_detail", track_ref=f"trk:{tid}", at_time_s=99999)
    assert st == "err" and err.code == "OUT_OF_COVERAGE" and "Valid times" in err.hint


def test_df_hypotheses_has_no_path_labels(ctx, site):
    tid = next(iter(site.tracks))
    t = float(site.points[tid]["t"][len(site.points[tid]["t"]) // 2])
    st, out = call(ctx, "df_hypotheses", track_ref=f"trk:{tid}", at_time_s=t)
    assert st == "ok"
    text = str(out.data)
    for leak in ("LOS", "reflect", "diffract", "path_type", "via"):
        assert leak not in text


def test_inside_zone_reports_overlap_for_tracks(ctx, site):
    tid = next(iter(site.tracks))
    _, out = call(ctx, "inside_zone", subject=f"trk:{tid}", zone_ref="zone:perimeter")
    assert "uncertainty_overlap_fraction" in out.data and 0 <= out.data["uncertainty_overlap_fraction"] <= 1
