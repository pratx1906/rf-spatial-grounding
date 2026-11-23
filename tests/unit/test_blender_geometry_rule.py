"""STRICT rule (2026-10-20): the detailed Blender world is the only production geometry. Every v2 RF scene must be
built from the Blender export and must never reference the OSM-extrusion RF scene (data/site_kandla/rf_scene).
Every data version is checked: v2 demo scenes on rf_scene_blender; v3 scenes on the simplified Blender copy
rf_scene_blender_rf (RF convergence plan). A version whose scenes are not exported yet is skipped."""

import json
from pathlib import Path

import pytest

from envbuild.variant_roots import V3_RF_SCENES, data_roots

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
DIRS = [SITE / "rf_cache_v2" / "scene"] + sorted((SITE / "rf_cache_v2" / "ensemble_scene").glob("*")) \
    + sorted((SITE / "world_cache_v2" / "rf_scene").glob("*"))
pytestmark = pytest.mark.skipif(not (SITE / "rf_scene_blender" / "objects.json").exists(), reason="Blender export missing")


@pytest.mark.parametrize("d", DIRS, ids=lambda d: d.name)
def test_v2_scene_is_blender_geometry(d):
    man = json.loads((d / "manifest.json").read_text())
    assert man["geometry"] == "blender"
    for band in ("ism_2400", "ism_5800", "sub_ghz_868_915"):
        xml = (d / f"scene_{band}.xml").read_text()
        assert "rf_scene_blender/meshes" in xml
        assert "/rf_scene/meshes" not in xml and "rf_scene\meshes" not in xml


def _version_dirs(v: str) -> list:
    r = data_roots(v)
    ds = [r["rf_cache"] / "scene"] + sorted((r["rf_cache"] / "ensemble_scene").glob("*")) \
        + sorted((r["world_cache"] / "rf_scene").glob("*"))
    ds = [d for d in ds if (d / "manifest.json").exists()]
    if not ds:
        return [pytest.param(v, None, marks=pytest.mark.skip(reason=f"{v} scenes not exported yet"), id=f"{v}-none")]
    return [pytest.param(v, d, id=f"{v}-{d.name}") for d in ds]


@pytest.mark.parametrize(("v", "d"), _version_dirs("v3"))
def test_v3_scene_is_simplified_blender_geometry(v, d):
    man = json.loads((d / "manifest.json").read_text())
    assert man["geometry"] == "blender" and man.get("data_version") == v
    srcs = {p.name for p in V3_RF_SCENES.values()}                    # rf_scene_blender_rf | _rfr (Task 3b)
    src = man.get("rf_scene", data_roots(v)["rf_scene"].name)
    assert src in srcs, src
    for band in ("ism_2400", "ism_5800", "sub_ghz_868_915"):
        xml = (d / f"scene_{band}.xml").read_text()
        assert f"{src}/meshes" in xml
        assert "rf_scene_blender/meshes" not in xml                    # never the full demo export
        assert "/rf_scene/meshes" not in xml and r"rf_scene\meshes" not in xml


def test_v3_blender_export_is_simplified_from_the_blender_world():
    f = data_roots("v3")["rf_scene"] / "objects.json"
    if not f.exists():
        pytest.skip("v3 simplified Blender export not made yet")
    m = json.loads(f.read_text())
    assert m["source"].endswith("kandla_world.blend") and "simplify" in m
    cols = {r["collection"] for r in m["shapes"] if r["group"] == "structure"}
    assert {"KW_tanks", "KW_buildings", "KW_buildings_ml", "KW_jetties", "KW_ships", "KW_pipelines", "KW_power"} <= cols


@pytest.mark.parametrize("geom", ["rf", "rfr"])
def test_v3_blender_sources_are_simplified_copies(geom):
    f = data_roots("v3", geom)["rf_scene"] / "objects.json"
    if not f.exists():
        pytest.skip(f"v3 {geom} export not made yet")
    m = json.loads(f.read_text())
    assert m["source"].endswith("kandla_world.blend") and "simplify" in m
    assert ("rf_relevance" in m["simplify"]["params"]) == (geom == "rfr")


def test_blender_export_covers_the_structure_collections():
    m = json.loads((SITE / "rf_scene_blender" / "objects.json").read_text())
    cols = {r["collection"] for r in m["shapes"] if r["group"] == "structure"}
    assert {"KW_tanks", "KW_buildings", "KW_buildings_ml", "KW_jetties", "KW_ships", "KW_pipelines", "KW_power"} <= cols
    assert m["tris"]["structure"] > 1_000_000


def test_osm_exporter_refuses_v2():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "envbuild.rf_variants", "--stack", "v2", "--export-all"], cwd=REPO,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "Blender" in (r.stdout + r.stderr)
