"""Kandla belief-visible asset configs: synthetic friendly emitters on real mounts, proposed aviation corridor."""

from pathlib import Path

import pytest
import yaml

from sim.belief.site_view import SiteView

REPO = Path(__file__).resolve().parents[2]
KANDLA = REPO / "data" / "site_kandla"
pytestmark = pytest.mark.skipif(not (KANDLA / "meta.yaml").exists(), reason="Kandla site not built")


def test_friendly_emitters_are_synthetic_and_mounted_on_real_objects():
    site = SiteView(KANDLA)
    assert len(site.friendly) >= 6
    for e in site.friendly:
        assert e["placement"] == "synthetic"
        assert e["band"] in site.bands
        assert e["mount_ref"].split(":", 1)[1] in site.objects
        assert {"id", "name", "kind", "x_m", "y_m", "z_m", "freq_mhz"} <= set(e)
    assert {e["kind"] for e in site.friendly} >= {"cctv_link", "port_wifi", "tank_telemetry", "lte_site"}


def test_aviation_corridor_is_a_proposed_zone():
    site = SiteView(KANDLA)
    z = [z for z in site.zones.values() if z.kind == "aviation"]
    assert z, "an aviation zone is needed for collateral checks"
    feats = yaml.safe_load((KANDLA / "zones.geojson").read_text())["features"]
    prov = next(f["properties"]["provenance"] for f in feats if f["properties"]["kind"] == "aviation")
    assert "HUMAN" in str(prov)
