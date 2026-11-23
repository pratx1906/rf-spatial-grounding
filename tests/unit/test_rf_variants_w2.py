"""W2 water level (demo-v2 S1.5, audit A Q7): the v2 W2 sea plane must sit below the flat land surface (z = 0),
not coplanar with it (a coplanar plane floods the port: water-first reflections 35,369 vs 11,009 in W0)."""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_v2_w2_water_is_below_quay_level():
    v = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())["variants_v2"]
    assert v["W2"]["water_z_m"] == -0.25
    assert all(spec["water_z_m"] < 0 for spec in v.values())
