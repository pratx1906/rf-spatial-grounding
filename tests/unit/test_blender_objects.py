"""v2 site catalogue from the Blender world: modelled heights, observed ships, no synthetic tankers."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (REPO / "data/site_kandla/rf_scene_blender/objects.json").exists(),
                                reason="Blender export missing")


def test_catalogue_uses_blender_heights_and_observed_ships():
    from envbuild.blender_objects import catalogue
    from envbuild.site_kandla import build

    objs, st = catalogue(build(13)["objs"])
    kinds = {}
    for o in objs:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    assert kinds["tank"] == 263 and kinds["ship"] == 13 and st["ships_dropped"] == 4
    assert not any(o.id.startswith("ship_s") for o in objs)
    assert st["height_from_blender"] > 400
    ship = next(o for o in objs if o.id == "ship_obs_3")
    assert ship.geom.area > 500 and 5 < ship.height_m < 60
