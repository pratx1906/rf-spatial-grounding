"""Belief-side RF maps from the trust ensemble (nominal + B1-B3, D-031): ghost risk, azimuth spread, trust."""

from pathlib import Path

import numpy as np
import pytest

from envbuild import rf_maps

REPO = Path(__file__).resolve().parents[2]
MAPS = REPO / "data" / "site_kandla" / "rf_cache" / "maps"


def member(az_strongest, ptype=0, n=3, h=1):
    """Synthetic node cache: strongest path at the given azimuth everywhere (plus one weak path)."""
    k = 2
    az = np.full((h, n, n, k), np.nan, "f4")
    az[..., 0] = az_strongest
    az[..., 1] = (az_strongest + 90) % 360
    hre = np.zeros((h, n, n, k), "f4")
    hre[..., 0], hre[..., 1] = 1.0, 0.1
    pt = np.full((h, n, n, k), 255, "u1")
    pt[..., 0], pt[..., 1] = ptype, 1
    return {"az_deg": az, "h_re": hre, "h_im": np.zeros_like(hre), "ptype": pt}


def geo(n=3, h=1, value=45.0):
    return np.full((h, n, n), value, "f4")


def test_identical_members_give_zero_spread_and_full_trust():
    out = rf_maps.node_metrics([member(45.0)] * 4, geo())
    assert np.allclose(out["az_spread_deg"], 0.0, atol=1e-3)
    assert np.allclose(out["ghost_risk"], 0.0)
    assert np.allclose(out["trust"], 1.0)


def test_ghost_risk_counts_members_whose_strongest_bearing_is_off_the_geometric_bearing():
    out = rf_maps.node_metrics([member(45.0), member(45.0), member(120.0, ptype=1), member(45.0)], geo())
    assert np.allclose(out["ghost_risk"], 0.25)
    assert np.all(out["az_spread_deg"] > 5)
    assert np.all(out["trust"] < 0.9)


def test_a_reflection_on_the_geometric_bearing_is_not_a_ghost():
    out = rf_maps.node_metrics([member(45.0, ptype=1)] * 4, geo())        # sea/ground bounce, same azimuth
    assert np.allclose(out["ghost_risk"], 0.0) and np.allclose(out["nlos_frac"], 1.0)


def test_cells_without_paths_in_any_member_are_nan():
    m = member(45.0)
    m["ptype"][:] = 255
    out = rf_maps.node_metrics([m] * 4, geo())
    assert np.isnan(out["ghost_risk"]).all() and np.isnan(out["trust"]).all()


def test_maps_module_never_names_the_world_side():
    assert "world_cache" not in Path(rf_maps.__file__).read_text()


@pytest.mark.skipif(not (MAPS / "ism_2400.npz").exists(), reason="maps not built")
def test_built_maps_use_the_coverage_union_grid_and_valid_ranges():
    for band in ("sub_ghz_868_915", "ism_2400", "ism_5800"):
        m = np.load(MAPS / f"{band}.npz")
        c = np.load(REPO / "data" / "site_kandla" / "rf_cache" / "coverage" / f"{band}.npz")
        for k in ("xs", "ys", "heights"):
            assert np.array_equal(m[k], c[k])
        g = m["ghost_risk"]
        assert g.shape == (len(m["sensors"]), len(m["heights"]), len(m["ys"]), len(m["xs"]))
        v = g[np.isfinite(g)]
        assert v.size and v.min() >= 0 and v.max() <= 1
        t = m["site_trust"][np.isfinite(m["site_trust"])]
        assert t.min() >= 0 and t.max() <= 1


@pytest.mark.skipif(not (MAPS / "ism_2400.npz").exists(), reason="maps not built")
def test_trust_at_keeps_its_contract():
    from sim.belief.trust import trust_at

    rf = REPO / "data" / "site_kandla" / "rf_cache"
    r = trust_at(rf, "S1", "ism_2400", 942.7 + 300, 956.4, 40.0)
    assert set(r) == {"trust", "spread_deg", "dominance", "flags", "backends"}
    assert 0 <= r["trust"] <= 1
    assert any("ensemble" in b for b in r["backends"])


@pytest.mark.skipif(not (MAPS / "ism_2400.npz").exists(), reason="maps not built")
def test_per_sensor_layers_stop_at_the_node_footprint_radius():
    import yaml

    m = np.load(MAPS / "ism_2400.npz")
    pose = {s["id"]: s["pose"] for s in yaml.safe_load((REPO / "data" / "site_kandla" / "sensors.yaml").read_text())["sensors"]}
    X, Y = np.meshgrid(m["xs"], m["ys"])
    for si, sid in enumerate(m["sensors"]):
        d = np.hypot(X - pose[str(sid)]["x_m"], Y - pose[str(sid)]["y_m"])
        inside = np.isfinite(m["ghost_risk"][si]).any(0)
        assert inside.any() and d[inside].max() <= rf_maps.FOOTPRINT_M + 15, sid    # half a 20 m cell of slack
