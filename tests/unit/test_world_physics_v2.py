"""World physics of the v2 RF scenes (demo-v2 S7.4, audit A Q7/Q8): ships float with the tide, ship shifts follow the
hull heading, and sea roughness (Ament) per sea state."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
BL = SITE / "rf_scene_blender"
need_blender = pytest.mark.skipif(not (BL / "objects.json").exists(), reason="Blender RF export missing")


def _hull(cx, cy, heading_rad, L=180.0, W=30.0, z0=-15.0, z1=28.5, n=12):
    """A box hull densely sampled along its length (like a real hull mesh), rotated by heading (from +x, ccw)."""
    xs = np.linspace(-L / 2, L / 2, n)
    pts = [(x, y, z) for x in xs for y in (-W / 2, W / 2) for z in (z0, z1)]
    p = np.array(pts, float)
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    out = p.copy()
    out[:, 0] = cx + p[:, 0] * c - p[:, 1] * s
    out[:, 1] = cy + p[:, 0] * s + p[:, 1] * c
    return out


# ---------------------------------------------------------------- ship heading and tide
@pytest.mark.parametrize("deg", [0.0, 30.0, 60.0, 95.0, 140.0])
def test_hull_axis_is_the_principal_axis_of_the_hull(deg):
    from envbuild.blender_rf import hull_axis

    ax = hull_axis(_hull(500.0, -200.0, math.radians(deg)))
    assert np.isclose(np.linalg.norm(ax), 1.0)
    ang = math.degrees(math.atan2(ax[1], ax[0])) % 180.0
    assert min(abs(ang - deg % 180), 180 - abs(ang - deg % 180)) < 0.5
    # sign convention: positive along the dominant bbox axis (the old shift direction), so signs keep their meaning
    dom = 0 if np.ptp(_hull(0, 0, math.radians(deg))[:, 0]) >= np.ptp(_hull(0, 0, math.radians(deg))[:, 1]) else 1
    assert ax[dom] > 0


def test_ship_offset_moves_along_the_heading_and_floats_with_constant_draft():
    from envbuild.blender_rf import MSL_M, ship_offset

    hull = _hull(1000.0, -900.0, math.radians(40.0))
    off = ship_offset(hull, shift_m=40.0, water_z_m=-6.0)
    assert np.allclose(off[:2], 40.0 * np.array([math.cos(math.radians(40)), math.sin(math.radians(40))]), atol=0.05)
    assert off[2] == pytest.approx(-6.0 - MSL_M)
    moved = hull + off
    # draft (water level - keel) is the same as at mean sea level
    assert (-6.0 - moved[:, 2].min()) == pytest.approx(MSL_M - hull[:, 2].min())
    assert np.allclose(ship_offset(hull, 0.0, MSL_M), 0.0)


@need_blender
def test_export_v2_ships_follow_the_tide_and_shift_along_their_hulls(tmp_path):
    from envbuild.blender_rf import MSL_M, _ply_read, export_v2, hull_axis

    cfg = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())
    objs = json.loads((BL / "objects.json").read_text())["shapes"]
    ships = [r for r in objs if r["object"].startswith("ship_obs_")]
    spec = {"side": "world", "water_z_m": -6.0, "ships": {"shift_along_m": {"ship_obs_3": 40.0}},
            "material_seed": None, "sea_state": 2}
    man = export_v2("Ttest", spec, cfg, out=tmp_path)
    xml = (tmp_path / "scene_ism_2400.xml").read_text()
    dz = -6.0 - MSL_M
    for r in ships:
        src, _ = _ply_read(BL / "meshes" / f"{r['shape']}.ply")
        assert f'id="mesh-{r["shape"]}"' in xml
        got, _ = _ply_read(tmp_path / "meshes" / f"{r['shape']}.ply")
        shift = np.zeros(3)
        if r["object"] == "ship_obs_3":
            steel, _ = _ply_read(BL / "meshes" / "ship_obs_3__steel.ply")
            shift[:2] = 40.0 * hull_axis(steel)[:2]
        assert np.allclose(got, src + shift + [0, 0, dz], atol=1e-3), r["shape"]
    assert man["sea_state"] == 2 and man["sea_sigma_h_m"] > 0
    assert man["hidden_features"]["ships_floating_dz_m"] == pytest.approx(dz)


@need_blender
def test_export_v2_at_mean_sea_level_keeps_ships_in_place(tmp_path):
    from envbuild.blender_rf import MSL_M, export_v2

    cfg = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())
    export_v2("Tmsl", {"side": "world", "water_z_m": MSL_M, "ships": "all", "material_seed": None}, cfg, out=tmp_path)
    xml = (tmp_path / "scene_ism_2400.xml").read_text()
    assert "rf_scene_blender/meshes/ship_obs_3__steel.ply" in xml        # referenced in place, not rewritten


# ---------------------------------------------------------------- sea roughness
def test_ament_rho_matches_the_formula_at_known_grazing_angles():
    from sim.rf.materials import ament_rho

    f, sig = 2.44e9, 0.075
    lam = 299792458.0 / f
    for psi_deg, expect in ((0.0, 1.0), (1.0, None), (5.0, None), (20.0, None), (90.0, None)):
        s = math.sin(math.radians(psi_deg))
        ref = math.exp(-2 * (2 * math.pi * sig * s / lam) ** 2)
        assert float(ament_rho(s, sig, f)) == pytest.approx(ref if expect is None else expect, rel=1e-9)
    # hand values: 5 deg -> g/2 = 2 (2 pi 0.075 0.0871557 / 0.1228657)^2 = 0.2236 -> rho 0.7996
    assert float(ament_rho(math.sin(math.radians(5.0)), sig, f)) == pytest.approx(0.7996, abs=2e-4)
    assert float(ament_rho(0.3, 0.0, f)) == 1.0


def test_sea_state_heights_are_monotone_and_bounded():
    from sim.rf.materials import sea_state_sigma_h

    s = [sea_state_sigma_h(k) for k in range(5)]
    assert s[0] == 0.0 and all(b > a for a, b in zip(s, s[1:]))
    assert s[4] * 4 <= 2.5                      # WMO sea state 4: H_1/3 <= 2.5 m
    with pytest.raises(ValueError):
        sea_state_sigma_h(7)


def test_roughness_factor_per_path_uses_the_incoming_grazing_angle():
    from envbuild.air7311_precompute import sea_roughness_factor
    from sim.rf.materials import ament_rho

    tx = np.array([0.0, 0.0, 30.0])
    f, sig, WATER, QUAY = 2.44e9, 0.2, 7, 3
    # paths (one receiver): 0 LOS; 1 sea bounce; 2 quay bounce; 3 quay then sea; 4 sea then sea
    inter = np.array([[[0, 1, 1, 1, 1]], [[0, 0, 0, 1, 1]]])
    objs = np.array([[[-1, WATER, QUAY, QUAY, WATER]], [[-1, -1, -1, WATER, WATER]]])
    v = np.zeros((2, 1, 5, 3))
    v[0, 0, 1] = (100.0, 0.0, -3.5)
    v[0, 0, 2] = (50.0, 0.0, 5.0)
    v[0, 0, 3] = (50.0, 20.0, 10.0)
    v[1, 0, 3] = (150.0, 20.0, -3.5)
    v[0, 0, 4] = (100.0, 0.0, -3.5)
    v[1, 0, 4] = (200.0, 0.0, -3.5)          # second sea bounce: incoming segment is horizontal -> sin psi = 0
    fac = sea_roughness_factor(inter, objs, v, tx, [WATER], sig, f)[0]
    s1 = 33.5 / math.hypot(100.0, 33.5)
    s3 = 13.5 / math.hypot(100.0, 13.5)
    assert fac[0] == 1.0 and fac[2] == 1.0
    assert fac[1] == pytest.approx(float(ament_rho(s1, sig, f)))
    assert fac[3] == pytest.approx(float(ament_rho(s3, sig, f)))
    assert fac[4] == pytest.approx(float(ament_rho(s1, sig, f)) * 1.0)
    assert np.all(sea_roughness_factor(inter, objs, v, tx, [WATER], 0.0, f) == 1.0)


def test_reduce_applies_roughness_before_the_top_k_selection():
    from envbuild.air7311_precompute import _reduce

    n_rx, n_p, f = 1, 3, 2.44e9
    tau = np.zeros((n_rx, 1, n_p))
    raw = {"ar": np.array([1.0, 0.9, 0.5]).reshape(n_rx, 1, 1, 1, n_p), "ai": np.zeros((n_rx, 1, 1, 1, n_p)),
           "phi_t": np.zeros((n_rx, 1, n_p)), "theta_t": np.full((n_rx, 1, n_p), math.pi / 2), "tau": tau,
           "interactions": np.array([[[[0, 1, 1]]]]).reshape(1, n_rx, 1, n_p),
           "objects": np.array([[[[-1, 7, 3]]]]).reshape(1, n_rx, 1, n_p),
           "vertices": np.array([[0, 0, 0], [100.0, 0, -3.5], [50.0, 0, 5]]).reshape(1, n_rx, 1, n_p, 3)}
    names = {7: "water", 3: "quay"}
    flat = _reduce(raw, names, n_rx, f, top_k=2)
    rough = _reduce(raw, names, n_rx, f, top_k=2, rough={"sigma_h_m": 0.5, "sea_objects": ["water"],
                                                        "tx": [0.0, 0.0, 30.0]})
    assert list(flat["ptype"][0]) == [0, 1] and flat["via"][0, 1] == 1            # the sea path is 2nd when flat
    assert rough["via"][0, 1] == 0                                                # rough: the quay path overtakes
    assert abs(rough["h_re"][0, 1]) == pytest.approx(0.5, rel=1e-5)


# ---------------------------------------------------------------- config and trace commands
def test_variants_v2_declare_a_sea_state_and_some_worlds_are_rough():
    v = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())["variants_v2"]
    assert all(spec.get("sea_state") in (0, 1, 2, 3, 4) for spec in v.values())
    assert v["N"]["sea_state"] == 0 and v["W0"]["sea_state"] == 0
    assert sum(1 for spec in v.values() if spec["side"] == "world" and spec["sea_state"] > 0) >= 2


def test_precompute_command_passes_the_sea_state():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pv", REPO / "scripts" / "precompute_variants.py")
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    v = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())["variants_v2"]
    rough = next(k for k, s in v.items() if s["sea_state"] > 0)
    cmd = pv.command(rough, v, "v2", 25000)
    assert cmd[cmd.index("--sea-state") + 1] == str(v[rough]["sea_state"])
    assert "--sea-state" not in pv.command("N", v, "v2", 25000)
    extra = pv.command("N", v, "v2", 25000, solver={"samples_per_src": 3_000_000, "max_depth": 3, "top_k": 32})
    assert extra[extra.index("--samples-per-src") + 1] == "3000000" and "--max-depth" in extra and "--top-k" in extra
