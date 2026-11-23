"""v2 cache stack (RF phase 2, Task 2): complete caches on the v1 grid, dual polarisation, drone-end angles, fixed
diffraction recorded in the manifest. Data tests: skipped until rf_cache_v2 is traced. The cache invariants run on
every data version whose nominal caches exist (v2 rf_cache_v2; v3 rf_cache_v3, RF convergence plan Task 5)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from envbuild.variant_roots import data_roots

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
V1, V2 = SITE / "rf_cache", SITE / "rf_cache_v2"
POSE = {s["id"]: s["pose"] for s in yaml.safe_load((SITE / "sensors.yaml").read_text())["sensors"]}


def _z(root, s, band="ism_2400"):
    return np.load(root / f"air7311_{s}" / "spec_diff" / f"{band}.npz")


def _traced(root: Path) -> bool:
    return (root / "air7311_S4" / "spec_diff" / "ism_2400.npz").exists()


need_v2 = pytest.mark.skipif(not _traced(V2), reason="v2 caches not traced")


@pytest.fixture(params=["v2", "v3"])
def cache(request):
    root = data_roots(request.param)["rf_cache"]
    if not _traced(root):
        pytest.skip(f"{request.param} caches not traced ({root.name})")
    return root


@pytest.mark.parametrize("s", ["S1", "S2", "S3", "S4", "S5"])
def test_grid_matches_v1_and_heights_add_the_ground_layer(s, cache):
    a, b = _z(V1, s), _z(cache, s)
    assert np.array_equal(a["xs"], b["xs"]) and np.array_equal(a["ys"], b["ys"])
    assert b["heights"].tolist() == [2.0, 5.0, 15.0, 40.0, 80.0]
    for k in ("h_re", "h_im", "tau_ns", "az_deg", "el_deg", "ptype", "via", "h_vh_re", "h_vh_im", "az_r_deg", "el_r_deg"):
        assert b[k].shape[:3] == (5, len(b["ys"]), len(b["xs"])), k


def test_manifest_records_dual_pol_and_the_diffraction_patch(cache):
    m = json.loads((cache / "air7311_S4" / "manifest.json").read_text())
    assert m["dual_pol"] is True and m["diffraction_patch"] == "sionna-2.2-utd-luebbers-v2"
    assert m["heights_m"] == [2.0, 5.0, 15.0, 40.0, 80.0]


def test_los_paths_equal_v1_and_are_co_polarised(cache):
    a, b = _z(V1, "S3"), _z(cache, "S3")
    los = (a["ptype"][:, :, :, 0] == 0) & (b["ptype"][1:, :, :, 0] == 0)
    va = np.abs(a["h_re"][:, :, :, 0] + 1j * a["h_im"][:, :, :, 0])[los]
    vb = np.abs(b["h_re"][1:, :, :, 0] + 1j * b["h_im"][1:, :, :, 0])[los]
    assert np.percentile(np.abs(20 * np.log10(vb / va)), 99) < 0.05
    vh = np.abs(b["h_vh_re"][1:, :, :, 0] + 1j * b["h_vh_im"][1:, :, :, 0])[los]
    assert np.percentile(20 * np.log10(np.maximum(vh, 1e-30) / vb), 99) < -30


def test_drone_end_angle_of_los_points_back_at_the_node(cache):
    b = _z(cache, "S5")
    p = POSE["S5"]
    X, Y = np.meshgrid(b["xs"], b["ys"])
    back = np.degrees(np.arctan2(p["x_m"] - X, p["y_m"] - Y)) % 360
    far = np.hypot(X - p["x_m"], Y - p["y_m"]) > 30
    for ih in range(5):
        los = (b["ptype"][ih, :, :, 0] == 0) & far
        d = np.abs((b["az_r_deg"][ih, :, :, 0] - back + 180) % 360 - 180)[los]
        assert np.percentile(d, 99) < 0.5


@need_v2
def test_the_diffraction_fix_changes_low_altitude_strongest_paths():
    def strongest_az(z, ih):
        pw = (z["h_re"][ih].astype("f8") ** 2 + z["h_im"][ih].astype("f8") ** 2) * (z["ptype"][ih] != 255)
        return np.take_along_axis(z["az_deg"][ih], pw.argmax(-1)[..., None], -1)[..., 0]
    a, b = _z(V1, "S4"), _z(V2, "S4")
    changed = (np.abs((strongest_az(a, 0) - strongest_az(b, 1) + 180) % 360 - 180) > 5).mean()
    assert changed > 0.02                     # measured 0.074 at 5 m; the solver noise floor is ~0.007
