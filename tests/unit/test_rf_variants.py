"""RF living-world variants: spec loading and the parametrised scene exporter (no Sionna needed)."""

import hashlib
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from envbuild import rf_variants as rv
from envbuild.export_rf import itu_params

REPO = Path(__file__).resolve().parents[2]
MESHES = REPO / "data" / "site_kandla" / "rf_scene" / "meshes"
pytestmark = pytest.mark.skipif(not (REPO / "data" / "site_kandla" / "rf_scene" / "scene.xml").exists(),
                                reason="nominal RF scene not exported")


def read_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    head, _, body = raw.partition(b"end_header\n")
    n = int(re.search(rb"element vertex (\d+)", head).group(1))
    return np.frombuffer(body[: n * 12], "<f4").reshape(n, 3)


def materials(xml: str) -> dict[str, tuple[float, float]]:
    out = {}
    for m in re.finditer(r'id="mat-sc_(\w+)">\s*<float name="relative_permittivity" value="([\d.e+-]+)"/>\s*'
                         r'<float name="conductivity" value="([\d.e+-]+)"/>', xml):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


@pytest.fixture(scope="module")
def variants():
    return rv.load_variants()


@pytest.fixture(scope="module")
def exported(tmp_path_factory, variants):
    root = tmp_path_factory.mktemp("rfv")
    out = {}
    for vid in ("W0", "W1", "W2", "B1"):
        out[vid] = (root / vid, rv.export_variant(variants[vid], root / vid))
    return out


def test_variants_load_with_disjoint_world_and_belief_seeds(variants):
    assert {"W0", "W1t", "W1", "W2", "W3", "B1", "B2", "B3"} <= set(variants)
    world = {v.material_seed for v in variants.values() if v.side == "world" and v.material_seed is not None}
    belief = {v.material_seed for v in variants.values() if v.side == "belief" and v.material_seed is not None}
    assert world and belief and not world & belief


def test_world_and_belief_out_dirs_are_separate(variants):
    assert "world_cache" in str(rv.scene_dir(variants["W1"]))
    assert "world_cache" not in str(rv.scene_dir(variants["B1"]))


def test_one_scene_file_per_band(exported):
    d, _ = exported["W1"]
    for band in rv.BANDS:
        assert (d / f"scene_{band}.xml").exists()


def test_water_plane_moves_and_skirt_spans_water_to_quay(exported):
    d, _ = exported["W2"]
    assert np.allclose(read_ply(d / "meshes" / "ground_water.ply")[:, 2], -0.5)
    skirt = np.concatenate([read_ply(p) for p in (d / "meshes").glob("bank_*.ply")])
    assert skirt[:, 2].min() == pytest.approx(-0.5) and skirt[:, 2].max() == pytest.approx(0.0)


def test_ships_float_and_absent_ships_are_dropped(exported):
    d, _ = exported["W1"]
    xml = (d / "scene_ism_2400.xml").read_text()
    assert "ship_s2" not in xml and "ship_s4" not in xml and "ship_s1" in xml
    hull = read_ply(d / "meshes" / "ship_s1.ply")
    assert hull[:, 2].min() == pytest.approx(-6.0) and hull[:, 2].max() == pytest.approx(16.0)


def test_extra_ship_is_added(exported):
    d, _ = exported["W2"]
    assert "ship_x1" in (d / "scene_ism_5800.xml").read_text()


def test_materials_are_itu_times_scale_per_band(exported, variants):
    for vid in ("W1", "B1"):
        d, _ = exported[vid]
        sc = variants[vid].material_scale
        for band, f in rv.BANDS.items():
            mats = materials((d / f"scene_{band}.xml").read_text())
            er, sg = itu_params("concrete", f / 1e9)
            assert mats["concrete"][0] == pytest.approx(er * sc["concrete"]["eps"], rel=1e-3)
            assert mats["concrete"][1] == pytest.approx(sg * sc["concrete"]["sigma"], rel=1e-3)
            assert mats["steel"][0] == pytest.approx(1.0)            # metal eps never scaled
            assert all(eps >= 1.0 for eps, _ in mats.values())
    m24 = materials((exported["W1"][0] / "scene_ism_2400.xml").read_text())
    m58 = materials((exported["W1"][0] / "scene_ism_5800.xml").read_text())
    assert m24["concrete"][1] != pytest.approx(m58["concrete"][1])   # band-centre values differ


def test_w0_reproduces_the_nominal_scene(exported):
    d, _ = exported["W0"]
    assert not list((d / "meshes").glob("bank_*.ply"))               # no skirt at water_z = 0
    xml = (d / "scene_ism_2400.xml").read_text()
    nominal = (REPO / "data" / "site_kandla" / "rf_scene" / "scene.xml").read_text()
    shapes = set(re.findall(r'id="mesh-([\w-]+)"', nominal))
    assert set(re.findall(r'id="mesh-([\w-]+)"', xml)) == shapes


def test_exporter_never_touches_the_nominal_meshes(exported):
    before = hashlib.sha256(b"".join(sorted(p.name.encode() + struct.pack("<d", p.stat().st_mtime)
                                           for p in MESHES.glob("*.ply")))).hexdigest()
    rv.export_variant(rv.load_variants()["W3"], exported["W0"][0].parent / "W3")
    after = hashlib.sha256(b"".join(sorted(p.name.encode() + struct.pack("<d", p.stat().st_mtime)
                                          for p in MESHES.glob("*.ply")))).hexdigest()
    assert before == after
