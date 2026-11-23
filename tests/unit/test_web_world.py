"""Console web world (data/site_kandla/web, built by envbuild/blender/bake_web.py and scripts/web_ground.py):
every baked tile's files exist, the browser download stays inside the budget (about 80 MB, no file over 100 MB),
and the illustrative markings are declared."""

import json
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "data" / "site_kandla" / "web"
pytestmark = pytest.mark.skipif(not (WEB / "manifest.json").exists(), reason="web world not built")


def _man() -> dict:
    return json.loads((WEB / "manifest.json").read_text())


def _loaded_files(man: dict) -> set:
    """What ss3d.js fetches: tiles (+ baked atlases), ground, terrain, sky, water normals, detail textures, vegetation."""
    files = {"terrain.json", "terrain.bin", "ground.jpg", "detail.jpg", "sky_1k.hdr", "waternormals.jpg"}
    for g, t in man["tiles"].items():
        if g == "sensors":
            continue
        files.add(t["file"])
        if t.get("baked"):
            files |= {t["baked"]["albedo"], t["baked"]["ao"]}
            if t["baked"].get("gi"):                      # bake_web.py --gi: baked indirect light
                files.add(t["baked"]["gi"])
    gr = man.get("ground") or {}
    files |= {gr[k] for k in ("albedo", "ao", "cls") if gr.get(k)}
    for v in (man.get("vegetation") or {}).values():
        files |= {v["file"], v.get("bin") or v["instances"]}
    files |= {f"tex/{p.name}" for p in (WEB / "tex").glob("*.jpg")}
    if man.get("look"):                                   # envbuild/blender/web_look.py: Blender view-transform LUT
        files.add(man["look"]["lut"]["file"])
    if man.get("water"):                                  # scripts/web_ground.py: Blender water flow (wake foam)
        files.add(man["water"]["flow"])
        if man["water"].get("shore"):                     # signed shore distance (Blender's shore foam band)
            files.add(man["water"]["shore"])
    return files


def test_manifest_files_exist():
    man = _man()
    missing = sorted(f for f in _loaded_files(man) if not (WEB / f).exists())
    assert not missing, missing


def test_download_budget():
    man = _man()
    total = sum((WEB / f).stat().st_size for f in _loaded_files(man) if (WEB / f).exists())
    assert total < 80e6, f"web world {total / 1e6:.1f} MB"
    big = [p.name for p in WEB.rglob("*") if p.is_file() and p.stat().st_size > 100e6]
    assert not big, big


def test_markings_declared_illustrative():
    man = _man()
    for g, t in man["tiles"].items():
        if t.get("markings"):
            assert "illustrative" in t["markings"] and "synthetic" in t["markings"], g


def test_look_from_blend():
    """The console's camera look is exported from the .blend (web_look.py), not guessed."""
    look = _man().get("look")
    if not look:
        pytest.skip("look not exported")
    assert look["view_transform"] == "AgX" and look["lut"]["n"] >= 33
    assert 0 < look["world"]["strength"] <= 2 and len(look["world"]["irradiance_up"]) == 3
    assert 0 <= look["sun"]["elevation_deg"] <= 90 and look["sun"]["strength"] > 0


def test_tanks_baked_indirect_light():
    """Render parity round 2: the tanks carry Cycles' indirect light (bake_web.py --gi), the water its shore distance."""
    man = _man()
    b = man["tiles"]["tanks"]["baked"]
    assert b.get("gi") and b.get("gi_scale", 0) > 0 and (WEB / b["gi"]).exists()
    w = man["water"]
    assert w.get("shore") and w["shore_span"] > 0 and (WEB / w["shore"]).exists()
