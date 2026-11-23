import shutil
from collections import namedtuple
from pathlib import Path

from scripts import disk_budget as db

Usage = namedtuple("Usage", "total used free")
TB = 1e12


def _fake_disk(monkeypatch, free_by_disk: dict[str, float], total=1.5 * TB):
    def fake(path):
        for prefix, free in free_by_disk.items():
            if str(path).startswith(prefix):
                return Usage(total, total - free, free)
        return Usage(total, total - free_by_disk["*"], free_by_disk["*"])
    monkeypatch.setattr(shutil, "disk_usage", fake)


def test_defaults_load_from_machine_yaml():
    b, min_free, paths = db.load_defaults()
    assert min_free == 0.15
    assert paths["data"] == "data"
    assert b.keep_checkpoints == 4


def test_estimate_scales_linearly_with_variants():
    base = db.Budget(variants=4).estimate_gb()["world_cache"]
    double = db.Budget(variants=8).estimate_gb()["world_cache"]
    assert abs(double - 2 * base) <= 0.02


def test_finer_grid_costs_more():
    assert db.Budget(fine_area_fraction=1.0).estimate_gb()["total"] > db.Budget().estimate_gb()["total"]
    assert db.Budget(fine_grid_m=5).estimate_gb()["total"] > db.Budget(fine_grid_m=10).estimate_gb()["total"]


def test_check_refuses_when_floor_would_be_crossed(tmp_path, monkeypatch):
    _fake_disk(monkeypatch, {"*": 0.3 * TB})              # 300 GB free of 1500 GB
    ok, info = db.check(est_total_gb=100, target=tmp_path, min_free_fraction=0.15)
    assert not ok and info["free_after_fraction"] == round(200 / 1500, 3)   # 13.3% < 15%
    ok, info = db.check(est_total_gb=50, target=tmp_path, min_free_fraction=0.15)
    assert ok and info["free_after_fraction"] == round(250 / 1500, 3)       # 16.7% >= 15%


def test_check_uses_tightest_physical_disk(tmp_path, monkeypatch):
    # ext4 virtual disk reports lots of space; the real C: drive is nearly full.
    monkeypatch.setattr(db, "physical_targets", lambda t: [t, Path("/mnt/c")])
    _fake_disk(monkeypatch, {"/mnt/c": 0.2 * TB, "*": 1.0 * TB})
    ok, info = db.check(est_total_gb=50, target=tmp_path, min_free_fraction=0.15)
    assert info["disk"] == "/mnt/c" and not ok


def test_cli_accepts_decimal_for_float_fields(tmp_path, monkeypatch):
    _fake_disk(monkeypatch, {"*": 1.0 * TB})
    assert db.main(["--target", str(tmp_path), "--fine-grid-m", "7.5", "--traces-gb", "7.5"]) == 0


def test_cli_check_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "physical_targets", lambda t: [t])
    _fake_disk(monkeypatch, {"*": 1.0 * TB})
    assert db.main(["--target", str(tmp_path), "--check", "--variants", "100000"]) == 2
    assert db.main(["--target", str(tmp_path), "--check"]) == 0
