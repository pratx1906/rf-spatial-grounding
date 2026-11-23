"""Data versions (RF convergence plan, Task 1): one helper, envbuild.variant_roots.data_roots, picks the scenario,
cache and scene roots from SCENESCOUT_DATA_VERSION (v2 = the frozen demo, default; v21; v3). Writers refuse the demo
roots unless explicitly allowed, and never write them under v21/v3."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from envbuild import variant_roots as vr

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"
DEMO = {"scenarios": REPO / "data" / "scenarios", "rf_cache": SITE / "rf_cache_v2",
        "world_cache": SITE / "world_cache_v2", "rf_scene": SITE / "rf_scene_blender"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(vr.ENV_VERSION, raising=False)
    monkeypatch.delenv(vr.ENV_ALLOW, raising=False)
    monkeypatch.delenv(vr.ENV_RF_GEOM, raising=False)


def _inside(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or a.resolve().is_relative_to(b.resolve())


# ------------------------------------------------------------------ roots
def test_default_is_demo_roots():
    r = vr.data_roots()
    for k, p in DEMO.items():
        assert r[k] == p, k
    assert r["track_model"] == REPO / "configs" / "track_class_v2.json"
    assert vr.data_roots("v2") == r


def test_env_var_selects_the_version(monkeypatch):
    monkeypatch.setenv(vr.ENV_VERSION, "v3")
    assert vr.data_version() == "v3"
    assert vr.data_roots()["rf_cache"] == SITE / "rf_cache_v3"
    monkeypatch.setenv(vr.ENV_VERSION, "v9")
    with pytest.raises(ValueError):
        vr.data_roots()


def test_v3_roots_are_separate():
    r3 = vr.data_roots("v3")
    assert r3["scenarios"] == REPO / "data" / "scenarios_v3"
    assert r3["rf_cache"] == SITE / "rf_cache_v3"
    assert r3["world_cache"] == SITE / "world_cache_v3"
    assert r3["rf_scene"] == SITE / "rf_scene_blender_rf"
    assert r3["track_model"] == REPO / "configs" / "track_class_v3.json"
    for k in ("scenarios", "rf_cache", "world_cache", "rf_scene", "track_model"):
        for d in vr.demo_roots():
            assert not _inside(r3[k], d), (k, d)


def test_v3_rfr_scene_source(monkeypatch):
    """Task 3b: the RF-relevance copy is an alternative v3 scene source; nothing else moves."""
    r, rr = vr.data_roots("v3"), vr.data_roots("v3", "rfr")
    assert rr["rf_scene"] == SITE / "rf_scene_blender_rfr"
    assert {k: p for k, p in rr.items() if k != "rf_scene"} == {k: p for k, p in r.items() if k != "rf_scene"}
    monkeypatch.setenv(vr.ENV_RF_GEOM, "rfr")
    assert vr.data_roots("v3")["rf_scene"] == SITE / "rf_scene_blender_rfr"
    assert vr.data_roots("v2")["rf_scene"] == DEMO["rf_scene"]          # v2/v21 ignore the env var
    with pytest.raises(ValueError):
        vr.data_roots("v2", "rfr")
    monkeypatch.setenv(vr.ENV_RF_GEOM, "bogus")
    with pytest.raises(ValueError):
        vr.data_roots("v3")
    for d in vr.demo_roots():
        assert not _inside(rr["rf_scene"], d)


def test_v21_scenarios_root():
    r21, r2 = vr.data_roots("v21"), vr.data_roots("v2")
    assert r21["scenarios"] == REPO / "data" / "scenarios_v21"
    assert not _inside(r21["scenarios"], r2["scenarios"])
    for k in ("rf_cache", "world_cache", "rf_scene"):           # v2.1 = S7 code on the current (v2) caches
        assert r21[k] == r2[k]
    assert r21["track_model"] == REPO / "configs" / "track_class_v21.json"


def test_scene_and_cache_roots_follow_the_version():
    assert vr.cache_root("W3", "world", "v2") == SITE / "world_cache_v2" / "rf" / "W3"
    assert vr.scene_dir("N", "nominal", "v2") == SITE / "rf_cache_v2" / "scene"
    assert vr.cache_root("W3", "world", "v2", version="v3") == SITE / "world_cache_v3" / "rf" / "W3"
    assert vr.scene_dir("B2", "belief", "v2", version="v3") == SITE / "rf_cache_v3" / "ensemble_scene" / "B2"
    assert vr.cache_root("W1", "world", "v1", version="v3") == SITE / "world_cache" / "rf" / "W1"   # v1 untouched


def test_df_run_dir_per_version():
    assert vr.df_run_dir("S4") == REPO / "runs" / "air7311_S4_v2"
    assert vr.df_run_dir("S4", "v21") == REPO / "runs" / "air7311_S4_v2"
    assert vr.df_run_dir("S4", "v3") == REPO / "runs" / "air7311_S4_v3"


def test_rf_cache_dir_string_per_version():
    assert vr.rel(vr.data_roots()["rf_cache"]) == "data/site_kandla/rf_cache_v2"
    assert vr.rel(vr.data_roots("v3")["rf_cache"]) == "data/site_kandla/rf_cache_v3"


# ------------------------------------------------------------------ writer guard
@pytest.mark.parametrize("p", [DEMO["scenarios"] / "kandla_s15_X", DEMO["rf_cache"] / "maps", DEMO["world_cache"],
                               DEMO["rf_scene"] / "meshes", REPO / "configs" / "track_class_v2.json",
                               REPO / "runs" / "air7311_S4_v2"])
def test_writers_refuse_demo_roots(p, monkeypatch):
    with pytest.raises(vr.DemoRootError):
        vr.guard_write(p)                                       # v2 without the override
    assert vr.guard_write(p, allow=True) == p                   # v2 with --allow-demo-roots
    monkeypatch.setenv(vr.ENV_ALLOW, "1")
    assert vr.guard_write(p) == p                               # v2 with SCENESCOUT_ALLOW_DEMO_ROOTS=1
    for v in ("v21", "v3"):
        with pytest.raises(vr.DemoRootError):
            vr.guard_write(p, allow=True, version=v)           # never under another data version


def test_writers_accept_new_roots(tmp_path):
    r3 = vr.data_roots("v3")
    for k in ("scenarios", "rf_cache", "world_cache", "rf_scene"):
        assert vr.guard_write(r3[k] / "x", version="v3") == r3[k] / "x"
    assert vr.guard_write(vr.data_roots("v21")["scenarios"] / "kandla_s1_X", version="v21")
    assert vr.guard_write(tmp_path / "anything") == tmp_path / "anything"


def test_blender_rf_export_refuses_the_demo_scene_dir():
    import yaml

    from envbuild.blender_rf import export_v2

    cfg = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())
    with pytest.raises(vr.DemoRootError):
        export_v2("N", cfg["variants_v2"]["N"], cfg)            # default out = rf_cache_v2/scene (demo)


def test_blender_export_args_and_out_guard():
    from envbuild.blender import export_rf as ex

    a = ex.parse_args(["blender", "-b", "x.blend", "-P", "export_rf.py", "--", "--simplify", "--angle-deg", "2",
                       "--min-extent-m", "0.2", "--out", str(SITE / "rf_scene_blender_rf")])
    assert a.simplify and a.angle_deg == 2.0 and a.min_extent_m == 0.2
    assert ex.resolve_out(a) == SITE / "rf_scene_blender_rf"
    d = ex.parse_args(["blender", "-P", "export_rf.py"])                       # no '--': defaults
    assert not d.simplify and d.angle_deg == 1.0 and d.min_extent_m == 0.10
    with pytest.raises(vr.DemoRootError):
        ex.resolve_out(d)                                                      # demo export needs the override
    assert ex.resolve_out(ex.parse_args(["--", "--allow-demo-roots"])) == SITE / "rf_scene_blender"
    with pytest.raises(vr.DemoRootError):                                      # simplified output never in place
        ex.resolve_out(ex.parse_args(["--", "--simplify", "--allow-demo-roots"]))
    r = ex.parse_args(["--", "--rf-relevance", "--out", str(SITE / "rf_scene_blender_rfr")])    # Task 3b
    assert r.simplify and r.rf_relevance and r.min_cross_m == 0.05 and r.slab_top == ["KW_buildings_ml"]
    assert ex.resolve_out(r) == SITE / "rf_scene_blender_rfr"
    assert ex.parse_args(["--", "--simplify"]).slab_top == []
    with pytest.raises(vr.DemoRootError):
        ex.resolve_out(ex.parse_args(["--", "--rf-relevance", "--allow-demo-roots"]))


def test_track_class_writers_refuse_demo(tmp_path):
    r = subprocess.run([sys.executable, "scripts/track_class_v2.py", "fit", "--train", "999", "--val", "998"],
                       cwd=REPO, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode != 0 and "demo root" in (r.stdout + r.stderr)
    # v3 never falls back to the demo classifier; checked on the resolved path, and the subprocess runs on an empty
    # temp root with a missing model so it can never write a real v3 set (it did once v3 had a model, 2026-10-26)
    assert vr.data_roots("v3")["track_model"] != vr.data_roots("v2")["track_model"]
    r = subprocess.run([sys.executable, "scripts/track_class_v2.py", "apply", "--dirs", "kandla_s15_X",
                        "--root", str(tmp_path), "--model", str(tmp_path / "no_model.json")], cwd=REPO,
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, vr.ENV_VERSION: "v3", vr.ENV_ALLOW: "1", "CUDA_VISIBLE_DEVICES": "",
                            })
    assert r.returncode != 0                            # no model: refuses instead of guessing one
    (tmp_path / "kandla_s15_X").mkdir()
    r = subprocess.run([sys.executable, "scripts/track_class_v2.py", "apply", "--root", str(tmp_path), "--all"],
                       cwd=REPO, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr          # a non-demo root needs no override
    assert '"applied_dirs": 1' in r.stdout


def test_build_scenarios_refuses_demo_roots_before_any_work():
    env = {k: v for k, v in os.environ.items() if k not in (vr.ENV_VERSION, vr.ENV_ALLOW)}
    r = subprocess.run([sys.executable, "scripts/build_scenarios.py", "--obs", "v2", "--seeds", "15", "--check-roots"],
                       cwd=REPO, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0 and "demo root" in (r.stdout + r.stderr)
    r = subprocess.run([sys.executable, "scripts/build_scenarios.py", "--obs", "v2", "--seeds", "15", "--check-roots"],
                       cwd=REPO, capture_output=True, text=True, timeout=120, env={**env, vr.ENV_VERSION: "v3"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "scenarios_v3" in r.stdout


def test_build_scenarios_dirs_follow_the_version(monkeypatch):
    from scripts.build_scenarios import v2_dir

    assert v2_dir(15, None) == REPO / "data" / "scenarios" / "kandla_s15_X"
    monkeypatch.setenv(vr.ENV_VERSION, "v21")
    assert v2_dir(15, "W4") == REPO / "data" / "scenarios_v21" / "kandla_s15_X_W4"


def test_object_catalogue_stays_on_the_full_blender_export(monkeypatch):
    import importlib

    monkeypatch.setenv(vr.ENV_VERSION, "v3")
    import envbuild.blender_objects as bo

    bo = importlib.reload(bo)
    assert bo.BL == SITE / "rf_scene_blender"              # tools/console catalogue, never the simplified RF copy


# ------------------------------------------------------------------ truth guard
def test_guard_forbids_world_cache_v3_even_before_it_exists(tmp_path):
    from harness.guard import forbidden_roots

    roots = forbidden_roots(tmp_path)
    for v in ("v2", "v3"):
        assert os.path.realpath(vr.data_roots(v)["world_cache"]) in roots
    assert os.path.realpath(SITE / "world_cache_v3") in roots


# ------------------------------------------------------------------ tasks and splits
def test_task_ids_are_prefixed_by_data_version():
    assert vr.task_tag() == "X"
    assert vr.task_tag("v21") == "X21"
    assert vr.task_tag("v3") == "X3"
    assert vr.task_tag("v3", "W4") == "X3W4"


def test_split_of_new_test_seeds():
    from scripts.make_tasks import split_of

    assert all(split_of(s) == "test" for s in range(51, 61))
    assert split_of(0) == "demo" and split_of(15) == "test" and split_of(13) == "val"
    assert split_of(1) == split_of(48) == split_of(50) == split_of(67) == split_of(121) == "train"
    # unseen-area tests (paper 2026-10-31): pilot 61-66, paired arms U 101-120 and F 201-220 are test only
    assert all(split_of(s) == "test" for s in list(range(61, 67)) + list(range(101, 121)) + list(range(201, 221)))


def test_varied_cast_is_exposed():
    import inspect

    from envbuild.site_kandla import write_site

    assert inspect.signature(write_site).parameters["cast"].default == "default"
    r = subprocess.run([sys.executable, "scripts/build_scenarios.py", "--help"], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert "--cast" in r.stdout and "varied" in r.stdout


def test_varied_cast_only_on_test_seeds():
    env = {**os.environ, vr.ENV_VERSION: "v3"}
    r = subprocess.run([sys.executable, "scripts/build_scenarios.py", "--obs", "v2", "--seeds", "3", "--cast",
                        "varied", "--check-roots"], cwd=REPO, capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode != 0 and "test" in (r.stdout + r.stderr)


def test_make_tasks_demo_file_is_never_written_outside_v2(tmp_path):
    from scripts.make_tasks import demo_tasks_path

    assert demo_tasks_path(tmp_path) == REPO / "data" / "demo_tasks_kandla.json"
    assert demo_tasks_path(tmp_path, "v3") == tmp_path / "demo.json"
    assert demo_tasks_path(tmp_path, "v21") == tmp_path / "demo.json"
