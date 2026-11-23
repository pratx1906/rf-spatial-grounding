"""demo-v2 S6.1: the gateway's demo-set switch (v1 = data/site_kandla + demo_tasks_kandla.json + demo_replays;
v2 = data/scenarios/kandla_s0_X + demo_tasks_kandla_v2.json + demo_replays_v2). Default v2 once v2 recordings exist."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


@pytest.fixture()
def gw(tmp_path, monkeypatch):
    from serve import gateway

    data = tmp_path / "data"
    sets = {"v1": {"site": data / "site_kandla", "tasks": data / "demo_tasks_kandla.json",
                   "replays": data / "demo_replays"},
            "v2": {"site": data / "scenarios" / "kandla_s0_X", "tasks": data / "demo_tasks_kandla_v2.json",
                   "replays": data / "demo_replays_v2"}}
    for s in sets.values():
        s["site"].mkdir(parents=True)
        s["replays"].mkdir(parents=True)
        s["tasks"].write_text("[]")
    monkeypatch.setattr(gateway, "DEMO_SETS", sets)
    monkeypatch.delenv("SCENESCOUT_DEMO_SET", raising=False)
    monkeypatch.delenv("SCENESCOUT_DEMO_TASKS", raising=False)
    monkeypatch.delenv("SCENESCOUT_SITE", raising=False)
    return gateway, sets


def _record(sets, name):
    (sets[name]["replays"] / "kX0-genuine_drone-00001__sft.json").write_text(json.dumps({"events": [], "summary": {}}))


def test_default_is_v1_until_v2_recordings_exist(gw):
    gateway, sets = gw
    assert gateway.active_demo_set() == "v1"
    _record(sets, "v2")
    assert gateway.active_demo_set() == "v2"


def test_env_forces_the_set(gw, monkeypatch):
    gateway, sets = gw
    _record(sets, "v2")
    monkeypatch.setenv("SCENESCOUT_DEMO_SET", "v1")
    assert gateway.active_demo_set() == "v1"
    monkeypatch.setenv("SCENESCOUT_DEMO_SET", "v2")
    _ = [f.unlink() for f in sets["v2"]["replays"].glob("*")]
    assert gateway.active_demo_set() == "v2"          # forced even without recordings
    monkeypatch.setenv("SCENESCOUT_DEMO_SET", "v9")
    with pytest.raises(ValueError):
        gateway.active_demo_set()


def test_site_tasks_and_replays_follow_the_site(gw, monkeypatch):
    gateway, sets = gw
    for name in ("v1", "v2"):
        monkeypatch.setattr(gateway, "SITE_DIR", sets[name]["site"])
        assert gateway._demo_file() == sets[name]["tasks"]
        assert gateway._replays_dir() == sets[name]["replays"]


def test_default_site_is_the_active_set(gw, monkeypatch):
    gateway, sets = gw
    assert gateway.default_site() == sets["v1"]["site"]
    _record(sets, "v2")
    assert gateway.default_site() == sets["v2"]["site"]
    monkeypatch.setenv("SCENESCOUT_SITE", "/somewhere/else")
    assert gateway.default_site() == Path("/somewhere/else")


def test_replays_endpoint_reads_the_active_set(gw, monkeypatch):
    from fastapi.testclient import TestClient

    gateway, sets = gw
    _record(sets, "v2")
    monkeypatch.setattr(gateway, "SITE_DIR", sets["v2"]["site"])
    c = TestClient(gateway.app)
    assert list(c.get("/api/replays").json()) == ["kX0-genuine_drone-00001"]
    assert c.get("/api/replay", params={"task_id": "kX0-genuine_drone-00001"}).status_code == 200
    monkeypatch.setattr(gateway, "SITE_DIR", sets["v1"]["site"])
    assert c.get("/api/replays").json() == {}


def test_warm_site_fills_the_console_boot_caches(monkeypatch):
    """S6: the console's first load waited ~5 s on a cold /api/blender_objects; the gateway warms it at start."""
    from serve import gateway

    calls = []
    monkeypatch.setattr(gateway, "site_geometry", lambda: calls.append("site"))
    monkeypatch.setattr(gateway, "tracks", lambda step_s, frame: calls.append(f"tracks {step_s} {frame}"))
    monkeypatch.setattr(gateway, "blender_objects", lambda: calls.append("blender"))
    res = gateway.warm_site()
    assert calls == ["site", "tracks 2.0 enu3", "blender"] and res["ok"]

    def boom():
        raise RuntimeError("no export")
    monkeypatch.setattr(gateway, "blender_objects", boom)
    assert gateway.warm_site()["ok"] is False                 # best effort: never raises


def test_v2b_preferred_and_its_questions_served_on_the_shared_scenario(gw, monkeypatch):
    gateway, sets = gw
    v2b = {"site": sets["v2"]["site"], "tasks": sets["v2"]["tasks"].with_name("demo_tasks_kandla_v2b.json"),
           "replays": sets["v2"]["replays"].with_name("demo_replays_v2b")}
    v2b["tasks"].write_text("[]")
    v2b["replays"].mkdir()
    monkeypatch.setitem(sets, "v2b", v2b)
    _record(sets, "v2")
    assert gateway.active_demo_set() == "v2"           # v2b has no recordings yet
    _record(sets, "v2b")
    assert gateway.active_demo_set() == "v2b"
    monkeypatch.setattr(gateway, "SITE_DIR", sets["v2"]["site"])   # v2 and v2b run on the same scenario
    assert gateway._demo_file() == v2b["tasks"]
    monkeypatch.setenv("SCENESCOUT_DEMO_SET", "v2")
    assert gateway._demo_file() == sets["v2"]["tasks"]
