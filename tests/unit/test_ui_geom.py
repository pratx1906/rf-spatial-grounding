"""UI events carry drawable geometry for episode-scoped results (footprints, weak zones, collateral victims)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve.ui_geom import ui_extras

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "scenarios" / "kandla_s13"
needs_seed = pytest.mark.skipif(not (SEED / "meta.yaml").exists(), reason="scenario seed 13 not built")


def _events(family):
    from harness.env import AkshaEnv
    from serve.agent import run_expert
    from tasks.kandla.gen import Gen

    g = Gen(SEED, 13, "ui")
    tasks = g.cc(1) + g.ss(1)
    g.close()
    t = next(x for x in tasks if x.template == family)
    return list(run_expert(AkshaEnv(SEED), t))


@needs_seed
def test_collateral_episode_streams_footprint_and_victims():
    evs = [e for e in _events("jam_protected") if e["type"] == "tool_result"]
    fp = next(e for e in evs if e["name"] == "effect_footprint")
    assert fp["ui"]["rings"] and len(fp["ui"]["rings"][0]) >= 4 and len(fp["ui"]["effector"]) == 2
    cc = next(e for e in evs if e["name"] == "collateral_check")
    assert all(v["pos"] and len(v["pos"]) == 2 for v in cc["ui"]["victims"])


@needs_seed
def test_weak_zone_episode_streams_polygons():
    evs = [e for e in _events("weak_df") if e["type"] == "tool_result" and e["name"] == "weak_zones"]
    assert evs and all(z["rings"] for z in evs[0]["ui"]["zones"])


# --- graded footprint bands and victim positions (unit, no scenario needed) -----------------------------------


def _env(fp=None, sensors=None, friendly=None):
    ctx = SimpleNamespace(episode={"fp": {"fp:x": fp}} if fp else {})
    site = SimpleNamespace(objects={}, zones={}, sensors=sensors or {}, friendly=friendly or [])
    return SimpleNamespace(executor=SimpleNamespace(ctx=ctx), site=site)


def _fp():
    xs = np.arange(-2000.0, 2001.0, 40.0)
    ys = np.arange(-2000.0, 2001.0, 40.0)
    X, Y = np.meshgrid(xs, ys)
    rx = 0.0 - 20 * np.log10(np.hypot(X, Y) + 1.0)          # 0 dBm at the effector, falling with range
    return {"rx": rx, "xs": xs, "ys": ys, "mask": rx >= -60.0, "threshold_dbm": -60.0, "effector": "sensor:S4"}


def test_footprint_has_three_graded_bands_inner_first():
    env = _env(_fp(), sensors={"S4": {"pose": {"x_m": 0.0, "y_m": 0.0}}})
    out = ui_extras(env, "effect_footprint", {"ok": True, "data": {"ref": "fp:x"}})
    assert [b["level_dbm"] for b in out["bands"]] == [-40.0, -50.0, -60.0]
    assert out["rings"] == out["bands"][-1]["rings"]
    radius = [max(np.hypot(*np.array(r).T).max() for r in b["rings"]) for b in out["bands"]]
    assert radius[0] < radius[1] < radius[2]                # stronger levels sit inside weaker ones
    assert out["effector"] == [0.0, 0.0]
    assert all(len(r) < 80 for b in out["bands"] for r in b["rings"])   # simplified, not a pixel trace


def test_victim_positions_include_own_sensors_and_names():
    env = _env(sensors={"S2": {"pose": {"x_m": 10.0, "y_m": -5.0}}},
               friendly=[{"id": "fe_lora_2", "x_m": 1.0, "y_m": 2.0}])
    res = {"ok": True, "data": {"violations": [
        {"ref": "sensor:S2", "kind": "own_sensor", "severity": "high", "predicted_level_dbm": -40.0},
        {"ref": "fe:fe_lora_2", "kind": "friendly_emitter", "severity": "high", "predicted_level_dbm": -46.2,
         "victim": "Tank gauging telemetry"}]}}
    v = ui_extras(env, "collateral_check", res)["victims"]
    assert v[0]["pos"] == [10.0, -5.0]
    assert v[1]["pos"] == [1.0, 2.0] and v[1]["name"] == "Tank gauging telemetry"
