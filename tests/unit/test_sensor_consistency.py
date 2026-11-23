"""sensor_consistency (plan 2026-10-28 T3): belief-side per-node bearing bias from leave-one-out fixes."""

import ast
import math
from pathlib import Path

import duckdb
import numpy as np
import pytest

from tools.errors import ToolError
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]


def test_consistency_imports_nothing_from_truth_or_world():
    tree = ast.parse((REPO / "tools" / "consistency.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    banned = ("sim.world", "sim.obs_model", "tasks.truth_access", "truth_access")
    assert not [m for m in mods if m.startswith(banned) or ".truth_access" in m], mods
    src = (REPO / "tools" / "consistency.py").read_text(encoding="utf-8")
    assert "world_cache" not in src and "truth/" not in src


class FakeSite:
    """The SiteView surface the tool reads: sensors, tracks, a bearings table, ref lookups."""

    def __init__(self, sensors: dict, tracks: list[str], rows: list[tuple]):
        self.sensors = {s: {"id": s, "type": "fixed_df", "pose": {"x_m": x, "y_m": y, "z_m": 20.0}}
                        for s, (x, y) in sensors.items()}
        self.tracks = {t: {} for t in tracks}
        self.objects, self.zones = {}, {}
        self.con = duckdb.connect()
        self.con.execute("CREATE TABLE bearings (hyp_id VARCHAR, track_id VARCHAR, sensor_id VARCHAR, t_s DOUBLE, "
                         "bearing_deg DOUBLE, snr_db DOUBLE, power_db DOUBLE)")
        self.con.executemany("INSERT INTO bearings VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def cursor(self):
        return self.con.cursor()

    def entity(self, kind, eid):
        return None

    def known_refs(self, kind):
        return [f"sensor:{s}" for s in self.sensors] if kind == "sensor" else [f"trk:{t}" for t in self.tracks]


def _brg(x0, y0, x1, y1):
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def synthetic_site(bias: dict[str, float], noise_deg: float = 0.7, seed: int = 0) -> FakeSite:
    rng = np.random.default_rng(seed)
    sensors = {"S1": (0.0, 0.0), "S2": (2000.0, 0.0), "S3": (2000.0, 2000.0), "S4": (0.0, 2000.0)}
    rows, tracks = [], []
    for k in range(6):
        tid = f"{1000 + k}"
        tracks.append(tid)
        x0, y0 = rng.uniform(300, 1700, 2)
        vx, vy = rng.uniform(-2, 2, 2)
        for t in range(0, 300, 10):
            x, y = x0 + vx * t, y0 + vy * t
            for s, (sx, sy) in sensors.items():
                b = _brg(sx, sy, x, y) + bias.get(s, 0.0) + rng.normal(0, noise_deg)
                rows.append((f"{s}_{tid}_{t}_1", tid, s, float(t), b % 360, 20.0, -60.0))
                if rng.random() < 0.15:          # a weaker multipath ghost: must not be taken as the node's bearing
                    rows.append((f"{s}_{tid}_{t}_2", tid, s, float(t), (b + rng.uniform(20, 90)) % 360, 12.0, -70.0))
    return FakeSite(sensors, tracks, rows)


def _run(site, **kw):
    load_all()
    spec = REGISTRY["sensor_consistency"]
    return spec.fn(ToolContext(site), spec.args(**kw))


def _nodes(out):
    return {n["sensor_ref"]: n for n in out.data["nodes"]}


def test_biased_node_flagged():
    out = _run(synthetic_site({"S2": 5.0}))
    nodes = _nodes(out)
    assert set(nodes) == {"sensor:S1", "sensor:S2", "sensor:S3", "sensor:S4"}
    assert nodes["sensor:S2"]["suspect"] and abs(nodes["sensor:S2"]["bias_deg"] - 5.0) <= 1.0, nodes["sensor:S2"]
    for s in ("sensor:S1", "sensor:S3", "sensor:S4"):
        assert not nodes[s]["suspect"], nodes[s]
    assert out.data["suspect_refs"] == ["sensor:S2"]
    assert "sensor:S2" in out.refs


def test_negative_bias_and_clean_site():
    nodes = _nodes(_run(synthetic_site({"S3": -5.0}, seed=1)))
    assert nodes["sensor:S3"]["suspect"] and abs(nodes["sensor:S3"]["bias_deg"] + 5.0) <= 1.0
    clean = _run(synthetic_site({}, seed=2))
    assert clean.data["suspect_refs"] == []
    assert all(abs(n["bias_deg"]) < 1.0 for n in clean.data["nodes"])


def test_filters_and_ref_echo():
    site = synthetic_site({"S2": 5.0})
    one = _run(site, sensor_ref="s2")
    assert [n["sensor_ref"] for n in one.data["nodes"]] == ["sensor:S2"] and one.refs[0] == "sensor:S2"
    trk = _run(site, track_ref="trk:1003", at_time_s=150)
    assert "trk:1003" in trk.refs and trk.data["track_ref"] == "trk:1003"
    assert trk.data["window_s"] == [60.0, 240.0]
    assert 0 < trk.data["n_epochs"] < 30
    with pytest.raises(ToolError) as e:
        _run(site, sensor_ref="sensor:S9")
    assert e.value.code == "UNKNOWN_REF"


def test_too_few_nodes_is_an_error():
    site = synthetic_site({})
    site.con.execute("DELETE FROM bearings WHERE sensor_id IN ('S3', 'S4')")
    with pytest.raises(ToolError) as e:
        _run(site)
    assert e.value.code == "NOT_FOUND"
