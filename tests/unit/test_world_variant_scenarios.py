"""World-variant scenarios (D-031): the belief side keeps the nominal RF model and sees no world state; truth is
unchanged; observations differ; variant task generation never touches the existing task splits."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "data" / "scenarios"
NOM = SC / "kandla_s15"
W0, W1 = SC / "kandla_s15_W0", SC / "kandla_s15_W1"
need = pytest.mark.skipif(not (W0.exists() and W1.exists() and NOM.exists()), reason="variant scenarios not built")


@need
def test_variant_scenario_keeps_the_nominal_belief_cache():
    from sim.belief.site_view import SiteView

    sv = SiteView(W1)
    assert sv.rf_dir == SiteView(NOM).rf_dir
    assert "world_cache" not in str(sv.rf_dir)


@need
def test_meta_carries_no_world_state():
    meta = yaml.safe_load((W1 / "meta.yaml").read_text())
    flat = json.dumps(meta).lower()
    for word in ("tide", "water_z", "rain", "weather", "ship_s", "berth"):
        assert word not in flat, word
    assert meta.get("world_variant_id") == "W1"
    assert (W1 / "truth" / "world_variant.yaml").exists()


def _bearings(d: Path):
    con = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    rows = con.execute("SELECT sensor_id, t_s, round(bearing_deg, 3) FROM bearings ORDER BY 1, 2, 3").fetchall()
    con.close()
    return rows


def _truth(d: Path):
    con = duckdb.connect(str(d / "truth" / "truth.duckdb"), read_only=True)
    rows = con.execute("SELECT emitter_id, t_s, round(x_m, 2), round(y_m, 2) FROM emitter_points ORDER BY 1, 2").fetchall()
    con.close()
    return rows


def _within(nominal: Path, variant: Path, tol: float) -> float:
    """Share of nominal bearings that have a bearing within tol deg in the same (sensor, t) group of the variant.
    Exact equality is not expected even for the W0 control: ray-launch noise changes a few cells, which shifts the
    receiver-noise draws of every later observation in that DF batch (runs/dbg/w0_bearings.py, D-031)."""
    from collections import defaultdict

    def groups(d):
        g = defaultdict(list)
        for sid, t, b in _bearings(d):
            g[(sid, t)].append(b)
        return g
    n, v = groups(nominal), groups(variant)
    d = [min(abs((b - x + 180) % 360 - 180) for x in v[k]) for k in set(n) & set(v) for b in n[k]]
    return sum(x < tol for x in d) / max(len(d), 1)


@need
def test_truth_identical_and_w0_control_bearings_match_nominal():
    assert _truth(W0) == _truth(NOM) == _truth(W1)
    w0 = _within(NOM, W0, 1.0)
    assert w0 >= 0.90, w0
    for v in ("W1t", "W1", "W2", "W3"):                       # the control is closer to nominal than any world draw
        d = SC / f"kandla_s15_{v}"
        if d.exists():
            assert _within(NOM, d, 1.0) < w0 - 0.03, v


@need
def test_w1_observations_differ():
    assert _within(NOM, W1, 1.0) <= 0.90


@need
def test_variant_task_generation_never_touches_existing_splits(tmp_path):
    test_jsonl = REPO / "data" / "tasks" / "test.jsonl"
    before = hashlib.sha256(test_jsonl.read_bytes()).hexdigest()
    out = tmp_path / "w1.jsonl"
    r = subprocess.run([sys.executable, "scripts/make_tasks.py", "--k", "2", "--seeds", "15", "--world-variant", "W1",
                        "--out", str(out)], cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    assert hashlib.sha256(test_jsonl.read_bytes()).hexdigest() == before
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows and all(x["task_id"].startswith("k15W1-") and x["world_variant"] == "W1" for x in rows)
    assert {x["area"] for x in rows} <= {"operator_query", "df_ambiguity", "drone_search", "replay"}


def test_variant_generation_refuses_train_and_demo_seeds(tmp_path):
    for seed in ("0", "3"):
        r = subprocess.run([sys.executable, "scripts/make_tasks.py", "--seeds", seed, "--world-variant", "W1",
                            "--out", str(tmp_path / "x.jsonl")], cwd=REPO, capture_output=True, text=True, timeout=300)
        assert r.returncode != 0
        assert "test-split only" in (r.stdout + r.stderr), r.stderr[-400:]   # refused, not crashed
