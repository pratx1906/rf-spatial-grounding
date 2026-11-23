"""eval.frozen_build (plan 2026-10-28 T5): seed text from data, stratified family x seed-cluster picks, verbatim copy."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from eval import frozen_build as fb

REPO = Path(__file__).resolve().parents[2]


def _rows(n_per=(5, 40, 200), seeds=(15, 16, 51, 52, 53)):
    rows = []
    for fi, n in enumerate(n_per):
        for i in range(n):
            s = seeds[i % len(seeds)]
            rows.append({"task_id": f"k{s}-f{fi}-{i:04d}", "area": "operator_query", "template": f"f{fi}",
                         "split": "test", "scenario_seed": s, "site": f"data/none/kandla_s{s}_X"})
    return rows


def test_seed_text():
    assert fb.seed_text([{"scenario_seed": s} for s in (60, 15, 16, 51, 52, 53, 54, 55, 56, 57, 58, 59)]) == \
        "15-16,51-60"
    assert fb.seed_text([{"scenario_seed": 15}]) == "15"


def test_family_quotas_water_fill():
    q = fb.family_quotas({"a": 5, "b": 40, "c": 200}, 80)
    assert q["a"] == 5 and sum(q.values()) == 80 and abs(q["b"] - q["c"]) <= 1
    assert fb.family_quotas({"a": 5}, 80) == {"a": 5}


def test_stratified_is_deterministic_and_balanced():
    rows = _rows()
    a = fb.pick_stratified(rows, 80, random.Random(1234))
    b = fb.pick_stratified(list(reversed(rows)), 80, random.Random(1234))
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b] and len(a) == 80
    c = Counter(r["template"] for r in a)
    assert c["f0"] == 5 and c["f1"] >= 30
    cl = Counter(fb.cluster_of(r["scenario_seed"]) for r in a if r["template"] == "f2")
    assert cl["15-16"] == pytest.approx(0.4 * sum(cl.values()), abs=1)       # proportional to availability


def test_build_writes_manifest_and_refuses_existing(tmp_path, monkeypatch):
    src = tmp_path / "test.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    monkeypatch.setattr(fb, "REPO", tmp_path)
    monkeypatch.setattr(fb, "HASHED", [])
    man = fb.build("x", "test.jsonl", target=60, root=tmp_path / "frozen")
    tf = tmp_path / "frozen" / "x" / "tasks.jsonl"
    assert man["n"] == 60 and man["tasks_sha256"] == hashlib.sha256(tf.read_bytes()).hexdigest()
    assert man["source"] == "test.jsonl (seeds 15-16,51-53)" and sum(man["per_family"].values()) == 60
    with pytest.raises(SystemExit):
        fb.build("x", "test.jsonl", target=60, root=tmp_path / "frozen")
    cp = fb.build("y", "test.jsonl", copy=True, root=tmp_path / "frozen")
    assert (tmp_path / "frozen" / "y" / "tasks.jsonl").read_bytes() == src.read_bytes() and cp["n"] == len(_rows())


@pytest.mark.parametrize("ver", ["rfg_core", "heldout_v1"])
def test_frozen_sets_match_their_manifest(ver):
    d = REPO / "eval" / "frozen" / ver
    if not (d / "manifest.json").exists():
        pytest.skip(f"{ver} not frozen")
    man = json.loads((d / "manifest.json").read_text())
    rows = [json.loads(x) for x in (d / "tasks.jsonl").read_text().splitlines()]
    assert man["tasks_sha256"] == hashlib.sha256((d / "tasks.jsonl").read_bytes()).hexdigest()
    assert man["n"] == len(rows) and all(r["split"] == "test" for r in rows)
    if ver == "rfg_core":
        assert {r["area"] for r in rows} == {"operator_query", "df_ambiguity", "drone_search"}
        assert {r["scenario_seed"] for r in rows} <= {15, 16, *range(51, 61)}
