"""S7.3: scenario-level (two-level) cluster bootstrap and confidence calibration in eval/run.py (audit G Q48, F Q41)."""

import random

import pytest

from eval import run as R


def _rows(scores, seeds, area="df_ambiguity", conf=None):
    return [{"task_id": f"k{s}X-fam-{i:05d}", "area": area, "score": float(v),
             "confidence": (conf[i] if conf is not None else None)}
            for i, (v, s) in enumerate(zip(scores, seeds))]


def test_scenario_key_parses_the_seed():
    assert R.scenario_of({"task_id": "k16X-jam_all-00371"}) == "16"
    assert R.scenario_of({"task_id": "k15XW4-direct_path-00001"}) == "15"
    assert R.scenario_of({"task_id": "k0-direct_path-00001"}) == "0"
    assert R.scenario_of({"task_id": "weird"}) == "weird"


def test_two_level_bootstrap_is_wider_than_task_level_when_scenarios_differ():
    rng = random.Random(1)
    a = [rng.random() < 0.9 for _ in range(200)] + [rng.random() < 0.5 for _ in range(200)]
    clusters = {"1": [float(x) for x in a[:200]], "2": [float(x) for x in a[200:]]}
    lo_c, hi_c = R.cluster_bootstrap(clusters, n=2000, seed=0)["ci95"]
    flat = [float(x) for x in a]
    lo_t, hi_t = R.cluster_bootstrap({str(i): [v] for i, v in enumerate(flat)}, n=2000, seed=0)["ci95"]
    assert (hi_c - lo_c) > 3 * (hi_t - lo_t)
    assert lo_c <= 0.6 and hi_c >= 0.8                       # spans the two scenario means


def test_cluster_bootstrap_is_deterministic_and_centred():
    cl = {"a": [0.0, 1.0, 1.0], "b": [1.0, 1.0], "c": [0.0, 1.0]}
    r1 = R.cluster_bootstrap(cl, n=500, seed=3)
    r2 = R.cluster_bootstrap(cl, n=500, seed=3)
    assert r1 == r2
    assert r1["mean"] == pytest.approx(5 / 7)
    assert r1["n_clusters"] == 3 and r1["n"] == 7


def test_paired_cluster_bootstrap_matches_the_task_mean_difference():
    a = _rows([1, 0, 1, 0, 1, 1], [15, 15, 15, 16, 16, 16])
    b = _rows([1, 1, 1, 1, 1, 0], [15, 15, 15, 16, 16, 16])
    res = R.paired_cluster_bootstrap(a, b, n=500, seed=0)
    assert res["_all"]["diff"] == pytest.approx((5 - 4) / 6, abs=1e-3)
    assert res["_all"]["n_clusters"] == 2
    lo, hi = res["_all"]["ci95"]
    assert lo <= res["_all"]["diff"] <= hi
    assert set(res) == {"_all", "df_ambiguity"}


def test_calibration_perfect_and_overconfident():
    rows = _rows([1, 0, 1, 0], [1, 1, 1, 1], conf=[1.0, 0.0, 1.0, 0.0])
    c = R.calibration(rows)
    assert c["ece"] == pytest.approx(0.0) and c["brier"] == pytest.approx(0.0)
    assert c["confident_wrong_rate"] == 0.0
    rows = _rows([0, 0, 1, 1], [1, 1, 1, 1], conf=[0.9, 0.8, 0.9, 0.8])
    c = R.calibration(rows)
    assert c["brier"] == pytest.approx((0.81 + 0.64 + 0.01 + 0.04) / 4)
    assert c["confident_wrong_rate"] == pytest.approx(0.5)     # conf >= 0.7 and score < 0.5
    assert c["confident_wrong_of_confident"] == pytest.approx(0.5)
    assert c["ece"] == pytest.approx(0.5 * 0.3 + 0.5 * 0.4)    # bins [0.8, 0.9) and [0.9, 1.0]
    assert sum(b["n"] for b in c["bins"]) == 4


def test_calibration_skips_rows_without_confidence():
    rows = _rows([1, 0], [1, 1], conf=[0.8, None])
    c = R.calibration(rows)
    assert c["n"] == 1 and c["n_missing"] == 1
    assert R.calibration(_rows([1], [1]))["n"] == 0


def test_summarise_keeps_old_keys_and_adds_calibration():
    rows = [{"task_id": f"k15X-f-{i}", "area": "df_ambiguity", "template": "f", "split": "test", "score": 1.0,
             "verifier_code": None, "abstained": False, "turns": 2, "latency_s": 1.0, "error": None, "intent": "x",
             "confidence": 0.8} for i in range(4)]
    s = R.summarise(rows)
    assert {"df_ambiguity", "_generalisation", "_families", "_all"} <= set(s)
    assert s["_all"]["macro_score"] == 1.0
    assert s["_calibration"]["n"] == 4
