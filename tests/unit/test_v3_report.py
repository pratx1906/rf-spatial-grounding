"""scripts/v3_report.py: pure parsers and statistics (CPU, no data needed)."""

import math

import pytest

from scripts import v3_report as R


def test_percentile_matches_linear_interpolation():
    v = [1.0, 2.0, 3.0, 4.0]
    assert R.percentile(v, 50) == pytest.approx(2.5)
    assert R.percentile(v, 90) == pytest.approx(3.7)
    assert R.percentile([], 50) is None


def test_seed_ranges_and_short_set():
    assert R.seed_ranges([16, 13, 14, 15, 51, 52, 60]) == "13-16,51-52,60"
    name = "kandla_s15_X,kandla_s16_X,kandla_s15_X_W4,kandla_s16_X_W4"
    assert R.short_set(name) == "15-16 X + 15-16 W4"
    assert R.short_set("val") == "val"


def test_scenario_seed():
    assert R.scenario_seed("k16X3W4-jam_all-00349") == 16
    assert R.scenario_seed("k51X3-locate-00001") == 51
    assert R.scenario_seed("weird") is None


def _set(tracks, spur, clut, cls):
    return {"tracks": tracks, "spurious": spur, "clutter": clut, "gospa": 100.0, "per_class": cls}


def test_summarise_tracks_pools_sets():
    a = _set(10, 1, 2, {"single": {"N": 2, "tracks_per_emitter": 1.0, "undetected": 0.5, "_in": [True, False],
                                   "_err": [10.0, 20.0]},
                        "controller": {"N": 1, "tracks_per_emitter": 0.0, "undetected": 1.0, "_in": [], "_err": []}})
    b = _set(10, 0, 0, {"swarm": {"N": 2, "tracks_per_emitter": 2.0, "undetected": 0.0, "_in": [True, True],
                                  "_err": [30.0, 40.0]}})
    s = R.summarise_tracks([a, b])
    assert s["tracks"] == 20 and s["false_track_rate"] == pytest.approx(0.05)
    assert s["nontarget_rate"] == pytest.approx(0.15)
    assert s["containment"] == pytest.approx(0.75)
    assert s["err_med"] == pytest.approx(25.0)
    assert s["tracks_per_drone"] == pytest.approx(1.5)          # (1*2 + 2*2) / 4
    assert s["per_class"]["controller"]["undetected"] == 1.0
    assert s["per_class"]["controller"]["containment"] is None
    assert R.summarise_tracks([]) == {}


def test_tracks_from_json():
    d = {"tracks": 5, "false_track_rate": 0.0, "nontarget_rate": 0.2, "gospa": 1.0, "containment": 0.9,
         "err_med": 3.0, "err_p90": 9.0,
         "per_class": {"single": {"N": 1, "tracks_per_emitter": 3.0}, "swarm": {"N": 3, "tracks_per_emitter": 1.0}}}
    t = R.tracks_from_json(d)
    assert t["err_p95"] is None and t["tracks_per_drone"] == pytest.approx(1.5)


def test_parse_q32_tdoa_expert_check():
    q = "\n".join(["[2026-10-24 16:35:10] bursts with t0 < 5 us: [1, 2]",
                   "[x] == per-burst top-peak bearing error |deg| (truth emitter of the burst's track) ==",
                   "[x] ('all',) n=10 med=1.5 rms=2.0 p95=9.0 max=12.0",
                   "[x] == top hyp label per (track, sensor, epoch) == {'LOS': 3}",
                   "[x] == fix horizontal error (m) at fix epochs ==",
                   "[x] ('n_sensors', 2) n=4 med=40.0 rms=50.0 p95=90.0 max=99.0",
                   "[x] == fix altitude error (m) z_est - z_true ==",
                   "[x] ('all',) n=4 med=20.0 rms=30.0 p95=60.0 max=70.0 bias=-12.5"])
    p = R.parse_q32(q)
    assert p["bearing"]["('all',)"]["med"] == 1.5
    assert p["fix"]["('n_sensors', 2)"]["n"] == 4
    assert p["z"]["('all',)"]["bias"] == -12.5
    assert p["t0"] == "[1, 2]" and "LOS" in p["top"]
    t = R.parse_tdoa("[a] all n=329 range-diff |err| m: med=54 rms=202795 p95=494279 max=1972322  frac<30m=0.36\n"
                     "[a] bursts (tracked, bw>=10 MHz) with toa_s NULL: 5066")
    assert t["all"]["med"] == 54 and t["all"]["frac30"] == 0.36 and t["null"] == "5066"
    e = R.parse_expert_check("[t] AREA collateral       n=  25 mean=1.000\n[t]   count_in_zone              n=  12 mean=0.682")
    assert e["area"]["collateral"] == (25, 1.0) and e["family"]["count_in_zone"] == (12, 0.682)


def test_parse_class_log_filters_by_model():
    log = "\n".join([
        "[t] === START T2: python scripts/track_class_v2.py eval --root R --model configs/a.json --dirs kandla_s15_X",
        '[t] {"set": "kandla_s15_X", "n": 5, "expert_rule_acc": 0.8}',
        "[t] === END T2 rc=0",
        "[t] === START T2: python scripts/track_class_v2.py eval --root R --model configs/b.json --dirs kandla_s15_X",
        '[t] {"set": "kandla_s15_X", "n": 5, "expert_rule_acc": 0.1}',
        "[t] === END T2 rc=0",
        '[t] {"set": "stray", "n": 1}'])
    out = R.parse_class_log(log, "configs/a.json")
    assert [(d["set"], d["expert_rule_acc"]) for d in out] == [("kandla_s15_X", 0.8)]


def test_cluster_ci_matches_eval_run():
    from eval.run import cluster_bootstrap

    cl = {13: [1.0, 0.0, 1.0], 15: [0.5, 0.5], 16: [1.0]}
    a, b = R.cluster_ci(cl, 500, 3), cluster_bootstrap(cl, 500, 3)
    assert a["mean"] == pytest.approx(b["mean"])
    assert [round(x, 4) for x in a["ci95"]] == b["ci95"]
    assert R.cluster_ci({})["mean"] is None


def test_unpaired_cluster_diff_constant_and_empty():
    d = R.unpaired_cluster_diff({1: [0.5, 0.5]}, {2: [0.75], 3: [0.75]}, 200)
    assert d["diff"] == pytest.approx(0.25) and d["ci95"] == [pytest.approx(0.25), pytest.approx(0.25)]
    assert R.unpaired_cluster_diff({}, {1: [1.0]})["diff"] is None


def test_clusters_of_filters_seeds():
    rows = [{"task_id": "k15X3-a-1", "score": 1}, {"task_id": "k51X3-a-2", "score": 0}, {"task_id": "?", "score": 1}]
    assert R.clusters_of(rows) == {15: [1.0], 51: [0.0]}
    assert R.clusters_of(rows, {51}) == {51: [0.0]}


def test_load_missing_inputs_is_pending(tmp_path):
    prof = {"label": "x", "root": str(tmp_path / "none"), "prefix": str(tmp_path / "none" / "p"),
            "model": str(tmp_path / "none.json"), "class_log": None, "frozen": "zz_missing",
            "world": str(tmp_path / "{pol}_{w}.json"), "groups": {"13-16": [13]}}
    D = R.load("v3", prof, live=False)
    assert D["q32"] is None and D["tdoa"] is None and D["expert_check"] is None and D["classifier"] is None
    assert all(v is None for v in D["evals"].values())
    assert R.fmt(None) == "pending" and not math.isnan(0.0)
