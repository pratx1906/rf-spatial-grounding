"""Paired mismatch task sets (plan 2026-10-28 T7, audit M1): area filter, pairing, determinism, test-only, no leaks."""

import json
from pathlib import Path

import pytest

import scripts.make_tasks as mt
from scripts.make_tasks import (HELD_OUT, MISMATCH_AREAS, NOT_FOR_SELECTION, PAIR_PART, PAIRED, mismatch_rows,
                                paired_params)

REPO = Path(__file__).resolve().parents[2]


def _dirs(seed, tag):
    from scripts.build_scenarios import v2_dir

    return v2_dir(seed, None, "v3"), v2_dir(seed, tag, "v3")


HAVE = all((d / "truth" / "world_variant.yaml").exists() or d.name.endswith("_X") for d in _dirs(15, "M2")) and \
    all((d / "observed.duckdb").exists() for d in _dirs(15, "M2"))
needs_data = pytest.mark.skipif(not HAVE, reason="v3 scenarios kandla_s15_X / _X_M2 not built")


def test_paired_families_are_track_id_free_and_scored_on_a_part():
    assert set(PAIRED) == {"zone_incursion", "count_in_zone", "near_object", "nearest_drone", "band_in_zone",
                           "swarm_in_zone"}
    assert set(PAIR_PART) == set(PAIRED) and set(PAIR_PART.values()) <= {"boolean", "count", "track"}
    assert MISMATCH_AREAS == {"operator_query", "df_ambiguity"}


def _main(args, monkeypatch):
    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    return mt.main(["--obs", "v2", *args])


def test_refuses_non_test_seeds_and_bad_tags(tmp_path, monkeypatch):
    for seeds in ("0", "3", "13", "17"):
        with pytest.raises(SystemExit):
            _main(["--seeds", seeds, "--tag", "M2", "--out-dir", str(tmp_path / "x")], monkeypatch)
    with pytest.raises(SystemExit):
        _main(["--seeds", "15", "--tag", "W4", "--out-dir", str(tmp_path / "x")], monkeypatch)
    with pytest.raises(SystemExit):
        _main(["--seeds", "15", "--tag", "M2", "--world-variant", "W4", "--out-dir", str(tmp_path / "x")], monkeypatch)
    assert not (tmp_path / "x").exists()


def test_refuses_existing_output(tmp_path, monkeypatch):
    (tmp_path / "test.jsonl").write_text("")
    with pytest.raises(SystemExit):
        _main(["--seeds", "15", "--tag", "M2", "--out-dir", str(tmp_path)], monkeypatch)
    assert (tmp_path / "test.jsonl").read_text() == ""


@needs_data
def test_params_identical_across_worlds_and_deterministic():
    n_dir, m_dir = _dirs(15, "M2")
    pn, pm = paired_params(n_dir, 15, 3), paired_params(m_dir, 15, 3)
    assert pn == pm and pn
    assert pn == paired_params(n_dir, 15, 3)
    assert {p["family"] for p in pn} == set(PAIRED)


@needs_data
def test_paired_rows_share_question_and_params(monkeypatch):
    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    n_dir, m_dir = _dirs(15, "M2")
    rn, _ = mismatch_rows(15, n_dir, "N", k=3, train_q=set(), paired_only=True)
    rm, _ = mismatch_rows(15, m_dir, "M2", k=3, train_q=set(), paired_only=True)
    bn, bm = {r["pair_id"]: r for r in rn}, {r["pair_id"]: r for r in rm}
    assert bn and set(bn) == set(bm)
    for pid, a in bn.items():
        b = bm[pid]
        assert a["question"] == b["question"] and a["params"] == b["params"] and a["template"] == b["template"]
        assert a["pair_part"] == b["pair_part"] == PAIR_PART[a["template"]]
        assert a["task_id"] != b["task_id"] and a["world_variant"] == "N" and b["world_variant"] == "M2"
        assert a["site"].endswith("kandla_s15_X") and b["site"].endswith("kandla_s15_X_M2")
        assert a["split"] == b["split"] == "test"


@needs_data
def test_full_rows_area_filter_and_ds_flag(monkeypatch):
    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    _, m_dir = _dirs(15, "M2")
    rows, ds = mismatch_rows(15, m_dir, "M2", k=2, train_q=set())
    assert rows and {r["area"] for r in rows} <= MISMATCH_AREAS
    assert {r["template"] for r in rows if r.get("pair_id")} <= set(PAIRED)
    for f in PAIRED:                                   # paired families come from the paired generator only
        assert all(r.get("pair_id") for r in rows if r["template"] == f)
    assert all(r["area"] == "drone_search" for r in ds)
    assert all(r["synthetic_bearings"] == r["template"].startswith("ds_") for r in ds)
    assert len({r["task_id"] for r in rows + ds}) == len(rows + ds)
    assert all(r["split"] == "test" and r["world_variant"] == "M2" for r in rows + ds)


@needs_data
def test_df_families_use_the_family_rng(monkeypatch):
    """DF tasks are unpaired but sampled with the same per-area RNG: identical in N vs N, and drawn by
    stable_seed(seed, area) (Gen.area with family_rng=True)."""
    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    n_dir, _ = _dirs(15, "M2")
    a, _ = mismatch_rows(15, n_dir, "N", k=2, train_q=set())
    b, _ = mismatch_rows(15, n_dir, "N", k=2, train_q=set())
    assert a == b


@pytest.mark.parametrize("name", ["x3_M2", "x3_M0", "x3_Npair"])
def test_built_sets_are_test_only_and_hold_no_train_like_split(name):
    d = REPO / "data" / "tasks" / name
    if not (d / "test.jsonl").exists():
        pytest.skip(f"{d} not built")
    rows = [json.loads(x) for x in (d / "test.jsonl").read_text().splitlines() if x.strip()]
    assert rows and all(r["split"] == "test" for r in rows)
    assert {r["area"] for r in rows} <= MISMATCH_AREAS
    assert not list(d.glob("train*.jsonl")) and not list(d.glob("val*.jsonl"))
    assert all(int(r["scenario_seed"]) in (15, 16) or 51 <= int(r["scenario_seed"]) <= 60 for r in rows)
    held = {r["template"] for r in rows} & NOT_FOR_SELECTION
    assert held <= HELD_OUT                            # held-out families appear only in these test-only sets


def test_pairs_line_up_in_built_sets():
    n = REPO / "data" / "tasks" / "x3_Npair" / "test.jsonl"
    m = REPO / "data" / "tasks" / "x3_M2" / "test.jsonl"
    if not (n.exists() and m.exists()):
        pytest.skip("paired sets not built")
    rd = lambda p: {r["pair_id"]: r for r in map(json.loads, p.read_text().splitlines()) if r.get("pair_id")}  # noqa: E731
    a, b = rd(n), rd(m)
    assert a and set(a) == set(b)
    assert all(a[k]["question"] == b[k]["question"] and a[k]["params"] == b[k]["params"] for k in a)


def test_train_like_sets_have_no_mismatch_or_pair_rows():
    """No mismatch-world or paired task is in any train / val / selection file."""
    for f in ("x3/train.jsonl", "x3/val.jsonl", "x3/val_clean.jsonl"):
        p = REPO / "data" / "tasks" / f
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                r = json.loads(line)
                assert "pair_id" not in r and not str(r.get("world_variant") or "").startswith("M")


def test_pairing_report_counts_twin_induced_failures():
    from eval.pairing import paired_report, part_ok

    def task(tid, pid, fam, part, key):
        return {"task_id": tid, "pair_id": pid, "template": fam, "pair_part": part, "answer_key": key}

    def ans(refs, b=None):
        return {"intent": "operator_query", "track_refs": refs, "boolean": b}

    nt = {"n1": task("n1", "p1", "zone_incursion", "boolean", {"track_refs": ["trk:1"], "boolean": True}),
          "n2": task("n2", "p2", "count_in_zone", "count", {"track_refs": ["trk:1", "trk:2"], "boolean": None}),
          "n3": {"task_id": "n3", "pair_id": None, "template": "locate", "answer_key": {}}}
    mt_ = {"m1": task("m1", "p1", "zone_incursion", "boolean", {"track_refs": ["trk:7"], "boolean": True}),
           "m2": task("m2", "p2", "count_in_zone", "count", {"track_refs": ["trk:8", "trk:9"], "boolean": None}),
           "m3": {"task_id": "m3", "pair_id": None, "template": "locate", "answer_key": {}}}
    row = lambda tid, s, a: {"task_id": tid, "template": (nt | mt_)[tid]["template"], "score": s, "answer": a}  # noqa: E731
    nres = {"rows": [row("n1", 1.0, ans(["trk:1"], True)), row("n2", 1.0, ans(["trk:1", "trk:2"])),
                     row("n3", 0.9, None)]}
    mres = {"rows": [row("m1", 0.0, ans([], False)), row("m2", 0.5, ans(["trk:8", "trk:3"])), row("m3", 0.4, None)]}
    rep = paired_report(nres, nt, mres, mt_)
    assert rep["pairs"] == 2 and rep["twin_fail"] == 2 and rep["part_twin_fail"] == 1   # count part still right
    assert rep["unpaired_mean_score"]["locate"] == {"N": 0.9, "M": 0.4}
    assert part_ok(mt_["m2"], ans(["trk:8", "trk:3"])) and not part_ok(mt_["m1"], ans([], False))
    assert not part_ok(mt_["m1"], {"intent": "abstain"})
