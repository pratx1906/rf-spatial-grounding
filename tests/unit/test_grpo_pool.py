"""RL prompt pools and the fixed RL validation subset (train/grpo_pool.py): selection set from val_clean only, targets
re-derived from val_clean scores, pools free of held-out families and disjoint from every eval set."""

import json
from collections import Counter
from pathlib import Path

import pytest

from train import grpo_pool as gp

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not gp.TRAIN.exists() or not gp.VAL_CLEAN.exists(), reason="x3 task files missing")


def test_val400_is_stratified_clean_and_deterministic():
    rows = gp.rows_of(gp.VAL_CLEAN)
    a = gp.val_subset(rows, 400, seed=0)
    assert len(a) == 400 and a == gp.val_subset(rows, 400, seed=0)
    assert a != gp.val_subset(rows, 400, seed=1)
    fams = {r["template"] for r in rows}
    assert {r["template"] for r in a} == fams                         # every val family represented
    assert not ({r["template"] for r in a} & gp.not_for_selection())
    assert {r["scenario_seed"] for r in a} <= {13, 14}


def test_targets_follow_the_declared_rule():
    model = {"ghost_check": (40, 0.10), "direct_all": (40, 0.50), "first_entry": (40, 0.99), "locate": (40, 0.85),
             "tiny": (2, 0.0), "near_object": (40, 0.2)}
    expert = {"ghost_check": (40, 1.0), "direct_all": (40, 0.56), "first_entry": (40, 1.0), "locate": (40, 0.92),
              "tiny": (2, 1.0), "near_object": (40, 1.0)}
    t = gp.derive_targets(model, expert)
    assert t["A"] == ["ghost_check"] and t["B"] == ["direct_all"] and t["replay"] == ["first_entry"]
    assert t["skipped_small_n"] == ["tiny"] and "near_object" not in json.dumps({k: t[k] for k in ("A", "B", "replay")})
    assert "val_clean" in t["rule"]["source"]


@pytest.fixture(scope="module")
def core():
    targets = {"A": ["ghost_check", "count_in_zone", "swarm_in_zone", "noisy_band"], "B": ["direct_all",
                                                                                           "nearest_drone"],
               "replay": ["first_entry", "last_seen", "exit_time", "actuation", "entity_not_found"]}
    return gp.build_core(gp.rows_of(gp.TRAIN), targets, seed=0, replay_frac=0.4)


def test_pool_composition(core):
    rows, info = core
    b = Counter(r["pool_bucket"] for r in rows)
    assert 0.3 <= b["replay"] / len(rows) <= 0.5
    assert b["target_A"] and b["target_B"] and b["abstain_pair"] and b["ds_wording"] and b["reword"]
    assert not ({r["template"] for r in rows} & gp.not_for_selection())
    assert "noisy_band" in info["unavailable"] and "first_into_zone" in info["unavailable"]
    assert len({r["task_id"] for r in rows}) == len(rows)
    qs = " ".join(r["question"] for r in rows if r["pool_bucket"] == "ds_wording").lower()
    assert sum(w in qs for w in ("allotted", "allowed", "you get", "moves.")) >= 2
    amb = [r for r in rows if r["template"].endswith("_ambiguous")]
    assert amb and all(r["clarification"] for r in amb)


def test_pool_is_disjoint_from_every_eval_set(core):
    rows, _ = core
    out = gp.check_disjoint(rows)
    assert out["n"] == len(rows) and out["eval_ids"] > 1000
    sets = gp.eval_sets()
    assert any("v6" in str(p) for p in sets) and any("heldout_v1" in str(p) for p in sets)
    leak = dict(gp.rows_of(sets[0])[0])
    with pytest.raises(ValueError):
        gp.check_disjoint(rows + [leak])
    with pytest.raises(ValueError):                   # a nominal train row on an eval seed
        gp.check_disjoint([{**rows[0], "task_id": "new", "site": "data/scenarios_v3/kandla_s15_X",
                            "scenario_seed": 15}])


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_written_pools_are_disjoint(name):
    p = gp.OUT / f"pool_{name}.jsonl"
    if not p.exists():
        pytest.skip(f"{p} not built yet")
    rows = gp.rows_of(p)
    gp.check_disjoint(rows)
    if name == "C":
        assert {r["world"] for r in rows} - {None}


def test_exploitable_keys_never_enter_a_pool(core):
    """AUDIT_CODE C6: no collateral / siting rows; no row an empty answer already wins, except the yes/no incursion
    families, whose "no" rows are capped at the "yes" rows."""
    from train.rewards.episode import VIOLATION_FAMILIES

    rows, _ = core
    assert not {r["area"] for r in rows} & {"collateral", "siting"}
    leaks = [r["task_id"] for r in rows if r["template"] not in VIOLATION_FAMILIES and gp.empty_answer_full_credit(r)]
    assert not leaks
    for f in VIOLATION_FAMILIES:
        yes = sum(1 for r in rows if r["template"] == f and r["answer_key"].get("boolean"))
        no = sum(1 for r in rows if r["template"] == f and r["answer_key"].get("boolean") is False)
        assert no <= yes, f


def test_empty_answer_detector():
    base = {"task_id": "x", "question": "q", "template": "band_in_zone", "area": "operator_query"}
    assert gp.empty_answer_full_credit({**base, "answer_key": {"track_refs": [], "boolean": None, "t_s": 1.0}})
    assert not gp.empty_answer_full_credit({**base, "answer_key": {"track_refs": ["trk:1"], "boolean": None}})
    ev = {"task_id": "y", "question": "q", "template": "entered_between", "area": "replay"}
    assert gp.empty_answer_full_credit({**ev, "answer_key": {"evt_refs": []}})
    assert not gp.empty_answer_full_credit({**ev, "answer_key": {"evt_refs": ["evt:1"]}})
    assert not gp.rl_eligible({"area": "collateral", "template": "jam_links", "answer_key": {"violations": ["a"]}})
