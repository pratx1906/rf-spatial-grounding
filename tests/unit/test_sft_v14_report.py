"""SFT v1.4 checkpoint selection (task score, not loss) and the Task 7 gate."""

from scripts.sft_v14_report import family_macro, gate, paired_macro_ci, select


def _r(tid, fam, score, seed=51):
    return {"task_id": f"k{seed}X3-{fam}-{tid}", "template": fam, "score": score, "area": "a"}


def test_family_macro_weights_families_not_tasks():
    rows = [_r(i, "big", 1.0) for i in range(9)] + [_r(0, "small", 0.0)]
    assert family_macro(rows) == 0.5


def test_select_prefers_the_best_macro_and_the_earliest_within_the_tie_band():
    a = [_r(0, "f", 0.80), _r(1, "g", 0.80)]
    b = [_r(0, "f", 0.90), _r(1, "g", 0.90)]
    c = [_r(0, "f", 0.903), _r(1, "g", 0.903)]
    out = select({"step30": c, "step15": b, "final": a})
    assert out["selected"] == "step15"                     # 0.903 vs 0.900: within 0.005 -> earliest
    assert select({"step15": a, "final": b})["selected"] == "final"


def test_gate_needs_all_three():
    ok = {"v3": {"diff": 0.0, "ci95_task": [-0.01, 0.01]}, "v6": {"diff": 0.03, "ci95_task": [0.01, 0.05]},
          "heldout_macro": {"base": 0.80, "new": 0.82}}
    assert gate(ok)["pass"]
    assert not gate({**ok, "v3": {"diff": -0.02, "ci95_task": [-0.04, 0.0]}})["pass"]
    assert not gate({**ok, "v6": {"diff": 0.01, "ci95_task": [-0.01, 0.03]}})["pass"]
    assert not gate({**ok, "heldout_macro": {"base": 0.80, "new": 0.79}})["pass"]
    assert not gate({})["pass"]


def test_paired_macro_ci_brackets_the_point_difference():
    base = [_r(i, f, 0.5, seed) for seed in (51, 52, 53) for f in ("x", "y") for i in range(5)]
    new = [{**r, "score": 0.7 if r["template"] == "x" else 0.5} for r in base]
    m = paired_macro_ci(base, new, n=300)
    assert m["diff"] == 0.1 and m["n_clusters"] == 3
    assert m["ci95_cluster"][0] <= 0.1 <= m["ci95_cluster"][1]
