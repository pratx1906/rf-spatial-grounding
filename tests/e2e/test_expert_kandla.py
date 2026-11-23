"""The scripted expert solves the Kandla families through AkshaEnv (SPEC 9.3 gate for M2 data)."""

from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "scenarios" / "kandla_s13"
pytestmark = pytest.mark.skipif(not (SEED / "meta.yaml").exists(), reason="scenario seed 13 not built")
GATE = {"operator_query": 0.9, "df_ambiguity": 0.7, "drone_search": 0.9, "siting": 0.9, "collateral": 0.9,
        "replay": 0.9, "unanswerable": 0.9}


def test_expert_meets_the_per_area_gate():
    from harness.env import AkshaEnv
    from tasks.expert_kandla import solve
    from tasks.kandla.gen_more import GenMore as Gen

    g = Gen(SEED, 13, "check")
    tasks = g.all(6)
    g.close()
    env = AkshaEnv(SEED)
    scores = defaultdict(list)
    for t in tasks:
        env.reset(t)
        solve(env, t)
        scores[t.area].append(env.answer_score if env.done else 0.0)
    means = {a: sum(v) / len(v) for a, v in scores.items()}
    assert set(GATE) <= set(means), means
    for a, m in means.items():
        assert m >= GATE[a], (a, round(m, 3), means)
