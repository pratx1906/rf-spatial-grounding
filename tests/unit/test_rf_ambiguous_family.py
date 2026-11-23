"""demo-v2 S5.4: the rf_ambiguous task family (offline generator) and its scripted-expert branch."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEED_DIR = REPO / "data" / "scenarios" / "kandla_s13"
pytestmark = pytest.mark.skipif(not (SEED_DIR / "meta.yaml").exists()
                                or not (REPO / "data/site_kandla/rf_cache/maps/ism_2400.npz").exists(),
                                reason="Kandla seed 13 or belief maps missing")


@pytest.fixture(scope="module")
def fam():
    from tasks.kandla.gen_more import GenMore

    g = GenMore(SEED_DIR, 13, "val")
    try:
        return g.rf_ambiguous(5)
    finally:
        g.close()


def test_generator_emits_abstain_keyed_tasks(fam):
    assert fam, "seed 13 has RF-ambiguous bearing sets (4 of 493 at the time of writing)"
    for t in fam:
        assert t.area == "unanswerable" and t.template == "rf_ambiguous"
        assert t.answer_key == {"reason": "insufficient_evidence"}
        assert "rf_ambiguous" in t.categories and "trk:" in t.question


def test_expert_checks_trust_and_abstains(fam):
    from harness.env import AkshaEnv
    from tasks.expert_kandla import solve

    env = AkshaEnv(SEED_DIR)
    for t in fam:
        env.reset(t)
        solve(env, t)
        tools = [e.get("tool") for e in env.tracer.events]
        assert "model_trust" in tools
        assert env.done and env.answer.intent == "abstain" and env.answer_score == 1.0
