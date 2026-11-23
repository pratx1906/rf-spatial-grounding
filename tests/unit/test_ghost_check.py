"""ghost_check family (RF phase 2, Task 4): gated to 8-member ensemble maps so v1 generation is unchanged; the
scripted expert solves it with model_trust(per_sensor) at the track estimate."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
S15 = REPO / "data" / "scenarios" / "kandla_s15"
pytestmark = pytest.mark.skipif(not (REPO / "data" / "site_kandla" / "rf_cache" / "maps" / "manifest.json").exists(),
                                reason="maps not built")


def test_v1_sites_never_get_the_family():
    from tasks.kandla.gen_more import GenMore

    g = GenMore(S15, 15, "test")
    assert g.maps_v2 is False
    g.close()


def test_expert_solves_ghost_check_on_both_answers(monkeypatch):
    from harness.env import AkshaEnv
    from tasks.expert_kandla import solve
    from tasks.kandla.gen_more import GenMore

    monkeypatch.setattr(GenMore, "maps_v2", property(lambda self: True))   # the v1 maps stand in (4 members)
    g = GenMore(S15, 15, "test")
    ts = g.rf_more(4)
    g.close()
    assert ts and {t.answer_key["boolean"] for t in ts} == {True, False}
    env = AkshaEnv(REPO / ts[0].site)
    scores = []
    for t in ts:
        env.reset(t)
        solve(env, t)
        scores.append(getattr(env, "answer_score", 0.0) if env.done else 0.0)
    assert min(scores) >= 0.95, scores
