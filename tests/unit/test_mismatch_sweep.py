"""S7.3 mismatch sweep: the belief-side perturbation models (audit G Q47)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mismatch_sweep as M  # noqa: E402


def test_heading_offsets_have_fixed_signs_and_the_magnitude():
    a = M.heading_offsets(["S1", "S2", "S3", "S4", "S5"], 3.0)
    b = M.heading_offsets(["S5", "S4", "S3", "S2", "S1"], 10.0)
    assert set(map(abs, a.values())) == {3.0}
    assert all(np.sign(a[s]) == np.sign(b[s]) for s in a)


def test_phase_bias_grows_with_the_phase_error():
    rng = np.random.default_rng(0)
    z = M.phase_bias_table(0.0, 0.054, 2.44e9, rng)
    assert np.abs(z).max() <= 1.0                       # scan step
    rms = []
    for s in (1.0, 5.0, 15.0):
        r = [np.sqrt(np.mean(M.phase_bias_table(s, 0.054, 2.44e9, np.random.default_rng(k)) ** 2)) for k in range(4)]
        rms.append(np.mean(r))
    assert rms[0] <= rms[1] < rms[2] and rms[2] > 2.0


def test_apply_bias_uses_the_bearing_index_and_wraps():
    t = np.zeros(360)
    t[359] = 3.0
    assert M.apply_bias(358.8, t) == pytest.approx(1.8)
    assert M.apply_bias(10.2, t) == pytest.approx(10.2)


PK = [{"bearing_deg": 40.0, "el_deg": 1.0, "rel_power_db": 0.0, "kind": "direct", "via": None},
      {"bearing_deg": 90.0, "el_deg": 1.0, "rel_power_db": -4.0, "kind": "reflection", "via": "tank_t001"}]


def test_too_tall_belief_removes_a_marginal_direct_peak():
    out = M.perturb_peaks(PK, 3.0, 5.0, 40.0)
    assert [p["kind"] for p in out] == ["reflection"] and out[0]["rel_power_db"] == 0.0
    assert M.perturb_peaks(PK, 8.0, 5.0, 40.0) == PK           # clearance survives
    assert M.perturb_peaks(PK, None, 10.0, 40.0) == PK         # nothing on the line


def test_too_short_belief_adds_a_direct_peak_when_barely_blocked():
    pk = PK[1:]
    out = M.perturb_peaks(pk, -3.0, -5.0, 41.3)
    assert out[0]["kind"] == "direct" and out[0]["bearing_deg"] == 41.3 and len(out) == 2
    assert M.perturb_peaks(pk, -8.0, -5.0, 41.3) == pk         # still blocked


def test_ghost_score_counts_sensors_with_a_choice():
    g = {"hyps": [{"hyp": "a", "sensor": "S1", "los": True}, {"hyp": "b", "sensor": "S1", "los": False},
                  {"hyp": "c", "sensor": "S2", "los": True}]}
    assert M._ghost(g, {"S1": "a"}) == 1.0
    assert M._ghost(g, {"S1": "b", "S2": "c"}) == 0.0           # S2 has one bearing: not scored
    assert M._ghost({"hyps": g["hyps"][2:]}, {}) is None
