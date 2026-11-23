"""S7.3 characterisation helpers (Pd crossing, burst matching, Pfa bound, confusion table)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import characterise_classifier as CC  # noqa: E402
import characterise_detection as CD  # noqa: E402
from sim.rf.bursts import Burst  # noqa: E402


def test_snr_at_interpolates_the_monotone_curve():
    s = [0, 2, 4, 6]
    assert CD.snr_at(s, [0.0, 0.2, 0.8, 1.0], 0.5) == pytest.approx(3.0)
    assert CD.snr_at(s, [0.0, 0.95, 0.9, 1.0], 0.9) == pytest.approx(2 * 0.9 / 0.95)   # dip ignored
    assert CD.snr_at(s, [0.0, 0.1, 0.2, 0.3], 0.5) is None
    assert CD.snr_at(s, [0.6, 0.7, 0.8, 1.0], 0.5) == 0.0


def test_match_needs_time_and_frequency_overlap_and_spares_edge_bursts():
    truth = [(1e-3, 2e-3, 9e6, 11e6)]
    edge = [(7.9e-3, 8e-3, 9e6, 11e6)]
    dets = [Burst(1.5e-3, 1e-3, 9.5e6, 10.5e6, 10.0), Burst(1.5e-3, 1e-3, 30e6, 31e6, 10.0),
            Burst(7.9e-3, 0.1e-3, 9e6, 11e6, 10.0)]
    hit, false = CD.match(dets, truth, edge=edge)
    assert hit == [0] and false == [1]


def test_clopper_pearson_zero_count_is_about_rule_of_three():
    lo, hi = CD.clopper_pearson(0, 1000)
    assert lo == 0.0 and hi == pytest.approx(3.682e-3, rel=0.01)       # 1 - 0.025^(1/1000)
    lo, hi = CD.clopper_pearson(5, 20)
    assert lo == pytest.approx(0.0866, abs=2e-3) and hi == pytest.approx(0.4910, abs=2e-3)


def test_confusion_table_rates():
    rows = [{"name": "wifi", "snr": 14, "labels": ["wifi", "wifi", "unknown", "missed"]},
            {"name": "wifi", "snr": 0, "labels": ["missed", "missed"]}]
    c = CC.confusion(rows)["wifi"]
    hi = c["high (12..20 dB)"]
    assert hi["wifi"] == 0.5 and hi["missed"] == 0.25 and hi["acc_detected"] == pytest.approx(2 / 3, abs=1e-3)
    assert hi["unknown_detected"] == pytest.approx(1 / 3, abs=1e-3)
    assert c["low (-6..2 dB)"]["acc_detected"] is None
