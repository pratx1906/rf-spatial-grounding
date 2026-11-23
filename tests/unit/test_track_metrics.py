import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("track_metrics", Path(__file__).resolve().parents[2] / "scripts" / "track_metrics.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)


def test_gospa_perfect_match_is_zero():
    X = np.array([[0.0, 0.0], [500.0, 0.0]])
    g = tm.gospa(X, X.copy())
    assert g["total"] == 0 and g["missed"] == 0 and g["false"] == 0


def test_gospa_missed_false_and_localisation_terms():
    X = np.array([[0.0, 0.0], [1000.0, 0.0]])
    Y = np.array([[30.0, 40.0], [5000.0, 0.0]])          # one 50 m match, one far false, one missed
    g = tm.gospa(X, Y, c=200.0, p=2)
    assert g["missed"] == 1 and g["false"] == 1
    assert abs(g["total"] - (50.0 ** 2 + 200.0 ** 2 / 2 * 2) ** 0.5) < 1e-9


def test_gospa_empty_sets():
    assert tm.gospa(np.zeros((0, 2)), np.zeros((0, 2)))["total"] == 0
    g = tm.gospa(np.zeros((2, 2)), np.zeros((0, 2)), c=200.0)
    assert g["missed"] == 2 and abs(g["total"] - (200.0 ** 2)**0.5) < 1e-9
