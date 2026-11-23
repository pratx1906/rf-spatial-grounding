"""The pull/reduce split of air7311_precompute.extract (overlapping GPU tracing with CPU extraction) must give the
same arrays as the original single-step extract (frozen in tests/unit/_legacy_extract.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from envbuild.air7311_precompute import extract

sys.path.insert(0, str(Path(__file__).parent))
from _legacy_extract import extract_legacy  # noqa: E402


class FakePaths:
    def __init__(self, n_rx, n_ant, n_p, depth, seed):
        r = np.random.default_rng(seed)
        self.a = (r.normal(size=(n_rx, n_ant, 1, 1, n_p)) * (r.random((n_rx, n_ant, 1, 1, n_p)) > 0.3),
                  r.normal(size=(n_rx, n_ant, 1, 1, n_p)))
        self.tau = r.uniform(1e-7, 1e-5, (n_rx, 1, n_p))
        for k in ("phi_t", "theta_t", "phi_r", "theta_r"):
            setattr(self, k, r.uniform(0, np.pi, (n_rx, 1, n_p)))
        self.interactions = r.choice([0, 1, 8], (depth, n_rx, 1, n_p))
        self.objects = r.choice([3, 7, 11, 42, 99], (depth, n_rx, 1, n_p))


@pytest.mark.parametrize("dual,n_p", [(False, 40), (True, 40), (False, 10)])
def test_split_extract_equals_legacy(dual, n_p):
    names = {3: "a", 7: "b", 11: "c", 42: "d"}                  # 99 is not a named object -> via -1
    p = FakePaths(50, 2 if dual else 1, n_p, 2, 7)
    new, old = extract(p, names, 50, 2.44e9, dual_pol=dual), extract_legacy(p, names, 50, 2.44e9, dual_pol=dual)
    assert set(new) == set(old)
    for k in old:
        assert np.array_equal(new[k], old[k], equal_nan=True) if old[k].dtype.kind != "U" else list(new[k]) == list(old[k]), k
