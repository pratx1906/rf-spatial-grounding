import itertools

import numpy as np

from sim.belief.assign import linear_sum_assignment


def _brute(c):
    n, m = c.shape
    best = np.inf
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            best = min(best, sum(c[i, j] for i, j in enumerate(perm)))
    else:
        for perm in itertools.permutations(range(n), m):
            best = min(best, sum(c[i, j] for j, i in enumerate(perm)))
    return best


def test_matches_brute_force_on_random_rectangular_matrices():
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (3, 3), (4, 6), (6, 4), (5, 5), (2, 7)]:
        for _ in range(5):
            c = rng.uniform(0, 10, shape)
            r, k = linear_sum_assignment(c)
            assert len(r) == min(shape) and len(set(r)) == len(r) and len(set(k)) == len(k)
            assert abs(c[r, k].sum() - _brute(c)) < 1e-9


def test_empty():
    r, k = linear_sum_assignment(np.zeros((0, 3)))
    assert len(r) == 0 and len(k) == 0
