"""Optimal one-to-one assignment (Hungarian / Kuhn-Munkres, shortest augmenting path with potentials).

numpy only (scipy is not in the pinned environments). linear_sum_assignment(cost) has the same contract as
scipy.optimize.linear_sum_assignment for finite rectangular matrices: min(n, m) pairs, minimum total cost.
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    if c.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    if not np.all(np.isfinite(c)):
        raise ValueError("cost must be finite")
    tr = c.shape[0] > c.shape[1]
    if tr:
        c = c.T
    n, m = c.shape                       # n <= m
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)       # p[j]: row (1-based) matched to column j; column 0 is the virtual start
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            uj = np.where(used)[0]
            u[p[uj]] += delta
            v[uj] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.where(p[1:] > 0)[0]
    rows = p[1:][cols] - 1
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if tr:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols
