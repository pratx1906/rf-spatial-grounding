"""GOSPA / OSPA multi-target tracking evaluation of the truth-free v3 tracker (OFFLINE; reads scenario truth).

Run (CPU only):
  wsl -d Ubuntu-24.04 --cd "<repo>" -e .venv-wsl/bin/python runs/s81/gospa/gospa_eval.py

Inputs per set data/scenarios_v3/kandla_s<seed>_X/:
  observed.duckdb  track_points (track_id, t_s, x_m, y_m)            -- tracker output (estimates)
  truth/truth.duckdb  emitters (emitter_id, kind, bands, swarm_id), emitter_points (emitter_id, t_s, x_m, y_m, z_m)
  data/site_kandla/world_cache_v2/spectrum_env.yaml  background emitter xy (clutter labelling, as track_metrics.py)

Conventions (identical to scripts/track_metrics.py, which produced runs/s77/v3_report.md):
  * 2 s grid (meta track_dt_s); steps = union of truth sample times and track point times, rounded to integer s.
  * A target exists at step k iff it has an emitter_points row at k (truth presence).
  * Track labelling: nearest emitter by median xy distance over the track's points (>= max(2, n/2) common times,
    median < 250 m), else `clutter` (background transmitter within 300 m median) else `spurious`.
  * Horizontal plane (x, y) only.

GOSPA (Garcia-Fernandez, Svensson, Morelande 2017), alpha = 2, p = 2, cutoff c in {100, 250} m:
  GOSPA^2 = sum_{assigned} d^2 + c^2/2 * (#missed + #false),   assignment minimises sum min(d, c)^2, pairs with d >= c
  are counted as one miss + one false.   OSPA (Schuhmacher 2008), c = 100 m, p = 2, steps with max(n, m) > 0.

Target sets (estimate sets in brackets):
  uav     drone + friendly_drone     [tracks labelled to a UAV emitter + spurious tracks]          (primary)
  uav_s   drone + friendly_drone     [all non-clutter tracks: tracks on controllers/decoys count false] (strict)
  all     every scenario emitter     [all non-clutter tracks]  (= track_metrics.py convention)
Oracle ("perfect localisation, real detection"): each estimate point of a track labelled to emitter e at a step where
  e exists is replaced by e's truth xy; everything else unchanged. Removes localisation error, keeps the tracker's
  detection, fragmentation (duplicates) and spurious tracks.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("SCENESCOUT_DATA_VERSION", "v3")
os.environ.setdefault("SCENESCOUT_RF_GEOM", "rfr")

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ROOT = REPO / "data" / "scenarios_v3"
GROUPS = {"nominal_13_16": [13, 14, 15, 16], "nominal_test_15_16": [15, 16], "varied_51_60": list(range(51, 61))}
CS = (100.0, 250.0)
T_MAX = 590    # last tracker output time (track_points t_s in 0..590); truth runs to 600 s
OSPA_C = 100.0
HOVER_MPS = 1.0
N_BOOT = 10000
CLASSES = ("single", "swarm", "friendly", "controller", "decoy")


# ------------------------------------------------------------------ metrics
def gospa(X: np.ndarray, Y: np.ndarray, c: float, p: int = 2) -> dict:
    """alpha = 2 GOSPA. Returns total, loc (sum d^p of pairs d < c), missed idx, false idx, pairs [(i, j, d)]."""
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return {"total": (c ** p / 2 * (n + m)) ** (1 / p), "loc": 0.0, "miss": list(range(n)),
                "false": list(range(m)), "pairs": []}
    D = np.hypot(X[:, None, 0] - Y[None, :, 0], X[:, None, 1] - Y[None, :, 1])
    r, k = linear_sum_assignment(np.minimum(D, c) ** p)
    pairs = [(int(i), int(j), float(D[i, j])) for i, j in zip(r, k) if D[i, j] < c]
    ai, aj = {q[0] for q in pairs}, {q[1] for q in pairs}
    loc = float(sum(q[2] ** p for q in pairs))
    miss = [i for i in range(n) if i not in ai]
    false = [j for j in range(m) if j not in aj]
    return {"total": (loc + c ** p / 2 * (len(miss) + len(false))) ** (1 / p), "loc": loc, "miss": miss,
            "false": false, "pairs": pairs}


def ospa(X: np.ndarray, Y: np.ndarray, c: float, p: int = 2) -> float | None:
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return None
    if n == 0 or m == 0:
        return c
    D = np.minimum(np.hypot(X[:, None, 0] - Y[None, :, 0], X[:, None, 1] - Y[None, :, 1]), c) ** p
    r, k = linear_sum_assignment(D)
    return float(((D[r, k].sum() + c ** p * abs(m - n)) / max(m, n)) ** (1 / p))


def _brute_gospa(X, Y, c, p=2):
    """Definition by enumeration over partial assignments (small sets only)."""
    n, m = len(X), len(Y)
    best = math.inf
    for k in range(min(n, m) + 1):
        for xi in itertools.combinations(range(n), k):
            for yj in itertools.permutations(range(m), k):
                s = sum(min(math.dist(X[a], Y[b]), c) ** p for a, b in zip(xi, yj))
                best = min(best, s + c ** p / 2 * (n + m - 2 * k))
    return best ** (1 / p)


def sanity() -> dict:
    rng = np.random.default_rng(0)
    X = rng.uniform(-500, 500, (4, 2))
    out = {}
    out["self_zero"] = gospa(X, X, 100)["total"]
    out["shift30_one_target"] = gospa(X[:1], X[:1] + [30, 0], 100)["total"]                     # expect 30
    out["shift30_four_targets_per_target"] = gospa(X, X + [0, 30], 100)["total"] / 2             # sqrt(4*900)/2=30
    out["one_missed_c100"] = gospa(X, X[1:], 100)["total"]                                       # 100/sqrt2 = 70.71
    out["one_false_c250"] = gospa(X[:1], np.vstack([X[:1], [[5000, 5000]]]), 250)["total"]        # 176.78
    out["far_pair_is_miss_plus_false_c100"] = gospa(X[:1], X[:1] + [400, 0], 100)["total"]       # 100
    out["ospa_self"] = ospa(X, X, 100)
    out["ospa_one_missed_of_4"] = ospa(X, X[1:], 100)                                            # sqrt(1e4/4) = 50
    errs = []
    for _ in range(300):
        A = rng.uniform(-200, 200, (rng.integers(0, 5), 2))
        B = rng.uniform(-200, 200, (rng.integers(0, 5), 2))
        for c in CS:
            errs.append(abs(gospa(A, B, c)["total"] - _brute_gospa(A, B, c)))
    out["max_abs_diff_vs_bruteforce_300x2"] = float(max(errs))
    ok = (out["self_zero"] < 1e-9 and abs(out["shift30_one_target"] - 30) < 1e-9
          and abs(out["shift30_four_targets_per_target"] - 30) < 1e-9
          and abs(out["one_missed_c100"] - 100 / math.sqrt(2)) < 1e-9
          and abs(out["one_false_c250"] - 250 / math.sqrt(2)) < 1e-9
          and abs(out["far_pair_is_miss_plus_false_c100"] - 100) < 1e-9 and out["ospa_self"] < 1e-9
          and abs(out["ospa_one_missed_of_4"] - 50) < 1e-9 and out["max_abs_diff_vs_bruteforce_300x2"] < 1e-6)
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------ data
def _bg() -> dict:
    env = REPO / "data" / "site_kandla" / "world_cache_v2" / "spectrum_env.yaml"
    return {e["id"]: (e["x_m"], e["y_m"]) for e in yaml.safe_load(env.read_text())["emitters"]} if env.exists() else {}


def _cls(kind: str, swarm) -> str:
    if kind == "drone":
        return "swarm" if swarm else "single"
    return {"friendly_drone": "friendly"}.get(kind, kind)


def load(seed: int, bg: dict) -> dict:
    d = ROOT / f"kandla_s{seed}_X"
    t = duckdb.connect(str(d / "truth" / "truth.duckdb"), read_only=True)
    em, bands, sw = {}, {}, {}
    for eid, kind, bj, swarm in t.execute("select emitter_id, kind, bands, swarm_id from emitters").fetchall():
        em[eid] = _cls(kind, swarm)
        sw[eid] = swarm
        bands[eid] = sorted((float(a), b) for a, b in json.loads(bj))
    P: dict = defaultdict(dict)
    for e, ts, x, y, z in t.execute("select emitter_id, t_s, x_m, y_m, z_m from emitter_points").fetchall():
        P[e][int(round(ts))] = (x, y, z)
    t.close()
    o = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    points = o.execute("select track_id, t_s, x_m, y_m, z_m, a_m, b_m, theta_deg from track_points").fetchall()
    o.close()
    tp = defaultdict(list)
    for r in sorted(points, key=lambda q: q[1]):
        tp[r[0]].append(tuple(r[1:8]))
    # --- label tracks: verbatim rule of scripts/track_metrics.py
    lab = {}
    for tid, rows in tp.items():
        best, bd = None, math.inf
        for e, pe in P.items():
            ds = [math.hypot(pe[k][0] - x, pe[k][1] - y) for (ts, x, y, *_r) in rows
                  if (k := int(round(ts))) in pe]
            if len(ds) >= max(2, len(rows) // 2) and np.median(ds) < bd:
                best, bd = e, float(np.median(ds))
        if best is not None and bd < 250:
            lab[tid] = best
        else:
            dm = min((np.median([math.hypot(bx - x, by - y) for (_ts, x, y, *_r) in rows]) for bx, by in bg.values()),
                     default=1e9)
            lab[tid] = "clutter" if dm < 300 else "spurious"
    return {"seed": seed, "em": em, "bands": bands, "P": P, "tp": tp, "lab": lab, "sw": sw}


def group_swarms(S: dict) -> dict:
    """Sensitivity: each swarm becomes ONE target at the centroid of its members present at that step; tracks
    labelled to any member are labelled to the group."""
    em, P, bands, lab = {}, defaultdict(dict), {}, {}
    gmap = {e: (f"grp:{S['sw'][e]}" if S["em"][e] == "swarm" else e) for e in S["em"]}
    acc = defaultdict(lambda: defaultdict(list))
    for e, pe in S["P"].items():
        g = gmap.get(e, e)
        em[g] = S["em"].get(e)
        bands.setdefault(g, S["bands"][e])
        for k, xyz in pe.items():
            acc[g][k].append(xyz)
    for g, d in acc.items():
        for k, v in d.items():
            P[g][k] = tuple(float(np.mean([q[i] for q in v])) for i in range(3))
    for tid, L in S["lab"].items():
        lab[tid] = gmap.get(L, L)
    return {**S, "em": em, "P": P, "bands": bands, "lab": lab}



def band_at(sched, k):
    b = sched[0][1]
    for t0, bb in sched:
        if t0 <= k:
            b = bb
    return b


def speed(pe, k):
    a, b = pe.get(k - 2, pe[k]), pe.get(k + 2, pe[k])
    dt = (k + 2 if (k + 2) in pe else k) - (k - 2 if (k - 2) in pe else k)
    return math.hypot(b[0] - a[0], b[1] - a[1]) / dt if dt else 0.0


# ------------------------------------------------------------------ per-set evaluation
CONFIGS = {
    "uav": ({"single", "swarm", "friendly"}, "labelled"),
    "uav_s": ({"single", "swarm", "friendly"}, "strict"),
    "all": (set(CLASSES), "strict"),
}


def new_acc():
    return defaultdict(float)


def eval_set(S: dict, configs: dict | None = None, tag: str | None = None, gate: dict | None = None,
             steps_ok: set | None = None) -> dict:
    """gate: {emitter: set of steps where it counts as a target} (None = every step it exists); a track labelled
    to a target that exists but is gated out at k is dropped (a real, undetected-by-definition emitter, not a false
    track). steps_ok: evaluate only these steps (None = all)."""
    em, P, tp, lab = S["em"], S["P"], S["tp"], S["lab"]
    steps = sorted({k for pe in P.values() for k in pe} | {int(round(r[0])) for rows in tp.values() for r in rows})
    assert max(int(round(r[0])) for rows in tp.values() for r in rows) <= T_MAX, "track output beyond T_MAX"
    tpk = defaultdict(list)
    for tid, rows in tp.items():
        if lab[tid] == "clutter":
            continue
        for r in rows:
            tpk[int(round(r[0]))].append((tid, r[1], r[2]))
    res = {}
    ts_rows = []
    for cfg0, (tcls, mode) in (configs or CONFIGS).items():
        cfg = tag or cfg0
        for oracle in (False, True):
            for c in CS + ((200.0,) if cfg == "all" and not oracle else ()):
                key = f"{cfg}|{'oracle' if oracle else 'tracker'}|c{int(c)}"
                A = new_acc()
                A["sets"] = 1
                dlist = []
                xcheck = cfg == "all" and c == 200.0 and not oracle   # full grid, as track_metrics.py
                for k in (steps if xcheck else [q for q in steps if q <= T_MAX]):
                    if steps_ok is not None and k not in steps_ok:
                        continue
                    es = [e for e in P if em.get(e) in tcls and k in P[e] and (gate is None or k in gate.get(e, ()))]
                    ys = [q for q in tpk.get(k, []) if mode == "strict" or lab[q[0]] == "spurious"
                          or (em.get(lab[q[0]]) in tcls and (gate is None or k not in P[lab[q[0]]]
                                                             or k in gate.get(lab[q[0]], ())))]
                    X = np.array([P[e][k][:2] for e in es]).reshape(-1, 2)
                    Yl = []
                    for tid, x, y in ys:
                        e = lab[tid]
                        if oracle and e in P and k in P[e]:
                            x, y = P[e][k][:2]
                        Yl.append((x, y))
                    Y = np.array(Yl).reshape(-1, 2)
                    g = gospa(X, Y, c)
                    nm, nf = len(g["miss"]), len(g["false"])
                    A["steps"] += 1
                    A["g_sum"] += g["total"]
                    A["g2_sum"] += g["total"] ** 2
                    A["loc2"] += g["loc"]
                    A["miss2"] += c * c / 2 * nm
                    A["false2"] += c * c / 2 * nf
                    A["n_tgt"] += len(es)
                    A["n_est"] += len(ys)
                    A["n_miss"] += nm
                    A["n_false"] += nf
                    A["n_pair"] += len(g["pairs"])
                    if len(es):
                        A["steps_with_tgt"] += 1
                    for j in g["false"]:
                        L = lab[ys[j][0]]
                        A["false_spurious" if L == "spurious" else
                          ("false_target_labelled" if em.get(L) in tcls else "false_other_emitter")] += 1
                    dlist += [q[2] for q in g["pairs"]]
                    # per-target attribution
                    paired = {q[0]: q[2] for q in g["pairs"]}
                    for i, e in enumerate(es):
                        cl = em[e]
                        bd = band_at(S["bands"][e], k)
                        mv = "static" if cl in ("controller", "decoy") else (
                            "hover" if speed(P[e], k) < HOVER_MPS else "moving")
                        contrib = paired[i] ** 2 if i in paired else c * c / 2
                        for dim in (f"cls:{cl}", f"band:{bd}", f"motion:{mv}"):
                            A[f"{dim}|n"] += 1
                            A[f"{dim}|c2"] += contrib
                            A[f"{dim}|miss"] += i not in paired
                            if i in paired:
                                A[f"{dim}|d2"] += paired[i] ** 2
                    # swarm phase split
                    nsw = sum(1 for e in es if em[e] == "swarm")
                    ph = "swarm_present" if nsw else "no_swarm"
                    A[f"phase:{ph}|steps"] += 1
                    A[f"phase:{ph}|n_tgt"] += len(es)
                    A[f"phase:{ph}|g2"] += g["total"] ** 2
                    A[f"phase:{ph}|loc2"] += g["loc"]
                    A[f"phase:{ph}|miss2"] += c * c / 2 * nm
                    A[f"phase:{ph}|false2"] += c * c / 2 * nf
                    if cfg in ("uav", "uavg") and c == 100.0:
                        o = ospa(X, Y, OSPA_C)
                        if o is not None:
                            A["ospa_sum"] += o
                            A["ospa_steps"] += 1
                        ts_rows.append({"cfg": cfg, "seed": S["seed"], "oracle": oracle, "t": k, "n_tgt": len(es), "n_swarm": nsw,
                                        "n_est": len(ys), "loc2": g["loc"], "miss2": c * c / 2 * nm,
                                        "false2": c * c / 2 * nf, "n_miss": nm, "n_false": nf})
                A["_d"] = dlist
                res[key] = A
    return {"acc": res, "ts": ts_rows}


# ------------------------------------------------------------------ aggregation
def summarise(accs: list[dict]) -> dict:
    T = defaultdict(float)
    for A in accs:
        for k, v in A.items():
            if k != "_d":
                T[k] += v
    d = [x for A in accs for x in A["_d"]]
    s = T["steps"]
    g2 = T["g2_sum"]
    out = {
        "sets": int(T["sets"]), "steps": int(s), "targets_per_step": T["n_tgt"] / s, "estimates_per_step": T["n_est"] / s,
        "gospa_mean_m": T["g_sum"] / s, "gospa_rms_m": math.sqrt(g2 / s),
        "gospa_per_target_m": math.sqrt(g2 / T["n_tgt"]) if T["n_tgt"] else None,
        "share_loc": T["loc2"] / g2 if g2 else None, "share_missed": T["miss2"] / g2 if g2 else None,
        "share_false": T["false2"] / g2 if g2 else None,
        "loc_rms_per_step_m": math.sqrt(T["loc2"] / s), "missed_rms_per_step_m": math.sqrt(T["miss2"] / s),
        "false_rms_per_step_m": math.sqrt(T["false2"] / s),
        "missed_per_step": T["n_miss"] / s, "false_per_step": T["n_false"] / s,
        "missed_rate": T["n_miss"] / T["n_tgt"] if T["n_tgt"] else None,
        "false_spurious_per_step": T["false_spurious"] / s,
        "false_target_labelled_per_step": T["false_target_labelled"] / s,
        "false_other_emitter_per_step": T["false_other_emitter"] / s,
        "assigned_rms_err_m": math.sqrt(T["loc2"] / T["n_pair"]) if T["n_pair"] else None,
        "assigned_med_err_m": float(np.median(d)) if d else None,
        "assigned_p95_err_m": float(np.percentile(d, 95)) if d else None,
    }
    if T["ospa_steps"]:
        out["ospa_mean_m"] = T["ospa_sum"] / T["ospa_steps"]
    for pre in ("cls", "band", "motion"):
        for key in sorted({k.split("|")[0] for k in T if k.startswith(pre + ":")}):
            n = T[f"{key}|n"]
            npair = n - T[f"{key}|miss"]
            out.setdefault(f"by_{pre}", {})[key.split(":", 1)[1]] = {
                "target_steps": int(n), "per_target_gospa_m": math.sqrt(T[f"{key}|c2"] / n),
                "missed_rate": T[f"{key}|miss"] / n,
                "assigned_rms_err_m": math.sqrt(T[f"{key}|d2"] / npair) if npair else None}
    for ph in ("swarm_present", "no_swarm"):
        st = T[f"phase:{ph}|steps"]
        if not st:
            continue
        g2p = T[f"phase:{ph}|g2"]
        out.setdefault("by_phase", {})[ph] = {
            "steps": int(st), "targets_per_step": T[f"phase:{ph}|n_tgt"] / st, "gospa_rms_m": math.sqrt(g2p / st),
            "gospa_per_target_m": math.sqrt(g2p / T[f"phase:{ph}|n_tgt"]) if T[f"phase:{ph}|n_tgt"] else None,
            "share_loc": T[f"phase:{ph}|loc2"] / g2p if g2p else None,
            "share_missed": T[f"phase:{ph}|miss2"] / g2p if g2p else None,
            "share_false": T[f"phase:{ph}|false2"] / g2p if g2p else None}
    return out


BOOT_KEYS = ("gospa_mean_m", "gospa_rms_m", "gospa_per_target_m", "share_loc", "share_missed", "share_false",
             "missed_rate", "false_per_step", "missed_per_step", "assigned_rms_err_m", "ospa_mean_m")


def boot(accs: list[dict], rng) -> dict:
    """Percentile CI over scenarios (sets resampled with replacement; ratio of sums recomputed)."""
    vals = defaultdict(list)
    n = len(accs)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        T = defaultdict(float)
        for i in idx:
            for k in ("steps", "g_sum", "g2_sum", "n_tgt", "loc2", "miss2", "false2", "n_miss", "n_false", "n_pair",
                      "ospa_sum", "ospa_steps"):
                T[k] += accs[i].get(k, 0.0)
        s, g2 = T["steps"], T["g2_sum"]
        v = {"gospa_mean_m": T["g_sum"] / s, "gospa_rms_m": math.sqrt(g2 / s),
             "gospa_per_target_m": math.sqrt(g2 / T["n_tgt"]) if T["n_tgt"] else np.nan,
             "share_loc": T["loc2"] / g2 if g2 else np.nan, "share_missed": T["miss2"] / g2 if g2 else np.nan,
             "share_false": T["false2"] / g2 if g2 else np.nan,
             "missed_rate": T["n_miss"] / T["n_tgt"] if T["n_tgt"] else np.nan,
             "false_per_step": T["n_false"] / s, "missed_per_step": T["n_miss"] / s,
             "assigned_rms_err_m": math.sqrt(T["loc2"] / T["n_pair"]) if T["n_pair"] else np.nan,
             "ospa_mean_m": T["ospa_sum"] / T["ospa_steps"] if T["ospa_steps"] else np.nan}
        for k in BOOT_KEYS:
            vals[k].append(v[k])
    return {k: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))] for k, x in vals.items()
            if not all(np.isnan(x))}


def cross_check(seeds: list[int]) -> dict:
    """Re-run scripts/track_metrics.py's own evaluate/combine on the same sets (its GOSPA: all emitters, c = 200)."""
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO))
    import track_metrics as tm  # noqa: E402
    bg = tm._bg()
    r = tm.combine([tm.evaluate(ROOT / f"kandla_s{s}_X", ROOT / f"kandla_s{s}_X", bg) for s in seeds])
    return {"gospa_c200_all_mean": r["gospa"], "fix_err_med": r["err_med"], "false_track_rate": r["false_track_rate"],
            "containment": r["containment"], "gospa_missed_per_step": r["gospa_missed"],
            "gospa_false_per_step": r["gospa_false"]}


def main() -> int:
    san = sanity()
    print("sanity", json.dumps(san))
    if not san["pass"]:
        print("SANITY FAIL")
        return 1
    bg = _bg()
    seeds = sorted({s for v in GROUPS.values() for s in v})
    per_set, ts_all, track_counts = {}, [], {}
    for s in seeds:
        S = load(s, bg)
        r = eval_set(S)
        rg = eval_set(group_swarms(S), {"uav": CONFIGS["uav"]}, "uavg")
        per_set[s] = {**r["acc"], **rg["acc"]}
        ts_all += r["ts"] + rg["ts"]
        labs = list(S["lab"].values())
        track_counts[s] = {"tracks": len(labs), "spurious": labs.count("spurious"), "clutter": labs.count("clutter"),
                           "n_uav": sum(1 for e, c in S["em"].items() if c in ("single", "swarm", "friendly")),
                           "n_swarm": sum(1 for c in S["em"].values() if c == "swarm")}
        print("set", s, track_counts[s], flush=True)
    rng = np.random.default_rng(81)
    full: dict = {"sanity": san, "track_counts": track_counts, "groups": {}, "per_set": {}, "cross_check": {}}
    keys = list(per_set[seeds[0]].keys())
    for gname, gs in GROUPS.items():
        full["groups"][gname] = {}
        for key in keys:
            accs = [per_set[s][key] for s in gs]
            sm = summarise(accs)
            if not key.startswith("all|oracle"):
                sm["ci95"] = boot(accs, rng) if len(gs) > 2 else {}
            full["groups"][gname][key] = sm
    for s in seeds:
        full["per_set"][s] = {k: {kk: vv for kk, vv in summarise([per_set[s][k]]).items()
                                  if not kk.startswith("by_")} for k in keys if k.startswith(("uav|", "uavg|"))}
    for gname in ("nominal_13_16", "varied_51_60"):
        full["cross_check"][gname] = {"track_metrics_py": cross_check(GROUPS[gname]),
                                      "this_script_all_c200_mean": full["groups"][gname]["all|tracker|c200"]["gospa_mean_m"]}
    (OUT / "gospa_full.json").write_text(json.dumps(full, indent=1, default=float))
    # time series (uav, c = 100), 60 s bins, mean over sets
    bins = {}
    for gname, gs in GROUPS.items():
        for oracle in (False, True):
            rows = [r for r in ts_all if r["seed"] in gs and r["oracle"] == oracle and r["cfg"] == "uav"]
            b = defaultdict(lambda: defaultdict(float))
            for r in rows:
                k = min(int(r["t"] // 60), 9)
                for f in ("n_tgt", "n_swarm", "n_est", "loc2", "miss2", "false2", "n_miss", "n_false"):
                    b[k][f] += r[f]
                b[k]["steps"] += 1
            bins[f"{gname}|{'oracle' if oracle else 'tracker'}"] = [
                {"t0": 60 * k, "steps": int(v["steps"]), "targets_per_step": v["n_tgt"] / v["steps"],
                 "swarm_per_step": v["n_swarm"] / v["steps"], "estimates_per_step": v["n_est"] / v["steps"],
                 "gospa_rms_m": math.sqrt((v["loc2"] + v["miss2"] + v["false2"]) / v["steps"]),
                 "gospa_per_target_m": math.sqrt((v["loc2"] + v["miss2"] + v["false2"]) / v["n_tgt"]) if v["n_tgt"] else None,
                 "share_loc": v["loc2"] / max(v["loc2"] + v["miss2"] + v["false2"], 1e-9),
                 "share_missed": v["miss2"] / max(v["loc2"] + v["miss2"] + v["false2"], 1e-9),
                 "share_false": v["false2"] / max(v["loc2"] + v["miss2"] + v["false2"], 1e-9),
                 "missed_per_step": v["n_miss"] / v["steps"], "false_per_step": v["n_false"] / v["steps"]}
                for k, v in sorted(b.items())]
    (OUT / "gospa_timeseries.json").write_text(json.dumps({"bins_60s": bins, "steps": ts_all}, default=float))
    try:
        figure(ts_all)
    except Exception as ex:  # noqa: BLE001
        print("figure skipped:", ex)
    # flat summary for the paper's \val system
    flat = {}

    def put(prefix, sm, ci=True):
        for k in ("gospa_mean_m", "gospa_rms_m", "gospa_per_target_m", "share_loc", "share_missed", "share_false",
                  "missed_rate", "missed_per_step", "false_per_step", "false_spurious_per_step",
                  "false_target_labelled_per_step", "false_other_emitter_per_step", "assigned_rms_err_m",
                  "assigned_med_err_m", "targets_per_step", "ospa_mean_m"):
            if sm.get(k) is not None:
                flat[f"{prefix}/{k}"] = sm[k]
        if ci:
            for k, v in sm.get("ci95", {}).items():
                flat[f"{prefix}/ci95/{k}"] = v

    for gname in GROUPS:
        for cfg in list(CONFIGS) + ["uavg"]:
            for c in CS:
                for kind in ("tracker", "oracle"):
                    sm = full["groups"][gname][f"{cfg}|{kind}|c{int(c)}"]
                    put(f"{gname}/{cfg}/{kind}/c{int(c)}", sm, ci=(cfg in ("uav", "uavg")))
                    if cfg == "uav" and kind == "tracker":
                        for dim in ("by_cls", "by_band", "by_motion", "by_phase"):
                            for name, v in sm.get(dim, {}).items():
                                for kk in ("per_target_gospa_m", "missed_rate", "assigned_rms_err_m", "gospa_per_target_m",
                                           "share_missed", "share_loc", "share_false"):
                                    if v.get(kk) is not None:
                                        flat[f"{gname}/uav/tracker/c{int(c)}/{dim[3:]}/{name}/{kk}"] = v[kk]
    flat["meta/n_sets_nominal_13_16"] = 4
    flat["meta/n_sets_varied_51_60"] = 10
    flat["meta/step_s"] = 2
    flat["meta/n_boot"] = N_BOOT
    flat["meta/sanity_pass"] = san["pass"]
    (OUT / "gospa_summary.json").write_text(json.dumps(flat, indent=1, sort_keys=True))
    print(json.dumps(full["cross_check"], indent=1))
    return 0


def figure(ts_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, (gname, gs) in zip(axs, [("nominal 13-16", GROUPS["nominal_13_16"]), ("varied 51-60", GROUPS["varied_51_60"])]):
        rows = [r for r in ts_all if r["seed"] in gs and not r["oracle"] and r["cfg"] == "uav"]
        T = sorted({r["t"] for r in rows})
        agg = {t: defaultdict(float) for t in T}
        for r in rows:
            for f in ("loc2", "miss2", "false2", "n_swarm", "n_tgt"):
                agg[r["t"]][f] += r[f] / len(gs)
        L = np.array([agg[t]["loc2"] for t in T])
        M = np.array([agg[t]["miss2"] for t in T])
        F = np.array([agg[t]["false2"] for t in T])
        ax.stackplot(T, L / 1e3, M / 1e3, F / 1e3, labels=["localisation", "missed", "false"],
                     colors=["#4C78A8", "#E45756", "#B0B0B0"])
        ax2 = ax.twinx()
        ax2.plot(T, [agg[t]["n_tgt"] for t in T], "k-", lw=1, label="UAVs present")
        ax2.plot(T, [agg[t]["n_swarm"] for t in T], "k--", lw=1, label="swarm UAVs present")
        ax2.set_ylabel("targets / set")
        ax.set_ylabel("GOSPA$^2$ (1e3 m$^2$)")
        ax.set_title(f"{gname}: GOSPA$^2$ decomposition, c = 100 m (mean over sets)", fontsize=9)
        if ax is axs[0]:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)
    axs[-1].set_xlabel("scenario time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "gospa_timeseries.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
