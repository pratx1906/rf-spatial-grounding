"""Grounding metrics for the RF grounding paper (plan 2026-10-28 T1; audit M6, MINOR 4, B2).

Offline only: these read eval rows (`answer` dumps written by eval/run.py) and the task answer keys; Hit@1 also reads
truth through tasks/truth_access.TruthStore (never at episode time). Conventions:
- an abstain, a missing answer or an answer without a position is a **miss** for Acc@d and Hit@1;
- Acc@d / median error: position families only (POSITION_FAMILIES, ds_*), tasks whose key has a position;
- Hit@1: single-key track families only (SINGLE_KEY_FAMILIES), on the first listed track, matched by emitter
  (two tracker tracks of one drone are the same target);
- pass@k: unbiased estimator (Chen et al. 2021) over the `sample`s of each task, success = score >= thr;
  pass^k: probability that k draws without replacement all succeed (tau-bench);
- selective metrics: risk score = 1 - confidence, abstain = maximal risk; AURC on the loss 1 - score.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.truth_access import TruthStore

POSITION_FAMILIES = ("locate", "locate_ghost", "locate_two_sensors", "pilot_locate")   # + every ds_* family
# One key track that the agent must find (the question does not name it). Frozen for the paper: multi-target OQ
# families (zone_incursion, tracks_in_band, ...) and families whose question names the track are excluded.
SINGLE_KEY_FAMILIES = ("nearest_drone", "comp_zone_band_nearest", "comp_sensor_pair")
RFG_AREAS = ("operator_query", "df_ambiguity")
ACC_D = (25, 50, 100)


def _is_position_family(template: str) -> bool:
    return template in POSITION_FAMILIES or template.startswith("ds_")


def _answer_xy(ans: dict | None) -> tuple[float, float] | None:
    if not ans or ans.get("intent") == "abstain":
        return None
    p = ans.get("position") if ans.get("intent") == "df_ambiguity" else \
        ans.get("final_position") if ans.get("intent") == "drone_search" else None
    if not p:
        return None
    return float(p["x_m"]), float(p["y_m"])


def position_metrics(rows: list[dict], tasks) -> dict:
    """Acc@25/50/100 m and median error over position-family rows whose key has a position. Misses (abstain, no
    answer, no position) count in n and as failures for Acc@d; the median is over answered rows."""
    by_id = {t.task_id: t for t in tasks}
    errs: list[float] = []
    n, missing = 0, 0
    for r in rows:
        t = by_id.get(r["task_id"])
        if t is None or not _is_position_family(t.template) or not (t.answer_key or {}).get("position"):
            continue
        n += 1
        xy = _answer_xy(r.get("answer"))
        if xy is None:
            missing += 1
            continue
        k = t.answer_key["position"]
        errs.append(math.hypot(xy[0] - k["x_m"], xy[1] - k["y_m"]))
    out = {f"acc{d}": (sum(e <= d for e in errs) / n if n else None) for d in ACC_D}
    out.update(median_err_m=statistics.median(errs) if errs else None, n=n, n_missing=missing)
    return out


def _base_track(ref: str) -> str:
    return ref.split("@", 1)[0].removeprefix("trk:")


def track_hit_metrics(rows: list[dict], tasks, scen_root: str | Path) -> dict:
    """Hit@1 on SINGLE_KEY_FAMILIES: the first answered track belongs to the same emitter as the key track (truth
    via TruthStore, per task site under scen_root). Unlabelled tracks fall back to the track id itself."""
    by_id = {t.task_id: t for t in tasks}
    stores: dict[str, dict] = {}

    def labels(site: str) -> dict:
        if site not in stores:
            ts = TruthStore(Path(scen_root) / site)
            try:
                stores[site] = ts.track_of()
            finally:
                ts.close()
        return stores[site]

    hits, n = 0, 0
    for r in rows:
        t = by_id.get(r["task_id"])
        if t is None or t.template not in SINGLE_KEY_FAMILIES or not (t.answer_key or {}).get("track_refs"):
            continue
        n += 1
        ans = r.get("answer") or {}
        refs = (ans.get("track_refs") or []) if ans.get("intent") == "operator_query" else []
        if not refs:
            continue
        lab = labels(t.site)
        got, key = _base_track(refs[0]), _base_track(t.answer_key["track_refs"][0])
        hits += (lab.get(got, (got,))[0] == lab.get(key, (key,))[0])
    return {"hit1": hits / n if n else None, "n": n}


def _per_task(rows: list[dict], thr: float) -> dict[str, list[bool]]:
    by = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(float(r["score"]) >= thr)
    return by


def pass_at_k(rows: list[dict], k: int, thr: float = 0.8) -> float:
    """Mean over tasks of 1 - C(n-c, k) / C(n, k) (n samples, c successes); needs n >= k for every task."""
    vals = []
    for tid, ok in _per_task(rows, thr).items():
        n, c = len(ok), sum(ok)
        if n < k:
            raise ValueError(f"{tid}: {n} samples < k={k}")
        vals.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return statistics.mean(vals) if vals else float("nan")


def pass_hat_k(rows: list[dict], k: int, thr: float = 0.8) -> float:
    """Mean over tasks of C(c, k) / C(n, k): all of k draws succeed (consistency, tau-bench pass^k)."""
    vals = []
    for tid, ok in _per_task(rows, thr).items():
        n, c = len(ok), sum(ok)
        if n < k:
            raise ValueError(f"{tid}: {n} samples < k={k}")
        vals.append(math.comb(c, k) / math.comb(n, k))
    return statistics.mean(vals) if vals else float("nan")


def _risk(r: dict) -> float:
    if r.get("abstained"):
        return 2.0                                   # abstain ranks above every stated confidence
    c = r.get("confidence")
    return 1.0 - float(c) if c is not None else 0.5


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def selective_metrics(rows: list[dict], fail_labels: dict | None = None, credit: float = 0.0,
                      fail_thr: float = 0.5) -> dict:
    """E7: AUROC of the risk score (abstain, else 1 - confidence) for predicting failure; AURC (area under the
    risk-coverage curve, loss 1 - score, ordered by confidence with abstains last); selective score (an abstain on
    an answerable task earns `credit`; on an unanswerable task it keeps its score).
    fail_labels: {task_id: bool} (e.g. the twin-induced failure label) over the rows it covers; None = each row's own
    failure (score < fail_thr)."""
    if fail_labels is None:
        lab = [(r, float(r["score"]) < fail_thr) for r in rows]
    else:
        lab = [(r, bool(fail_labels[r["task_id"]])) for r in rows if r["task_id"] in fail_labels]
    auroc = _auroc([_risk(r) for r, _ in lab], [y for _, y in lab])
    order = sorted(rows, key=_risk)
    risks, loss = [], 0.0
    for i, r in enumerate(order, 1):
        loss += 1.0 - float(r["score"])
        risks.append(loss / i)
    sel = [credit if (r.get("abstained") and r.get("area") != "unanswerable") else float(r["score"]) for r in rows]
    return {"auroc": auroc, "aurc": statistics.mean(risks) if risks else None,
            "sel_score": statistics.mean(sel) if sel else None, "credit": credit, "n": len(rows),
            "n_labelled": len(lab), "n_fail": sum(y for _, y in lab)}


def rfg_score(summary: dict) -> dict:
    """Headline RF grounding score: n-weighted mean of the OQ + DF area scores (truth-keyed, ray-traced). DS is
    reported separately (its bearings are synthetic)."""
    parts = [summary[a] for a in RFG_AREAS if a in summary]
    n = sum(p["n"] for p in parts)
    ds = summary.get("drone_search")
    return {"rfg": sum(p["score"] * p["n"] for p in parts) / n if n else None, "n": n,
            "ds": ds["score"] if ds else None, "n_ds": ds["n"] if ds else 0}
