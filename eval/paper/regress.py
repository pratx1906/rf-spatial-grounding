"""P9 (b): what makes a task hard. Joins difficulty features (eval/paper/features.py) with an eval/run.py output and fits
success ~ features with (1) an L2 logistic regression (standardised features, bootstrap CIs clustered by scenario
seed) and (2) a small random forest with permutation importance (drop in held-out AUC, 5-fold CV grouped by seed).

    python -m eval.paper.regress --features runs/paper/results/P9/features_v6.parquet \
        --eval runs/eval/sft_v11_v6_t7.json --name sft_v11_v6 [--thr 0.8] [--out runs/paper/results/P9]

numpy + pandas only (scikit-learn is not installed in either environment); matplotlib figures when available.
Writes <out>/<name>_logit.csv, _importance.csv, _bins.csv, _summary.json and figures <name>_*.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NUMERIC = ["emitter_z_m", "nearest_sensor_m", "log10_gdop", "n_sensors", "hyps_per_look", "ghost_frac_label",
           "ghost_frac_5deg", "map_ghost_risk", "map_nlos_frac", "map_az_spread_deg", "n_emitters_active",
           "n_tracks_active", "los_look_frac", "strongest_los_frac", "n_track_refs"]
CATEG = ["band", "area"]


# ------------------------------------------------------------------------------------------------- data

def load(features: str | Path, eval_json: str | Path, thr: float) -> pd.DataFrame:
    fp = Path(REPO / features)
    f = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_json(fp, lines=True)
    rows = json.loads(Path(REPO / eval_json).read_text())["rows"]
    e = pd.DataFrame(rows)
    e = e.groupby("task_id", as_index=False).agg(score=("score", "mean"), turns=("turns", "mean"),
                                                 tool_calls=("tool_calls", "mean"))
    d = f.merge(e, on="task_id", how="inner")
    d["success"] = (d["score"] >= thr).astype(int)
    if "gdop_m_per_deg" in d:
        d["log10_gdop"] = np.log10(d["gdop_m_per_deg"].astype(float))
    return d


def design(d: pd.DataFrame, numeric: list[str], categ: list[str]) -> tuple[np.ndarray, list[str]]:
    """Standardised numeric columns (NaN -> 0 after standardising, plus a missing indicator when any is missing) and
    one-hot categories (first level dropped)."""
    cols, names = [], []
    for c in numeric:
        if c not in d:
            continue
        x = d[c].astype(float).to_numpy()
        m = np.isnan(x)
        if m.all() or np.nanstd(x) == 0:
            continue
        z = (x - np.nanmean(x)) / np.nanstd(x)
        cols.append(np.where(m, 0.0, z))
        names.append(c)
        if m.any():
            mf = m.astype(float)
            twin = next((n for n, col in zip(names, cols, strict=True)
                         if n.endswith("_missing") and np.array_equal(col, mf)), None)
            if twin is None:                     # one indicator per distinct missingness pattern
                cols.append(mf)
                names.append(c + "_missing")
    for c in categ:
        if c not in d:
            continue
        lv = sorted(d[c].dropna().astype(str).unique())
        for v in lv[1:]:
            cols.append((d[c].astype(str) == v).to_numpy(float))
            names.append(f"{c}={v}")
    return (np.stack(cols, 1) if cols else np.zeros((len(d), 0))), names


# ------------------------------------------------------------------------------------------------- models

def logit_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalised logistic regression by Newton/IRLS; returns [intercept, coefs...] (intercept unpenalised)."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    P = np.eye(A.shape[1]) * l2
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + P @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + P + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def logit_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Tree:
    """CART classifier (Gini), depth-limited, random feature subsets; leaves hold P(y=1)."""

    def __init__(self, depth=4, min_leaf=8, mtry=None, rng=None):
        self.depth, self.min_leaf, self.mtry = depth, min_leaf, mtry
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X, y):
        self.node = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, dep):
        p = float(y.mean()) if len(y) else 0.5
        if dep >= self.depth or len(y) < 2 * self.min_leaf or p in (0.0, 1.0):
            return p
        k = X.shape[1]
        feats = self.rng.choice(k, size=min(k, self.mtry or k), replace=False)
        best = None
        for j in feats:
            xs = X[:, j]
            for t in np.unique(np.quantile(xs, np.linspace(0.05, 0.95, 15))):
                L = xs <= t
                nl, nr = L.sum(), (~L).sum()
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                pl, pr = y[L].mean(), y[~L].mean()
                gini = nl * pl * (1 - pl) + nr * pr * (1 - pr)
                if best is None or gini < best[0]:
                    best = (gini, j, t, L)
        if best is None:
            return p
        _, j, t, L = best
        return (j, t, self._grow(X[L], y[L], dep + 1), self._grow(X[~L], y[~L], dep + 1))

    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            n = self.node
            while isinstance(n, tuple):
                n = n[2] if x[n[0]] <= n[1] else n[3]
            out[i] = n
        return out


class Forest:
    def __init__(self, n=100, depth=4, min_leaf=8, seed=0):
        self.n, self.depth, self.min_leaf, self.seed = n, depth, min_leaf, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        mtry = max(1, int(math.sqrt(X.shape[1])))
        self.trees = []
        for _ in range(self.n):
            idx = rng.integers(0, len(y), len(y))
            self.trees.append(Tree(self.depth, self.min_leaf, mtry, rng).fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


def group_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Fold index per row, whole groups (scenario seeds) per fold; falls back to random rows with < k groups."""
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    if len(g) >= k:
        perm = rng.permutation(g)
        fold_of = {v: i % k for i, v in enumerate(perm)}
        f = np.array([fold_of[v] for v in groups])
    else:
        f = rng.permutation(len(groups)) % k
    return [np.where(f == i)[0] for i in range(k)]


def permutation_importance(X, y, groups, names, model="forest", k=5, repeats=5, seed=0) -> pd.DataFrame:
    """Drop in held-out AUC when one column (or a feature with its _missing twin) is permuted, CV by group."""
    rng = np.random.default_rng(seed)
    base_feats = sorted({n.removesuffix("_missing") if not n.startswith(("band=", "area=")) else n.split("=")[0]
                         for n in names})
    cols_of = {b: [i for i, n in enumerate(names) if n == b or n == b + "_missing" or n.startswith(b + "=")]
               for b in base_feats}
    drops = {b: [] for b in base_feats}
    aucs = []
    for te in group_folds(groups, k, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        if model == "forest":
            m = Forest(seed=seed).fit(X[tr], y[tr])
            pred = m.predict
        else:
            w = logit_fit(X[tr], y[tr])
            pred = lambda Z, w=w: logit_predict(w, Z)  # noqa: E731
        a0 = auc(y[te], pred(X[te]))
        aucs.append(a0)
        for b, cols in cols_of.items():
            for _ in range(repeats):
                Z = X[te].copy()
                perm = rng.permutation(len(te))
                Z[:, cols] = Z[perm][:, cols]
                drops[b].append(a0 - auc(y[te], pred(Z)))
    imp = pd.DataFrame({"feature": list(drops), "auc_drop_mean": [np.mean(v) if v else np.nan for v in drops.values()],
                        "auc_drop_sd": [np.std(v) if v else np.nan for v in drops.values()]})
    imp = imp.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)
    imp.attrs["cv_auc"] = float(np.nanmean(aucs)) if aucs else float("nan")
    return imp


def bootstrap_logit(X, y, groups, names, B=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w0 = logit_fit(X, y)
    g = np.unique(groups)
    idx_of = {v: np.where(groups == v)[0] for v in g}
    W = []
    for _ in range(B):
        pick = rng.choice(g, len(g), replace=True)
        idx = np.concatenate([idx_of[v] for v in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        W.append(logit_fit(X[idx], y[idx]))
    W = np.array(W)
    lo, hi = np.percentile(W, [2.5, 97.5], axis=0)
    return pd.DataFrame({"term": ["intercept"] + names, "coef": w0, "ci_lo": lo, "ci_hi": hi,
                         "odds_ratio_per_sd": np.exp(w0)})


def binned(d: pd.DataFrame, col: str, q: int = 4) -> pd.DataFrame:
    x = d[col].astype(float)
    ok = x.notna()
    if ok.sum() < 2 * q or x[ok].nunique() < 2:
        return pd.DataFrame()
    b = pd.qcut(x[ok], q=q, duplicates="drop")
    g = d[ok].groupby(b, observed=True)
    return pd.DataFrame({"feature": col, "bin": [str(i) for i in g.size().index], "n": g.size().values,
                         "success": g.success.mean().values.round(3), "score": g.score.mean().values.round(3)})


# ------------------------------------------------------------------------------------------------- run

def analyse(d: pd.DataFrame, name: str, out: Path, track_only: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sub = d[d.has_track == 1].reset_index(drop=True) if track_only else d
    groups = sub["seed"].fillna(-1).astype(int).to_numpy()
    y = sub["success"].to_numpy()
    res = {"name": name, "n": int(len(sub)), "success_rate": round(float(y.mean()), 3) if len(y) else None}
    tables = {}
    for tag, numeric, categ in (("phys", NUMERIC, ["band"]), ("phys_area", NUMERIC, CATEG)):
        X, names = design(sub, numeric, categ)
        lg = bootstrap_logit(X, y, groups, names)
        imp_f = permutation_importance(X, y, groups, names, "forest")
        imp_l = permutation_importance(X, y, groups, names, "logit")
        lg.to_csv(out / f"{name}_{tag}_logit.csv", index=False)
        imp_f.to_csv(out / f"{name}_{tag}_importance_forest.csv", index=False)
        imp_l.to_csv(out / f"{name}_{tag}_importance_logit.csv", index=False)
        tables[tag] = (lg, imp_f)
        res[tag] = {"cv_auc_forest": round(imp_f.attrs["cv_auc"], 3), "cv_auc_logit": round(imp_l.attrs["cv_auc"], 3),
                    "top_forest": imp_f.head(5)[["feature", "auc_drop_mean"]].round(4).to_dict("records"),
                    "top_logit": imp_l.head(5)[["feature", "auc_drop_mean"]].round(4).to_dict("records"),
                    "sig_logit_terms": lg[(lg.term != "intercept") & ((lg.ci_lo > 0) | (lg.ci_hi < 0))]
                    .round(3)[["term", "coef", "ci_lo", "ci_hi"]].to_dict("records")}
    bins = pd.concat([binned(sub, c) for c in ["log10_gdop", "ghost_frac_5deg", "ghost_frac_label", "hyps_per_look",
                                                "nearest_sensor_m", "emitter_z_m", "n_tracks_active",
                                                "los_look_frac"] if c in sub], ignore_index=True)
    bins.to_csv(out / f"{name}_bins.csv", index=False)
    strata = {}
    for col in ("los_stratum", "band", "area", "template"):
        if col in d:
            g = d.groupby(d[col].fillna("none"))
            strata[col] = {k: {"n": int(len(v)), "success": round(float(v.success.mean()), 3),
                               "score": round(float(v.score.mean()), 3)} for k, v in g}
    res["strata"] = strata
    (out / f"{name}_summary.json").write_text(json.dumps(res, indent=1))
    figures(sub, tables, bins, name, out)
    return res


def figures(sub, tables, bins, name, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    lg, imp = tables["phys"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    t = imp.head(10)[::-1]
    ax[0].barh(t.feature, t.auc_drop_mean, xerr=t.auc_drop_sd, color="#4a6fa5")
    ax[0].set_xlabel("drop in held-out AUC when permuted")
    ax[0].set_title(f"Random forest, physics features (CV AUC {imp.attrs['cv_auc']:.2f})", fontsize=10)
    c = lg[(lg.term != "intercept") & ~lg.term.str.endswith("_missing")].sort_values("coef")
    ax[1].errorbar(c.coef, c.term, xerr=[c.coef - c.ci_lo, c.ci_hi - c.coef], fmt="o", color="#333", ms=4)
    ax[1].axvline(0, color="#999", lw=0.8)
    ax[1].set_xlabel("log-odds of success per SD (95 % cluster bootstrap CI)")
    ax[1].set_title("Logistic regression", fontsize=10)
    fig.savefig(out / f"{name}_drivers.png", dpi=150)
    plt.close(fig)
    feats = list(dict.fromkeys(bins.feature)) if len(bins) else []
    if feats:
        n = len(feats)
        fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 3), sharey=True, constrained_layout=True)
        for a, f in zip(np.atleast_1d(axs), feats, strict=False):
            b = bins[bins.feature == f]
            a.plot(range(len(b)), b.success, "o-", color="#4a6fa5")
            a.set_xticks(range(len(b)), [s.replace(", ", ",\n") for s in b.bin], fontsize=6)
            a.set_title(f, fontsize=8)
            a.set_ylim(0, 1)
        np.atleast_1d(axs)[0].set_ylabel("success rate")
        fig.savefig(out / f"{name}_bins.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--eval", required=True, help="eval/run.py output json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--thr", type=float, default=0.8, help="success = score >= thr")
    ap.add_argument("--out", default="runs/paper/results/P9")
    a = ap.parse_args(argv)
    d = load(a.features, a.eval, a.thr)
    res = analyse(d, a.name, Path(REPO / a.out))
    print(json.dumps({k: res[k] for k in ("name", "n", "success_rate", "phys", "phys_area")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
