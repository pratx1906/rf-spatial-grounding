"""Fit / evaluate / apply the v2 track classifier (sim/obs_model/track_class.py). Offline only (WSL).

    python scripts/track_class_v2.py fit   --train 1-12,17-28 --val 13-14   # writes configs/track_class_v2.json
    python scripts/track_class_v2.py eval  --dirs kandla_s15_X kandla_s16_X
    python scripts/track_class_v2.py apply --all                              # rewrites tracks.klass_probs/confidence

`fit` and `eval` read world labels (truth/track_labels) to train and score; `apply` reads observed tables only.

--root (scenario root) and --model (classifier JSON) default to the data version (SCENESCOUT_DATA_VERSION,
envbuild.variant_roots.data_roots): v2 data/scenarios + configs/track_class_v2.json (demo), v21 data/scenarios_v21 +
configs/track_class_v21.json, v3 data/scenarios_v3 + configs/track_class_v3.json. `fit` (writes the model) and `apply`
(rewrites the sets) refuse the demo roots unless --allow-demo-roots (v2 only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_scenarios import parse_seeds  # noqa: E402
from envbuild.variant_roots import add_allow_flag, data_roots, guard_write  # noqa: E402
from sim.obs_model import track_class as tc  # noqa: E402

SC = REPO / "data" / "scenarios"          # demo default; main() sets the data version's root (--root)


def _load(d: Path, labels: bool):
    con = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    feats = tc.track_features(con)
    con.close()
    lab = {}
    if labels:
        t = duckdb.connect(str(d / "truth" / "truth.duckdb"), read_only=True)
        lab = dict(t.execute("select track_id, emitter_kind from track_labels").fetchall())
        t.close()
    return feats, lab


def _xy(dirs):
    X, y = [], []
    for d in dirs:
        f, lab = _load(d, True)
        for tid, v in f.items():
            X.append(v)
            y.append(tc.CLASSES.index(tc.label_class(lab.get(tid, "spurious"))))
    return np.array(X), np.array(y)


def _report(m, X, y, name):
    P = tc.predict(m, X)
    pred = P.argmax(1)
    dr = tc.CLASSES.index("drone")
    is_d = y == dr
    called = pred == dr
    out = {"set": name, "n": int(len(y)), "acc": round(float((pred == y).mean()), 3),
           "drone_recall": round(float((called & is_d).sum() / max(is_d.sum(), 1)), 3),
           "drone_precision": round(float((called & is_d).sum() / max(called.sum(), 1)), 3),
           "nondrone_called_drone": round(float((called & ~is_d).sum() / max((~is_d).sum(), 1)), 3)}
    conf = np.array([tc.confidence(p[dr], int(n)) for p, n in zip(P, X[:, 0], strict=True)])
    rule = called & (conf >= 0.6)                         # tasks/expert_kandla genuine_drone rule (MINC)
    out["expert_rule_acc"] = round(float((rule == is_d).mean()), 3)
    # AUC of p(drone) for drone vs not
    s = P[:, dr]
    pos, neg = s[is_d], s[~is_d]
    if len(pos) and len(neg):
        out["auc_drone"] = round(float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()), 3)
    print(json.dumps(out))
    return out


def _dirs(spec: str, suffix: str = "_X"):
    return [SC / f"kandla_s{s}{suffix}" for s in parse_seeds(spec) if (SC / f"kandla_s{s}{suffix}" / "observed.duckdb").exists()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval", "apply"])
    ap.add_argument("--train", default="1-12,17-28")
    ap.add_argument("--val", default="13-14")
    ap.add_argument("--dirs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--root", default=None, help="scenario root (default: the data version's)")
    ap.add_argument("--model", default=None, help="classifier JSON (default: the data version's)")
    add_allow_flag(ap)
    a = ap.parse_args()
    global SC
    roots = data_roots()
    SC = Path(a.root) if a.root else roots["scenarios"]
    model = Path(a.model) if a.model else roots["track_model"]
    if a.cmd == "fit":
        guard_write(model, a.allow_demo_roots)
        tr, va = _dirs(a.train), _dirs(a.val)
        if not tr:
            raise SystemExit(f"no training sets under {SC} for --train {a.train}")
        X, y = _xy(tr)
        m = tc.fit(X, y, l2=a.l2)
        m["train_dirs"] = [d.name for d in tr]
        m["metrics"] = [_report(m, X, y, "train")]
        if va:
            Xv, yv = _xy(va)
            m["metrics"].append(_report(m, Xv, yv, "val"))
        model.write_text(json.dumps(m, indent=1))
        print("wrote", model)
        return 0
    m = tc.load_model(model)
    dirs = sorted(SC.glob("kandla_s*_X*")) if a.all else [SC / d for d in (a.dirs or [])]
    if a.cmd == "eval":
        X, y = _xy(dirs)
        _report(m, X, y, ",".join(d.name for d in dirs))
        return 0
    for d in dirs:
        guard_write(d, a.allow_demo_roots)
    n = 0
    for d in dirs:
        if not (d / "observed.duckdb").exists():
            continue
        f, _ = _load(d, False)
        con = duckdb.connect(str(d / "observed.duckdb"))
        ns = dict(con.execute("select track_id, n_sensors from tracks").fetchall())
        for tid, v in f.items():
            p = tc.predict(m, v)[0]
            probs = {c: round(float(q), 3) for c, q in zip(tc.CLASSES, p, strict=True)}
            con.execute("update tracks set klass_probs = ?, confidence = ? where track_id = ?",
                        [json.dumps(probs), tc.confidence(probs["drone"], ns[tid] or 0), tid])
            n += 1
        con.close()
    print(json.dumps({"applied_dirs": len(dirs), "tracks": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
