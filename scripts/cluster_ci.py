"""S7.3: scenario-level (two-level cluster) CIs for the SFT v1.1 / v1.2 / v1.3 comparisons and answer-confidence
calibration, recomputed from the stored eval rows (runs/eval/*.json; no model calls).

    python scripts/cluster_ci.py            # -> runs/eval/cluster_ci.json (+ printed tables)

Clusters are scenario seeds (task id k<seed>...). Every frozen set (v3, v4, v4_W4, v4_W5) holds tasks from only two
test scenarios (15, 16), so a two-level bootstrap resamples two clusters: the CI then mostly reflects how different
the two scenarios are, which is the honest uncertainty for a claim about NEW scenarios (audit G Q48).
"pooled" stacks v4 + W4 + W5 per model pair (the same two seeds in three hidden worlds; clustered by seed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.run import calibration, cluster_bootstrap, paired_bootstrap, paired_cluster_bootstrap, scenario_of  # noqa: E402

EV = REPO / "runs" / "eval"
MODELS = ["v11", "v12", "v13"]
SETS = {"v3": "sft_{m}_v3.json", "v4": "sft_{m}_v4.json", "v4_W4": "sft_{m}_W4.json", "v4_W5": "sft_{m}_W5.json"}
PAIRS = [("v11", "v12"), ("v11", "v13"), ("v12", "v13")]


def rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def per_scenario_diff(ra: list[dict], rb: list[dict]) -> dict:
    ia = {r["task_id"]: r for r in ra}
    d: dict = {}
    for r in rb:
        if r["task_id"] in ia:
            d.setdefault(scenario_of(r), []).append(r["score"] - ia[r["task_id"]]["score"])
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(d.items())}


def main() -> int:
    out = {"pairs": {}, "scores": {}, "calibration": {}}
    for set_, pat in SETS.items():
        for m in MODELS:
            p = EV / pat.format(m=m)
            if not p.exists():
                continue
            rs = rows(p)
            cl: dict = {}
            for r in rs:
                cl.setdefault(scenario_of(r), []).append(r["score"])
            cb = cluster_bootstrap(cl)
            per_sc = {k: round(sum(v) / len(v), 3) for k, v in sorted(cl.items())}
            out["scores"][f"{m}/{set_}"] = {"score": round(cb["mean"], 3), "ci95_cluster": cb["ci95"],
                                            "per_scenario": per_sc, "n": cb["n"]}
            out["calibration"][f"{m}/{set_}"] = calibration(rs)
        for a, b in PAIRS:
            pa, pb = EV / pat.format(m=a), EV / pat.format(m=b)
            if pa.exists() and pb.exists():
                ra, rb = rows(pa), rows(pb)
                t, c = paired_bootstrap(ra, rb)["_all"], paired_cluster_bootstrap(ra, rb)
                out["pairs"][f"{b}-{a}/{set_}"] = {"n": t["n"], "diff": t["diff"], "ci95_task": t["ci95"],
                                                   "ci95_cluster": c["_all"]["ci95"],
                                                   "n_clusters": c["_all"]["n_clusters"],
                                                   "per_scenario_diff": per_scenario_diff(ra, rb),
                                                   "per_area_cluster": {k: v for k, v in c.items() if k != "_all"}}
    for a, b in PAIRS:                      # pooled over the v4 worlds (N, W4, W5): clusters = seeds
        ra, rb = [], []
        for set_ in ("v4", "v4_W4", "v4_W5"):
            pa, pb = EV / SETS[set_].format(m=a), EV / SETS[set_].format(m=b)
            if pa.exists() and pb.exists():
                ra += rows(pa)
                rb += rows(pb)
        if ra:
            t, c = paired_bootstrap(ra, rb)["_all"], paired_cluster_bootstrap(ra, rb)["_all"]
            out["pairs"][f"{b}-{a}/pooled_v4_worlds"] = {"n": t["n"], "diff": t["diff"], "ci95_task": t["ci95"],
                                                         "ci95_cluster": c["ci95"], "n_clusters": c["n_clusters"],
                                                         "per_scenario_diff": per_scenario_diff(ra, rb)}
    p = EV / "expert_v4.json"
    if p.exists():
        out["calibration"]["expert/v4"] = calibration(rows(p))
    (EV / "cluster_ci.json").write_text(json.dumps(out, indent=1))
    print("pair                      n    diff  CI task           CI cluster (2 seeds)   per scenario")
    for k, v in out["pairs"].items():
        print(f"{k:26s}{v['n']:5d} {v['diff']:+.3f}  {str(v['ci95_task']):18s} {str(v['ci95_cluster']):20s} "
              f"{v['per_scenario_diff']}")
    print("\nscore                 mean   CI cluster        per scenario")
    for k, v in out["scores"].items():
        print(f"{k:20s} {v['score']:.3f}  {str(v['ci95_cluster']):18s} {v['per_scenario']}")
    print("\ncalibration           n    ECE    Brier  conf  score  conf-wrong (of conf>=0.7)  distinct")
    for k, v in out["calibration"].items():
        if v.get("n"):
            print(f"{k:20s}{v['n']:4d}  {v['ece']:.3f}  {v['brier']:.3f}  {v['mean_conf']:.2f}  {v['mean_score']:.2f}  "
                  f"{v['confident_wrong_rate']:.3f} ({v['confident_wrong_of_confident']})  {v['n_distinct_conf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
