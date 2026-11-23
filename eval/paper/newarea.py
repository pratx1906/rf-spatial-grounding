"""Unseen-area test (paper 2026-10-31): new area (newarea_v1, seeds 61-66, cast newarea) vs the familiar area (v6 rows on
seeds 51-60, the same randomised cast generator anchored on the trained tank farms / jetties).
    python -m eval.paper.newarea --out runs/paper/results/NA/compare.json
Per model: overall and area-macro mean score on both sets, the difference with a cluster bootstrap over scenario seeds
(the two sets are independent: resample seeds within each set), per-area scores, and the score on the questions that
name a new zone. Relative = new-area score / expert new-area score (how much of the solvable part the model gets).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
MODELS = {"expert": ("P1/expert_v6", "NA/expert_newarea_v1"), "sft11": ("P1/sft11_v6", "NA/sft11_newarea_v1"),
          "sft14": ("P1/sft14_v6", "NA/sft14_newarea_v1"), "rl_b3": ("P5/rl_b3_v6", "NA/rl_b3_newarea_v1")}
NEW_ZONES = ("south_warehouse", "jetty_j6", "South warehouse", "Jetty J6", "J6 zone")


def _rows(d: str) -> list[dict]:
    p = R / d / "eval.json"
    return json.loads(p.read_text())["rows"] if p.exists() else []


def _seed(tid: str) -> int:
    return int(re.match(r"k(\d+)", tid).group(1))


def _macro(rows: list[dict]) -> float:
    areas = sorted({r["area"] for r in rows})
    return float(np.mean([np.mean([float(r["score"]) for r in rows if r["area"] == a]) for a in areas]))


def _boot(a: list[dict], b: list[dict], n: int = 5000) -> list[float]:
    rng = np.random.default_rng(0)
    ga = {}; gb = {}
    for r in a:
        ga.setdefault(_seed(r["task_id"]), []).append(r)
    for r in b:
        gb.setdefault(_seed(r["task_id"]), []).append(r)
    ka, kb = list(ga), list(gb)
    out = []
    for _ in range(n):
        ra = [r for k in rng.choice(ka, len(ka)) for r in ga[k]]
        rb = [r for k in rng.choice(kb, len(kb)) for r in gb[k]]
        out.append(_macro(ra) - _macro(rb))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/NA/compare.json")
    a = ap.parse_args()
    tasks = {json.loads(x)["task_id"]: x for x in Path("eval/frozen/newarea_v1/tasks.jsonl").read_text().splitlines()
             if x.strip()}
    res = {}
    for m, (fam, new) in MODELS.items():
        f = [r for r in _rows(fam) if 51 <= _seed(r["task_id"]) <= 60]
        nw = _rows(new)
        if not f or not nw:
            continue
        nz = [r for r in nw if any(z in tasks.get(r["task_id"], "") for z in NEW_ZONES)]
        pk = set(json.loads((R / "NA/verify/pocket_task_ids.json").read_text()))
        pr = [r for r in nw if r["task_id"] in pk]
        areas = sorted({r["area"] for r in f + nw})
        res[m] = {"familiar_n": len(f), "new_n": len(nw),
                  "familiar_mean": round(float(np.mean([float(r["score"]) for r in f])), 4),
                  "new_mean": round(float(np.mean([float(r["score"]) for r in nw])), 4),
                  "familiar_macro": round(_macro(f), 4), "new_macro": round(_macro(nw), 4),
                  "macro_delta_new_minus_familiar": round(_macro(nw) - _macro(f), 4), "ci95": _boot(nw, f),
                  "new_zone_questions": {"n": len(nz), "mean": round(float(np.mean([float(r["score"]) for r in nz])), 4)
                                         if nz else None},
                  "pocket_keyed": {"n": len(pr), "mean": round(float(np.mean([float(r["score"]) for r in pr])), 4) if pr else None},
                  "per_area": {ar: {"familiar": round(float(np.mean([float(r["score"]) for r in f if r["area"] == ar] or [np.nan])), 3),
                                    "new": round(float(np.mean([float(r["score"]) for r in nw if r["area"] == ar] or [np.nan])), 3)}
                               for ar in areas}}
    if "expert" in res:
        for m in res:
            res[m]["new_macro_rel_expert"] = round(res[m]["new_macro"] / res["expert"]["new_macro"], 4)
            res[m]["familiar_macro_rel_expert"] = round(res[m]["familiar_macro"] / res["expert"]["familiar_macro"], 4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for m, v in res.items():
        print(f"{m:7s} familiar {v['familiar_macro']:.3f}  new {v['new_macro']:.3f}  delta {v['macro_delta_new_minus_familiar']:+.3f} "
              f"{v['ci95']}  new-zone qs {v['new_zone_questions']}  pocket-keyed {v['pocket_keyed']}  rel-expert fam/new "
              f"{v.get('familiar_macro_rel_expert')}/{v.get('new_macro_rel_expert')}")


if __name__ == "__main__":
    main()
