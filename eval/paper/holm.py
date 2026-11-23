"""P9 / M7: the three pre-registered primary contrasts with Holm correction (plan 2026-10-28, M7 item 9).
    python -m eval.paper.holm --out runs/paper/results/P9full/holm.json
(i)   best RL vs best SFT on v6 (paired by task, score);
(ii)  best RL vs best prompted ladder model on v6 (paired by task, score);
(iii) twin-failure rate with vs without sensor_consistency on M2c (paired by pair_id; the only M set run both ways).
"Best" = highest v6 mean among the candidates, chosen here and reported. Cluster bootstrap over scenario seeds (the
number after "k" in the task id), two-sided p from the bootstrap distribution, Holm over the three p values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

R = Path("runs/paper/results")
RL = {"rl_b3": R / "P5/rl_b3_v6/eval.json", "rl_b3s1": R / "P5/rl_b3s1_v6/eval.json", "rl_a3": R / "P5/rl_a3_v6/eval.json"}
SFT = {"sft11": R / "P1/sft11_v6/eval.json", "sft14": R / "P1/sft14_v6/eval.json"}
LADDER = {"qwen3-4b": R / "P1/base_v6/eval.json", "qwen3-8b": R / "P2/qwen3-8b/v6.json",
          "qwen3-4b-awq": R / "P2/qwen3-4b-awq/v6.json"}


def _scores(p: Path) -> dict[str, float]:
    return {r["task_id"]: float(r["score"]) for r in json.loads(p.read_text())["rows"]}


def _seed(tid: str) -> str:
    return re.match(r"k(\d+)", tid).group(1)


def _paired(a: dict, b: dict, ids=None, n: int = 10000, rng_seed: int = 0) -> dict:
    ids = sorted(set(a) & set(b)) if ids is None else ids
    d = np.array([a[i] - b[i] for i in ids], float)
    seeds = np.array([_seed(i) for i in ids])
    groups = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n)
    for k in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        boot[k] = d[idx].mean()
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"n": len(ids), "clusters": len(groups), "delta": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "p": round(float(max(p, 1 / n)), 5)}


def _best(c: dict) -> tuple[str, dict]:
    s = {k: _scores(p) for k, p in c.items() if p.exists()}
    k = max(s, key=lambda k: np.mean(list(s[k].values())))
    return k, s[k], {k: round(float(np.mean(list(v.values()))), 4) for k, v in s.items()}


def _pid(set_name: str) -> dict[str, str]:
    rows = [json.loads(x) for x in Path(f"eval/frozen/{set_name}/tasks.jsonl").read_text().splitlines() if x.strip()]
    return {t["task_id"]: t["pair_id"] for t in rows if t.get("pair_id")}


def _twin(pairing: Path, thr: float = 0.8) -> dict[str, float]:
    """nominal task id -> 1 if twin-induced failure (ok on nominal, not ok on mismatch), as in eval.pairing."""
    d = json.loads(pairing.read_text())
    n, m = (json.loads(Path(d[k]).read_text())["rows"] for k in ("nominal", "mismatch"))
    pn, pm = _pid("v6_Npair"), _pid("v6_M2c")
    sm = {pm[r["task_id"]]: float(r["score"]) for r in m if r["task_id"] in pm and int(r.get("sample") or 0) == 0}
    return {r["task_id"]: float(float(r["score"]) >= thr and sm[pn[r["task_id"]]] < thr) for r in n
            if r["task_id"] in pn and pn[r["task_id"]] in sm and int(r.get("sample") or 0) == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper/results/P9full/holm.json")
    a = ap.parse_args()
    rl_k, rl, rl_means = _best(RL)
    sft_k, sft, sft_means = _best(SFT)
    lad_k, lad, lad_means = _best(LADDER)
    c = {"i_bestRL_vs_bestSFT_v6": {"a": rl_k, "b": sft_k, **_paired(rl, sft)},
         "ii_bestRL_vs_bestPrompted_v6": {"a": rl_k, "b": lad_k, **_paired(rl, lad)}}
    off, on = _twin(R / "P5/pairing/rl_b3_Npair_vs_M2c.json"), _twin(R / "P5/pairing/rl_b3_cons_Npair_vs_M2c.json")
    # Npair task ids differ between the two runs only if the sets differ; pair on the nominal pair_id instead
    pn = _pid("v6_Npair"); on = {pn[k]: v for k, v in on.items()}; off = {pn[k]: v for k, v in off.items()}
    tid = {v: k for k, v in pn.items()}
    on = {tid[k]: v for k, v in on.items()}; off = {tid[k]: v for k, v in off.items()}
    ids = sorted(set(off) & set(on))
    c["iii_twinfail_with_minus_without_consistency_M2c"] = {"model": "rl_b3", **_paired(on, off, ids=ids)}
    order = sorted(c, key=lambda k: c[k]["p"])
    m, prev = len(order), 0.0
    for j, k in enumerate(order):
        prev = max(prev, min(1.0, (m - j) * c[k]["p"]))
        c[k]["p_holm"] = round(prev, 5); c[k]["reject_0.05"] = prev < 0.05
    out = {"means_v6": {**rl_means, **sft_means, **lad_means}, "contrasts": c,
           "method": "cluster bootstrap over scenario seeds (10000), two-sided, Holm over 3"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
