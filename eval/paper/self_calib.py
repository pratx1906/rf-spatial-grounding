"""T-FIX arm T3 (docs/paper/TWIN_FIX_PREREG.md): truth-free self-calibration of DF heading bias per scenario.
    python -m eval.paper.self_calib --tags N M0 M1 M2 M3 M4 --seeds 51-60 --out runs/paper/results/TFIX/self_calib.json
Runs the opt-in sensor_consistency tool over the whole scenario (observations + twin only, inside the truth guard) and
writes, per scenario dir, the per-node bias estimate and the suspect flags. The correction applied by the T3 rebuild
is the estimated bias of SUSPECT nodes only (0 elsewhere). The realised world biases (truth/world_variant.yaml) are
read afterwards, offline, only to report the recovery error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from eval.ablations import apply_ablation
from harness.config import EnvConfig
from harness.env import AkshaEnv
from tasks.expert import _call

SC = Path("data/scenarios_v3")


def _dir(seed: int, tag: str) -> Path:
    return SC / (f"kandla_s{seed}_X" if tag == "N" else f"kandla_s{seed}_X_{tag}")


def estimate(site: Path) -> dict:
    env = AkshaEnv(site, config=apply_ablation(EnvConfig.load(), "with_consistency"))
    try:
        env.reset({"task_id": "selfcal", "area": "operator_query", "template": "zone_incursion", "question": "selfcal",
                   "answer_key": {}, "split": "test", "scenario_seed": 0, "site": str(site), "params": {}})
        r = _call(env, "sensor_consistency", "self-calibration over the whole scenario")
    finally:
        env.close() if hasattr(env, "close") else None
    if not r["ok"]:
        return {"ok": False, "error": r.get("error")}
    nodes = {n["sensor_ref"].split(":")[1]: n for n in r["data"]["nodes"]}
    return {"ok": True, "n_epochs": r["data"]["n_epochs"], "suspects": r["data"]["suspect_refs"],
            "nodes": {s: {k: n.get(k) for k in ("bias_deg", "spread_deg", "n", "suspect")} for s, n in nodes.items()},
            "correction_deg": {s: round(float(n.get("bias_deg") or 0.0), 3) for s, n in nodes.items() if n.get("suspect")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["N", "M0", "M1", "M2", "M3", "M4"])
    ap.add_argument("--seeds", default="51-60")
    ap.add_argument("--out", default="runs/paper/results/TFIX/self_calib.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rep = {}
    for tag in a.tags:
        for s in range(lo, hi + 1):
            d = _dir(s, tag)
            if not (d / "observed.duckdb").exists():
                continue
            e = estimate(d)
            wv = d / "truth" / "world_variant.yaml"          # offline report only, after the estimate
            if e.get("ok") and wv.exists():
                real = {k: v["heading_bias_deg"] for k, v in (yaml.safe_load(wv.read_text()).get("nodes") or {}).items()}
                e["realised_bias_deg"] = real
                e["abs_err_deg"] = {k: round(abs((v["bias_deg"] or 0.0) - real.get(k, 0.0)), 3)
                                    for k, v in e["nodes"].items() if v["bias_deg"] is not None}
            rep[d.name] = e
            print(d.name, e.get("suspects"), e.get("correction_deg"), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
