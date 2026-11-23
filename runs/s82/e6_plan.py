"""E6 SELFCAL planner (docs/paper/SELFCAL_PREREG.md): run the frozen estimator and write hw-override YAMLs.
    pass 1: python runs/s82/e6_plan.py --pass 1 --tags M0 M5 M1 M6 M2 M7 M8
            estimates on <TAG> dirs -> runs/s82/e6/overrides/<TAG>p_s<seed>.yaml (correction = estimate)
    pass 2: python runs/s82/e6_plan.py --pass 2 ...
            estimates on <TAG>p dirs -> <TAG>j_s<seed>.yaml (correction = pass-1 correction + pass-2 estimate);
            plan.json lists the scenarios whose correction changed by > 0.25 deg (the others are copied from <TAG>p).
Observations + twin only; truth is read after the estimate, only for the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "runs/s82"))
from anchored_calib import calibrate  # noqa: E402

SC = REPO / "data/scenarios_v3"
OUT = REPO / "runs/s82/e6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="p", type=int, required=True, choices=(1, 2))
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", default="51-60")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    (OUT / "overrides").mkdir(parents=True, exist_ok=True)
    plan_f = OUT / "plan.json"
    plan = json.loads(plan_f.read_text()) if plan_f.exists() else {}
    for tag in a.tags:
        sig = yaml.safe_load((REPO / f"configs/hardware/mismatch/{tag}.yaml").read_text())["heading_datum"]["bias_deg_1sigma"]
        rebuild = []
        for s in range(lo, hi + 1):
            src_tag = tag if a.p == 1 else f"{tag}p"
            d = SC / f"kandla_s{s}_X_{src_tag}"
            if not (d / "observed.duckdb").exists():
                continue
            est = calibrate(d)
            prev = {}
            if a.p == 2:
                prev = yaml.safe_load((OUT / f"overrides/{tag}p_s{s}.yaml").read_text())["heading_datum"]["correction_deg"]
            corr = dict(prev)
            for sid, v in est.items():
                if v["source"] != "unresolved":
                    corr[sid] = round(prev.get(sid, 0.0) + float(v["bias_deg"]), 3)
            changed = a.p == 1 or any(abs(corr.get(k, 0.0) - prev.get(k, 0.0)) > 0.25 for k in set(corr) | set(prev))
            new_tag = f"{tag}p" if a.p == 1 else f"{tag}j"
            (OUT / f"overrides/{new_tag}_s{s}.yaml").write_text(yaml.safe_dump(
                {"heading_datum": {"bias_deg_1sigma": sig, "correction_deg": corr}}, sort_keys=True))
            rec = {"pass": a.p, "tag": new_tag, "seed": s, "correction_deg": corr, "changed": changed,
                   "sources": {k: v["source"] for k, v in est.items()}}
            (OUT / "estimates.jsonl").open("a").write(json.dumps(rec) + "\n")
            if changed:
                rebuild.append(s)
            print(new_tag, s, corr, "rebuild" if changed else "copy", flush=True)
        plan[f"{tag}{'p' if a.p == 1 else 'j'}"] = rebuild
    plan_f.write_text(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
