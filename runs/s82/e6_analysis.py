"""E6 SELFCAL analysis as pre-registered (docs/paper/SELFCAL_PREREG.md). Output runs/s82/e6_results.json.
Uncalibrated = s81 sweep evals (runs/eval/s81/<pol>_<k>c.json); calibrated = runs/eval/s82/e6_<pol>_<k>jc.json; both paired
against the same nominal runs/eval/s81/<pol>_Npair.json. Metric: twin-induced failure on the world key, per pair_id.
    python runs/s82/e6_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.pairing import WorldKeys, _load, _seed_of_pair, key_agrees, part_ok_world  # noqa: E402

THR = 0.8
B = 10_000
W = [("M0", 0.5), ("M5", 1), ("M1", 3), ("M6", 5), ("M2", 10), ("M7", 15), ("M8", 30)]
POL = ["expert", "sft14", "sft14s1"]


def pp(n_file, m_file, wk):
    nres, nt = _load(n_file)
    mres, mt = _load(m_file)
    by = lambda res, t: {t[r["task_id"]]["pair_id"]: (r, t[r["task_id"]]) for r in res["rows"]
                         if t[r["task_id"]].get("pair_id") and r.get("sample", 0) == 0}
    pn, pm = by(nres, nt), by(mres, mt)
    out = {}
    for pid in set(pn) & set(pm):
        (rn, tn), (rm, tm) = pn[pid], pm[pid]
        wn, wm = wk.key(tn), wk.key(tm)
        if wn is None or wm is None:
            continue
        a = part_ok_world(tn, rn.get("answer"), wn, wk.track_emitter)
        b = part_ok_world(tm, rm.get("answer"), wm, wk.track_emitter)
        out[pid] = {"wtf": float(a and not b), "ok_m": float(b),
                    "key_wrong_m": float(not key_agrees(tm, wm, wk.track_emitter))}
    return out


def boot(v, rng):
    cl = defaultdict(list)
    for pid, x in v.items():
        cl[_seed_of_pair(pid)].append(x)
    ks = sorted(cl)
    arr = [np.asarray(cl[k]) for k in ks]
    pt = float(np.mean(np.concatenate(arr)))
    bs = np.array([np.mean(np.concatenate([rng.choice(arr[j], len(arr[j])) for j in rng.choice(len(ks), len(ks))]))
                   for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"mean": round(pt, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "p_boot": round(float(p), 4),
            "n": len(v), "clusters": len(ks)}


def main():
    rng = np.random.default_rng(20261107)
    wk = WorldKeys()
    unc, cal = {}, {}
    for p in POL:
        n = REPO / f"runs/eval/s81/{p}_Npair.json"
        for t, _ in W:
            unc[(p, t)] = pp(n, REPO / f"runs/eval/s81/{p}_{t}c.json", wk)
            cal[(p, t)] = pp(n, REPO / f"runs/eval/s82/e6_{p}_{t}jc.json", wk)
    res = {"rates": {}, "reduction": {}}

    def sftmean(d, t, key):
        a, b = d[("sft14", t)], d[("sft14s1", t)]
        return {pid: (a[pid][key] + b[pid][key]) / 2 for pid in a if pid in b}

    for t, sig in W:
        for p in POL + ["sft"]:
            for key in ("wtf", "ok_m", "key_wrong_m"):
                if p == "sft":
                    u, c = sftmean(unc, t, key), sftmean(cal, t, key)
                else:
                    u = {k: v[key] for k, v in unc[(p, t)].items()}
                    c = {k: v[key] for k, v in cal[(p, t)].items()}
                res["rates"][f"{p}/{t}/{key}"] = {"uncal": round(float(np.mean(list(u.values()))), 4),
                                                   "cal": round(float(np.mean(list(c.values()))), 4),
                                                   "n_uncal": len(u), "n_cal": len(c)}
                common = {k: u[k] - c[k] for k in u if k in c}   # positive = calibration reduced it
                res["reduction"][f"{p}/{t}/{key}"] = boot(common, rng)
    r = res["reduction"]
    # C1: M2, SFT and expert, Holm over the two
    ps = sorted([("sft", r["sft/M2/wtf"]["p_boot"]), ("expert", r["expert/M2/wtf"]["p_boot"])], key=lambda x: x[1])
    holm, stop = {}, False
    for i, (k, pv) in enumerate(ps):
        ok = (not stop) and pv <= 0.05 / (len(ps) - i)
        stop = stop or not ok
        holm[k] = ok
    res["C1_holm"] = holm
    res["C1_pass"] = bool(all(holm.values()) and r["sft/M2/wtf"]["mean"] > 0 and r["expert/M2/wtf"]["mean"] > 0)
    # C2: pooled over M1, M6, M2, M7 (pair ids are distinct per world: key them by world)
    pooled = {}
    for t in ("M1", "M6", "M2", "M7"):
        u, c = sftmean(unc, t, "wtf"), sftmean(cal, t, "wtf")
        for k in u:
            if k in c:
                pooled[f"{k}@{t}"] = u[k] - c[k]
    # cluster on scenario seed (strip the world suffix)
    cl_seed = {k: v for k, v in pooled.items()}
    res["C2_pooled_sft_reduction"] = boot({k.split("@")[0] + "@" + k.split("@")[1]: v for k, v in cl_seed.items()}, rng)
    res["C2_pass"] = bool(res["C2_pooled_sft_reduction"]["ci95"][0] > 0)
    # C3: M0 control, calibrated - uncalibrated <= +0.02 (upper bound), SFT and expert
    res["C3"] = {p: {"cal_minus_uncal": round(-r[f"{p}/M0/wtf"]["mean"], 4),
                     "upper_ci": round(-r[f"{p}/M0/wtf"]["ci95"][0], 4)} for p in ("sft", "expert")}
    res["C3_pass"] = bool(all(v["upper_ci"] <= 0.02 for v in res["C3"].values()))
    # C4: twin key error on M2 (policy-free; use the expert's pairs)
    res["C4_M2_key_wrong_reduction"] = r["expert/M2/key_wrong_m"]
    res["C4_pass"] = bool(r["expert/M2/key_wrong_m"]["ci95"][0] > 0)
    res["C5_M8"] = {k: r[f"{k}/M8/wtf"] for k in ("sft", "expert")}
    (REPO / "runs/s82/e6_results.json").write_text(json.dumps(res, indent=1))
    print("world  pol     wtf uncal -> cal   reduction [CI]        key_wrong uncal -> cal")
    for t, sig in W:
        for p in ("expert", "sft"):
            a, b = res["rates"][f"{p}/{t}/wtf"], res["rates"][f"{p}/{t}/key_wrong_m"]
            rr = r[f"{p}/{t}/wtf"]
            print(f"{t}({sig:>4}) {p:6s} {a['uncal']:.3f} -> {a['cal']:.3f}  {rr['mean']:+.3f} [{rr['ci95'][0]:+.3f},{rr['ci95'][1]:+.3f}]"
                  f"   {b['uncal']:.3f} -> {b['cal']:.3f}  (n {a['n_uncal']}/{a['n_cal']})")
    print({k: res[k] for k in ("C1_pass", "C1_holm", "C2_pass", "C3", "C3_pass", "C4_pass")})
    print("C2", res["C2_pooled_sft_reduction"])


if __name__ == "__main__":
    main()
