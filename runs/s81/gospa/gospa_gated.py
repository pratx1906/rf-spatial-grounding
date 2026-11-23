"""Detectability-gated GOSPA (OFFLINE; run AFTER gospa_eval.py, which it imports; merges keys into gospa_summary.json).

Why a gate from observations and not from truth transmissions: in the v3 simulator (sim/obs_model/observe_v2.py)
every emitter with a truth position transmits at EVERY observation epoch (EPOCH_S = 10 s); there is no duty-cycle
or silence model at epoch level and the truth DB has no per-emitter burst table. A "transmitted within W" gate from
truth is therefore a no-op. The gate here is "was DETECTED", from the observation products:

  burst attribution (independent of the tracker): each observed burst (observed.duckdb:bursts: sensor_id, t_s, f_mhz,
  bearing_deg) is assigned to the candidate with the smallest |bearing - truth bearing from that sensor|, among
  (a) scenario emitters present at t_s whose band at t_s (truth `bands` schedule) equals the burst's band, and
  (b) background transmitters (world_cache_v2/spectrum_env.yaml) in that band (> 30 m from the sensor);
  accepted if the difference <= TOL deg (5; 10 as sensitivity). A burst won by a background transmitter attributes
  to no drone.
  gate (W, n): a UAV is a target at step k iff bursts attributed to it were detected by >= n distinct sensors with
  burst epoch in [k - W, k]. W = 2 s is only meaningful at steps with an epoch in that window (k mod 10 in {0, 2}),
  so W = 2 is evaluated on those steps only; W = 10 covers every step.
  Tracks labelled to a UAV that exists but is gated out at k are dropped (not counted false).

Validation: on bursts the tracker tagged (bursts.track_id) whose track is truth-labelled to a scenario emitter, the
fraction whose bearing attribution names the same emitter.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("SCENESCOUT_DATA_VERSION", "v3")
os.environ.setdefault("SCENESCOUT_RF_GEOM", "rfr")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gospa_eval as G  # noqa: E402

OUT = G.OUT
EPOCH_S = 10
BAND_OF = (("sub_ghz_868_915", 840.0, 960.0), ("ism_2400", 2300.0, 2550.0), ("ism_5800", 5600.0, 5950.0))  # sim.rf.spectrum
UAV = {"single", "swarm", "friendly"}
GATES = {  # name: (W s, min sensors, tol deg)
    "W2_s1": (2, 1, 5.0), "W2_s2": (2, 2, 5.0), "W10_s1": (10, 1, 5.0), "W10_s2": (10, 2, 5.0),
    "W10_s1_tol10": (10, 1, 10.0),
}


def band_of(f):
    return next((b for b, lo, hi in BAND_OF if lo <= f <= hi), None)


def bearing(x0, y0, x1, y1):
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def adiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def attribute(S: dict, bgfull: list, tol: float) -> tuple[dict, dict]:
    d = G.ROOT / f"kandla_s{S['seed']}_X"
    sens = {s["id"]: (s["pose"]["x_m"], s["pose"]["y_m"]) for s in yaml.safe_load((d / "sensors.yaml").read_text())["sensors"]}
    o = duckdb.connect(str(d / "observed.duckdb"), read_only=True)
    rows = o.execute("select sensor_id, t_s, f_mhz, bearing_deg, track_id from bursts where bearing_deg is not null").fetchall()
    o.close()
    det = defaultdict(lambda: defaultdict(set))       # emitter -> epoch -> sensors
    agree = n_val = n_att = 0
    cat = defaultdict(int)
    for sid, ts, f, brg, tid in rows:
        b = band_of(f)
        k = int(round(ts))
        if b is None or sid not in sens:
            continue
        sx, sy = sens[sid]
        best, bd = None, 1e9
        for e, pe in S["P"].items():
            if k in pe and G.band_at(S["bands"][e], k) == b:
                dd = adiff(brg, bearing(sx, sy, pe[k][0], pe[k][1]))
                if dd < bd:
                    best, bd = e, dd
        for bid, bx, by, bf in bgfull:
            if band_of(bf) == b and math.hypot(bx - sx, by - sy) > 30:
                dd = adiff(brg, bearing(sx, sy, bx, by))
                if dd < bd:
                    best, bd = "bg:" + bid, dd
        att = best if bd <= tol else None
        if att is not None and not att.startswith("bg:"):
            det[att][k].add(sid)
            n_att += 1
        L = S["lab"].get(tid) if tid is not None else None
        if L in S["em"]:
            n_val += 1
            agree += att == L
            if k not in S["P"][L]:
                cat["label_absent"] += 1
            elif G.band_at(S["bands"][L], k) != b:
                cat["label_other_band"] += 1                    # tracker tagged a burst outside its emitter's band
            elif att == L:
                cat["agree"] += 1
            elif att is not None and not att.startswith("bg:") and S["sw"].get(att) and S["sw"].get(att) == S["sw"].get(L):
                cat["same_swarm_member"] += 1
            elif att is None:
                cat["none_within_tol"] += 1
            elif att.startswith("bg:"):
                cat["background_wins"] += 1
            else:
                cat["other_emitter"] += 1
    return det, {"bursts": len(rows), "attributed_to_scenario_emitter": n_att, "validation_bursts": n_val,
                 "validation_agree": agree, **{f"cat_{k}": v for k, v in cat.items()}}


def gate_sets(S, det, W, n):
    gate = {}
    for e, pe in S["P"].items():
        if S["em"][e] not in UAV:
            continue
        ep = det.get(e, {})
        g = set()
        for k in pe:
            ss = set()
            for t0 in range(int(math.ceil((k - W) / EPOCH_S)) * EPOCH_S, k + 1, EPOCH_S):
                ss |= ep.get(t0, set())
            if len(ss) >= n:
                g.add(k)
        gate[e] = g
    return gate


def main() -> int:
    env = yaml.safe_load((G.REPO / "data/site_kandla/world_cache_v2/spectrum_env.yaml").read_text())["emitters"]
    bgfull = [(e["id"], e["x_m"], e["y_m"], e["f_mhz"]) for e in env]
    bg = G._bg()
    seeds = sorted({s for v in G.GROUPS.values() for s in v})
    per_set, diag, val = {}, defaultdict(dict), defaultdict(lambda: defaultdict(int))
    for s in seeds:
        S = G.load(s, bg)
        per_set[s] = {}
        dets = {}
        for tol in sorted({g[2] for g in GATES.values()}):
            dets[tol], v = attribute(S, bgfull, tol)
            for kk, vv in v.items():
                val[tol][kk] += vv
        uav_steps = sum(len([k for k in pe if k <= G.T_MAX]) for e, pe in S["P"].items() if S["em"][e] in UAV)
        for name, (W, n, tol) in GATES.items():
            gate = gate_sets(S, dets[tol], W, n)
            steps_ok = None if W >= EPOCH_S else {k for k in range(0, G.T_MAX + 1, 2) if (k % EPOCH_S) <= W}
            kept = sum(len([k for k in g if k <= G.T_MAX and (steps_ok is None or k in steps_ok)]) for g in gate.values())
            tot = sum(len([k for k in pe if k <= G.T_MAX and (steps_ok is None or k in steps_ok)])
                      for e, pe in S["P"].items() if S["em"][e] in UAV)
            diag[s][name] = {"uav_steps": tot, "kept": kept}
            r = G.eval_set(S, {"uav": G.CONFIGS["uav"]}, f"uav_{name}", gate=gate, steps_ok=steps_ok)
            per_set[s].update(r["acc"])
            if name.startswith("W2"):       # ungated reference on the same W = 2 step subset
                r0 = G.eval_set(S, {"uav": G.CONFIGS["uav"]}, "uav_W2steps_ungated", steps_ok=steps_ok)
                per_set[s].update(r0["acc"])
        print("set", s, {k: round(v["kept"] / max(v["uav_steps"], 1), 3) for k, v in diag[s].items()}, flush=True)
    rng = np.random.default_rng(82)
    for v in val.values():
        comp = sum(v.get(f"cat_{c}", 0) for c in ("agree", "same_swarm_member", "none_within_tol", "background_wins",
                                                    "other_emitter"))
        v["comparable_bursts"] = comp
        v["agree_rate_comparable"] = v.get("cat_agree", 0) / max(comp, 1)
        v["agree_rate_comparable_swarm_level"] = (v.get("cat_agree", 0) + v.get("cat_same_swarm_member", 0)) / max(comp, 1)
    out = {"validation": {str(t): dict(v, agree_rate=v["validation_agree"] / max(v["validation_bursts"], 1),
                                        attributed_frac=v["attributed_to_scenario_emitter"] / max(v["bursts"], 1))
                          for t, v in val.items()},
           "gates": {k: {"W_s": v[0], "min_sensors": v[1], "tol_deg": v[2]} for k, v in GATES.items()},
           "diag": diag, "groups": {}}
    keys = sorted(per_set[seeds[0]].keys())
    for gname, gs in G.GROUPS.items():
        out["groups"][gname] = {}
        for key in keys:
            accs = [per_set[s][key] for s in gs]
            sm = G.summarise(accs)
            sm["ci95"] = G.boot(accs, rng) if len(gs) > 2 else {}
            out["groups"][gname][key] = sm
        for name in GATES:
            kept = sum(diag[s][name]["kept"] for s in gs)
            tot = sum(diag[s][name]["uav_steps"] for s in gs)
            out["groups"][gname][f"kept_frac|{name}"] = kept / tot
    (OUT / "gospa_gated.json").write_text(json.dumps(out, indent=1, default=float))
    # merge flat keys (new keys only) into gospa_summary.json
    summ_p = OUT / "gospa_summary.json"
    flat = json.loads(summ_p.read_text())
    for gname in G.GROUPS:
        for key, sm in out["groups"][gname].items():
            if key.startswith("kept_frac|"):
                flat[f"{gname}/gated/{key.split('|')[1]}/uav_target_steps_kept_frac"] = sm
                continue
            cfg, kind, c = key.split("|")
            name = cfg.removeprefix("uav_")
            pre = f"{gname}/gated/{name}/uav/{kind}/{c}"
            for k in ("gospa_mean_m", "gospa_rms_m", "gospa_per_target_m", "share_loc", "share_missed", "share_false",
                      "missed_rate", "missed_per_step", "false_per_step", "false_spurious_per_step",
                      "false_target_labelled_per_step", "assigned_rms_err_m", "assigned_med_err_m", "targets_per_step",
                      "ospa_mean_m"):
                if sm.get(k) is not None:
                    flat[f"{pre}/{k}"] = sm[k]
            for k, v in sm.get("ci95", {}).items():
                flat[f"{pre}/ci95/{k}"] = v
    for t, v in out["validation"].items():
        flat[f"meta/gate_attribution/tol{int(float(t))}/agree_rate"] = v["agree_rate"]
        flat[f"meta/gate_attribution/tol{int(float(t))}/validation_bursts"] = v["validation_bursts"]
        flat[f"meta/gate_attribution/tol{int(float(t))}/agree_rate_comparable"] = v["agree_rate_comparable"]
        flat[f"meta/gate_attribution/tol{int(float(t))}/agree_rate_comparable_swarm_level"] = v["agree_rate_comparable_swarm_level"]
    summ_p.write_text(json.dumps(flat, indent=1, sort_keys=True))
    print(json.dumps(out["validation"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
