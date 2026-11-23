"""Print the markdown tables of docs/paper/TRACKER_GOSPA.md from gospa_full.json."""
import json
from pathlib import Path
f = json.loads((Path(__file__).parent / "gospa_full.json").read_text())
G = f["groups"]
def ci(sm, k, d=1):
    c = sm.get("ci95", {}).get(k)
    return f" [{c[0]:.{d}f}, {c[1]:.{d}f}]" if c else ""
def row(g, key):
    s = G[g][key]
    return (f"| {g} | {key} | {s['targets_per_step']:.1f} / {s['estimates_per_step']:.1f} | {s['gospa_mean_m']:.0f}{ci(s,'gospa_mean_m',0)} | "
            f"{s['gospa_per_target_m']:.1f}{ci(s,'gospa_per_target_m')} | {s['share_loc']:.2f} / {s['share_missed']:.2f} / {s['share_false']:.2f} | "
            f"{s['missed_rate']:.2f}{ci(s,'missed_rate',2)} | {s['false_per_step']:.2f} ({s['false_spurious_per_step']:.2f} / {s['false_target_labelled_per_step']:.2f} / {s['false_other_emitter_per_step']:.2f}) | "
            f"{s['assigned_rms_err_m']:.1f} / {s['assigned_med_err_m']:.1f} | " + (f"{s['ospa_mean_m']:.1f}{ci(s,'ospa_mean_m')}" if 'ospa_mean_m' in s else "-") + " |")
print("| group | config | targets / estimates per step | GOSPA mean m [CI] | per-target GOSPA m [CI] | share loc / missed / false | missed rate [CI] | false per step (spurious / target-labelled / other emitter) | assigned err RMS / median m | OSPA m |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c in ("c100", "c250"):
    for g in G:
        for cfg in ("uav", "uavg", "uav_s", "all"):
            for kind in ("tracker", "oracle"):
                print(row(g, f"{cfg}|{kind}|{c}"))
for g in ("nominal_13_16", "varied_51_60"):
    s = G[g]["uav|tracker|c100"]
    for dim in ("by_cls", "by_band", "by_motion"):
        for k, v in s[dim].items():
            print(f"| {g} | {dim[3:]} {k} | {v['target_steps']} | {v['per_target_gospa_m']:.1f} | {v['missed_rate']:.2f} | {v['assigned_rms_err_m']:.1f} |")
    for k, v in s["by_phase"].items():
        print(f"| {g} | {k} | {v['steps']} | {v['targets_per_step']:.1f} | {v['gospa_rms_m']:.0f} | {v['gospa_per_target_m']:.1f} | {v['share_loc']:.2f} / {v['share_missed']:.2f} / {v['share_false']:.2f} |")
print(json.dumps(f["cross_check"]))

# ---- detectability-gated variant (gospa_gated.py)
gp = Path(__file__).parent / "gospa_gated.json"
if gp.exists():
    H = json.loads(gp.read_text())
    print("\n| group | c | gate | UAV target-steps kept | targets / estimates per step | GOSPA mean m [CI] | per-target GOSPA m [CI] | share loc / missed / false | missed rate [CI] | false per step | oracle per-target m | OSPA m |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in ("nominal_13_16", "varied_51_60", "nominal_test_15_16"):
        for c in ("c100", "c250"):
            rows = [("ungated", G[g][f"uav|tracker|{c}"], G[g][f"uav|oracle|{c}"], 1.0)]
            for n in ("W2steps_ungated", "W2_s1", "W2_s2", "W10_s1", "W10_s2", "W10_s1_tol10"):
                rows.append((n, H["groups"][g][f"uav_{n}|tracker|{c}"], H["groups"][g][f"uav_{n}|oracle|{c}"],
                             H["groups"][g].get(f"kept_frac|{n}", 1.0)))
            for n, s, o, kf in rows:
                print(f"| {g} | {c} | {n} | {kf:.2f} | {s['targets_per_step']:.1f} / {s['estimates_per_step']:.1f} | "
                      f"{s['gospa_mean_m']:.0f}{ci(s, 'gospa_mean_m', 0)} | {s['gospa_per_target_m']:.1f}{ci(s, 'gospa_per_target_m')} | "
                      f"{s['share_loc']:.2f} / {s['share_missed']:.2f} / {s['share_false']:.2f} | "
                      f"{s['missed_rate']:.2f}{ci(s, 'missed_rate', 2)} | {s['false_per_step']:.2f} | "
                      f"{o['gospa_per_target_m']:.1f} | " + (f"{s['ospa_mean_m']:.1f}" if s.get('ospa_mean_m') else "-") + " |")
    print(json.dumps({t: {k: v for k, v in x.items()} for t, x in H["validation"].items()}))
