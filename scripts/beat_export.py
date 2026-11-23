"""Export what Blender needs to render the demo beats as 4K cinematic clips.

    python scripts/beat_export.py      (WSL venv)
    -> data/site_kandla/truth/render/drones.json   truth trajectories for the world render (offline, via TruthStore)
    -> runs/blender/beats/<task_id>.json           belief-side evidence geometry of the recorded SceneScout answer

The evidence comes only from the recorded SFT run (data/demo_replays/<task>__sft.json): what the agent saw and
claimed, in site ENU metres. The world (drones flying) is the simulation's truth, rendered offline like every other
Blender still; it is never shown to the agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.belief.site_view import SiteView  # noqa: E402
from tasks.truth_access import TruthStore  # noqa: E402

SITE = REPO / "data" / "site_kandla"
OUT = REPO / "runs" / "blender" / "beats"
BEATS = ["genuine_drone", "direct_path", "ds_hover", "suggest_one", "jam_links"]


def track_at(site: SiteView, ref: str, t: float):
    tid = ref.split(":", 1)[-1].split("@")[0]
    p = site.points.get(tid)
    if p is None or not len(p["t"]):
        return None
    i = int(np.clip(np.searchsorted(p["t"], t), 0, len(p["t"]) - 1))
    return {"ref": f"trk:{tid}", "t_s": float(p["t"][i]), "pos": [float(p["x"][i]), float(p["y"][i]), float(p["z"][i])],
            "ell": {"a_m": float(p["a"][i]), "b_m": float(p["b"][i]), "theta_deg": float(p["theta"][i])}}


def ring(geom):
    g = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda x: x.area)
    return [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords]


def beat(site: SiteView, task: dict) -> dict:
    rec = json.loads((REPO / "data" / "demo_replays" / f"{task['task_id']}__sft.json").read_text())
    ev = rec["events"]
    ans = next((e for e in ev if e["type"] == "answer"), {}).get("answer") or {}
    res = [e for e in ev if e["type"] == "tool_result" and (e.get("result") or {}).get("ok", True)]
    data = lambda name: [(e["result"] or {}).get("data") or {} for e in res if e["name"] == name]  # noqa: E731
    ui = lambda name: [e.get("ui") or {} for e in res if e["name"] == name]  # noqa: E731
    sensors = {sid: [s["pose"]["x_m"], s["pose"]["y_m"], s["pose"].get("z_m", 10.0)] for sid, s in site.sensors.items()}
    b = {"task_id": task["task_id"], "template": task["template"], "question": task["question"],
         "intent": ans.get("intent"), "confidence": ans.get("confidence"), "sensors": sensors,
         "wall_s": rec["summary"]["wall_s"]}
    p = task.get("params") or {}
    t = float(p.get("t") or 300.0)
    if task["template"] == "genuine_drone":
        d = data("track_detail")[0]
        t = float(d.get("t_s") or t)
        b["track"] = track_at(site, d["ref"], t)
        b["track"]["klass_probs"] = d.get("klass_probs")
        b["track"]["confidence"] = d.get("confidence")
        b["track"]["band"] = (d.get("signal") or {}).get("band")
        b["verdict"] = "yes" if ans.get("boolean") else "no"
    elif task["template"] == "direct_path":
        dh = data("df_hypotheses")
        chosen = {r.split("@")[0] for r in ans.get("chosen_hyp_refs", [])}
        rejected = {r.split("@")[0] for r in ans.get("rejected_hyp_refs", [])}
        rays, seen = [], set()
        for d in dh:
            t = float(d.get("t_s") or t)
            for s in d.get("sensors", []):
                for h in s.get("hyps", []):
                    r = h["ref"].split("@")[0]
                    if r in seen:
                        continue
                    seen.add(r)
                    st = "chosen" if r in chosen else "rejected" if r in rejected else "other"
                    rays.append({"sensor": s["sensor"].split(":")[1], "bearing_deg": h["bearing_deg"], "status": st, "ref": h["ref"]})
        b["rays"] = rays
        fc = data("fit_candidates")
        if fc and fc[0].get("candidates"):
            c = fc[0]["candidates"][0]
            b["fit"] = {"ref": c["ref"], "pos": [c["position"]["x_m"], c["position"]["y_m"], c["position"].get("z_m") or 0.0],
                        "ell": c["uncertainty"], "residual_deg": c.get("residual_deg_rms")}
        b["track"] = track_at(site, p.get("track") or dh[0]["track"], t)
        b["focus_sensor"] = p.get("sensor")
    elif task["template"] == "ds_hover":
        st, moves = data("drone_state"), data("move_drone")
        path = [[st[0]["pose"]["x_m"], st[0]["pose"]["y_m"], st[0]["pose"]["z_m"]]]
        ests = [{"pos": [st[0]["estimate"]["position"]["x_m"], st[0]["estimate"]["position"]["y_m"], 0.0], "ell": st[0]["estimate"]["uncertainty"]}]
        meas = []
        for m in moves:
            path.append([m["pose"]["x_m"], m["pose"]["y_m"], m["pose"]["z_m"]])
            ests.append({"pos": [m["estimate"]["position"]["x_m"], m["estimate"]["position"]["y_m"], 0.0], "ell": m["estimate"]["uncertainty"]})
            meas.append({"from": path[-1], "bearing_deg": m["measurement"]["bearing_deg"], "wp": m["wp"]})
        t = float(st[0].get("t_s") or t)
        b["drone"] = {"path": path, "estimates": ests, "measurements": meas,
                      "final": {"pos": [ans["final_position"]["x_m"], ans["final_position"]["y_m"], 0.0], "ell": ans["uncertainty"]}}
        b["track"] = track_at(site, st[0].get("target") or p.get("track"), min(t, 600.0))
    elif task["template"] == "suggest_one":
        c = data("suggest_node")[0]["candidates"][0]
        b["node"] = {"ref": c["ref"], "pos": [c["position"]["x_m"], c["position"]["y_m"], c["position"].get("z_m") or 12.0],
                     "gain": c.get("predicted_gain"), "err_before": data("suggest_node")[0]["baseline"]["region_mean_df_err_deg"],
                     "err_after": c.get("region_mean_df_err_after_deg")}
        zid = p["zone"].split(":", 1)[1]
        b["zone"] = {"ref": p["zone"], "name": site.zones[zid].name, "ring": ring(site.zones[zid].geom)}
    elif task["template"] == "jam_links":
        fp, u = data("effect_footprint")[0], ui("effect_footprint")[0]
        b["footprint"] = {"ref": fp["ref"], "band": fp["band"], "eirp_dbm": fp["eirp_dbm"], "threshold_dbm": fp["threshold_dbm"],
                          "rings": u.get("rings", []), "effector": u.get("effector") or sensors[p["effector"].split(":")[1]][:2]}
        b["victims"] = [v for v in ui("collateral_check")[0].get("victims", []) if v.get("pos")]
        b["effector"] = p["effector"].split(":")[1]
    b["t_s"] = t
    return b


def main() -> int:
    ts = TruthStore(SITE)
    (SITE / "truth" / "render").mkdir(parents=True, exist_ok=True)
    (SITE / "truth" / "render" / "drones.json").write_text(json.dumps(ts.trajectories()))
    ts.close()
    site = SiteView(SITE)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((REPO / "data" / "demo_tasks_kandla.json").read_text())
    for tmpl in BEATS:
        task = next(t for t in tasks if t["template"] == tmpl)
        b = beat(site, task)
        (OUT / f"{task['task_id']}.json").write_text(json.dumps(b, indent=1))
        print(tmpl, task["task_id"], "t", b["t_s"], [k for k in b if k in ("rays", "fit", "track", "drone", "node", "footprint", "victims")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
