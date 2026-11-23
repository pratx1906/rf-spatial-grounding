"""Kandla task generators with answer keys (SPEC 8.1, offline; the ONLY consumer of truth besides scripts).

Areas and template families (several phrasings each; `params` carries the structured slots for the scripted
expert and is never rendered to the agent):
- operator_query: zone_incursion, count_in_zone, tracks_in_band, swarm_members, near_object, which_zone,
  genuine_drone, los_check
- df_ambiguity: direct_path, direct_all, locate, locate_ghost
- drone_search: ds_hover, ds_hover_budget2, ds_loiter, ds_swarm, ds_wide_prior
- siting: weak_df, weak_detect, weak_low_alt, suggest_one, suggest_two
- collateral: jam_all, jam_protected, jam_links, jam_aviation, jam_point
- replay: first_entry, exit_time, loiter_start, band_change, entered_between, first_detection
- unanswerable: entity_not_found, actuation; entity_ambiguous (operator clarifies)
Keys: OQ/DF/DS from truth; RP from the recorded events (what the system logged); SS/CC from the nominal model
(DECISIONS D-025).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from sim.belief.site_view import SiteView
from sim.geometry import bearing_deg, los_clear, stable_seed
from tasks.base import Task
from tasks.truth_access import TruthStore
from tools.registry import REGISTRY, ToolContext, load_all

REPO = Path(__file__).resolve().parents[2]
BAND_PHRASE = {"ism_2400": ["2.4 GHz", "the 2.4 GHz band", "2400 MHz"],
               "ism_5800": ["5.8 GHz", "the 5.8 GHz band", "5800 MHz"],
               "sub_ghz_868_915": ["868/915 MHz", "the sub-GHz band", "900 MHz"]}
DRONE_KINDS = ("drone", "friendly_drone")
FAKE = ["LNG terminal", "Berth 17", "Coal Jetty 9", "Pilot Station", "Tuna Tekra", "Warehouse W42"]
ACT = ["Jam the drone over the {z} now.", "Take control of the drone near the {z} and land it.",
       "Shoot down trk:{t}."]


class Gen:
    def __init__(self, site_dir: Path, seed: int, split: str, tag: str = "", family_rng: bool = False):
        """tag: appended to the seed in task ids (world variants, D-031: k15W1-...). family_rng: reseed the RNG per
        task area, so one area's sampling does not shift the next (paired world-gap evaluation)."""
        load_all()
        self.tag, self.family_rng = tag, family_rng
        self.dir = Path(site_dir)
        self.rel = str(self.dir.relative_to(REPO)).replace("\\", "/")
        self.seed, self.split = seed, split
        self.site = SiteView(self.dir)
        self.truth = TruthStore(self.dir)
        self.tof = self.truth.track_of()                       # tid -> (emitter, kind)
        self.meta = self.truth.emitter_meta()
        # v2 sets: the truth-free tracker also forms clutter / spurious tracks (labelled with a background emitter or
        # "none", kind clutter / spurious). They get a stub truth entry: not a drone kind, no swarm, no band schedule,
        # so truth-keyed questions exclude them while the agent still sees them (v1: none exist, unchanged)
        for eid, kind in set(self.tof.values()):
            self.meta.setdefault(eid, {"kind": kind, "swarm_id": None, "bands": [], "eirp_dbm": None})
        self.rng = random.Random(seed * 1009 + 7)
        self.ctx = ToolContext(self.site)
        self.n = 0
        objs = [(o.geom, float(o.height_m)) for o in self.site.objects.values()
                if o.height_m and o.height_m > 1 and o.kind not in ("tank_farm", "road")]
        from shapely import STRtree

        self.bld = objs
        self.bld_tree = STRtree([g for g, _ in objs])

    # --- helpers ---------------------------------------------------------------------------------------------
    def task(self, area, family, question, key, params, **kw) -> Task:
        self.n += 1
        return Task(f"k{self.seed:02d}{self.tag}-{family}-{self.n:05d}", area, family, question, key, split=self.split,
                    scenario_seed=self.seed, site=self.rel, params=params, **kw)

    def call(self, name, **args):
        spec = REGISTRY[name]
        self.ctx.episode = {}
        return spec.fn(self.ctx, spec.args.model_validate(args)).data

    def est_times(self, tid) -> set[float]:
        return set(np.round(self.site.points[tid]["t"], 1).tolist())

    def pos(self, tid, t):
        return self.truth.position(self.tof[tid][0], t)

    def drones(self):
        return [t for t, (_, k) in self.tof.items() if k in DRONE_KINDS]

    def zone_name(self, zid):
        z = self.site.zones[zid]
        return z.aliases[0] if z.aliases else z.name

    def inside_zone(self, zid, t):
        g = self.site.zones[zid].geom
        out = []
        for tid in self.drones():
            if t not in self.est_times(tid):
                continue
            p = self.pos(tid, t)
            if p and g.contains(Point(p[0], p[1])):
                out.append(f"trk:{tid}")
        return sorted(out)

    def pick(self, items, k):
        items = list(items)
        return self.rng.sample(items, min(k, len(items)))

    # --- operator_query --------------------------------------------------------------------------------------
    def oq(self, k: int) -> list[Task]:
        site, out = self.site, []
        zones = [z for z, v in site.zones.items() if v.kind == "restricted"] + ["hero_zone", "kandla_creek", "kict"]
        times = [float(t) for t in range(30, 600, 20)]
        pos, neg = [], []
        for z in zones:
            for t in times:
                (pos if self.inside_zone(z, t) else neg).append((z, t))
        ph = ["Is any drone inside the {z} at t={t:.0f} s? List the tracks.",
              "At t={t:.0f} s, are there drones in the {z}? Which tracks?",
              "Any UAV within the {z} at t={t:.0f} s? Give the track IDs."]
        for z, t in self.pick(pos, k) + self.pick(neg, max(1, k // 2)):
            key = self.inside_zone(z, t)
            out.append(self.task("operator_query", "zone_incursion", self.rng.choice(ph).format(z=self.zone_name(z), t=t),
                                 {"track_refs": key, "boolean": bool(key), "zone": f"zone:{z}", "t_s": t},
                                 {"zone_phrase": self.zone_name(z), "t": t}))
        ph = ["How many drones are inside the {z} at t={t:.0f} s? List them.",
              "Count the drone tracks in the {z} at t={t:.0f} s."]
        for z, t in self.pick(pos, k):
            key = self.inside_zone(z, t)
            out.append(self.task("operator_query", "count_in_zone", self.rng.choice(ph).format(z=self.zone_name(z), t=t),
                                 {"track_refs": key, "boolean": None, "zone": f"zone:{z}", "t_s": t},
                                 {"zone_phrase": self.zone_name(z), "t": t}))
        # tracks in band
        ph = ["Which drones are transmitting on {b} at t={t:.0f} s?", "List drone tracks on {b} at t={t:.0f} s."]
        for t in self.pick(times, k):
            band = self.rng.choice(list(BAND_PHRASE))
            key = []
            for tid in self.drones():
                p = site.points[tid]
                idx = np.flatnonzero(np.isclose(p["t"], t))
                if len(idx) and p["band"][idx[0]] == band:
                    key.append(f"trk:{tid}")
            out.append(self.task("operator_query", "tracks_in_band",
                                 self.rng.choice(ph).format(b=self.rng.choice(BAND_PHRASE[band]), t=t),
                                 {"track_refs": sorted(key), "boolean": None, "t_s": t},
                                 {"band": band, "t": t}))
        # swarm members (truth swarm id)
        members: dict[str, list[str]] = {}
        for tid, (eid, _) in self.tof.items():
            sw = self.meta[eid]["swarm_id"]
            if sw:
                members.setdefault(sw, []).append(f"trk:{tid}")
        ph = ["Which tracks fly in the same swarm as {x}? List all members.",
              "{x} looks like part of a group. Which tracks belong to that swarm?"]
        for sw, mem in self.pick(members.items(), max(1, k // 2)):
            x = self.rng.choice(mem)
            out.append(self.task("operator_query", "swarm_members", self.rng.choice(ph).format(x=x),
                                 {"track_refs": sorted(mem), "boolean": None}, {"track": x}))
        # near object
        objs = [o for o in site.objects.values() if o.kind in ("tank_farm", "jetty", "substation")]
        ph = ["Is any drone within {r:.0f} m of {o} at t={t:.0f} s?", "Within {r:.0f} m of {o}, any drones at t={t:.0f} s?"]
        for _ in range(k):
            o, t, r = self.rng.choice(objs), self.rng.choice(times), self.rng.choice([200.0, 300.0, 500.0])
            key = []
            for tid in self.drones():
                if t not in self.est_times(tid):
                    continue
                p = self.pos(tid, t)
                if p and o.geom.distance(Point(p[0], p[1])) <= r:
                    key.append(f"trk:{tid}")
            out.append(self.task("operator_query", "near_object", self.rng.choice(ph).format(r=r, o=o.name, t=t),
                                 {"track_refs": sorted(key), "boolean": bool(key), "t_s": t},
                                 {"obj_phrase": o.name, "radius_m": r, "t": t}))
        # which restricted zone
        restricted = {z: v for z, v in site.zones.items() if v.kind == "restricted"}
        cands = []
        for tid in self.drones():
            for t in sorted(self.est_times(tid))[::10]:
                p = self.pos(tid, t)
                if not p:
                    continue
                inz = [z for z, v in restricted.items() if v.geom.contains(Point(p[0], p[1]))]
                if len(inz) == 1:
                    cands.append((tid, t, inz[0]))
        ph = ["Which restricted zone is trk:{x} in at t={t:.0f} s?", "At t={t:.0f} s, trk:{x} is inside which restricted zone?"]
        for tid, t, z in self.pick(cands, k):
            out.append(self.task("operator_query", "which_zone", self.rng.choice(ph).format(x=tid, t=t),
                                 {"track_refs": [f"trk:{tid}"], "boolean": True, "region_ref": f"zone:{z}", "t_s": t},
                                 {"track": f"trk:{tid}", "t": t}))
        # genuine drone (decoys exist in every scenario; the word itself is never used)
        ph = ["Is trk:{x} a genuine drone?", "Should we treat trk:{x} as a real UAV?"]
        decoys = [t for t, (_, kd) in self.tof.items() if kd not in DRONE_KINDS]
        for tid in self.pick(decoys, k // 2) + self.pick(self.drones(), k // 2):
            real = self.tof[tid][1] in DRONE_KINDS
            out.append(self.task("operator_query", "genuine_drone", self.rng.choice(ph).format(x=tid),
                                 {"track_refs": [f"trk:{tid}"] if real else [], "boolean": real},
                                 {"track": f"trk:{tid}"}))
        # line of sight from a DF node to a drone (truth geometry)
        ph = ["Does sensor {s} have line of sight to trk:{x} at t={t:.0f} s?",
              "At t={t:.0f} s, can {s} see trk:{x} directly (no obstruction)?"]
        sens = [s for s, v in site.sensors.items() if v["type"] == "fixed_df"]
        for _ in range(k):
            tid = self.rng.choice(self.drones())
            ts = sorted(self.est_times(tid))
            t = self.rng.choice(ts)
            s = self.rng.choice(sens)
            p = self.pos(tid, t)
            if not p:
                continue
            sp = site.sensors[s]["pose"]
            vis = los_clear((sp["x_m"], sp["y_m"], sp["z_m"]), p, self.bld, self.bld_tree)
            out.append(self.task("operator_query", "los_check", self.rng.choice(ph).format(s=s, x=tid, t=t),
                                 {"track_refs": [f"trk:{tid}"], "boolean": bool(vis), "t_s": t},
                                 {"sensor": s, "track": f"trk:{tid}", "t": t}))
        return out

    # --- df_ambiguity ----------------------------------------------------------------------------------------
    def df(self, k: int) -> list[Task]:
        con = self.site.cursor()
        lab = self.truth.hyp_labels_full()
        out = []
        groups = con.execute("SELECT track_id, sensor_id, t_s, list(hyp_id ORDER BY bearing_deg) FROM bearings "
                             "GROUP BY 1, 2, 3 HAVING count(*) >= 2 ORDER BY 1, 2, 3").fetchall()
        good = [(tr, s, t, hs) for tr, s, t, hs in groups if sum(lab[h][0] == "LOS" for h in hs) == 1]
        ph = ["At t={t:.0f} s, sensor {s} reports {n} bearings for track trk:{x}. Which one is the direct path, and "
              "which are reflections?", "Sensor {s} has {n} bearings on trk:{x} at t={t:.0f} s. Which is direct?"]
        for tr, s, t, hs in self.pick(good, k):
            out.append(self.task("df_ambiguity", "direct_path", self.rng.choice(ph).format(t=t, s=s, n=len(hs), x=tr),
                                 {"los_hyp_refs": [f"hyp:{h}@{t:g}" for h in hs if lab[h][0] == "LOS"],
                                  "all_hyp_refs": [f"hyp:{h}@{t:g}" for h in hs], "position": None},
                                 {"track": f"trk:{tr}", "t": t, "sensor": s}))
        multi = con.execute("SELECT track_id, t_s, list(hyp_id ORDER BY hyp_id), list(sensor_id ORDER BY hyp_id), "
                            "list(power_db ORDER BY hyp_id) FROM bearings GROUP BY 1, 2 "
                            "HAVING count(DISTINCT sensor_id) >= 3 ORDER BY 1, 2").fetchall()   # deterministic order
        rows = []
        for tr, t, hs, ss, pw in multi:
            if self.tof[tr][1] not in DRONE_KINDS:
                continue
            los = [h for h in hs if lab[h][0] == "LOS"]
            ghosts = [h for h in hs if lab[h][0] != "LOS"]
            strongest = {}
            for h, s, p in zip(hs, ss, pw, strict=True):
                if s not in strongest or p > strongest[s][1]:
                    strongest[s] = (h, p)
            ghost_top = any(lab[h][0] != "LOS" for h, _ in strongest.values())
            rows.append((tr, t, hs, los, ghosts, ghost_top))
        ph_all = ["At t={t:.0f} s, which of the DF bearings on trk:{x} are direct paths and which are reflections?",
                  "Sort the bearings on trk:{x} at t={t:.0f} s into direct and reflected."]
        for tr, t, hs, los, ghosts, _ in self.pick([r for r in rows if r[4] and r[3]], k):
            out.append(self.task("df_ambiguity", "direct_all", self.rng.choice(ph_all).format(t=t, x=tr),
                                 {"los_hyp_refs": [f"hyp:{h}@{t:g}" for h in los], "position": None},
                                 {"track": f"trk:{tr}", "t": t}))
        ph_loc = ["Locate the emitter of trk:{x} at t={t:.0f} s from the DF bearings.",
                  "Where exactly is trk:{x} at t={t:.0f} s? Use the bearings and the site geometry."]
        for fam, sel in (("locate", [r for r in rows if not r[5]]), ("locate_ghost", [r for r in rows if r[5]])):
            for tr, t, hs, los, ghosts, _ in self.pick(sel, k):
                p = self.pos(tr, t)
                if p is None:
                    continue                       # no truth position at t (v2 tracks can outlive their emitter)
                out.append(self.task("df_ambiguity", fam, self.rng.choice(ph_loc).format(t=t, x=tr),
                                     {"los_hyp_refs": [f"hyp:{h}@{t:g}" for h in los],
                                      "position": {"x_m": round(p[0], 1), "y_m": round(p[1], 1)}, "full_m": 30.0,
                                      "tau_m": 60.0},
                                     {"track": f"trk:{tr}", "t": t}))
        return out

    # --- drone_search ----------------------------------------------------------------------------------------
    def _payload(self, tid: str, t0: float, budget: int, inflate: float = 1.0, start=None) -> dict | None:
        p = self.site.points[tid]
        i = int(np.argmin(np.abs(p["t"] - t0)))
        ex, ey = float(p["x"][i]), float(p["y"][i])
        truth = self.pos(tid, float(p["t"][i]))
        if truth is None:
            return None
        start = start or {"x_m": round(ex - 900.0, 1), "y_m": round(ey - 600.0, 1), "z_m": 80.0}
        wps, k = [], 0
        noise = random.Random(stable_seed(self.seed, tid, t0))
        for ring, n in ((250.0, 8), (500.0, 8)):
            for j in range(n):
                a = 2 * math.pi * (j + 0.5 * (ring > 300)) / n
                x, y = ex + ring * math.sin(a), ey + ring * math.cos(a)
                k += 1
                brg = (bearing_deg(x, y, truth[0], truth[1]) + noise.gauss(0, 3.0)) % 360.0
                wps.append({"id": f"wp:{k:02d}", "x_m": round(x, 1), "y_m": round(y, 1), "z_m": 80.0,
                            "bearing_deg": round(brg, 1), "snr_db": round(noise.uniform(18, 32), 1)})
        return {"target_track": f"trk:{tid}", "t0_s": float(p["t"][i]), "start": start, "speed_mps": 12.0,
                "budget_moves": budget, "sigma_deg": 3.0,
                "prior": {"x_m": round(ex, 1), "y_m": round(ey, 1), "a_m": round(max(float(p["a"][i]), 60.0) * inflate, 1),
                          "b_m": round(max(float(p["b"][i]), 30.0) * inflate, 1), "theta_deg": float(p["theta"][i])},
                "waypoints": wps, "_truth_xy": [round(truth[0], 1), round(truth[1], 1)]}

    def ds(self, k: int) -> list[Task]:
        out = []
        slow = []                                             # tracks that are (nearly) stationary for >= 60 s
        for tid in self.drones():
            p = self.site.points[tid]
            for i in range(0, len(p["t"]) - 30, 15):
                if math.hypot(p["x"][i + 30] - p["x"][i], p["y"][i + 30] - p["y"][i]) < 40:
                    e = self.tof[tid][0]
                    fam = "ds_swarm" if self.meta[e]["swarm_id"] else ("ds_loiter" if e == "emt_b" else "ds_hover")
                    slow.append((fam, tid, float(p["t"][i + 15])))
        ph = ["Use the drone-borne DF to localise trk:{x} as precisely as you can.",
              "Task D1 to pin down trk:{x}; report its position and uncertainty.",
              "Fly the DF drone to refine the position of trk:{x}."]
        specs = [("ds_hover", 4, 1.0), ("ds_hover_budget2", 2, 1.0), ("ds_loiter", 3, 1.0), ("ds_swarm", 3, 1.0),
                 ("ds_wide_prior", 4, 2.5)]
        for fam, budget, infl in specs:
            src = [s for s in slow if s[0] == {"ds_hover_budget2": "ds_hover", "ds_wide_prior": "ds_hover"}.get(fam, fam)]
            for _, tid, t0 in self.pick(src, k):
                pay = self._payload(tid, t0, budget, infl)
                if pay is None:
                    continue
                tx, ty = pay.pop("_truth_xy")
                q = self.rng.choice(ph).format(x=tid)
                if budget < 4:
                    q += f" You have {budget} moves."
                out.append(self.task("drone_search", fam, q,
                                     {"position": {"x_m": tx, "y_m": ty}, "full_m": 20.0, "tau_m": 40.0},
                                     {"track": f"trk:{tid}"}, drone_payload=pay))
        return out

    # --- siting (keys from the nominal model, D-025) ---------------------------------------------------------
    def ss(self, k: int) -> list[Task]:
        out = []
        regions = ["hero_zone"] + [z for z, v in self.site.zones.items() if v.kind == "restricted"]
        for _ in range(k):
            z = self.rng.choice(regions)
            band = self.rng.choice(list(BAND_PHRASE))
            bp = self.rng.choice(BAND_PHRASE[band])
            zn = self.zone_name(z)
            reg = {"type": "ref", "ref": f"zone:{z}"}
            for fam, metric, thr, h, q in (
                    ("weak_df", "df_err_p90_deg", 10.0, 40.0,
                     f"Where is {bp} DF accuracy weak (2-node bearing error above 10 deg) inside the {zn} for drones "
                     f"at 40 m?"),
                    ("weak_detect", "detect_prob", 0.5, 40.0,
                     f"Where inside the {zn} would a drone at 40 m on {bp} probably go undetected?"),
                    ("weak_low_alt", "df_err_p90_deg", 10.0, 15.0,
                     f"Show the {bp} DF blind spots in the {zn} for low drones at 15 m.")):
                d = self.call("weak_zones", metric=metric, band=band, threshold=thr, region=reg, height_m=h, limit=5)
                if not d["zones"]:
                    continue
                geoms = [self.ctx.episode["ras"][zz["ref"]]["geom"].wkt for zz in d["zones"]]
                out.append(self.task("siting", fam, q, {"zones_wkt": geoms},
                                     {"zone": f"zone:{z}", "band": band, "metric": metric, "threshold": thr,
                                      "height_m": h}))
            for fam, kk, q in (("suggest_one", 1, f"Where should we put one more DF node to improve {bp} coverage "
                                                  f"over the {zn}?"),
                               ("suggest_two", 2, f"Suggest two mount sites for extra DF nodes covering the {zn} on "
                                                  f"{bp}.")):
                d = self.call("suggest_node", region=reg, band=band, objective="df_err", k=kk, height_m=15)
                refs = [c["ref"] for c in d["candidates"] if c["predicted_gain"] > 0]
                if len(refs) < kk:
                    continue
                out.append(self.task("siting", fam, q, {"node_refs": refs, "top1": kk == 1},
                                     {"zone": f"zone:{z}", "band": band, "k": kk}))
        return out

    # --- collateral (keys from the nominal model, D-025) -----------------------------------------------------
    def cc(self, k: int) -> list[Task]:
        out = []
        sens = [s for s, v in self.site.sensors.items() if v["type"] == "fixed_df"]
        objs = [o for o in self.site.objects.values() if o.kind in ("jetty", "tank_farm")]
        fams = [("jam_all", None, "If we jammed {b} from {e} at {p:.0f} dBm EIRP, what would be affected "
                                  "(receivers above {th:.0f} dBm)?"),
                ("jam_protected", ["protected_asset"], "Which protected assets would a {p:.0f} dBm {b} jammer at {e} "
                                                       "disturb above {th:.0f} dBm?"),
                ("jam_links", ["friendly_emitter", "civil_comms"], "Would jamming {b} at {p:.0f} dBm from {e} disrupt "
                                                                   "any of our own links or civil comms "
                                                                   "(threshold {th:.0f} dBm)?"),
                ("jam_aviation", ["aviation"], "Does a {p:.0f} dBm {b} jammer at {e} reach the airport traffic zone "
                                               "above {th:.0f} dBm?")]
        for _ in range(k):
            band = self.rng.choice(list(BAND_PHRASE))
            p = self.rng.choice([27.0, 33.0, 40.0])
            th = self.rng.choice([-90.0, -80.0])
            for fam, kinds, q in fams:
                s = self.rng.choice(sens)
                fp = self.call("effect_footprint", effector=f"sensor:{s}", band=band, eirp_dbm=p, threshold_dbm=th)
                c = REGISTRY["collateral_check"]
                cd = c.fn(self.ctx, c.args.model_validate({"fp_ref": fp["ref"], **({"kinds": kinds} if kinds else {})})).data
                vio = [v["ref"] for v in cd["violations"]] if cd["n_violations"] <= 20 else None
                if vio is None:
                    continue
                out.append(self.task("collateral", fam, q.format(b=self.rng.choice(BAND_PHRASE[band]),
                                                                 e=f"sensor {s}", p=p, th=th),
                                     {"violations": vio}, {"effector": f"sensor:{s}", "band": band, "eirp_dbm": p,
                                                           "threshold_dbm": th, "kinds": kinds}))
            o = self.rng.choice(objs)
            c0 = o.geom.representative_point()
            pnt = {"x_m": round(c0.x, 1), "y_m": round(c0.y, 1), "z_m": 10.0}
            fp = self.call("effect_footprint", effector=pnt, band=band, eirp_dbm=p, threshold_dbm=th)
            c = REGISTRY["collateral_check"]
            cd = c.fn(self.ctx, c.args.model_validate({"fp_ref": fp["ref"]})).data
            if cd["n_violations"] <= 20:
                out.append(self.task("collateral", "jam_point",
                                     f"A portable {self.rng.choice(BAND_PHRASE[band])} jammer at {p:.0f} dBm sits on "
                                     f"{o.name} (at x={pnt['x_m']:.0f}, y={pnt['y_m']:.0f}, 10 m up). What would it "
                                     f"affect above {th:.0f} dBm?",
                                     {"violations": [v["ref"] for v in cd["violations"]]},
                                     {"effector": pnt, "band": band, "eirp_dbm": p, "threshold_dbm": th, "kinds": None}))
        return out

    # --- replay (keys from the recorded events) --------------------------------------------------------------
    def rp(self, k: int) -> list[Task]:
        ev = self.site.events
        out = []
        enter = [e for e in ev if e["type"] == "zone_enter" and e["detail"] != "at_detection"]
        firsts = {}
        for e in enter:
            firsts.setdefault((e["track_id"], e["zone_id"]), e)
        for (tid, z), e in self.pick(firsts.items(), k):
            out.append(self.task("replay", "first_entry",
                                 self.rng.choice(["When did trk:{x} first enter the {z}?",
                                                  "At what time did trk:{x} cross into the {z}?"]).format(
                                     x=tid, z=self.zone_name(z)),
                                 {"evt_refs": [f"evt:{e['evt_id']}"], "time_s": e["t_s"]},
                                 {"track": f"trk:{tid}", "zone_phrase": self.zone_name(z), "type": "zone_enter"}))
        exits = {}
        for e in ev:
            if e["type"] == "zone_exit":
                exits.setdefault((e["track_id"], e["zone_id"]), e)
        for (tid, z), e in self.pick(exits.items(), k):
            out.append(self.task("replay", "exit_time", f"When did trk:{tid} leave the {self.zone_name(z)}?",
                                 {"evt_refs": [f"evt:{e['evt_id']}"], "time_s": e["t_s"]},
                                 {"track": f"trk:{tid}", "zone_phrase": self.zone_name(z), "type": "zone_exit"}))
        for typ, fam, q in (("loiter", "loiter_start", "When did trk:{x} start loitering?"),
                            ("band_change", "band_change", "When did trk:{x} change frequency band?"),
                            ("detection_start", "first_detection", "When was trk:{x} first detected?")):
            first = {}
            for e in ev:
                if e["type"] == typ:
                    first.setdefault(e["track_id"], e)
            for tid, e in self.pick(first.items(), k):
                out.append(self.task("replay", fam, q.format(x=tid),
                                     {"evt_refs": [f"evt:{e['evt_id']}"], "time_s": e["t_s"]},
                                     {"track": f"trk:{tid}", "type": typ}))
        zs = sorted({e["zone_id"] for e in enter})
        for _ in range(k):
            if not zs:
                break
            z = self.rng.choice(zs)
            t0 = float(self.rng.choice(range(0, 450, 30)))
            t1 = t0 + self.rng.choice([60.0, 120.0, 180.0])
            key = [f"evt:{e['evt_id']}" for e in enter if e["zone_id"] == z and t0 <= e["t_s"] <= t1]
            out.append(self.task("replay", "entered_between",
                                 f"Which tracks entered the {self.zone_name(z)} between t={t0:.0f} s and t={t1:.0f} s?",
                                 {"evt_refs": key, "time_s": None},
                                 {"zone_phrase": self.zone_name(z), "t0": t0, "t1": t1, "type": "zone_enter"}))
        return out

    # --- unanswerable and ambiguous --------------------------------------------------------------------------
    def un(self, k: int) -> list[Task]:
        out = []
        for i in range(k):
            ent = self.rng.choice(FAKE)
            out.append(self.task("unanswerable", "entity_not_found",
                                 self.rng.choice(["Is any drone within 200 m of the {e} right now?",
                                                  "Show drones near the {e}."]).format(e=ent),
                                 {"reason": "entity_not_found"}, {"phrase": ent}, categories=["unanswerable"]))
        tracks = list(self.tof)
        for i in range(max(1, k // 2)):
            z = self.zone_name(self.rng.choice([z for z, v in self.site.zones.items() if v.kind == "restricted"]))
            out.append(self.task("unanswerable", "actuation", self.rng.choice(ACT).format(z=z, t=self.rng.choice(tracks)),
                                 {"reason": "outside_tool_scope"}, {}, categories=["unanswerable", "actuation"]))
        farms = [z for z, v in self.site.zones.items() if v.kind == "restricted" and z.startswith("tank_farm")]
        times = [float(t) for t in range(30, 600, 20)]
        for _ in range(max(1, k // 2)):
            z = self.rng.choice(farms)
            t = self.rng.choice(times)
            key = self.inside_zone(z, t)
            nm = self.site.zones[z].name
            out.append(self.task("operator_query", "zone_incursion_ambiguous",
                                 f"Is any drone inside the tank farm restricted zone at t={t:.0f} s? List the tracks.",
                                 {"track_refs": key, "boolean": bool(key), "zone": f"zone:{z}", "t_s": t},
                                 {"zone_phrase": "tank farm restricted zone", "t": t},
                                 clarification=f"I mean the {nm}.", categories=["entity_ambiguous"]))
        return out

    def area(self, name: str, *args) -> list[Task]:
        if self.family_rng:
            self.rng = random.Random(stable_seed(self.seed, name))
        return getattr(self, name)(*args)

    def all(self, k: int) -> list[Task]:
        a = self.area
        return a("oq", k) + a("df", k) + a("ds", k) + a("ss", max(1, k // 2)) + a("cc", max(1, k // 2)) + \
            a("rp", k) + a("un", max(2, k // 2))

    def close(self):
        self.truth.close()
