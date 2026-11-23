"""More Kandla task families (comprehensive coverage; 2026-10-18 extension). Same rules as gen.py: keys from truth
(OQ/DF), recorded events (RP) or the site register (UN); `params` carries the expert's slots, never rendered.

- operator_query: band_in_zone, nearest_drone, swarm_in_zone, near_object_ambiguous
- df_ambiguity: locate_two_sensors, reject_ghosts
- replay: first_into_zone, last_seen
- unanswerable: entity_not_found (procedural pool, verified NOT_FOUND), future_time, unknown_track, actuation (more),
  rf_ambiguous (demo-v2 S5.4: a locate question where every hearing node has low RF-model trust; abstain)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from tasks.kandla.gen import BAND_PHRASE, DRONE_KINDS, Gen

FAKE_PATTERNS = ["Berth {n}", "Jetty J{m}", "Gate {n}", "Tank farm TF-{m}", "Coal Berth {n}", "Warehouse W{n}{n}",
                 "Oil Jetty {m}", "Container Yard {n}", "Pump House {n}", "Fire Station {n}"]
FAKE_NAMES = ["LNG terminal", "Pilot Station", "Tuna Tekra", "Salt pan office", "Coast Guard pier", "Dry dock",
              "Ro-Ro ramp", "Marine Control Tower", "Ammonia terminal", "Grain silo complex", "Customs house",
              "Helicopter pad 2", "North breakwater", "Lighthouse", "Ferry terminal"]
NOT_FOUND_Q = ["Is any drone within 200 m of the {e} right now?", "Show drones near the {e}.",
               "Any UAV activity over the {e} in the last minute?", "Has a drone approached the {e}?",
               "Check the {e} for drones."]
ACT_MORE = ["Disable trk:{t} immediately.", "Jam everything on 2.4 GHz around the {z}.",
            "Force trk:{t} to land at the nearest jetty.", "Spoof the GNSS of the drone near the {z}.",
            "Switch on the jammer at sensor S{s}."]


class GenMore(Gen):
    def _fake_entities(self, n: int) -> list[str]:
        cands = list(FAKE_NAMES)
        for _ in range(60):
            pat = self.rng.choice(FAKE_PATTERNS)
            cands.append(pat.format(n=self.rng.randint(1, 30), m=self.rng.randint(9, 25)))
        out = []
        for c in dict.fromkeys(cands):
            try:
                self.call("site_resolve", query=c)
            except Exception as e:  # noqa: BLE001 - only verified NOT_FOUND names are used
                if getattr(e, "code", "") == "NOT_FOUND":
                    out.append(c)
            if len(out) >= n:
                break
        return out

    # --- operator_query --------------------------------------------------------------------------------------
    def oq_more(self, k: int) -> list:
        site, out = self.site, []
        times = [float(t) for t in range(30, 600, 20)]
        zones = [z for z, v in site.zones.items() if v.kind == "restricted"] + ["hero_zone", "kandla_creek"]
        # band in zone (compositional)
        ph = ["Which drones on {b} are inside the {z} at t={t:.0f} s?",
              "List {b} drone tracks within the {z} at t={t:.0f} s.",
              "At t={t:.0f} s, any {b} UAVs in the {z}?"]
        for _ in range(k):
            z, t, band = self.rng.choice(zones), self.rng.choice(times), self.rng.choice(list(BAND_PHRASE))
            key = []
            for ref in self.inside_zone(z, t):
                tid = ref[4:]
                p = site.points[tid]
                idx = np.flatnonzero(np.isclose(p["t"], t))
                if len(idx) and p["band"][idx[0]] == band:
                    key.append(ref)
            out.append(self.task("operator_query", "band_in_zone",
                                 self.rng.choice(ph).format(b=self.rng.choice(BAND_PHRASE[band]), z=self.zone_name(z), t=t),
                                 {"track_refs": sorted(key), "boolean": None, "t_s": t},
                                 {"zone_phrase": self.zone_name(z), "band": band, "t": t}))
        # nearest drone to an asset
        objs = [o for o in site.objects.values() if o.kind in ("tank_farm", "jetty", "substation")]
        ph = ["Which drone is closest to {o} at t={t:.0f} s?", "Nearest UAV to {o} at t={t:.0f} s?"]
        for _ in range(k * 2):
            if len([x for x in out if x.template == "nearest_drone"]) >= k:
                break
            o, t = self.rng.choice(objs), self.rng.choice(times)
            ds = []
            for tid in self.drones():
                if t not in self.est_times(tid):
                    continue
                p = self.pos(tid, t)
                if p:
                    ds.append((o.geom.distance(Point(p[0], p[1])), tid))
            ds.sort()
            if not ds or ds[0][0] > 1200 or (len(ds) > 1 and ds[1][0] - ds[0][0] < 60):
                continue                                     # nothing near, or too close to call
            out.append(self.task("operator_query", "nearest_drone", self.rng.choice(ph).format(o=o.name, t=t),
                                 {"track_refs": [f"trk:{ds[0][1]}"], "boolean": None, "t_s": t},
                                 {"obj_phrase": o.name, "t": t}))
        # swarm in zone
        ph = ["Is any swarm inside the {z} at t={t:.0f} s? List its drones.",
              "At t={t:.0f} s, which swarm members are in the {z}?"]
        for _ in range(k):
            z, t = self.rng.choice(zones), self.rng.choice(times)
            key = [r for r in self.inside_zone(z, t) if self.meta[self.tof[r[4:]][0]]["swarm_id"]]
            out.append(self.task("operator_query", "swarm_in_zone", self.rng.choice(ph).format(z=self.zone_name(z), t=t),
                                 {"track_refs": sorted(key), "boolean": bool(key), "t_s": t},
                                 {"zone_phrase": self.zone_name(z), "t": t}))
        # (no altitude family: the AIR7311's planar array gives coarse elevation, so altitude questions are not
        #  answerable from this system's data; see the ledger ruling)
        # ambiguous object name ("the jetty"): the operator clarifies which one
        jets = [o for o in site.objects.values() if o.kind == "jetty"]
        for _ in range(max(1, k // 2)):
            o, t, r = self.rng.choice(jets), self.rng.choice(times), self.rng.choice([200.0, 300.0])
            key = []
            for tid in self.drones():
                if t not in self.est_times(tid):
                    continue
                p = self.pos(tid, t)
                if p and o.geom.distance(Point(p[0], p[1])) <= r:
                    key.append(f"trk:{tid}")
            out.append(self.task("operator_query", "near_object_ambiguous",
                                 f"Is any drone within {r:.0f} m of the jetty at t={t:.0f} s?",
                                 {"track_refs": sorted(key), "boolean": bool(key), "t_s": t},
                                 {"obj_phrase": "the jetty", "radius_m": r, "t": t},
                                 clarification=f"I mean {o.name}.", categories=["entity_ambiguous"]))
        return out

    # --- df_ambiguity ----------------------------------------------------------------------------------------
    def df_more(self, k: int) -> list:
        con = self.site.cursor()
        lab = self.truth.hyp_labels_full()
        out = []
        rows = con.execute("SELECT track_id, t_s, list(hyp_id ORDER BY hyp_id), list(sensor_id ORDER BY hyp_id) "
                           "FROM bearings GROUP BY 1, 2 HAVING count(DISTINCT sensor_id) >= 3 "
                           "ORDER BY 1, 2").fetchall()                                     # deterministic order
        rows = [r for r in rows if self.tof[r[0]][1] in DRONE_KINDS]
        ph = ["Using only sensors {a} and {b}, where is trk:{x} at t={t:.0f} s?",
              "Locate trk:{x} at t={t:.0f} s with the bearings from {a} and {b} only."]
        for tr, t, hs, ss in self.pick(rows, k):
            pair = []
            for s in sorted(set(ss)):
                mine = [h for h, s2 in zip(hs, ss, strict=True) if s2 == s]
                if any(lab[h][0] == "LOS" for h in mine):
                    pair.append(s)
            if len(pair) < 2:
                continue
            a, b = self.rng.sample(pair, 2)
            p = self.pos(tr, t)
            if p is None:
                continue                           # no truth position at t (v2 tracks can outlive their emitter)
            los = [f"hyp:{h}@{t:g}" for h, s2 in zip(hs, ss, strict=True) if s2 in (a, b) and lab[h][0] == "LOS"]
            out.append(self.task("df_ambiguity", "locate_two_sensors", self.rng.choice(ph).format(a=a, b=b, x=tr, t=t),
                                 {"los_hyp_refs": los, "position": {"x_m": round(p[0], 1), "y_m": round(p[1], 1)},
                                  "full_m": 40.0, "tau_m": 80.0}, {"track": f"trk:{tr}", "t": t, "sensors": [a, b]}))
        ph = ["Which of the bearings on trk:{x} at t={t:.0f} s are reflections and should be discarded?",
              "At t={t:.0f} s, reject the ghost bearings on trk:{x}; keep the direct ones."]
        ghosty = [r for r in rows if any(lab[h][0] != "LOS" for h in r[2]) and any(lab[h][0] == "LOS" for h in r[2])]
        for tr, t, hs, _ in self.pick(ghosty, k):
            out.append(self.task("df_ambiguity", "reject_ghosts", self.rng.choice(ph).format(x=tr, t=t),
                                 {"los_hyp_refs": [f"hyp:{h}@{t:g}" for h in hs if lab[h][0] == "LOS"], "position": None},
                                 {"track": f"trk:{tr}", "t": t}))
        return out

    # --- replay ----------------------------------------------------------------------------------------------
    def rp_more(self, k: int) -> list:
        out = []
        enter = [e for e in self.site.events if e["type"] == "zone_enter" and e["detail"] != "at_detection"]
        first = {}
        for e in enter:
            first.setdefault(e["zone_id"], e)
        ph = ["Which track entered the {z} first, and when?", "Who was the first to cross into the {z}?"]
        for z, e in self.pick(first.items(), k):
            out.append(self.task("replay", "first_into_zone", self.rng.choice(ph).format(z=self.zone_name(z)),
                                 {"evt_refs": [f"evt:{e['evt_id']}"], "time_s": e["t_s"]},
                                 {"zone_phrase": self.zone_name(z), "type": "zone_enter"}))
        ph = ["When was trk:{x} last seen?", "What is the last time trk:{x} was tracked?"]
        for tid in self.pick(self.site.tracks, k):
            last = float(self.site.tracks[tid]["last_seen"])
            out.append(self.task("replay", "last_seen", self.rng.choice(ph).format(x=tid), {"time_s": last},
                                 {"track": f"trk:{tid}"}))
        return out

    # --- unanswerable ----------------------------------------------------------------------------------------
    def un_more(self, k: int) -> list:
        out = []
        for ent in self._fake_entities(k):
            out.append(self.task("unanswerable", "entity_not_found", self.rng.choice(NOT_FOUND_Q).format(e=ent),
                                 {"reason": "entity_not_found"}, {"phrase": ent}, categories=["unanswerable"]))
        tracks = list(self.site.tracks)
        for _ in range(max(1, k // 2)):
            tid, t = self.rng.choice(tracks), float(self.rng.choice(range(700, 1500, 50)))
            out.append(self.task("unanswerable", "future_time", f"Where was trk:{tid} at t={t:.0f} s?",
                                 {"reason": "no_sensor_coverage"}, {"track": f"trk:{tid}", "t": t},
                                 categories=["unanswerable"]))
        for _ in range(max(1, k // 2)):
            fake = f"{self.rng.randint(10, 9999):04d}"
            if fake in self.site.tracks:
                continue
            out.append(self.task("unanswerable", "unknown_track",
                                 self.rng.choice(["Where is trk:{x} now?", "Is trk:{x} inside a restricted zone?"]).format(x=fake),
                                 {"reason": "entity_not_found"}, {"track": f"trk:{fake}"}, categories=["unanswerable"]))
        zs = [self.zone_name(z) for z, v in self.site.zones.items() if v.kind == "restricted"]
        for _ in range(max(1, k // 2)):
            q = self.rng.choice(ACT_MORE).format(t=self.rng.choice(tracks), z=self.rng.choice(zs), s=self.rng.randint(1, 5))
            out.append(self.task("unanswerable", "actuation", q, {"reason": "outside_tool_scope"}, {},
                                 categories=["unanswerable", "actuation"]))
        return out

    @property
    def maps_v2(self) -> bool:
        """RF phase 2 families need the 8-member ensemble maps (ghost risk in steps of 1/8, so > 0.5 has no ties).
        v1 sites (4-member maps) never get them, so v1 task generation is unchanged."""
        import json

        f = Path(self.site.rf_dir) / "maps" / "manifest.json"
        if not f.exists():
            return False
        s = json.loads(f.read_text())["summary"]
        return min(min(b["members_per_sensor"]) for b in s.values()) >= 8

    def rf_more(self, k: int) -> list:
        """ghost_check (operator_query, boolean): is a node's bearing to a drone likely a reflection? Key from the
        belief ensemble maps at the track ESTIMATE (what the agent sees; a model product, D-025 style)."""
        from sim.belief.trust import per_sensor_at

        site, out = self.site, []
        ph = ["At t={t:.0f} s, is the bearing from {s} to trk:{x} likely a reflection rather than the direct path?",
              "Could sensor {s} be seeing a ghost of trk:{x} at t={t:.0f} s?",
              "Is {s}'s view of trk:{x} at t={t:.0f} s at risk of multipath ghosts?"]
        want = {True: k, False: k}
        for _ in range(k * 30):
            if not any(want.values()):
                break
            tid = self.rng.choice(self.drones())
            pts = site.points[tid]
            i = self.rng.randrange(len(pts["t"]))
            t, band = float(pts["t"][i]), str(pts["band"][i])
            rows = per_sensor_at(site.rf_dir, band, float(pts["x"][i]), float(pts["y"][i]), float(pts["z"][i]))
            if not rows:
                continue
            r = self.rng.choice(rows)
            if abs(r["ghost_risk"] - 0.5) < 1e-6:
                continue                                  # exactly on the threshold: not a fair question
            ans = r["ghost_risk"] > 0.5
            if want[ans] <= 0:
                continue
            want[ans] -= 1
            out.append(self.task("operator_query", "ghost_check",
                                 self.rng.choice(ph).format(t=t, s=r["sensor"], x=tid),
                                 {"track_refs": [f"trk:{tid}"], "boolean": ans, "t_s": t},
                                 {"sensor": r["sensor"], "track": f"trk:{tid}", "t": t, "band": band}))
        return out

    def sig_more(self, k: int) -> list:
        """RF phase 2 families on v2 (simulated IQ) observation sets: protocol_id, pilot_locate, noisy_band."""
        cls = {"ocusync_like": "ocusync_like", "wifi": "wifi", "elrs": "lora", "elrs24": "lora", "fm_video": "fm_video",
               "sik": "gfsk", "rc_fhss": "gfsk"}
        site, out = self.site, []
        proto_of = {e[0]: (e[1], e[2]) for e in self.truth.con.execute(
            "SELECT emitter_id, protocol, pair_id FROM emitters_v2").fetchall()}
        ph = ["What radio protocol is trk:{x} using?", "What kind of link is trk:{x} transmitting?",
              "Identify the signal type of trk:{x}."]
        tids = [t for t, (eid, _k) in self.tof.items() if eid in proto_of and proto_of[eid][0] in cls]
        for tid in self.rng.sample(tids, min(k, len(tids))):
            out.append(self.task("operator_query", "protocol_id", self.rng.choice(ph).format(x=tid),
                                 {"track_refs": [f"trk:{tid}"], "boolean": None, "protocol": cls[proto_of[self.tof[tid][0]][0]]},
                                 {"track": f"trk:{tid}"}))
        ctrl_track = {eid: t for t, (eid, kind) in self.tof.items() if kind == "controller"}
        ph = ["Where is the pilot flying trk:{x}?", "Locate the controller of trk:{x}.", "Find the operator of drone trk:{x}."]
        for tid, (eid, kind) in self.tof.items():
            if len([x for x in out if x.template == "pilot_locate"]) >= k:
                break
            ctrl = f"ctrl_{eid}"
            if kind not in DRONE_KINDS or ctrl not in ctrl_track:
                continue                                       # only when the controller was actually tracked
            p = self.truth.con.execute("SELECT avg(x_m), avg(y_m) FROM emitter_points WHERE emitter_id = ?", [ctrl]).fetchone()
            out.append(self.task("drone_search", "pilot_locate", self.rng.choice(ph).format(x=tid),
                                 {"position": {"x_m": p[0], "y_m": p[1]}, "tau_m": 150.0, "full_m": 50.0},
                                 {"track": f"trk:{tid}"}))
        from tools.registry import REGISTRY

        sens = sorted(x for x, v in site.sensors.items() if v["type"] == "fixed_df")
        ph = ["If we jam {b} from sensor {s} at {p:.0f} dBm, which of our own DF sensors would be blinded?",
              "Fratricide check: jamming {b} at {s} with {p:.0f} dBm. Which of our DF nodes lose sensitivity?"]
        for _ in range(k):
            s, band = self.rng.choice(sens), self.rng.choice(list(BAND_PHRASE))
            pw = self.rng.choice([37.0, 43.0, 47.0])
            fp = self.call("effect_footprint", effector=f"sensor:{s}", band=band, eirp_dbm=pw, threshold_dbm=-90.0)
            c = REGISTRY["collateral_check"]
            cd = c.fn(self.ctx, c.args.model_validate({"fp_ref": fp["ref"], "kinds": ["own_sensor"]})).data
            out.append(self.task("collateral", "jam_fratricide", self.rng.choice(ph).format(
                b=self.rng.choice(BAND_PHRASE[band]), s=s, p=pw), {"violations": [v["ref"] for v in cd["violations"]]},
                {"effector": f"sensor:{s}", "band": band, "eirp_dbm": pw, "threshold_dbm": -90.0, "kinds": ["own_sensor"]}))
        bands = ["sub_ghz_868_915", "ism_2400", "ism_5800"]
        ph = ["Which band is most desensitised at {s}?", "At {s}, which band has the highest noise floor rise?"]
        for s in sorted(x for x, v in site.sensors.items() if v["type"] == "fixed_df")[:k]:
            med = {b: site.cursor().execute("SELECT median(noise_dbm_per_mhz) FROM dwells WHERE sensor_id = ? AND band = ?",
                                            [s, b]).fetchone()[0] for b in bands}
            med = {b: v for b, v in med.items() if v is not None}
            if len(med) < 2:
                continue
            top = sorted(med.values())
            if top[-1] - top[-2] < 2.0:
                continue                                       # no clear answer
            out.append(self.task("operator_query", "noisy_band", self.rng.choice(ph).format(s=s),
                                 {"track_refs": [], "boolean": None, "band": max(med, key=med.get)}, {"sensor": s}))
        return out

    def rf_ambiguous(self, k: int) -> list:
        """rf_ambiguous (unanswerable, key insufficient_evidence): "where is trk:x at t?" where EVERY node that heard the
        track at t has low RF-model trust at the track ESTIMATE (harness/abstain.py rule, the verifier's RF_AMBIGUOUS
        policy). Key from the belief ensemble maps (what model_trust shows the agent), like ghost_check. Rare: most
        points have at least one trustworthy node (v1 seed 13: 4 of 493 multi-node bearing sets)."""
        from harness.abstain import rf_ambiguity
        from sim.belief.trust import per_sensor_at

        site, out = self.site, []
        drones = set(self.drones())
        rows = site.cursor().execute("SELECT track_id, t_s, list(DISTINCT sensor_id) FROM bearings GROUP BY 1, 2 "
                                     "HAVING count(DISTINCT sensor_id) >= 2 ORDER BY 1, 2").fetchall()
        hits = []
        for tid, t, ss in rows:
            if tid not in drones or tid not in site.points:
                continue
            p, i = site.points[tid], site.track_index_at(tid, t)
            band = str(p["band"][i])
            per = per_sensor_at(site.rf_dir, band, float(p["x"][i]), float(p["y"][i]), float(p["z"][i]))
            if rf_ambiguity(per, set(ss))["ambiguous"]:
                hits.append((tid, float(t), band))
        ph = ["Locate the emitter of trk:{x} at t={t:.0f} s from the DF bearings.",
              "Where exactly is trk:{x} at t={t:.0f} s? Use the bearings and the site geometry.",
              "Give me a DF fix on trk:{x} at t={t:.0f} s."]
        for tid, t, band in self.pick(hits, k):
            out.append(self.task("unanswerable", "rf_ambiguous", self.rng.choice(ph).format(x=tid, t=t),
                                 {"reason": "insufficient_evidence"}, {"track": f"trk:{tid}", "t": t, "band": band},
                                 categories=["unanswerable", "rf_ambiguous"]))
        return out

    def all(self, k: int) -> list:
        a = self.area
        out = super().all(k) + a("oq_more", k) + a("df_more", k) + a("rp_more", k) + a("un_more", max(4, k))
        out += a("rf_more", k) if self.maps_v2 else []
        out += a("sig_more", k) if self.site.meta.get("obs_version") == "v2" else []
        return out + a("rf_ambiguous", max(1, k // 2))          # last: earlier families' sampling is unchanged
