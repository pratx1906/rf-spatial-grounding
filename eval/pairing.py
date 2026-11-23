"""Paired N-vs-M analysis of the hardware-mismatch sets (plan 2026-10-28 T7/T8, audit M1). Offline, keys only.

    python -m eval.pairing runs/paper/pilot/expert_Npair.json runs/paper/pilot/expert_M2.json [--thr 0.8]

Pairs rows of two eval/run.py outputs on `pair_id` (written by scripts/make_tasks.py --tag; the same question, params
and template in both worlds). Per pair:
- `ok_n` / `ok_m`: score >= thr (the task's own full score);
- `part_n` / `part_m`: the comparable part only (boolean / count / track, `pair_part`), from the dumped answer;
- twin-induced failure: ok on N and not ok on M (also reported on the part).
DF families are unpaired: per-family mean scores of both sets side by side.

Paper Revision 2 / 3.1 additions (new output keys; the old ones are unchanged):
- `world_key` (AUDIT_CODE C3; on by default in the CLI, --no-world-key to skip): the comparable part re-keyed on the
  TRUE emitters (every drone emitter of the world at t, tracked or not; offline truth via tasks/truth_access only).
  The task keys count only drones the tracker followed at t, so a drone that the mismatch made the tracker lose turns
  the M key to "no", and a "no" answer is scored correct. Reported: part_ok / part_twin_fail under the world key next
  to the tracker key, and how often the two keys disagree (key_disagree_n / _m).
- `act_abstain` (AgentAbstain-style paired accuracy, arXiv 2607.10059): a pair counts as correct only if the agent
  answers the nominal (answerable, consistent) twin correctly (score >= thr, not abstained / flagged) AND abstains or
  flags (requires_human_confirm) on the mismatch twin. Meaningful on wrong-twin worlds M1-M9 (`m_twin_world`); on the
  M0 control it measures false flags. Cluster bootstrap CI over scenario seeds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIR_OBJ_KINDS = ("tank_farm", "jetty", "substation")     # scripts/make_tasks.paired_params objects


def _base(ref: str) -> str:
    return ref.split("@", 1)[0]


def part_ok(task: dict, answer: dict | None) -> bool:
    """The comparable part of a paired answer: the boolean, the track count, or the single track."""
    if not answer or answer.get("intent") != "operator_query":
        return False
    key, part = task["answer_key"], task["pair_part"]
    got = sorted({_base(r) for r in answer.get("track_refs") or []})
    if part == "boolean":
        b = answer.get("boolean")
        b = bool(got) if b is None else b
        return bool(b) == bool(key["boolean"])
    if part == "count":
        return len(got) == len(key["track_refs"])
    if part == "track":
        return got[:1] == sorted({_base(r) for r in key["track_refs"]})[:1]
    raise ValueError(part)


def flagged(answer: dict | None) -> bool:
    """Abstained, or flagged for a human (requires_human_confirm; collateral answers always carry it, so not those)."""
    if not answer:
        return False
    return answer.get("intent") == "abstain" or (bool(answer.get("requires_human_confirm"))
                                                 and answer.get("intent") not in (None, "collateral"))


def _seed_of_pair(pid: str) -> str:
    return pid.split("-", 1)[0].lstrip("s")


# --- C3: the true-emitter ("world") key of a paired task (offline truth only) ------------------------------------
class WorldKeys:
    """World keys of paired tasks from each world's truth (tasks/truth_access.TruthStore) and site geometry, with the
    rules of scripts/make_tasks.paired_task applied to every true drone emitter instead of the tracked ones.
    Offline only: never imported by the agent stack."""

    def __init__(self, repo: Path = REPO):
        self.repo = repo
        self._site: dict = {}

    def _load_site(self, site: str) -> dict:
        if site not in self._site:
            from sim.belief.site_view import SiteView
            from tasks.kandla.gen import DRONE_KINDS
            from tasks.truth_access import TruthStore

            d = self.repo / site
            sv, ts = SiteView(d), TruthStore(d)
            try:
                meta = {e: m for e, m in ts.emitter_meta().items() if m["kind"] in DRONE_KINDS}
                tof = {f"trk:{t}": e for t, (e, _k) in ts.track_of().items()}
                pos: dict = {}
                for e, t, x, y in ts.con.execute("SELECT emitter_id, t_s, x_m, y_m FROM emitter_points").fetchall():
                    if e in meta:
                        pos.setdefault(round(float(t), 3), []).append((e, float(x), float(y)))
            finally:
                ts.close()
            zones: dict = {}
            for zid, z in sv.zones.items():
                for nm in {z.name, *z.aliases[:1]}:
                    zones.setdefault(nm, set()).add(zid)
            objs: dict = {}
            for oid, o in sv.objects.items():
                if o.kind in PAIR_OBJ_KINDS:
                    objs.setdefault(o.name, set()).add(oid)
            self._site[site] = {"sv": sv, "meta": meta, "tof": tof, "pos": pos, "zones": zones, "objs": objs}
        return self._site[site]

    def track_emitter(self, site: str, ref: str) -> str | None:
        return self._load_site(site)["tof"].get(_base(ref))

    def key(self, task: dict) -> dict | None:
        """{emitters: sorted true drone emitters satisfying the question at t, boolean, count, nearest}, or None when
        the slots cannot be resolved (an ambiguous zone or object name)."""
        from shapely.geometry import Point

        from scripts.make_tasks import _band_at

        S = self._load_site(task["site"])
        sv, fam, p, k = S["sv"], task["template"], task["params"], task["answer_key"]
        t = float(p["t"])
        pts = S["pos"].get(round(t, 3), [])
        zone = obj = None
        if "zone_phrase" in p:
            zid = (k.get("zone") or "").split(":", 1)[-1] or None
            if zid is None:
                cand = S["zones"].get(p["zone_phrase"], set())
                zid = next(iter(cand)) if len(cand) == 1 else None
            if zid is None or zid not in sv.zones:
                return None
            zone = sv.zones[zid].geom
        if "obj_phrase" in p:
            cand = S["objs"].get(p["obj_phrase"], set())
            if len(cand) != 1:
                return None
            obj = sv.objects[next(iter(cand))].geom
        if fam in ("zone_incursion", "count_in_zone"):
            em = [e for e, x, y in pts if zone.contains(Point(x, y))]
        elif fam == "band_in_zone":
            em = [e for e, x, y in pts if zone.contains(Point(x, y)) and _band_at(S["meta"][e], t) == p["band"]]
        elif fam == "swarm_in_zone":
            em = [e for e, x, y in pts if zone.contains(Point(x, y)) and S["meta"][e]["swarm_id"]]
        elif fam == "near_object":
            em = [e for e, x, y in pts if obj.distance(Point(x, y)) <= float(p["radius_m"])]
        elif fam == "nearest_drone":
            ds = sorted((obj.distance(Point(x, y)), e) for e, x, y in pts)
            em = [ds[0][1]] if ds else []
        else:
            return None
        em = sorted(set(em))
        return {"emitters": em, "boolean": bool(em), "count": len(em), "nearest": em[0] if em else None}


def part_ok_world(task: dict, answer: dict | None, wk: dict, track_emitter) -> bool:
    """The comparable part of a paired answer judged against the world key (true emitters)."""
    if not answer or answer.get("intent") != "operator_query":
        return False
    part = task["pair_part"]
    got = sorted({_base(r) for r in answer.get("track_refs") or []})
    if part == "boolean":
        b = answer.get("boolean")
        b = bool(got) if b is None else b
        return bool(b) == wk["boolean"]
    if part == "count":
        return len(got) == wk["count"]
    if part == "track":
        return bool(got) and wk["nearest"] is not None and track_emitter(task["site"], got[0]) == wk["nearest"]
    raise ValueError(part)


def key_agrees(task: dict, wk: dict, track_emitter) -> bool:
    """Does the task's (tracker) key give the same comparable part as the world key?"""
    k = task["answer_key"]
    return part_ok_world(task, {"intent": "operator_query", "track_refs": k.get("track_refs") or [],
                                "boolean": k.get("boolean")}, wk, track_emitter)


def _load(path: str | Path) -> tuple[dict, dict]:
    res = json.loads(Path(path).read_text())
    tf = REPO / res["set"]
    rows = [json.loads(x) for x in tf.read_text().splitlines() if x.strip()]
    return res, {t["task_id"]: t for t in rows}


def paired_report(n_res: dict, n_tasks: dict, m_res: dict, m_tasks: dict, thr: float = 0.8,
                  world: WorldKeys | None = None) -> dict:
    def by_pair(res, tasks):
        out = {}
        for r in res["rows"]:
            t = tasks[r["task_id"]]
            if t.get("pair_id") and r.get("sample", 0) == 0:
                out[t["pair_id"]] = (r, t)
        return out

    pn, pm = by_pair(n_res, n_tasks), by_pair(m_res, m_tasks)
    fam = defaultdict(lambda: defaultdict(int))
    aa_cl: dict = defaultdict(list)
    for pid in sorted(set(pn) & set(pm)):
        (rn, tn), (rm, tm) = pn[pid], pm[pid]
        f = fam[tn["template"]]
        okn, okm = rn["score"] >= thr, rm["score"] >= thr
        pa, pb = part_ok(tn, rn.get("answer")), part_ok(tm, rm.get("answer"))
        f["n"] += 1
        f["ok_n"] += okn
        f["ok_m"] += okm
        f["twin_fail"] += okn and not okm
        f["twin_fix"] += okm and not okn
        f["part_ok_n"] += pa
        f["part_ok_m"] += pb
        f["part_twin_fail"] += pa and not pb
        # act / abstain pair (AgentAbstain): answer N correctly, abstain or flag on M
        act, flg = okn and not flagged(rn.get("answer")), flagged(rm.get("answer"))
        f["act_ok_n"] += act
        f["flag_m"] += flg
        f["act_abstain_ok"] += act and flg
        aa_cl[_seed_of_pair(pid)].append(float(act and flg))
        if world is not None:
            wn, wm = world.key(tn), world.key(tm)
            if wn is None or wm is None:
                f["world_unresolved"] += 1
                continue
            te = world.track_emitter
            wa, wb = part_ok_world(tn, rn.get("answer"), wn, te), part_ok_world(tm, rm.get("answer"), wm, te)
            f["world_n"] += 1
            f["world_part_ok_n"] += wa
            f["world_part_ok_m"] += wb
            f["world_part_twin_fail"] += wa and not wb
            f["key_disagree_n"] += not key_agrees(tn, wn, te)
            f["key_disagree_m"] += not key_agrees(tm, wm, te)
    tot = defaultdict(int)
    for f in fam.values():
        for k, v in f.items():
            tot[k] += v

    def mean_by_family(res, tasks):
        acc = defaultdict(list)
        for r in res["rows"]:
            if not tasks[r["task_id"]].get("pair_id"):
                acc[r["template"]].append(r["score"])
        return {k: round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}

    un_n, un_m = mean_by_family(n_res, n_tasks), mean_by_family(m_res, m_tasks)
    rep = {"thr": thr, "pairs": tot["n"], "unmatched_pairs": len(set(pn) ^ set(pm)),
           "twin_fail": tot["twin_fail"], "twin_fail_rate": round(tot["twin_fail"] / tot["n"], 4) if tot["n"] else None,
           "part_twin_fail": tot["part_twin_fail"], "totals": dict(tot),
           "per_family": {k: dict(v) for k, v in sorted(fam.items())},
           "unpaired_mean_score": {k: {"N": un_n.get(k), "M": un_m.get(k)} for k in sorted(set(un_n) | set(un_m))}}
    m_world = sorted({str(t.get("world_variant") or "") for t in m_tasks.values() if t.get("pair_id")})
    aa = {"n": tot["n"], "act_ok_n": tot["act_ok_n"], "flag_m": tot["flag_m"], "pairs_ok": tot["act_abstain_ok"],
          "rate": round(tot["act_abstain_ok"] / tot["n"], 4) if tot["n"] else None,
          "m_world": m_world, "m_twin_world": any(re.fullmatch(r"M[1-9]", w) for w in m_world)}
    if tot["n"]:
        from eval.run import cluster_bootstrap

        cb = cluster_bootstrap(aa_cl)
        aa["ci95_cluster"], aa["n_clusters"] = cb["ci95"], cb["n_clusters"]
    rep["act_abstain"] = aa
    if world is not None:
        wn = tot["world_n"]
        rep["world_key"] = {
            "pairs": wn, "unresolved": tot["world_unresolved"],
            "part_ok_n": tot["world_part_ok_n"], "part_ok_m": tot["world_part_ok_m"],
            "part_twin_fail": tot["world_part_twin_fail"],
            "part_twin_fail_rate": round(tot["world_part_twin_fail"] / wn, 4) if wn else None,
            "key_disagree_n": tot["key_disagree_n"], "key_disagree_m": tot["key_disagree_m"],
            "tracker_key": {"part_twin_fail": tot["part_twin_fail"],
                            "part_twin_fail_rate": round(tot["part_twin_fail"] / tot["n"], 4) if tot["n"] else None}}
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nominal")
    ap.add_argument("mismatch")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-world-key", action="store_true", help="skip the true-emitter key (C3; reads truth offline)")
    a = ap.parse_args(argv)
    (nr, nt), (mr, mt) = _load(a.nominal), _load(a.mismatch)
    try:
        body = paired_report(nr, nt, mr, mt, a.thr, world=None if a.no_world_key else WorldKeys())
    except Exception as e:  # noqa: BLE001  the world key is an add-on: never lose the tracker-key report to it
        body = {**paired_report(nr, nt, mr, mt, a.thr), "world_key": {"error": f"{type(e).__name__}: {e}"}}
    rep = {"nominal": a.nominal, "mismatch": a.mismatch, **body}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
