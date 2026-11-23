"""Generate Kandla tasks for every scenario seed, run the scripted expert as a check, write split files.

    python scripts/make_tasks.py [--k 6] [--seeds 1-16] [--check 40]

Splits (SPEC 8.2): scenario seeds 1-12 and 17+ train, 13-14 val, 15-16 test, plus 51-60 test (fresh varied-cast test
seeds of the RF convergence plan, never used for training); held-out families (novel phrasing / new template, test
only): HELD_OUT. Demo tasks: seed 0 -> data/demo_tasks_kandla.json (2+ per area; a demo root: needs
--allow-demo-roots). Data versions v21/v3 (SCENESCOUT_DATA_VERSION, --obs v2 only): sets from the version's scenario
root, task ids tagged k<seed>X21-/k<seed>X3-, demo tasks to <out-dir>/demo.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from envbuild.variant_roots import add_allow_flag, data_version, guard_write, task_tag  # noqa: E402
from scripts.build_scenarios import parse_seeds, seed_dir  # noqa: E402
from tasks.kandla.gen_more import GenMore as Gen  # noqa: E402

HELD_OUT = {"near_object", "locate_ghost", "ds_wide_prior", "weak_low_alt", "jam_point", "loiter_start"}
COMPOSITIONAL: set[str] = set()     # compositional hold-out families (plan 2026-10-28; none yet)
NOT_FOR_SELECTION = HELD_OUT | COMPOSITIONAL       # never in train, val or any checkpoint-selection set (audit M9)


def selection_rows(rows: list[dict]) -> list[dict]:
    """The rows a train/val/selection set may hold: held-out and compositional families stripped (audit M9)."""
    return [r for r in rows if r["template"] not in NOT_FOR_SELECTION]


def write_clean_val(src: str | Path, dst: str | Path) -> dict:
    """Filter an existing val file into a clean selection set (no regeneration; never overwrites). Returns counts."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    if src.resolve() == dst.resolve():
        raise SystemExit("--clean-val: source and destination are the same file")
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    keep = selection_rows(rows)
    dst.write_text("".join(json.dumps(r) + "\n" for r in keep))
    dropped = defaultdict(int)
    for r in rows:
        if r["template"] in NOT_FOR_SELECTION:
            dropped[r["template"]] += 1
    return {"src": str(src), "dst": str(dst), "kept": len(keep), "dropped": dict(sorted(dropped.items()))}


TEST_SEEDS_V3 = range(51, 61)       # RF convergence plan P0-6: fresh test seeds (41-48 were v1 train seeds)
NEWAREA_SEEDS = list(range(61, 67)) + list(range(101, 121)) + list(range(201, 221))   # unseen-area tests (paper 2026-10-31), test only


def split_of(seed: int) -> str:
    if seed == 0:
        return "demo"
    if seed in TEST_SEEDS_V3 or seed in NEWAREA_SEEDS:
        return "test"
    return "train" if seed <= 12 or seed >= 17 else "val" if seed <= 14 else "test"


def demo_tasks_path(out: Path, version: str | None = None) -> Path:
    """The demo task file: data/demo_tasks_kandla.json for v2 (a demo root), <out-dir>/demo.json for v21/v3."""
    return REPO / "data" / "demo_tasks_kandla.json" if data_version(version) == "v2" else Path(out) / "demo.json"


VARIANT_AREAS = {"operator_query", "df_ambiguity", "drone_search", "replay"}   # scenario-dependent keys only


def variant_tasks(a) -> int:
    """World-variant test tasks for the paired world-gap evaluation. Never writes data/tasks/{train,val,test}.jsonl
    or the demo file; refuses seed 0 and non-test seeds; area-scoped RNG; ids tagged k<seed><W>-...; drops
    scenario-independent areas (siting/collateral/unanswerable keys come from nominal products, D-025) and any
    question text that appears in the training split."""
    from scripts.build_scenarios import variant_dir

    if not a.out:
        raise SystemExit("--world-variant needs --out")
    seeds = parse_seeds(a.seeds)
    bad = [s for s in seeds if s == 0 or split_of(s) != "test"]
    if bad:
        raise SystemExit(f"world-variant tasks are test-split only; refused seeds {bad}")
    train_q = set()
    tf = REPO / "data" / "tasks" / "train.jsonl"
    if tf.exists():
        with tf.open() as f:
            train_q = {json.loads(x)["question"] for x in f}
    rows, dropped = [], 0
    for seed in seeds:
        d = seed_dir(seed) if a.world_variant == "N" else variant_dir(seed, a.world_variant)
        if not d.exists():
            raise SystemExit(f"missing scenario {d}: build it with scripts/build_scenarios.py --world-variant")
        g = Gen(d, seed, "test", tag=a.world_variant, family_rng=True)
        ts = g.all(a.k)
        g.close()
        seen = set()
        for t in ts:
            if t.area not in VARIANT_AREAS or t.question in seen:
                continue
            seen.add(t.question)
            if t.question in train_q:
                dropped += 1
                continue
            r = t.to_row()
            r["world_variant"] = a.world_variant
            rows.append(r)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["area"]] += 1
    print(json.dumps({"variant": a.world_variant, "tasks": len(rows), "dropped_train_overlap": dropped,
                      "areas": dict(cnt), "out": str(out)}))
    return 0


# --- paired mismatch sets (plan 2026-10-28 T7, audit M1) -----------------------------------------------------------
MISMATCH_AREAS = {"operator_query", "df_ambiguity"}      # truth-keyed and ray-traced: the only M-sensitive areas
# track-ID-free operator_query families paired across worlds on (seed, template, params); scored on this part of the
# answer, which is comparable across worlds even when the observed track ids differ
PAIR_PART = {"zone_incursion": "boolean", "near_object": "boolean", "swarm_in_zone": "boolean",
             "count_in_zone": "count", "band_in_zone": "count", "nearest_drone": "track"}
PAIRED = tuple(PAIR_PART)
TRAIN_REF = {"v2": "x", "v21": "x21", "v3": "x3"}         # data/tasks/<dir>/train.jsonl of each data version


def _band_at(meta: dict, t: float) -> str | None:
    """An emitter's scheduled band at t (truth bands = [(from_t, band), ...])."""
    cur = None
    for t0, b in sorted(meta.get("bands") or [], key=lambda x: x[0]):
        if t0 <= t:
            cur = b
    return cur


def paired_params(site_dir: Path, seed: int, k: int) -> list[dict]:
    """World-independent slots of the paired families: drawn from the site (zones, objects) and the TRUE emitter
    positions only, never from observed tracks, with one RNG per family (stable_seed(seed, "pair", family)). The
    emitters are identical in every hardware-mismatch world of a seed, so N and every M get the same list."""
    from shapely.geometry import Point

    from sim.belief.site_view import SiteView
    from sim.geometry import stable_seed
    from tasks.kandla.gen import BAND_PHRASE, DRONE_KINDS
    from tasks.truth_access import TruthStore

    site, truth = SiteView(site_dir), TruthStore(site_dir)
    try:
        meta = {e: m for e, m in truth.emitter_meta().items() if m["kind"] in DRONE_KINDS}
        times = [float(t) for t in range(30, 600, 20)]
        xy = {}
        for t in times:
            for e in sorted(meta):
                p = truth.position(e, t)
                if p:
                    xy.setdefault(t, []).append((e, p[0], p[1]))
    finally:
        truth.close()
    zn = lambda z: site.zones[z].aliases[0] if site.zones[z].aliases else site.zones[z].name  # noqa: E731
    restricted = sorted(z for z, v in site.zones.items() if v.kind == "restricted")
    oq_zones = restricted + ["hero_zone", "kandla_creek", "kict"]       # Gen.oq
    more_zones = restricted + ["hero_zone", "kandla_creek"]             # GenMore.oq_more
    objs = [(oid, o) for oid, o in sorted(site.objects.items()) if o.kind in ("tank_farm", "jetty", "substation")]
    inside = lambda z, t: [e for e, x, y in xy.get(t, []) if site.zones[z].geom.contains(Point(x, y))]  # noqa: E731
    bands = list(BAND_PHRASE)
    out: list[dict] = []

    def rng(fam):
        return random.Random(stable_seed(seed, "pair", fam))

    def add(fam, slots, **qargs):
        out.append({"family": fam, "slots": slots, "q": qargs})

    def pick(r, items, n):
        return r.sample(items, min(n, len(items)))

    pos = [(z, t) for z in oq_zones for t in times if inside(z, t)]
    neg = [(z, t) for z in oq_zones for t in times if not inside(z, t)]
    r = rng("zone_incursion")
    for z, t in pick(r, pos, k) + pick(r, neg, max(1, k // 2)):
        add("zone_incursion", {"zone": z, "t": t}, ph=r.randrange(3), z=zn(z), t=t)
    r = rng("count_in_zone")
    for z, t in pick(r, pos, k):
        add("count_in_zone", {"zone": z, "t": t}, ph=r.randrange(2), z=zn(z), t=t)
    r = rng("near_object")
    for _ in range(k):
        (oid, o), t, rad = r.choice(objs), r.choice(times), r.choice([200.0, 300.0, 500.0])
        add("near_object", {"obj": oid, "t": t, "radius_m": rad}, ph=r.randrange(2), o=o.name, t=t, r=rad)
    r, n = rng("nearest_drone"), 0
    for _ in range(k * 2):
        if n >= k:
            break
        (oid, o), t = r.choice(objs), r.choice(times)
        ds = sorted(o.geom.distance(Point(x, y)) for _, x, y in xy.get(t, []))
        ph = r.randrange(2)
        if not ds or ds[0] > 1200 or (len(ds) > 1 and ds[1] - ds[0] < 60):
            continue                                     # nothing near, or too close to call (on the true emitters)
        n += 1
        add("nearest_drone", {"obj": oid, "t": t}, ph=ph, o=o.name, t=t)
    more_pos = [(z, t) for z in more_zones for t in times if inside(z, t)]
    r = rng("band_in_zone")                              # half where a true drone is inside, on its scheduled band
    for i in range(k):
        if i < (k + 1) // 2 and more_pos:
            z, t = r.choice(more_pos)
            band = _band_at(meta[r.choice(sorted(inside(z, t)))], t) or r.choice(bands)
        else:
            z, t, band = r.choice(more_zones), r.choice(times), r.choice(bands)
        add("band_in_zone", {"zone": z, "t": t, "band": band}, ph=r.randrange(3),
            b=r.choice(BAND_PHRASE[band]), z=zn(z), t=t)
    sw_pos = [(z, t) for z, t in more_pos if any(meta[e]["swarm_id"] for e in inside(z, t))]
    r = rng("swarm_in_zone")                             # half where a true swarm member is inside
    for i in range(k):
        z, t = r.choice(sw_pos) if i < (k + 1) // 2 and sw_pos else (r.choice(more_zones), r.choice(times))
        add("swarm_in_zone", {"zone": z, "t": t}, ph=r.randrange(2), z=zn(z), t=t)
    for i, p in enumerate(out):
        p["pair_id"] = f"s{seed}-{p['family']}-{sum(q['family'] == p['family'] for q in out[:i]):03d}"
    return out


PAIR_PH = {
    "zone_incursion": ["Is any drone inside the {z} at t={t:.0f} s? List the tracks.",
                       "At t={t:.0f} s, are there drones in the {z}? Which tracks?",
                       "Any UAV within the {z} at t={t:.0f} s? Give the track IDs."],
    "count_in_zone": ["How many drones are inside the {z} at t={t:.0f} s? List them.",
                      "Count the drone tracks in the {z} at t={t:.0f} s."],
    "near_object": ["Is any drone within {r:.0f} m of {o} at t={t:.0f} s?",
                    "Within {r:.0f} m of {o}, any drones at t={t:.0f} s?"],
    "nearest_drone": ["Which drone is closest to {o} at t={t:.0f} s?", "Nearest UAV to {o} at t={t:.0f} s?"],
    "band_in_zone": ["Which drones on {b} are inside the {z} at t={t:.0f} s?",
                     "List {b} drone tracks within the {z} at t={t:.0f} s.",
                     "At t={t:.0f} s, any {b} UAVs in the {z}?"],
    "swarm_in_zone": ["Is any swarm inside the {z} at t={t:.0f} s? List its drones.",
                      "At t={t:.0f} s, which swarm members are in the {z}?"]}


def paired_task(g, p: dict):
    """One paired task in the world of Gen `g`: the question text and params of `p`, the key from this world's truth
    and observed tracks with exactly the rule of the original family (Gen.oq / GenMore.oq_more)."""
    import numpy as np
    from shapely.geometry import Point

    fam, s, q = p["family"], p["slots"], dict(p["q"])
    text = PAIR_PH[fam][q.pop("ph")].format(**q)
    t = s["t"]
    obj = g.site.objects[s["obj"]] if "obj" in s else None

    def near(r):
        out = []
        for tid in g.drones():
            if t in g.est_times(tid):
                pp = g.pos(tid, t)
                if pp and obj.geom.distance(Point(pp[0], pp[1])) <= r:
                    out.append(f"trk:{tid}")
        return sorted(out)

    if fam in ("zone_incursion", "count_in_zone"):
        key = g.inside_zone(s["zone"], t)
        ans = {"track_refs": key, "boolean": bool(key) if fam == "zone_incursion" else None,
               "zone": f"zone:{s['zone']}", "t_s": t}
        params = {"zone_phrase": g.zone_name(s["zone"]), "t": t}
    elif fam == "near_object":
        key = near(s["radius_m"])
        ans, params = {"track_refs": key, "boolean": bool(key), "t_s": t}, \
            {"obj_phrase": obj.name, "radius_m": s["radius_m"], "t": t}
    elif fam == "nearest_drone":
        ds = []
        for tid in g.drones():
            if t in g.est_times(tid):
                pp = g.pos(tid, t)
                if pp:
                    ds.append((obj.geom.distance(Point(pp[0], pp[1])), tid))
        ds.sort()
        ans = {"track_refs": [f"trk:{ds[0][1]}"] if ds else [], "boolean": None, "t_s": t}
        params = {"obj_phrase": obj.name, "t": t}
    elif fam == "band_in_zone":
        key = []
        for ref in g.inside_zone(s["zone"], t):
            pts = g.site.points[ref[4:]]
            idx = np.flatnonzero(np.isclose(pts["t"], t))
            if len(idx) and pts["band"][idx[0]] == s["band"]:
                key.append(ref)
        ans = {"track_refs": sorted(key), "boolean": None, "t_s": t}
        params = {"zone_phrase": g.zone_name(s["zone"]), "band": s["band"], "t": t}
    elif fam == "swarm_in_zone":
        key = [r for r in g.inside_zone(s["zone"], t) if g.meta[g.tof[r[4:]][0]]["swarm_id"]]
        ans = {"track_refs": sorted(key), "boolean": bool(key), "t_s": t}
        params = {"zone_phrase": g.zone_name(s["zone"]), "t": t}
    else:
        raise ValueError(fam)
    return g.task("operator_query", fam, text, ans, params)


def mismatch_rows(seed: int, d: Path, tag: str, k: int, train_q: set[str],
                  paired_only: bool = False, split: str = "test") -> tuple[list[dict], list[dict]]:
    """(OQ + DF rows, drone_search rows) of one seed in world dir `d` (tag N = the nominal _X). Unpaired families as in
    the world-variant path (Gen with family_rng: one RNG per area method, so DF is drawn with the same RNG in every
    world); the six paired families come from `paired_params` only. Questions in `train_q` are dropped."""
    g = Gen(d, seed, split, tag=task_tag(None, tag), family_rng=True)
    try:
        ts = [] if paired_only else g.all(k)
        pairs = paired_params(d, seed, k)
        pt = [(paired_task(g, p), p) for p in pairs]
    finally:
        g.close()
    rows, ds, seen = [], [], set()
    for t, p in pt:
        if t.question in seen or t.question in train_q:
            continue                                     # text is world-independent: the same pairs drop everywhere
        seen.add(t.question)
        rows.append({**t.to_row(), "world_variant": tag, "pair_id": p["pair_id"], "pair_part": PAIR_PART[t.template]})
    for t in ts:
        if t.template in PAIR_PART or t.question in seen or t.question in train_q:
            continue
        if t.area not in MISMATCH_AREAS and t.area != "drone_search":
            continue
        seen.add(t.question)
        r = {**t.to_row(), "world_variant": tag, "pair_id": None}
        if t.area == "drone_search":
            r["synthetic_bearings"] = t.template.startswith("ds_")     # truth + N(0, 3 deg), not ray-traced (M1/M6)
            ds.append(r)
        else:
            rows.append(r)
    return rows, ds


def mismatch_tasks(a) -> int:
    """Paired hardware-mismatch test sets (T7): --obs v2 --tag M<k> reads kandla_s<N>_X_M<k>; --tag N the nominal
    kandla_s<N>_X (the paired nominal set). Writes <out-dir>/test.jsonl (OQ + DF) and test_ds.jsonl (drone_search,
    synthetic bearings flagged); refuses existing files, non-test seeds and missing or unfinished worlds."""
    import re

    from scripts.build_scenarios import v2_dir

    if a.world_variant:
        raise SystemExit("--tag (mismatch worlds) and --world-variant do not combine")
    if not re.fullmatch(r"N|M[0-9A-Za-z_]*", a.tag):
        raise SystemExit(f"--tag {a.tag!r}: N (nominal pair) or a mismatch tag M1..M4")
    if not a.out_dir:
        raise SystemExit("--tag needs --out-dir")
    seeds = parse_seeds(a.seeds)
    split = "train" if getattr(a, "train_pool", False) else "test"
    bad = [s for s in seeds if s == 0 or split_of(s) != split]
    if bad:
        raise SystemExit(f"mismatch task sets are {split}-split only here (--train-pool: train seeds); refused {bad}")
    out = Path(a.out_dir)
    files = [out / f"{split}.jsonl", out / f"{split}_ds.jsonl"]
    if any(f.exists() for f in files):
        raise SystemExit(f"{out}: task files exist; refusing to overwrite")
    dirs = {s: v2_dir(s, None if a.tag == "N" else a.tag) for s in seeds}
    for s, d in dirs.items():
        need = [d / "observed.duckdb", d / "meta.yaml"] + ([] if a.tag == "N" else [d / "truth" / "world_variant.yaml"])
        if not all(f.exists() for f in need):
            raise SystemExit(f"seed {s}: {d} missing or unfinished (needs {[f.name for f in need]})")
    tf = REPO / "data" / "tasks" / TRAIN_REF[data_version()] / "train.jsonl"
    train_q = set()
    if tf.exists() and split == "test":                  # train pools keep train questions
        with tf.open() as f:
            train_q = {json.loads(x)["question"] for x in f}
    rows, ds = [], []
    for s in seeds:
        r, d = mismatch_rows(s, dirs[s], a.tag, a.k, train_q, split=split)
        if split == "train":                             # held-out / compositional families never enter training
            r = [x for x in r if x["template"] not in NOT_FOR_SELECTION]
            d = [x for x in d if x["template"] not in NOT_FOR_SELECTION]
        rows += r
        ds += d
        print(f"seed {s}: {len(r)} OQ+DF ({sum(bool(x['pair_id']) for x in r)} paired), {len(d)} DS", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for f, rr in zip(files, (rows, ds), strict=True):
        f.write_text("".join(json.dumps(r) + "\n" for r in rr))
    fam = defaultdict(int)
    for r in rows + ds:
        fam[r["template"]] += 1
    print(json.dumps({"tag": a.tag, "seeds": a.seeds, "k": a.k, "tasks": len(rows), "paired": sum(bool(r["pair_id"])
                      for r in rows), "ds": len(ds), "families": dict(sorted(fam.items())), "out": str(out)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="samples per family per seed")
    ap.add_argument("--seeds", default="0-16")
    ap.add_argument("--world-variant", default=None,
                    help="RF living-world variant (D-031): W0/W1t/W1/W2/W3, or N for the nominal twin; test seeds only")
    ap.add_argument("--out", default=None, help="with --world-variant: the ONLY file written")
    ap.add_argument("--obs", choices=["v1", "v2"], default="v1",
                    help="v2: RF phase 2 sets (kandla_s<N>_X[_W]); all splits written to --out-dir only")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--clean-val", nargs=2, metavar=("SRC", "DST"), default=None,
                    help="only filter an existing val file into DST without held-out/compositional families, exit")
    ap.add_argument("--tag", default=None,
                    help="--obs v2 paired mismatch test sets (T7): M1..M4 reads kandla_s<N>_X_<tag>, N the nominal "
                         "_X; writes <out-dir>/test.jsonl (OQ + DF) and test_ds.jsonl only")
    ap.add_argument("--train-pool", action="store_true",
                    help="with --tag: mismatch tasks on TRAIN seeds for the RL pool (Revision 1 item 3); writes "
                         "<out-dir>/train.jsonl and train_ds.jsonl with split=train")
    add_allow_flag(ap)
    a = ap.parse_args(argv)
    if a.clean_val:
        print(json.dumps(write_clean_val(*a.clean_val)))
        return 0
    version = data_version()
    if version != "v2" and a.obs != "v2":
        raise SystemExit(f"data version {version} has v2-observation sets only (--obs v2)")
    if a.tag:
        if a.obs != "v2":
            raise SystemExit("--tag is an --obs v2 option")
        return mismatch_tasks(a)
    if a.world_variant and a.obs == "v1":
        return variant_tasks(a)
    out = REPO / "data" / "tasks"
    if a.obs == "v2":
        from scripts.build_scenarios import v2_dir

        if not a.out_dir or Path(a.out_dir).resolve() == out.resolve():
            raise SystemExit("--obs v2 needs --out-dir (never the v1 data/tasks directory)")
        out = Path(a.out_dir)
        tag = task_tag(version, a.world_variant)        # k15X- (v2), k15X21- (v21), k15X3- (v3)
        dir_of = lambda s: v2_dir(s, a.world_variant)  # noqa: E731
    else:
        tag, dir_of = "", seed_dir
    seeds = parse_seeds(a.seeds)
    if 0 in seeds:
        guard_write(demo_tasks_path(out, version), a.allow_demo_roots)      # refuse before generating anything
    out.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list] = defaultdict(list)
    # eval splits first: their question texts are banned from train (siting/collateral/unanswerable keys do not
    # depend on the scenario, so the same text would carry the same answer into training)
    seeds = sorted(seeds, key=lambda s: split_of(s) == "train")
    eval_q: set[str] = set()
    dropped = 0
    for seed in seeds:
        if not dir_of(seed).exists():
            print(f"seed {seed}: no scenario at {dir_of(seed)}", flush=True)
            continue
        g = Gen(dir_of(seed), seed, split_of(seed), tag=tag)
        ts = g.all(a.k)
        g.close()
        seen = set()
        for t in ts:
            sp = split_of(seed)
            if t.question in seen:
                continue                                    # one copy of each question per scenario
            seen.add(t.question)
            if t.template in NOT_FOR_SELECTION and sp in ("train", "val"):
                continue                                    # whole families held out of training and selection (M9)
            t.split = sp
            if sp in ("val", "test"):
                eval_q.add(t.question)
            elif sp == "train" and t.question in eval_q:
                dropped += 1
                continue
            by_split[sp].append(t.to_row())
        print(f"seed {seed}: {len(ts)} tasks", flush=True)
    print("train tasks dropped (question also in val/test):", dropped)
    for sp, rows in by_split.items():
        if sp == "demo":
            rng = random.Random(0)
            per = defaultdict(list)
            for r in rows:
                per[r["area"]].append(r)
            demo = [x for area in sorted(per) for x in rng.sample(per[area], min(3, len(per[area])))]
            guard_write(demo_tasks_path(out, version), a.allow_demo_roots).write_text(json.dumps(demo, indent=1))
            print("demo", len(demo))
            continue
        with open(out / f"{sp}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        cnt = defaultdict(int)
        for r in rows:
            cnt[r["area"]] += 1
        print(sp, len(rows), dict(cnt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
