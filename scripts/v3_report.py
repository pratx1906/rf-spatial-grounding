"""Task 6 report tables (RF convergence plan): demo v2 vs v2.1 vs v3 (CPU only, read only).

Reads what runs/rfconv/t6/post.sh writes for v3 and the same artifacts for demo v2 and v2.1, and prints markdown:
gates, track quality (test seeds 13-16 and the varied-cast seeds 51-60 split out), per-class tracks, TDOA, DF q32,
classifier held-out metrics, expert check per area / family, expert + SFT v1.1/v1.2/v1.3 on the frozen set with
two-level (scenario, task) cluster CIs, and the world gap N vs W4-W8. Any missing input prints "pending".

  python scripts/v3_report.py                                  # demo v2 | v21 | v3
  python scripts/v3_report.py --version v21                    # validation: v21 takes the "new" column
  python scripts/v3_report.py --version v3 --root data/scenarios_v3 --no-live

Track quality is recomputed live with scripts/track_metrics.evaluate (offline, reads truth/; never inside an env
step) so the two seed groups and the fix-error p95 can be split out; --no-live falls back to the combined
<prefix>_track_metrics.json written by post.sh.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PENDING = "pending"
TEST_SEEDS = list(range(13, 17))
VARIED_SEEDS = list(range(51, 61))
CLASSES = ("single", "swarm", "friendly", "controller", "decoy")
SFT = ("sft_v11", "sft_v12", "sft_v13")
WORLDS = ("W4", "W5", "W6", "W7", "W8")

# Per data version: scenario root, audit prefix (runs/s75 or runs/s77), classifier model + the log holding its eval
# lines, frozen eval tag (runs/eval/<policy>_<tag>.json) and world-variant eval files.
PROFILES = {
    "v2": {"label": "demo v2", "root": "data/scenarios", "prefix": "runs/s75/demo_v2",
           "model": "configs/track_class_v2.json", "class_log": None, "frozen": "v4",
           "world": "runs/eval/{pol}_{w}.json", "groups": {"13-16": TEST_SEEDS}},
    "v21": {"label": "v2.1", "root": "data/scenarios_v21", "prefix": "runs/s75/v21",
            "model": "configs/track_class_v21.json", "class_log": "runs/rfconv/logs/T2.log", "frozen": "v5",
            "world": "runs/eval/{pol}_v5_{w}.json", "groups": {"13-16": TEST_SEEDS}},
    "v3": {"label": "v3", "root": "data/scenarios_v3", "prefix": "runs/s77/v3",
           "model": "configs/track_class_v3.json", "class_log": "runs/rfconv/logs/T6.log", "frozen": "v6",
           "world": "runs/eval/{pol}_v6_{w}.json", "groups": {"13-16": TEST_SEEDS, "51-60": VARIED_SEEDS}},
}


# ---------------------------------------------------------------------------------------------------- pure helpers
def fmt(x, d: int = 3) -> str:
    """Number to fixed decimals; None / missing -> 'pending'."""
    if x is None:
        return PENDING
    if isinstance(x, str):
        return x
    return f"{x:.{d}f}"


def strip_ts(line: str) -> str:
    """Drop the run_logged.sh '[YYYY-mm-dd HH:MM:SS] ' prefix."""
    return re.sub(r"^\[[^\]]*\] ", "", line)


def percentile(v: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy default) without numpy."""
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def scenario_seed(task_id: str) -> int | None:
    """Scenario seed of a task id (k<seed>X3W4-family-n); None if it does not parse."""
    m = re.match(r"k(\d+)", str(task_id))
    return int(m.group(1)) if m else None


def seed_ranges(seeds) -> str:
    """[13, 14, 15, 51, 52] -> '13-15,51-52'."""
    s = sorted(set(seeds))
    out, i = [], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(f"{s[i]}" if i == j else f"{s[i]}-{s[j]}")
        i = j + 1
    return ",".join(out)


def short_set(name: str) -> str:
    """Classifier set name 'kandla_s15_X,kandla_s16_X,kandla_s15_X_W4,...' -> '15-16 X + 15-16 W4'."""
    by: dict = defaultdict(list)
    for part in name.split(","):
        m = re.fullmatch(r"kandla_s(\d+)_X(?:_(W\d+))?", part.strip())
        if not m:
            return name
        by[m.group(2) or "X"].append(int(m.group(1)))
    return " + ".join(f"{seed_ranges(v)} {k}" for k, v in sorted(by.items(), key=lambda kv: (kv[0] != "X", kv[0])))


def summarise_tracks(res: list[dict]) -> dict:
    """Pool per-set track_metrics.evaluate() results: totals and rates, containment, fix error median/p90/p95,
    tracks per drone (single + swarm emitters) and the never-tracked share per class."""
    if not res:
        return {}
    tracks = sum(r["tracks"] for r in res)
    spur = sum(r["spurious"] for r in res)
    clut = sum(r["clutter"] for r in res)
    out = {"sets": len(res), "tracks": tracks, "false_track_rate": spur / max(tracks, 1),
           "nontarget_rate": (spur + clut) / max(tracks, 1),
           "gospa": statistics.mean(r["gospa"] for r in res), "per_class": {}}
    ins_all, err_all = [], []
    drone_n = drone_tr = 0.0
    for c in CLASSES:
        v = [r["per_class"][c] for r in res if c in r.get("per_class", {})]
        n = sum(x["N"] for x in v)
        if not n:
            continue
        ins = [b for x in v for b in x.get("_in", [])]
        er = [b for x in v for b in x.get("_err", [])]
        ins_all += ins
        err_all += er
        tpe = sum(x["tracks_per_emitter"] * x["N"] for x in v) / n
        und = sum(x["undetected"] * x["N"] for x in v) / n
        if c in ("single", "swarm"):
            drone_n += n
            drone_tr += tpe * n
        out["per_class"][c] = {"N": n, "tracks_per_emitter": tpe, "undetected": und,
                               "containment": statistics.mean(ins) if ins else None,
                               "err_med": percentile(er, 50), "err_p90": percentile(er, 90)}
    out["containment"] = statistics.mean(ins_all) if ins_all else None
    out["err_med"], out["err_p90"], out["err_p95"] = (percentile(err_all, q) for q in (50, 90, 95))
    out["tracks_per_drone"] = drone_tr / drone_n if drone_n else None
    return out


def tracks_from_json(d: dict) -> dict:
    """The combined track_metrics.py --json output in summarise_tracks() shape (no p95: the file has none)."""
    pc = d.get("per_class", {})
    dn = sum(pc[c]["N"] for c in ("single", "swarm") if c in pc)
    return {**{k: d.get(k) for k in ("tracks", "false_track_rate", "nontarget_rate", "gospa", "containment",
                                     "err_med", "err_p90")}, "err_p95": None, "per_class": pc,
            "tracks_per_drone": (sum(pc[c]["tracks_per_emitter"] * pc[c]["N"] for c in ("single", "swarm") if c in pc)
                                 / dn) if dn else None}


def parse_q32(text: str) -> dict:
    """d_q32 text -> {'bearing'|'fix'|'z': {key: {n, med, rms, p95, max[, bias]}}, 'top': str, 't0': str}."""
    out: dict = {}
    sec = None
    for line in text.splitlines():
        line = strip_ts(line)
        if line.startswith("=== "):
            continue
        if line.startswith("=="):
            sec = ("bearing" if "bearing" in line else "fix" if "horizontal" in line
                   else "z" if "altitude" in line else "top")
            if sec == "top":
                out["top"] = line.split("==")[-1].strip()
            continue
        m = re.match(r"^(\(.*?\)) n=(\d+)(.*)$", line)
        if m and sec in ("bearing", "fix", "z"):
            vals = dict(re.findall(r"(\w+)=(-?[\d.]+)", m.group(3)))
            out.setdefault(sec, {})[m.group(1)] = {"n": int(m.group(2)), **{k: float(v) for k, v in vals.items()}}
        elif line.startswith("bursts with t0"):
            out["t0"] = line.split(":", 1)[1].strip()
    return out


def parse_tdoa(text: str) -> dict:
    """d_tdoa text -> {'all'|'clean': dict(n, med, rms, p95, max, frac30), 'null': str}."""
    out: dict = {}
    for line in text.splitlines():
        line = strip_ts(line)
        m = re.match(r"^(\w+) n=(\d+) range-diff \|err\| m: med=(\d+) rms=(\d+) p95=(\d+) max=(\d+)\s+"
                     r"frac<30m=([\d.]+)", line)
        if m:
            g = m.groups()
            out[g[0]] = {"n": int(g[1]), "med": int(g[2]), "rms": int(g[3]), "p95": int(g[4]), "max": int(g[5]),
                         "frac30": float(g[6])}
        elif "toa_s NULL" in line:
            out["null"] = line.split(":")[-1].strip()
    return out


def parse_expert_check(text: str) -> dict:
    """expert_check.py output -> {'area': {name: (n, mean)}, 'family': {name: (n, mean)}}."""
    out: dict = {"area": {}, "family": {}}
    for line in text.splitlines():
        line = strip_ts(line)
        m = re.match(r"^AREA (\w+)\s+n=\s*(\d+) mean=([\d.]+|nan)", line)
        if m:
            out["area"][m.group(1)] = (int(m.group(2)), float(m.group(3)))
            continue
        m = re.match(r"^\s+(\w+)\s+n=\s*(\d+) mean=([\d.]+|nan)", line)
        if m:
            out["family"][m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return out


def parse_class_log(text: str, model: str) -> list[dict]:
    """JSON metric lines of track_class_v2.py fit/eval runs (in a run_logged.sh log) that used `model`; the last run
    of each set name wins."""
    keep, on = {}, False
    for raw in text.splitlines():
        line = strip_ts(raw)
        if line.startswith("=== START"):
            on = "track_class_v2.py" in line and model in line and (" fit " in line or " eval " in line)
            continue
        if line.startswith("=== END"):
            on = False
            continue
        if on and line.startswith("{") and '"set"' in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            keep[d["set"]] = d
    return list(keep.values())


def cluster_ci(clusters: dict, n: int = 2000, seed: int = 0) -> dict:
    """Two-level bootstrap (scenarios, then tasks within each) of a pooled mean; same as eval.run.cluster_bootstrap."""
    keys = sorted(k for k, v in clusters.items() if v)
    vals = [list(clusters[k]) for k in keys]
    allv = [x for v in vals for x in v]
    if not allv:
        return {"n": 0, "n_clusters": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        tot, cnt = 0.0, 0
        for _k in range(len(vals)):
            v = vals[rng.randrange(len(vals))]
            s = rng.choices(v, k=len(v))
            tot += sum(s)
            cnt += len(s)
        means.append(tot / cnt)
    means.sort()
    return {"n": len(allv), "n_clusters": len(keys), "mean": statistics.mean(allv),
            "ci95": [means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]]}


def unpaired_cluster_diff(a: dict, b: dict, n: int = 2000, seed: int = 0) -> dict:
    """mean(b) - mean(a) for two independent clustered samples ({scenario: [scores]}): each bootstrap draw resamples
    both sides with the two-level scheme. Used for the world gap (W tasks are different questions from N tasks)."""
    def draw(rng, vals):
        tot, cnt = 0.0, 0
        for _ in range(len(vals)):
            v = vals[rng.randrange(len(vals))]
            s = rng.choices(v, k=len(v))
            tot += sum(s)
            cnt += len(s)
        return tot / cnt
    va = [list(v) for _, v in sorted(a.items()) if v]
    vb = [list(v) for _, v in sorted(b.items()) if v]
    if not va or not vb:
        return {"diff": None, "ci95": [None, None]}
    rng = random.Random(seed)
    d = sorted(draw(rng, vb) - draw(rng, va) for _ in range(n))
    ma = statistics.mean(x for v in va for x in v)
    mb = statistics.mean(x for v in vb for x in v)
    return {"diff": mb - ma, "ci95": [d[int(0.025 * n)], d[min(n - 1, int(0.975 * n))]]}


def clusters_of(rows: list[dict], seeds: set[int] | None = None) -> dict:
    """{scenario seed: [scores]} for eval rows, optionally only the given seeds."""
    out = defaultdict(list)
    for r in rows:
        s = scenario_seed(r.get("task_id", ""))
        if s is not None and (seeds is None or s in seeds):
            out[s].append(float(r["score"]))
    return dict(out)


def ci_str(c: dict, d: int = 3) -> str:
    if c.get("mean", c.get("diff")) is None:
        return PENDING
    m = c.get("mean", c.get("diff"))
    lo, hi = c["ci95"]
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# --------------------------------------------------------------------------------------------------------- loading
def _read(p: str | Path | None) -> str | None:
    if not p:
        return None
    p = REPO / p
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def _json(p: str | Path | None):
    t = _read(p)
    try:
        return json.loads(t) if t else None
    except json.JSONDecodeError:
        return None


def live_tracks(root: Path, seeds: list[int]) -> dict | None:
    """Recompute track metrics for the seeds that exist under root (None if none exist)."""
    dirs = [root / f"kandla_s{s}_X" for s in seeds]
    dirs = [d for d in dirs if (d / "observed.duckdb").exists() and (d / "truth" / "truth.duckdb").exists()]
    if not dirs:
        return None
    from scripts import track_metrics as tm

    bg = tm._bg()
    res = [tm.evaluate(d, d, bg) for d in dirs]
    out = summarise_tracks(res)
    out["seeds"] = len(dirs)
    return out


def load(version: str, prof: dict, live: bool) -> dict:
    root = REPO / prof["root"]
    D: dict = {"label": prof["label"], "version": version, "root": prof["root"], "tracks": {}}
    for g, seeds in prof["groups"].items():
        t = None
        if live:
            try:
                t = live_tracks(root, seeds)
            except Exception as e:  # noqa: BLE001
                print(f"<!-- live track metrics {version} {g} failed: {type(e).__name__}: {e} -->")
        if t is None and not live and g == "13-16":
            j = _json(prof["prefix"] + "_track_metrics.json")
            t = tracks_from_json(j) if j else None
        D["tracks"][g] = t
    if not live:   # the post.sh file pools 13-16 and 51-60 for v3
        j = _json(prof["prefix"] + "_track_metrics.json")
        D["tracks_file"] = tracks_from_json(j) if j else None
    q = _read(prof["prefix"] + "_d_q32.txt")
    D["q32"] = parse_q32(q) if q else None
    t = _read(prof["prefix"] + "_d_tdoa.txt")
    D["tdoa"] = parse_tdoa(t) if t else None
    e = _read(prof["prefix"] + "_expert_check.txt")
    D["expert_check"] = parse_expert_check(e) if e else None
    m = _json(prof["model"])
    cl = list((m or {}).get("metrics", []))
    log = _read(prof["class_log"])
    if log:
        names = {c["set"] for c in cl}
        cl += [c for c in parse_class_log(log, prof["model"]) if c["set"] not in names or c["set"] not in ("train", "val")]
    seen, uniq = set(), []
    for c in cl:
        if c["set"] not in seen:
            seen.add(c["set"])
            uniq.append(c)
    D["classifier"] = uniq if m or log else None
    D["evals"] = {}
    for pol in ("expert",) + SFT:
        d = _json(f"runs/eval/{pol}_{prof['frozen']}.json")
        D["evals"][pol] = d.get("rows") if d else None
    D["world"] = {}
    for pol in ("expert",) + SFT:
        for w in WORLDS:
            d = _json(prof["world"].format(pol=pol, w=w))
            D["world"][(pol, w)] = d.get("rows") if d else None
    return D


# -------------------------------------------------------------------------------------------------------- printing
def table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def track_cols(V: list[dict]) -> list[tuple[str, dict | None]]:
    cols = []
    for D in V:
        for g, t in D["tracks"].items():
            cols.append((f"{D['label']} {g}", t))
        if "tracks_file" in D:
            cols.append((f"{D['label']} file (pooled)", D["tracks_file"]))
    return cols


def print_gates(N: dict) -> None:
    t = N["tracks"].get("13-16") or N.get("tracks_file") or {}
    cl = N["classifier"] or []
    held = [c for c in cl if c["set"] not in ("train",)]
    acc = [c.get("expert_rule_acc") for c in held if c["set"] != "val"]
    rows = [["build", "42/42 rc 0 (+ 51-60)", "see runs/rfconv/logs (not parsed)", "-"],
            ["classifier held-out expert-rule accuracy", ">= 0.75",
             ", ".join(f"{short_set(c['set'])}: {c.get('expert_rule_acc')}" for c in held) or PENDING,
             PENDING if not acc else ("PASS" if min(acc) >= 0.75 else "FAIL")],
            ["false (spurious) track rate, 13-16", "< 0.05", fmt(t.get("false_track_rate")),
             PENDING if t.get("false_track_rate") is None else ("PASS" if t["false_track_rate"] < 0.05 else "FAIL")],
            ["2-sigma containment, 13-16", "0.80-0.93", fmt(t.get("containment")),
             PENDING if t.get("containment") is None else ("PASS" if 0.80 <= t["containment"] <= 0.93 else "FAIL")]]
    print(f"## Gates ({N['label']})\n")
    print(table(["gate", "target", N["label"], "result"], rows))


def print_tracks(V: list[dict]) -> None:
    cols = track_cols(V)
    print("\n## Track quality (scripts/track_metrics.py; seeds 13-16 test, 51-60 varied cast)\n")
    spec = [("sets", "sets", 0), ("tracks", "tracks", 0), ("false (spurious) rate", "false_track_rate", 3),
            ("non-target (spurious + clutter)", "nontarget_rate", 3), ("GOSPA mean (m)", "gospa", 1),
            ("2-sigma containment (nominal 0.865)", "containment", 3), ("fix err median (m)", "err_med", 1),
            ("fix err p90 (m)", "err_p90", 1), ("fix err p95 (m)", "err_p95", 1),
            ("tracks per drone (single + swarm)", "tracks_per_drone", 2)]
    rows = [[lab] + [fmt((t or {}).get(k), d) for _, t in cols] for lab, k, d in spec]
    for c in CLASSES:
        rows.append([f"never tracked, {c}"] + [fmt(((t or {}).get("per_class", {}).get(c) or {}).get("undetected"), 2)
                                               for _, t in cols])
    print(table(["metric"] + [c for c, _ in cols], rows))
    print("\n| class | " + " | ".join(f"{c} tracks/emitter, containment, err med/p90 m" for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for c in CLASSES:
        cells = []
        for _, t in cols:
            v = (t or {}).get("per_class", {}).get(c)
            cells.append(PENDING if not v else f"{v['tracks_per_emitter']:.2f}, {fmt(v.get('containment'))}, "
                                               f"{fmt(v.get('err_med'), 1)}/{fmt(v.get('err_p90'), 0)}")
        print(f"| {c} | " + " | ".join(cells) + " |")


def print_tdoa(V: list[dict]) -> None:
    print("\n## TDOA range-difference error (d_tdoa; first-arrival bursts with toa_s, bw >= 10 MHz)\n")
    rows = []
    for D in V:
        r = (D["tdoa"] or {}).get("all")
        rows.append([D["label"]] + ([str(r[k]) for k in ("n", "med", "rms", "p95", "max")] + [f"{r['frac30']:.2f}",
                                    D["tdoa"].get("null", "-")] if r else [PENDING] * 7))
    print(table(["set", "n", "median m", "RMS m", "p95 m", "max m", "within 30 m", "toa_s NULL bursts"], rows))


Q32_KEYS = {"bearing": ["('all',)", "('band', 'sub_ghz_868_915')", "('band', 'ism_2400')", "('band', 'ism_5800')",
                        "('los', 'LOS')", "('los', 'NLOS')"] + [f"('sensor', 'S{i}')" for i in range(1, 6)],
            "fix": ["('all',)"] + [f"('n_sensors', {i})" for i in range(2, 6)],
            "z": ["('all',)"]}


def print_q32(V: list[dict]) -> None:
    title = {"bearing": "Per-burst bearing error |deg|", "fix": "Fix horizontal error (m) at fix epochs",
             "z": "Fix altitude error z_est - z_true (m)"}
    for sec, keys in Q32_KEYS.items():
        print(f"\n## DF q32: {title[sec]} (n, median / p95)\n")
        rows = []
        for k in keys:
            cells = []
            for D in V:
                v = ((D["q32"] or {}).get(sec) or {}).get(k)
                cells.append(f"{v['n']}, {v['med']:.1f} / {v['p95']:.1f}" + (f", bias {v['bias']:.1f}" if "bias" in v else "")
                             if v else PENDING)
            rows.append([k.strip("()").replace("'", "").rstrip(",")] + cells)
        print(table(["split"] + [D["label"] for D in V], rows))
    print("\nNote: the v3 q32/TDOA files pool seeds 13-16 and 51-60 (runs/s77/d_*_v3.py); demo v2 and v2.1 are 13-16.")


def print_classifier(V: list[dict]) -> None:
    print("\n## Track classifier (scripts/track_class_v2.py; train 1-12,17-28, val 13-14)\n")
    rows = []
    for D in V:
        if not D["classifier"]:
            rows.append([D["label"], PENDING] + [PENDING] * 6)
            continue
        for c in D["classifier"]:
            rows.append([D["label"], short_set(c["set"]), str(c.get("n")), fmt(c.get("acc")), fmt(c.get("expert_rule_acc")),
                         fmt(c.get("drone_precision")), fmt(c.get("drone_recall")), fmt(c.get("auc_drone"))])
    print(table(["data", "set", "n", "acc", "expert-rule acc", "drone precision", "drone recall", "AUC"], rows))


def print_expert_check(V: list[dict]) -> None:
    print("\n## Expert check (scripts/expert_check.py --k 3; n, mean)\n")
    ex = [D["expert_check"] or {"area": {}, "family": {}} for D in V]
    areas = sorted({a for e in ex for a in e["area"]})
    cell = lambda v: PENDING if v is None else f"{v[0]}, {v[1]:.3f}"  # noqa: E731
    print(table(["area"] + [D["label"] for D in V], [[a] + [cell(e["area"].get(a)) for e in ex] for a in areas]
                or [[PENDING] * (len(V) + 1)]))
    fams = sorted({f for e in ex for f in e["family"]})
    print("\nPer family (`*` marks < 0.75 in the newest column):\n")
    rows = []
    for f in fams:
        last = ex[-1]["family"].get(f)
        rows.append([f + (" *" if last and last[1] < 0.75 else "")] + [cell(e["family"].get(f)) for e in ex])
    print(table(["family"] + [D["label"] for D in V], rows or [[PENDING] * (len(V) + 1)]))


def print_evals(V: list[dict], n_boot: int) -> None:
    print("\n## Frozen eval: mean score with two-level (scenario, task) cluster 95 % CI\n")
    rows = []
    for D in V:
        prof = PROFILES[D["version"]]
        for pol in ("expert",) + SFT:
            r = D["evals"].get(pol)
            groups = [("all", None)]
            if r is not None:
                present = {scenario_seed(x["task_id"]) for x in r}
                split = [(g, sd) for g, sd in prof["groups"].items() if present & set(sd)]
                if len(split) > 1:
                    groups += split
            for g, seeds in groups:
                if r is None:
                    rows.append([D["label"], prof["frozen"], pol, PENDING, PENDING, PENDING, PENDING])
                    continue
                cl = clusters_of(r, set(seeds) if seeds else None)
                if not cl:
                    continue
                g = f"{g} ({seed_ranges(cl)})"
                c = cluster_ci(cl, n_boot)
                rr = [x for x in r if seeds is None or scenario_seed(x["task_id"]) in seeds]
                by = defaultdict(list)
                for x in rr:
                    by[x["area"]].append(x["score"])
                macro = statistics.mean(statistics.mean(v) for v in by.values())
                rows.append([D["label"], prof["frozen"], pol, g, str(c["n"]), ci_str(c), f"{macro:.3f}"])
    print(table(["data", "set", "policy", "seeds", "n", "mean [cluster CI]", "macro"], rows))
    print("\nPaired differences (same tasks), two-level cluster CI:\n")
    rows = []
    try:
        from eval.run import paired_cluster_bootstrap as pcb
    except Exception:  # noqa: BLE001
        pcb = None
    for D in V:
        e = D["evals"]
        for a, b in [("expert", "sft_v11"), ("sft_v11", "sft_v12"), ("sft_v11", "sft_v13")]:
            if e.get(a) is None or e.get(b) is None or pcb is None:
                rows.append([D["label"], f"{b} - {a}", PENDING, PENDING])
                continue
            res = pcb(e[a], e[b], n_boot)["_all"]
            rows.append([D["label"], f"{b} - {a}", str(res["n"]), f"{res['diff']:+.3f} [{res['ci95'][0]:+.3f}, "
                                                                  f"{res['ci95'][1]:+.3f}]"])
    print(table(["data", "comparison", "n", "diff [cluster CI]"], rows))


def print_world(V: list[dict], n_boot: int) -> None:
    print("\n## World gap: W - N (unpaired; N restricted to the W set's scenario seeds), cluster 95 % CI\n")
    rows, missing = [], []
    for D in V:
        for pol in ("expert",) + SFT:
            n_rows = D["evals"].get(pol)
            for w in WORLDS:
                w_rows = D["world"].get((pol, w))
                if n_rows is None or w_rows is None:
                    missing.append(f"{D['label']} {pol} {w}")
                    continue
                wc = clusters_of(w_rows)
                nc = clusters_of(n_rows, set(wc))
                if not nc:
                    rows.append([D["label"], pol, w, "no N tasks on these seeds", PENDING, PENDING])
                    continue
                rows.append([D["label"], pol, w, ci_str(cluster_ci(nc, n_boot)), ci_str(cluster_ci(wc, n_boot)),
                             ci_str(unpaired_cluster_diff(nc, wc, n_boot))])
    print(table(["data", "policy", "world", "N mean", "W mean", "W - N"], rows or [[PENDING] * 6]))
    if missing:
        print(f"\nPending (no N or W eval file): {', '.join(missing)}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default="v3", choices=["v21", "v3"], help="data version in the 'new' column")
    ap.add_argument("--root", default=None, help="scenario root override for the new column")
    ap.add_argument("--prefix", default=None, help="audit file prefix override (e.g. runs/s75/v21)")
    ap.add_argument("--no-live", action="store_true", help="read <prefix>_track_metrics.json instead of recomputing")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    order = ["v2", "v21"] + (["v3"] if a.version == "v3" else [])
    profs = {k: dict(PROFILES[k]) for k in order}
    if a.root:
        profs[a.version]["root"] = a.root
    if a.prefix:
        profs[a.version]["prefix"] = a.prefix
    V = [load(k, profs[k], not a.no_live) for k in order]
    print(f"# Task 6 tables: {' vs '.join(D['label'] for D in V)} (simulation only)\n")
    print("Roots: " + "; ".join(f"{D['label']} `{D['root']}`" for D in V) + "\n")
    print_gates(V[-1])
    print_tracks(V)
    print_tdoa(V)
    print_q32(V)
    print_classifier(V)
    print_expert_check(V)
    print_evals(V, a.boot)
    print_world(V, a.boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
