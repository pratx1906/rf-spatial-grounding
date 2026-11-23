"""Curate the demo question set (seed 0) for the operator console: the six-beat demo script first, then extras.

    python scripts/demo_set.py            -> data/demo_tasks_kandla.json        (v1 observations, data/site_kandla)
    python scripts/demo_set.py --obs v2   -> data/demo_tasks_kandla_v2.json     (RF phase 2, data/scenarios/kandla_s0_X)

Beats follow docs/design/FINAL.md (from the programme-director audit): only families where SFT v1 scores 1.0 or
close on frozen eval v3, each question checked with the scripted expert (score >= 0.95) before it is kept.
The demo split (seed 0) is never used for training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness.env import AkshaEnv  # noqa: E402
from scripts.build_scenarios import seed_dir, v2_dir  # noqa: E402
from tasks.expert_kandla import solve  # noqa: E402
from tasks.kandla.gen_more import GenMore as Gen  # noqa: E402

BEATS = ["genuine_drone", "direct_path", "ds_hover", "suggest_one", "jam_links", "actuation"]
EXTRAS = ["jam_aviation", "ds_swarm", "first_entry", "exit_time", "entity_not_found", "band_in_zone"]
WEAK = {"first_into_zone", "locate_ghost", "count_in_zone", "weak_low_alt", "tracks_in_band"}
# v2: the same six beats; a beat with no expert-solved question falls back to the closest family (Ruling in the ledger)
FALLBACK = {"genuine_drone": ["ghost_check", "protocol_id"], "direct_path": ["direct_all", "locate"],
            "ds_hover": ["ds_hover_budget2", "ds_loiter"], "suggest_one": ["suggest_two", "weak_df"],
            "jam_links": ["jam_fratricide", "jam_protected"], "actuation": ["entity_not_found"]}
MIN_SCORE = {"v1": 0.95, "v2": 0.9}
# v2 direct_path beat: the direct bearing must stand at least this far from every reflection at that sensor. Seed 0
# v2 has direct_path questions whose three bearings lie within 2-4 deg at equal SNR (no visible story, and the model
# picks the wrong near-twin); demo-v2 Ruling S6.2.
MIN_DIRECT_GAP_DEG = 10.0


def paths(obs: str) -> tuple[Path, str, Path]:
    """(scenario dir, task-id tag, output file) of the demo set for an observation version."""
    if obs == "v1":
        return seed_dir(0), "", REPO / "data" / "demo_tasks_kandla.json"
    if obs == "v2":
        return v2_dir(0, None), "X", REPO / "data" / "demo_tasks_kandla_v2.json"
    raise ValueError(f"unknown observation version {obs!r}")


def expert_score(env: AkshaEnv, t) -> float:
    env.reset(t)
    try:
        solve(env, t)
    except Exception:  # noqa: BLE001
        return 0.0
    return getattr(env, "answer_score", 0.0) if env.done else 0.0


def direct_gap_deg(env: AkshaEnv, t) -> float:
    """Smallest bearing gap (deg) between the direct-path hyps and the reflections the question's sensor reports."""
    ref = (t.answer_key.get("all_hyp_refs") or [""])[0]       # hyp:<sensor>_<track>_<tttt>_<n>@<t>
    if not ref.startswith("hyp:"):
        return 360.0
    sen, trk = ref[4:].split("_")[:2]
    tr, ts = f"trk:{trk}", float(ref.split("@")[1])
    env.reset(t)
    env.turn = 1
    msg = env.call_tools([{"name": "df_hypotheses",
                           "arguments": {"track_ref": tr, "at_time_s": ts, "sensor_refs": [f"sensor:{sen}"]}}])
    hyps = json.loads(msg[0]["content"].splitlines()[0])["data"]["sensors"][0]["hyps"]
    los = set(t.answer_key.get("los_hyp_refs") or [])
    d = [h["bearing_deg"] for h in hyps if h["ref"] in los]
    o = [h["bearing_deg"] for h in hyps if h["ref"] not in los]
    return min((abs((a - b + 180.0) % 360.0 - 180.0) for a in d for b in o), default=360.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["v1", "v2"], default="v1")
    ap.add_argument("--k", type=int, default=8, help="questions per family drawn from the generator")
    a = ap.parse_args()
    sdir, tag, out_f = paths(a.obs)
    if not sdir.exists():
        raise SystemExit(f"missing scenario {sdir}")
    g = Gen(sdir, 0, "demo", tag=tag)
    tasks = g.all(a.k)
    g.close()
    env = AkshaEnv(REPO / tasks[0].site)
    rng = random.Random(7)
    by_fam: dict[str, list] = {}
    for t in tasks:
        by_fam.setdefault(t.template, []).append(t)
    out, seen = [], set()
    lo = MIN_SCORE[a.obs]
    for fam, n in [(f, 1) for f in BEATS] + [(f, 1) for f in EXTRAS]:
        assert fam not in WEAK
        cands = by_fam.get(fam, [])
        if a.obs == "v2" and fam in FALLBACK:
            fb = [t for f in FALLBACK[fam] for t in by_fam.get(f, [])]
        else:
            fb = []
        rng.shuffle(cands)
        kept = 0
        for t in cands + fb:
            if t.question in seen or len(t.question) > 160 or " the the " in t.question:
                continue                    # (v2 zone names can start with "the": generator text "the the creek")
            if a.obs == "v2" and t.template == "direct_path" and direct_gap_deg(env, t) < MIN_DIRECT_GAP_DEG:
                continue
            sc = expert_score(env, t)
            if sc >= lo:
                t.split = "demo"
                out.append(t.to_row())
                seen.add(t.question)
                kept += 1
                print(f"{fam:18s} {sc:.2f}  {t.template if t.template != fam else ''}  {t.question}", flush=True)
                if kept >= n:
                    break
        if not kept:
            print(f"{fam:18s} --   no expert-verified question at seed 0")
    out_f.write_text(json.dumps(out, indent=1))
    print("demo tasks:", len(out), "->", out_f.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
