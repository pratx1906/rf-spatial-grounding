"""Pipeline status for long multi-task runs. Stdlib only; works on Windows and WSL.

Each task writes runs/<program>/status/<task>.json; the live dashboard (scripts/pipeline_dashboard.py) reads them.
Programs: rfconv (default, RF convergence plan 2026-10-23, runs/rfconv) and paper (paper program 2026-10-28,
runs/paper). Pick one with `--program paper` (before the sub-command) or the env var SCENESCOUT_PIPELINE=paper.

    python scripts/pipeline_status.py --program paper set P1 --state running --pct 10

    python scripts/pipeline_status.py set T2 --state running --pct 40 --eta-min 35 --note "build 17/42"
    python scripts/pipeline_status.py metric T2 false_track_rate 0.009
    python scripts/pipeline_status.py result T2 "42/42 builds rc 0"
    python scripts/pipeline_status.py show

States: queued | running | done | failed | blocked. Logs: runs/<program>/logs/<task>.log (each task writes its own).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROGRAMS = {"rfconv": REPO / "runs" / "rfconv", "paper": REPO / "runs" / "paper"}
ENV_VAR = "SCENESCOUT_PIPELINE"


def program_root(program: str | None = None) -> Path:
    """runs/<program> for a known program name; the env var SCENESCOUT_PIPELINE when None; rfconv by default."""
    name = program or os.environ.get(ENV_VAR) or "rfconv"
    if name not in PROGRAMS:
        raise ValueError(f"unknown pipeline program {name!r}; one of {sorted(PROGRAMS)}")
    return PROGRAMS[name]


def set_program(program: str | None = None) -> Path:
    """Point this module's STATUS / LOGS at a program's root (module globals, used by load/save/update)."""
    global ROOT, STATUS, LOGS
    ROOT = program_root(program)
    STATUS, LOGS = ROOT / "status", ROOT / "logs"
    return ROOT


ROOT = STATUS = LOGS = Path()
set_program()


def _path(task: str) -> Path:
    STATUS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    return STATUS / f"{task}.json"


def load(task: str) -> dict:
    p = _path(task)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"task": task, "state": "queued", "pct": 0, "metrics": {}, "results": [], "history": []}


def save(d: dict) -> None:
    p = _path(d["task"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, p)                       # atomic: the dashboard never reads a half-written file


def update(task: str, **kw) -> dict:
    d = load(task)
    now = time.time()
    if kw.get("state") == "running" and d.get("state") != "running":
        d["started"] = now
    if kw.get("state") in ("done", "failed"):
        d["ended"] = now
        if kw.get("state") == "done":
            kw.setdefault("pct", 100)
    for k, v in kw.items():
        if v is not None:
            d[k] = v
    d["updated"] = now
    d.setdefault("history", []).append({"t": now, **{k: v for k, v in kw.items() if v is not None}})
    d["history"] = d["history"][-200:]
    save(d)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", choices=sorted(PROGRAMS), default=None,
                    help=f"status root runs/<program> (default: ${ENV_VAR} or rfconv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("task")
    s.add_argument("--state", choices=["queued", "running", "done", "failed", "blocked"])
    s.add_argument("--pct", type=float)
    s.add_argument("--eta-min", type=float)
    s.add_argument("--note")
    s.add_argument("--title")
    m = sub.add_parser("metric")
    m.add_argument("task")
    m.add_argument("key")
    m.add_argument("value")
    r = sub.add_parser("result")
    r.add_argument("task")
    r.add_argument("text")
    sub.add_parser("show")
    a = ap.parse_args()
    set_program(a.program)
    if a.cmd == "set":
        eta = time.time() + a.eta_min * 60 if a.eta_min is not None else None
        update(a.task, state=a.state, pct=a.pct, eta=eta, note=a.note, title=a.title)
    elif a.cmd == "metric":
        d = load(a.task)
        try:
            val = float(a.value)
        except ValueError:
            val = a.value
        d.setdefault("metrics", {})[a.key] = val
        d["updated"] = time.time()
        save(d)
    elif a.cmd == "result":
        d = load(a.task)
        d.setdefault("results", []).append(a.text)
        d["updated"] = time.time()
        save(d)
    else:
        for p in sorted(STATUS.glob("*.json")) if STATUS.exists() else []:
            d = json.loads(p.read_text())
            print(f"{d['task']:5s} {d.get('state', '?'):8s} {d.get('pct', 0):5.0f}%  {d.get('note', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
