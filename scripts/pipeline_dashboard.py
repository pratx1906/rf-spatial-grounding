"""Live pipeline dashboard (stdlib only). Programs: rfconv (RF-convergence plan, the default, http://127.0.0.1:8095)
and paper (paper program 2026-10-28, `--program paper`, http://127.0.0.1:8096).

Shows plan progress and live ETA, the task queue (state, %, ETA, dependencies), per-task metrics/results/log,
SFT training curves and today's eval outputs, live hardware (GPU, CPU, RAM, WSL, disk), job processes, the ledger
tail and the end results. Binds 127.0.0.1 only; never writes anything.

Start (Windows Python, no third-party packages; detached, survives the terminal):
    Start-Process -WindowStyle Hidden -FilePath .venv-sionna-win\\Scripts\\python.exe `
        -ArgumentList 'scripts\\pipeline_dashboard.py','--port','8095' -WorkingDirectory .
    Start-Process -WindowStyle Hidden -FilePath .venv-sionna-win\\Scripts\\python.exe `
        -ArgumentList 'scripts\\pipeline_dashboard.py','--program','paper' -WorkingDirectory .    # port 8096
Foreground:  .venv-sionna-win\\Scripts\\python.exe scripts\\pipeline_dashboard.py [--port 8095]
Stop:        Get-NetTCPConnection -LocalPort 8095 -State Listen | % { Stop-Process -Id $_.OwningProcess }

Inputs (all optional; missing files are shown as empty, never an error):
    runs/rfconv/status/<task>.json   written by scripts/pipeline_status.py (set/metric/result)
    runs/rfconv/logs/<task>.log      e.g. via  bash scripts/run_logged.sh <task> <cmd...>
    docs/superpowers/plans/2026-10-23-rf-convergence/LEDGER.md
    runs/sft/<run>/log.jsonl (updated in the last 6 h), runs/eval/*.json (written today)
  paper program: runs/paper instead of runs/rfconv, ledger docs/superpowers/plans/2026-10-28-paper-program/LEDGER.md, and
    runs/paper/queue/{pending,running,done}/   GPU queue (runs/paper/queue/runner.sh; job format in its README.md)
    runs/paper/results/P4*/**/log.jsonl        GRPO curves (reward mean, val score, abstain rate, KL; missing ok)
    runs/paper/results/**/summary.json         results table (scripts/paper_manifest.py)
API: /api/hw (every 3 s), /api/tasks, /api/log?task=T2&n=200, /api/train, /log/<task> (full log, text), /api/meta;
     paper: /api/queue, /api/grpo, /api/results.
Overhead: one sampler thread; nvidia-smi at most every 3 s, wsl every 10 s, Windows process list every 15 s.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
RF = REPO / "runs" / "rfconv"
STATUS, LOGS = RF / "status", RF / "logs"
LEDGER = REPO / "docs" / "superpowers" / "plans" / "2026-10-23-rf-convergence" / "LEDGER.md"
PAGE = Path(__file__).with_name("pipeline_page.html")
FONTS = REPO / "ui" / "kandla" / "fonts"
NVSMI = r"C:\Windows\System32\nvidia-smi.exe"
DISTRO = "Ubuntu-24.04"
NOWIN = 0x08000000 if sys.platform == "win32" else 0        # CREATE_NO_WINDOW: no console flashes when detached

# Plan tasks: estimate in hours (PLAN.md Estimates + ruling P1-4 for T5; T2b user-added) = weight, and dependencies.
PLAN = {
    "T1": (1.0, []),
    "T2": (3.0, ["T1"]),
    "T2b": (2.0, []),
    "T3": (2.0, ["T1"]),
    "T4": (3.0, ["T2", "T3"]),
    "T5": (6.0, ["T4"]),
    "T6": (4.0, ["T5", "T2b"]),
}
ORDER = ["T0", "T1", "T2", "T2b", "T3", "T4", "T5", "T6", "D"]
PROGRAM = "rfconv"

# Paper program (docs/superpowers/plans/2026-10-28-paper-program/PLAN.md): GPU hours as weights, CPU tasks ~1 h.
PLAN_PAPER = {
    "P0": (1.0, []),
    "P1": (2.5, ["P0"]),
    "P2": (4.0, ["P0"]),
    "P3": (1.0, []),
    "P4": (18.0, ["P3"]),
    "P5": (6.0, ["P1", "P2", "P4", "P6a"]),
    "P6a": (1.0, []),
    "P7": (2.0, ["P4"]),
    "P8": (5.0, ["P0"]),
    "P9": (1.0, ["P1", "P4", "P5"]),
    # P10 (human questions) removed and P11 (VLM) dropped for the 31 Oct deadline (user, 2026-10-29)
    "P12": (1.0, ["P0", "P1", "P2", "P3", "P4", "P5", "P6a", "P7", "P8", "P9"]),
}
PROGRAMS = {
    "rfconv": {"root": REPO / "runs" / "rfconv", "plan": PLAN, "order": ORDER, "port": 8095, "panels": [],
               "title": "RF convergence + v3 data (plan 2026-10-23)",
               "ledger": REPO / "docs" / "superpowers" / "plans" / "2026-10-23-rf-convergence" / "LEDGER.md"},
    "paper": {"root": REPO / "runs" / "paper", "plan": PLAN_PAPER, "order": list(PLAN_PAPER), "port": 8096,
              "panels": ["queue", "grpo", "results"], "title": "Paper program (plan 2026-10-28)",
              "ledger": REPO / "docs" / "superpowers" / "plans" / "2026-10-28-paper-program" / "LEDGER.md"},
}


def configure(program: str) -> dict:
    """Point the module globals (status/log dirs, ledger, plan, order) at one program; returns its config."""
    global RF, STATUS, LOGS, LEDGER, PLAN, ORDER, PROGRAM
    cfg = PROGRAMS[program]
    PROGRAM, RF, LEDGER, PLAN, ORDER = program, cfg["root"], cfg["ledger"], cfg["plan"], cfg["order"]
    STATUS, LOGS = RF / "status", RF / "logs"
    return cfg

JOB_RE = re.compile(r"build_scenarios|precompute|(?<!venv-)sionna|vllm|serve\.llm|blender|eval\.run|eval/run|train\.sft|"
                    r"train\.grpo|make_tasks|convergence_sweep|df_eval|coverage_build|rf_maps|blender_rf|"
                    r"export_rf|run_logged|gap_eval|mismatch_sweep|track_class|queue/runner|gpu_lock\.py run", re.I)
SKIP_RE = re.compile(r"pipeline_dashboard|mcp-for-blender|ps -eo", re.I)
SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def run(cmd: list[str], timeout: float) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=NOWIN)
        return r.stdout.decode("utf-8", "replace").replace("\x00", "")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- hardware (Windows ctypes + subprocess, cached)
class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


class _MEM(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("total", ctypes.c_uint64),
                ("avail", ctypes.c_uint64), ("pt", ctypes.c_uint64), ("pa", ctypes.c_uint64), ("vt", ctypes.c_uint64),
                ("va", ctypes.c_uint64), ("ve", ctypes.c_uint64)]


def _systimes():
    try:
        i, k, u = _FT(), _FT(), _FT()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        f = lambda x: (x.hi << 32) | x.lo  # noqa: E731
        return f(i), f(k), f(u)
    except Exception:  # noqa: BLE001
        return None


def _winmem():
    try:
        m = _MEM()
        m.dwLength = ctypes.sizeof(_MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return {"used_gb": round((m.total - m.avail) / 2**30, 1), "total_gb": round(m.total / 2**30, 1)}
    except Exception:  # noqa: BLE001
        return None


def _gpu():
    out = run([NVSMI, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
               "--format=csv,noheader,nounits"], 5)
    try:
        p = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        num = lambda s: float(s) if s not in ("[N/A]", "N/A", "") else None  # noqa: E731
        return {"name": p[0], "util": num(p[1]), "mem_mb": num(p[2]), "mem_total_mb": num(p[3]), "temp": num(p[4]),
                "power": num(p[5]), "power_limit": num(p[6])}
    except Exception:  # noqa: BLE001
        return None


def _wsl():
    out = run(["wsl.exe", "-d", DISTRO, "-e", "sh", "-c",
               "cat /proc/loadavg; echo @@; cat /proc/meminfo; echo @@; nproc; echo @@; "
               "ps -eo pid=,etimes=,rss=,args= --cols 500"], 8)
    if not out or out.count("@@") < 3:
        return None
    load, mem, nproc, ps = out.split("@@", 3)
    mi = {}
    for ln in mem.splitlines():
        m = re.match(r"(\w+):\s+(\d+)", ln)
        if m:
            mi[m.group(1)] = int(m.group(2))
    procs = []
    for ln in ps.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)", ln)
        if not m:
            continue
        args = m.group(4)
        if JOB_RE.search(args) and not SKIP_RE.search(args):
            procs.append({"where": "WSL", "pid": int(m.group(1)), "elapsed_s": int(m.group(2)),
                          "rss_gb": round(int(m.group(3)) / 2**20, 2), "cmd": args[:400]})
    tot, av = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
    return {"load": [float(x) for x in load.split()[:3]] if load.split() else None, "nproc": int(nproc.strip() or 0),
            "mem_used_gb": round((tot - av) / 2**20, 1), "mem_total_gb": round(tot / 2**20, 1), "procs": procs}


def _winprocs():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe' or Name='blender.exe'\" | "
          "Select-Object ProcessId,CreationDate,CommandLine,WorkingSetSize | ConvertTo-Json -Compress")
    out = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps], 12)
    if out is None:
        return None
    try:
        rows = json.loads(out) if out.strip() else []
    except ValueError:
        return None
    rows = rows if isinstance(rows, list) else [rows]
    now, procs = time.time(), []
    for r in rows:
        cmd = r.get("CommandLine") or ""
        if not JOB_RE.search(cmd) or SKIP_RE.search(cmd):
            continue
        cd = r.get("CreationDate")
        t0 = None
        if isinstance(cd, str) and (m := re.search(r"/Date\((\d+)", cd)):
            t0 = int(m.group(1)) / 1000
        elif isinstance(cd, dict) and "value" in cd:
            m = re.search(r"/Date\((\d+)", str(cd["value"]))
            t0 = int(m.group(1)) / 1000 if m else None
        procs.append({"where": "Windows", "pid": r.get("ProcessId"), "elapsed_s": int(now - t0) if t0 else None,
                      "rss_gb": round((r.get("WorkingSetSize") or 0) / 2**30, 2), "cmd": cmd[:400]})
    return procs


class Sampler(threading.Thread):
    """GPU/CPU/RAM every 3 s, WSL every 10 s, Windows processes every 15 s; 10 min of history."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.hist = collections.deque(maxlen=200)
        self.hw = {"t": None}
        self.wsl, self.wsl_t = None, None
        self.wp, self.wp_t = None, None
        self.prev = _systimes()

    @staticmethod
    def _loop(fn, period):
        while True:
            t0 = time.time()
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(max(0.5, period - (time.time() - t0)))

    def run(self):
        # slow probes in their own threads so a slow wsl/powershell call never delays the 3 s GPU sample
        threading.Thread(target=self._loop, args=(self.tick_wsl, 10.0), daemon=True).start()
        threading.Thread(target=self._loop, args=(self.tick_win, 15.0), daemon=True).start()
        self._loop(self.tick, 3.0)

    def tick_wsl(self):
        w = _wsl()
        if w is not None:
            with self.lock:
                self.wsl, self.wsl_ok = w, time.time()

    def tick_win(self):
        p = _winprocs()
        if p is not None:
            with self.lock:
                self.wp, self.wp_ok = p, time.time()

    def tick(self):
        cur, cpu = _systimes(), None
        if cur and self.prev:
            di, dk, du = (c - p for c, p in zip(cur, self.prev))
            cpu = round(100 * (1 - di / (dk + du)), 1) if dk + du > 0 else None
        self.prev = cur
        g = _gpu()
        try:
            du = shutil.disk_usage("C:\\")
            disk = {"free_gb": round(du.free / 2**30, 1), "total_gb": round(du.total / 2**30, 1)}
        except OSError:
            disk = None
        t = time.time()
        hw = {"t": t, "gpu": g, "gpu_t": t if g else self.hw.get("gpu_t"), "cpu": cpu, "ram": _winmem(), "disk": disk}
        with self.lock:
            self.hw = hw
            self.hist.append([round(t, 1), g and g["util"], g and g["mem_mb"] and round(g["mem_mb"] / 1024, 2), cpu])

    def snapshot(self):
        with self.lock:
            w = self.wsl or {}
            return {**self.hw, "wsl": {k: v for k, v in w.items() if k != "procs"} or None,
                    "wsl_t": getattr(self, "wsl_ok", None), "hist": list(self.hist), "now": time.time()}

    def procs(self):
        with self.lock:
            return {"procs": (self.wsl or {}).get("procs", []) + (self.wp or []),
                    "wsl_t": getattr(self, "wsl_ok", None), "win_t": getattr(self, "wp_ok", None)}


SAMPLER = Sampler()


# ---------------------------------------------------------------- pipeline status
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tail(p: Path, n: int, max_bytes: int = 512 * 1024) -> list[str]:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if size > max_bytes:
        lines = lines[1:]
    return [x for x in lines if x.strip()][-n:]


def tasks_data() -> dict:
    now = time.time()
    st = {}
    for p in sorted(STATUS.glob("*.json")) if STATUS.exists() else []:
        d = _read_json(p)
        if isinstance(d, dict):
            st[d.get("task", p.stem)] = d
    for k in PLAN:
        st.setdefault(k, {"task": k, "state": "queued", "pct": 0, "metrics": {}, "results": []})

    # projected finish per plan task: done -> ended; running -> eta / extrapolation; else after deps
    fin, start, src = {}, {}, {}

    def finish(k):
        if k in fin:
            return fin[k]
        d, (est, deps) = st[k], PLAN[k]
        s, pct = d.get("state"), float(d.get("pct") or 0)
        if s == "done":
            fin[k], src[k] = d.get("ended") or d.get("updated") or now, "done"
            start[k] = d.get("started")
        elif s == "running":
            st0 = d.get("started") or now
            start[k] = st0
            if d.get("eta") and d["eta"] > now - 60:
                fin[k], src[k] = max(d["eta"], now), "task ETA"
            elif pct >= 3:
                fin[k], src[k] = now + (now - st0) * (100 - pct) / pct, "rate"
            else:
                fin[k], src[k] = max(st0 + est * 3600, now + 0.25 * est * 3600), "plan estimate"
        else:
            s0 = max([now] + [finish(x) for x in deps])
            start[k], fin[k], src[k] = s0, s0 + est * 3600, "plan estimate"
        return fin[k]

    for k in PLAN:
        finish(k)
    for k, d in st.items():                     # tasks outside the plan (T0, D): their own ETA only
        if k not in PLAN and d.get("state") == "running" and d.get("eta"):
            fin[k], src[k] = max(d["eta"], now), "task ETA"
    wsum = sum(e for e, _ in PLAN.values())
    pct = sum(e * (100 if st[k].get("state") == "done" else float(st[k].get("pct") or 0)) for k, (e, _) in PLAN.items()) / wsum
    rows = []
    for k in sorted(st, key=lambda x: (ORDER.index(x) if x in ORDER else 99, x)):
        d = st[k]
        log = LOGS / f"{k}.log"
        try:
            lm = log.stat().st_mtime
        except OSError:
            lm = None
        rows.append({"task": k, "title": d.get("title", k), "state": d.get("state", "queued"),
                     "pct": d.get("pct") or 0, "note": d.get("note", ""), "started": d.get("started"),
                     "ended": d.get("ended"), "updated": d.get("updated"), "eta": d.get("eta"),
                     "proj_start": start.get(k) if k in PLAN and d.get("state") not in ("done", "running") else None,
                     "proj_finish": fin.get(k), "eta_src": src.get(k), "deps": PLAN.get(k, (0, []))[1],
                     "weight_h": PLAN.get(k, (None,))[0], "in_plan": k in PLAN, "metrics": d.get("metrics", {}),
                     "results": d.get("results", []), "log_mtime": lm})
    running = [r for r in rows if r["state"] == "running"]
    failed = [r for r in rows if r["state"] == "failed" and r["in_plan"]]
    if running:
        head = " | ".join(f"{r['task']} {r['title']}: {r['note'] or 'running'} ({r['pct']:.0f}%)" for r in running)
    elif all(st[k].get("state") == "done" for k in PLAN):
        head = "Plan complete"
    else:
        nxt = [k for k in PLAN if st[k].get("state") not in ("done",)]
        ready = [k for k in nxt if all(st[x].get("state") == "done" for x in PLAN[k][1])]
        head = f"Idle. Next ready: {', '.join(ready) or 'none'}" + (" (waiting for the user's go)" if ready else "")
    if failed:
        head = "FAILED: " + ", ".join(r["task"] for r in failed) + ". " + head
    led = _tail(LEDGER, 15)
    try:
        led_m = LEDGER.stat().st_mtime
    except OSError:
        led_m = None
    return {"t": now, "pct": round(pct, 1), "finish": max(fin[k] for k in PLAN), "headline": head, "tasks": rows,
            "ledger": led, "ledger_mtime": led_m,
            "status_mtime": max([r["updated"] or 0 for r in rows] or [0]) or None, **SAMPLER.procs()}


# ---------------------------------------------------------------- training / eval
_EV_CACHE: dict[str, tuple[float, dict]] = {}


def _eval_summary(p: Path, mtime: float):
    c = _EV_CACHE.get(str(p))
    if c and c[0] == mtime:
        return c[1]
    d = _read_json(p) or {}
    s = d.get("summary", {}) if isinstance(d, dict) else {}
    al = s.get("_all", {}) if isinstance(s, dict) else {}
    out = {"score": al.get("score"), "macro": al.get("macro_score"), "n": al.get("n"), "wall_s": al.get("wallclock_s"),
           "policy": d.get("policy") if isinstance(d, dict) else None, "set": d.get("set") if isinstance(d, dict) else None}
    _EV_CACHE[str(p)] = (mtime, out)
    return out


def train_data() -> dict:
    now = time.time()
    runs = []
    for f in sorted((REPO / "runs" / "sft").glob("*/log.jsonl")):
        try:
            m = f.stat().st_mtime
        except OSError:
            continue
        if now - m > 6 * 3600:
            continue
        rows = []
        for x in f.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(x))
            except ValueError:
                pass
        start = max((i for i, r in enumerate(rows) if r.get("step") == 0 and "val_loss" in r), default=0)
        rows = rows[start:]
        steps = [r for r in rows if "loss" in r and "step" in r]
        vals = [[r["step"], r["val_loss"]] for r in rows if "val_loss" in r]
        smooth = [sum(s["loss"] for s in steps[max(0, i - 4): i + 1]) / len(steps[max(0, i - 4): i + 1])
                  for i in range(len(steps))]
        runs.append({"run": f.parent.name, "mtime": m, "total": max([s["step"] for s in steps] + [1]),
                     "steps": [{"step": s["step"], "loss": s["loss"], "elapsed_s": s.get("elapsed_s")} for s in steps],
                     "vals": vals, "smooth": smooth})
    today = dt.date.today()
    evals = []
    for p in sorted((REPO / "runs" / "eval").glob("*.json")):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if dt.date.fromtimestamp(m) == today:
            evals.append({"name": p.stem, "mtime": m, **_eval_summary(p, m)})
    evals.sort(key=lambda e: e["mtime"])
    running = []
    for pr in SAMPLER.procs()["procs"]:
        if re.search(r"eval[./]run", pr["cmd"]):
            o = re.search(r"--out\s+(\S+)", pr["cmd"])
            s = re.search(r"--set\s+(\S+)", pr["cmd"])
            n = None
            if s:
                try:
                    n = sum(1 for _ in open(REPO / s.group(1), encoding="utf-8"))
                except OSError:
                    pass
            running.append({"out": o.group(1) if o else "?", "set": s.group(1) if s else "?", "n": n,
                            "elapsed_s": pr["elapsed_s"], "where": pr["where"]})
    return {"t": now, "runs": runs, "evals": evals, "eval_running": running}


# ---------------------------------------------------------------- paper program: GPU queue, GRPO curves, results
JOB_NAME_RE = re.compile(r"^\d+_([A-Za-z0-9]+)_")


def job_task(name: str) -> str:
    """Task id of a queue job file: 010_P1_base_v6.sh -> P1; anything else -> 'misc' (same rule as runner.sh)."""
    m = JOB_NAME_RE.match(name)
    return m.group(1) if m else "misc"


def _mtime(p: Path):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def queue_data() -> dict:
    """runs/<program>/queue: pending (lexical order = run order), running (+ its live json), done (record json)."""
    q, now = RF / "queue", time.time()
    pending = [{"name": p.name, "task": job_task(p.name), "queued": _mtime(p)} for p in sorted((q / "pending").glob("*.sh"))]
    running = []
    for p in sorted((q / "running").glob("*.sh")):
        d = _read_json(p.with_suffix(".json")) or {}
        running.append({"name": p.name, "task": job_task(p.name), "queued": d.get("queued"), "started": d.get("started"),
                        "peak_vram_mb": d.get("peak_vram_mb"),
                        "elapsed_s": now - d["started"] if d.get("started") else None})
    done = []
    for p in (q / "done").glob("*.json"):
        d = _read_json(p)
        if isinstance(d, dict):
            done.append({k: d.get(k) for k in ("name", "task", "queued", "started", "ended", "exit", "wall_s",
                                                "peak_vram_mb", "episodes", "episodes_per_s", "note")})
    done.sort(key=lambda d: d.get("ended") or 0, reverse=True)
    hb = _read_json(q / "runner.json") or {}
    return {"t": now, "pending": pending, "running": running, "done": done[:60],
            "runner": {**hb, "alive": bool(hb.get("t") and now - hb["t"] < 120)}, "stop_requested": (q / "STOP").exists()}


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}/"))
        else:
            out[f"{pre}{k}"] = v
    return out


GRPO_X = ("iter", "iteration", "step", "update", "global_step")
GRPO_SERIES = {
    "reward": ("reward_mean", "reward/mean", "mean_reward", "train/reward_mean", "reward"),
    "val": ("val_score", "val/score", "val_before_update/score", "val_after_update/score", "validation/score",
            "val_mean_score"),
    "abstain": ("abstain_rate", "abstain/rate", "train/abstain_rate", "counters/abstain_rate", "hacking/abstain_rate"),
    "kl": ("kl", "kl_mean", "kl/mean", "approx_kl", "train/kl"),
}


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else None


def parse_grpo_log(lines: list[str]) -> dict:
    """{n, x_max, reward/val/abstain/kl: [[x, y], ...]} from GRPO log.jsonl lines; bad lines and missing fields skipped.
    x = the first of iter/iteration/step/update/global_step, else the line number."""
    out: dict = {k: [] for k in GRPO_SERIES}
    xs = []
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if not isinstance(r, dict):
            continue
        f = _flat(r)
        x = next((_num(f[k]) for k in GRPO_X if _num(f.get(k)) is not None), float(i + 1))
        xs.append(x)
        for name, keys in GRPO_SERIES.items():
            y = next((_num(f[k]) for k in keys if _num(f.get(k)) is not None), None)
            if y is not None:
                out[name].append([x, y])
    return {"n": len(xs), "x_max": max(xs) if xs else 0, **out}


def grpo_data() -> dict:
    base, runs = RF / "results", []
    for f in sorted(base.glob("P4*/**/log.jsonl")):
        if f.parent.name.startswith("_") or "." in f.parent.name:      # set aside (_x) or archived failed runs (x.reason)
            continue
        try:
            m, lines = f.stat().st_mtime, f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        runs.append({"run": f.parent.relative_to(base).as_posix(), "mtime": m, **parse_grpo_log(lines)})
    return {"t": time.time(), "runs": runs}


def results_data() -> dict:
    """One row per runs/<program>/results/**/summary.json (scripts/paper_manifest.py writes them)."""
    base, rows = RF / "results", []
    for f in sorted(base.glob("**/summary.json")):
        d = _read_json(f)
        if not isinstance(d, dict):
            continue
        ci = d.get("ci95")
        rel = f.parent.relative_to(base).as_posix()
        rows.append({"run": rel, "task": rel.split("/", 1)[0], "model": d.get("model"), "set": d.get("set"),
                     "ablation": d.get("ablation"), "n": d.get("n"), "score": d.get("score"),
                     "macro": d.get("macro_score"), "ci95": ci if isinstance(ci, list) and len(ci) == 2 else None,
                     "ci_kind": d.get("ci_kind"), "abstain_rate": d.get("abstain_rate"), "mtime": _mtime(f)})
    rows.sort(key=lambda r: (r["task"], str(r["set"]), str(r["model"]), str(r["ablation"])))
    return {"t": time.time(), "rows": rows}


# ---------------------------------------------------------------- HTTP
class H(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/hw":
                return self._json(SAMPLER.snapshot())
            if u.path == "/api/tasks":
                return self._json(tasks_data())
            if u.path == "/api/train":
                return self._json(train_data())
            if u.path == "/api/meta":
                cfg = PROGRAMS[PROGRAM]
                return self._json({"program": PROGRAM, "title": cfg["title"], "panels": cfg["panels"],
                                   "logs": LOGS.relative_to(REPO).as_posix(), "ledger": LEDGER.relative_to(REPO).as_posix()})
            if u.path == "/api/queue":
                return self._json(queue_data())
            if u.path == "/api/grpo":
                return self._json(grpo_data())
            if u.path == "/api/results":
                return self._json(results_data())
            if u.path == "/api/log":
                task = q.get("task", [""])[0]
                n = max(1, min(2000, int(q.get("n", ["200"])[0] or 200)))
                if not SAFE.match(task):
                    return self._json({"task": task, "lines": [], "mtime": None, "t": time.time()})
                p = LOGS / f"{task}.log"
                try:
                    m, size = p.stat().st_mtime, p.stat().st_size
                except OSError:
                    m, size = None, 0
                return self._json({"task": task, "lines": _tail(p, n), "mtime": m, "size": size, "t": time.time()})
            if u.path.startswith("/log/"):
                task = u.path[5:]
                p = LOGS / f"{task}.log"
                if not SAFE.match(task) or not p.exists():
                    return self._send(404, b"no log yet", "text/plain; charset=utf-8")
                return self._send(200, p.read_bytes(), "text/plain; charset=utf-8")
            if u.path.startswith("/fonts/"):
                name = u.path[7:]
                p = FONTS / name
                if SAFE.match(name) and name.endswith((".woff2", ".css")) and p.exists():
                    return self._send(200, p.read_bytes(), "font/woff2" if name.endswith(".woff2") else "text/css")
                return self._send(404, b"", "text/plain")
            if u.path in ("/", "/index.html"):
                try:
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                except OSError:
                    return self._send(500, b"pipeline_page.html missing", "text/plain")
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # noqa: BLE001
            try:
                return self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")
            except Exception:  # noqa: BLE001
                return None

    def log_message(self, *_):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", choices=sorted(PROGRAMS), default="rfconv")
    ap.add_argument("--port", type=int, default=None, help="default 8095 (rfconv), 8096 (paper)")
    a = ap.parse_args()
    cfg = configure(a.program)
    a.port = a.port or cfg["port"]
    SAMPLER.start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    print(f"SceneScout pipeline dashboard ({a.program}): http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
