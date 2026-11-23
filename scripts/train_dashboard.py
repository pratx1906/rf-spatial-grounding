"""Live training dashboard (stdlib only): loss curve, validation loss, ETA, GPU utilisation. Refreshes every 5 s.

    python scripts/train_dashboard.py --run sft_v1 --total 113 [--port 8095]     (Windows; reads nvidia-smi)

Open http://localhost:8095. Local only (binds 127.0.0.1). Reads runs/sft/<run>/log.jsonl and the console log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NVSMI = r"C:\Windows\System32\nvidia-smi.exe"

PAGE_FILE = Path(__file__).with_name("dashboard_page.html")


def gpu():
    try:
        out = subprocess.run([NVSMI, "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return {"util": int(float(out[0])), "mem": int(float(out[1])), "power": round(float(out[2]))}
    except Exception:  # noqa: BLE001
        return None


def data(run: str, total: int) -> dict:
    f = REPO / "runs" / "sft" / run / "log.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    start = max((i for i, r in enumerate(rows) if r.get("step") == 0 and "val_loss" in r), default=0)
    rows = rows[start:]
    steps = [r for r in rows if "loss" in r]
    vals = [[r["step"], r["val_loss"]] for r in rows if "val_loss" in r]
    smooth = [sum(s["loss"] for s in steps[max(0, i - 4): i + 1]) / len(steps[max(0, i - 4): i + 1])
              for i in range(len(steps))]
    eta = None
    if len(steps) >= 3:                       # rate from the recent steps (step 1 includes the initial validation)
        a, b = steps[-min(10, len(steps) - 1) - 1], steps[-1]
        eta = (b["elapsed_s"] - a["elapsed_s"]) / max(1, b["step"] - a["step"]) * (total - b["step"]) / 60
    health = "OK"
    if any(s["loss"] != s["loss"] for s in steps):
        health = "NaN loss"
    elif len(vals) >= 3 and vals[-1][1] > vals[-2][1] > vals[-3][1]:
        health = "val loss rising"
    con = REPO / "runs" / "dbg" / f"{run}.log"
    tail = ""
    if con.exists():
        lines = [x for x in con.read_text(errors="replace").replace("\r", "\n").splitlines()
                 if x.strip() and "Loading weights" not in x]
        tail = "\n".join(lines[-8:])
        if "Traceback" in tail or "Error" in tail:
            health = "ERROR (see console)"
    rate = None
    if len(steps) >= 3:
        a_, b_ = steps[-min(10, len(steps) - 1) - 1], steps[-1]
        rate = (b_["elapsed_s"] - a_["elapsed_s"]) / max(1, b_["step"] - a_["step"])
    import datetime as _dt
    clock = (_dt.datetime.now() + _dt.timedelta(minutes=eta)).strftime("%H:%M") if eta is not None else None
    saved = (REPO / "runs" / "dbg" / f"{run}.log").exists() and "[sft] saved" in (REPO / "runs" / "dbg" / f"{run}.log").read_text(errors="replace")
    return {"run": run, "total": total, "steps": steps, "vals": vals, "smooth": smooth, "eta_min": eta,
            "sec_per_step": rate, "finish_clock": clock, "saved": saved, "gpu": gpu(), "health": health, "console": tail}


EVAL_SETS = [("v3", 603), ("v4", 611), ("W4", 300), ("W5", 300), ("demo", 12)]
REQ_PER_TASK = 3.4                       # chat requests per question (p50 3 turns; the eval runs at concurrency 16)


def _summ(p: Path):
    try:
        s = json.loads(p.read_text())["summary"]
        al = s.get("_all", {})
        return {"all": al.get("score"), "macro": al.get("macro_score"),
                "areas": {k: v["score"] for k, v in s.items()
                                                     if isinstance(v, dict) and not k.startswith("_") and "score" in v}}
    except Exception:  # noqa: BLE001
        return None


def evaldata(t0: float) -> dict:
    """Evaluation pipeline (S2.5): stages from runs/dbg/s25/eval.sh, progress from the vLLM request log."""
    ev, d = REPO / "runs" / "eval", REPO / "runs" / "dbg" / "s25"
    stages = []
    for n, size in EVAL_SETS:
        for model, out in (("v1.3", f"sft_v13_{n}.json"), ("v1.2", f"sft_v12_{n}.json"),
                           ("v1.1", "sft_v11_v3_s25.json" if n == "v3" else f"sft_v11_{n}.json")):
            stages.append({"set": n, "model": model, "size": size, "out": ev / out, "log": d / (Path(out).stem + ".log")})
    for n, size in (("v3", 603), ("v4", 611)):
        for v in ("v11", "v12", "v13"):
            p = ev / f"sft_{v}_{n}_spec4.json"
            if p.exists() or (d / f"{p.stem}.log").exists():
                stages.append({"set": f"{n} spec-ngram", "model": "v" + v[1] + "." + v[2:], "size": size, "out": p,
                               "log": d / f"{p.stem}.log"})
    def nreq(name):
        x = d / name
        return x.read_text(errors="replace").count("POST /v1/chat/completions") if x.exists() else 0
    reqs = {"main": nreq("llm_eval.log"), "spec": nreq("llm_spec.log")}
    used = {"main": 0.0, "spec": 0.0}                # requests attributed to finished stages (estimate)
    rows, results = [], {}
    for s in stages:
        srv = "spec" if "spec" in s["set"] else "main"
        done = s["out"].exists() and s["out"].stat().st_mtime >= t0
        running = not done and s["log"].exists() and s["log"].stat().st_mtime >= t0
        prog = 1.0 if done else 0.0
        if done:
            used[srv] += REQ_PER_TASK * s["size"]
        if running:
            prog = max(0.0, min(0.99, (reqs[srv] - used[srv]) / (REQ_PER_TASK * s["size"])))
        sm = _summ(s["out"]) if done else None
        if sm:
            results.setdefault(s["set"], {})[s["model"]] = sm
        rows.append({"set": s["set"], "model": s["model"], "size": s["size"], "state": "done" if done else
                     "running" if running else "queued", "progress": round(prog, 3), "score": sm and sm["all"]})
    dec = d / "decide.json"
    led = REPO / "docs" / "superpowers" / "plans" / "2026-10-21-demo-v2.LEDGER.md"
    ledger = [x for x in led.read_text(encoding="utf-8").splitlines() if x.startswith(("S", "Ruling"))][-8:] if led.exists() else []
    out = (d / "eval.out").read_text(errors="replace") if (d / "eval.out").exists() else ""
    return {"stages": rows, "results": results, "decision": dec.read_text() if dec.exists() and dec.stat().st_size else None,
            "done": "EVAL_DONE" in out, "ledger": ledger, "requests": reqs["main"] + reqs["spec"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sft_v1")
    ap.add_argument("--total", type=int, default=113)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--eval-since", default="2026-10-21 19:09", help="eval outputs older than this are not this run's")
    a = ap.parse_args()
    import datetime as _dt
    t0 = _dt.datetime.strptime(a.eval_since, "%Y-%m-%d %H:%M").timestamp()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/eval"):
                body = json.dumps(evaldata(t0)).encode()
                ctype = "application/json"
            elif self.path.startswith("/data"):
                body = json.dumps(data(a.run, a.total)).encode()
                ctype = "application/json"
            else:
                body = PAGE_FILE.read_bytes()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"SceneScout training dashboard: http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
