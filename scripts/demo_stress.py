"""Demo stress checks.

Live (demo-v2 S6.3): run each demo beat N times against a running gateway and report latency, errors and fps.

    python scripts/demo_stress.py --gateway http://127.0.0.1:8080 [--repeats 5] [--beats 1-6] [--mode model]
                                  [--fps] [--out runs/eval/demo_stress_live.json]

    HTTP runs (default): POST /api/ask {"q", "mode", "task_id"} and read the SSE stream, one question at a time.
    --fps: the same runs driven through the console in a browser (Playwright, as the presenter would: D, then the
           beat number), recording the event stream and the frame rate while the console draws the beat. Needs
           `uv run --no-project --with playwright --with httpx python scripts/demo_stress.py ...` on Windows.
    --mode scripted runs the scripted expert instead of the model (checks the gateway and demo set without vLLM).
    Per run: time to the first event, first tool call, answer and done; turns; status (verified/flagged/none);
    verifier flags; error codes; offline score against the demo answer key when the demo file is local.

Paraphrase (older check, kept): each demo question as written plus N rewordings, run in-process and scored.

    python scripts/demo_stress.py --models sft[,sft11] [--n 3]      (vLLM must serve every named model)
    -> runs/eval/demo_stress.json and a per-question table on stdout
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEMO_FILES = {"v1": REPO / "data" / "demo_tasks_kandla.json", "v2": REPO / "data" / "demo_tasks_kandla_v2.json"}


# ---------------------------------------------------------------------------------------------- live (S6.3)
def parse_beats(spec: str, n: int) -> list[int]:
    """'1-6' / '1,3,5' / 'all' -> 0-based indices into the demo list (beats are the first six entries)."""
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    bad = [i for i in out if not 1 <= i <= n]
    if bad:
        raise ValueError(f"beats {bad} outside 1..{n}")
    return [i - 1 for i in out]


def summarize_run(events: list[tuple[float, dict]], t0: float) -> dict:
    """One run from its (arrival time, event) list: latencies (s from the request), status, errors, turns."""
    def first(pred):
        return next((round(t - t0, 3) for t, e in events if pred(e)), None)

    ans = next((e for _, e in events if e.get("type") == "answer"), None)
    errs = [e.get("code") or "error" for _, e in events if e.get("type") == "error"]
    return {"t_first_event_s": first(lambda e: True),
            "t_first_tool_s": first(lambda e: e.get("type") == "tool_call"),
            "t_answer_s": first(lambda e: e.get("type") == "answer"),
            "t_done_s": first(lambda e: e.get("type") == "done"),
            "tool_calls": sum(e.get("type") == "tool_call" for _, e in events),
            "turns": ans.get("turns") if ans else None, "end_reason": ans.get("end_reason") if ans else None,
            "status": ans.get("status", "none") if ans else "none", "flags": (ans or {}).get("flags") or [],
            "errors": errs, "answer": (ans or {}).get("answer"),
            "complete": any(e.get("type") == "done" for _, e in events)}


def offline_score(task_row: dict | None, answer: dict | None) -> float | None:
    """Score against the local demo answer key (episode-scoped refs are not resolved: siting ras: refs score 0)."""
    if task_row is None:
        return None
    try:
        from tasks.base import Task
        from tasks.scoring import score_answer
        from tools.control import SubmitArgs

        ans = SubmitArgs.model_validate({"answer": answer}).answer if answer else None
        return round(float(score_answer(Task.from_row(task_row), ans)), 3)
    except Exception:  # noqa: BLE001 - a scoring problem must not hide the latency numbers
        return None


def http_run(client, gw: str, d: dict, mode: str, timeout_s: float) -> dict:
    """One question through POST /api/ask; retries a 429 (previous question still closing) for up to 10 s."""
    body = {"q": d["question"], "mode": mode, "task_id": d["task_id"]}
    for _ in range(20):
        t0 = time.perf_counter()
        events: list[tuple[float, dict]] = []
        with client.stream("POST", f"{gw}/api/ask", json=body, timeout=timeout_s) as r:
            if r.status_code == 429:
                time.sleep(0.5)
                continue
            if r.status_code != 200:
                return {"http": r.status_code, "errors": [f"http_{r.status_code}"], "status": "none",
                        "complete": False, "wall_s": round(time.perf_counter() - t0, 3)}
            for line in r.iter_lines():
                if line.startswith("data: "):
                    events.append((time.perf_counter(), json.loads(line[6:])))
        out = summarize_run(events, t0)
        out.update(http=200, wall_s=round(time.perf_counter() - t0, 3))
        return out
    return {"http": 429, "errors": ["busy"], "status": "none", "complete": False}


EV_HOOK = """
window.__ev = [];
const _ES = window.EventSource;
window.EventSource = function (url, cfg) {
  const es = new _ES(url, cfg);
  window.__ev.push({t: performance.now(), e: {type: "_open", url: String(url)}});
  es.addEventListener("message", (m) => { try { window.__ev.push({t: performance.now(), e: JSON.parse(m.data)}); } catch (_) {} });
  return es;
};
window.EventSource.prototype = _ES.prototype;
window.__frames = [];
(function loop(t) { window.__frames.push(t); if (window.__frames.length > 20000) window.__frames.splice(0, 10000);
                    requestAnimationFrame(loop); })(performance.now());
"""


def fps_stats(frames: list[float]) -> dict:
    """Frame times (ms timestamps) -> mean fps, 5th-percentile fps (1 / p95 frame time) and the worst frame."""
    if len(frames) < 3:
        return {"fps_mean": None, "fps_p5": None, "worst_frame_ms": None}
    dt = sorted(b - a for a, b in zip(frames, frames[1:]))
    p95 = dt[min(len(dt) - 1, int(0.95 * len(dt)))]
    return {"fps_mean": round(1000.0 * (len(frames) - 1) / (frames[-1] - frames[0]), 1),
            "fps_p5": round(1000.0 / p95, 1) if p95 > 0 else None, "worst_frame_ms": round(dt[-1], 1)}


def close_gate(pg) -> str | None:
    """An open operator gate takes every key until the operator decides (console3.js), so the next beat's D + digit
    would be swallowed. Decide as the presenter does: Esc (defer, review step), then Enter (confirm) once the confirm
    delay has passed. Returns None (no gate), "recorded" (confirm 200, in the audit log) or "not_recorded"."""
    if not pg.evaluate("!!(window.__scenescout && window.__scenescout.gateOpen)"):
        return None
    pg.keyboard.press("Escape")
    try:
        pg.wait_for_function("(() => { const g = window.__scenescout.gate; return !!(g && g.pending && g.pending.token); })()",
                             timeout=10000)
        pg.wait_for_timeout(1300)                                 # SCENESCOUT_GATE_DELAY_S (1.0 s) plus margin
        pg.keyboard.press("Enter")
        pg.wait_for_function("window.__scenescout.gateOpen === false", timeout=10000)
    except Exception:  # noqa: BLE001
        return "not_recorded"
    txt = pg.evaluate("(document.getElementById('gRes') || {}).innerText || ''")
    return "recorded" if "written to the audit log" in txt else "not_recorded"


def browser_runs(gw: str, demo: list[dict], beats: list[int], repeats: int, timeout_s: float, shots: Path | None):
    """Drive the console like the presenter (D, then the beat digit); record events and frames per run."""
    from playwright.sync_api import sync_playwright

    rows = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        cerr: list[str] = []
        pg.on("console", lambda m: cerr.append(m.text) if m.type == "error" else None)
        pg.add_init_script(EV_HOOK)
        t_nav = time.perf_counter()
        pg.goto(f"{gw}/")
        pg.wait_for_function("document.getElementById('curtain').classList.contains('gone')", polling="raf",
                             timeout=120000)
        boot = {"interactive_s": round(time.perf_counter() - t_nav, 2),
                "boot_ms": pg.evaluate("window.__scenescout && window.__scenescout.bootMs")}
        pg.wait_for_function("document.getElementById('curtain').hidden === true", timeout=60000)
        pg.wait_for_timeout(1500)
        pg.evaluate("window.__frames = []")
        pg.wait_for_timeout(3000)
        streaming = fps_stats(pg.evaluate("window.__frames"))      # remaining groups + vegetation still streaming
        t_all = time.perf_counter()
        pg.evaluate("(window.__scenescout.W && window.__scenescout.W.ready) ? window.__scenescout.W.ready.then(() => true) : true")
        boot["world_all_s"] = round(time.perf_counter() - t_nav, 2)
        boot["waited_for_rest_s"] = round(time.perf_counter() - t_all, 2)
        pg.wait_for_timeout(1500)
        pg.evaluate("window.__frames = []")
        pg.wait_for_timeout(3000)
        idle = fps_stats(pg.evaluate("window.__frames"))
        idle["while_streaming"] = streaming
        for i in beats:
            if i >= 9:
                raise ValueError("the console's demo menu takes digits 1-9")
            for k in range(repeats):
                pg.evaluate("window.__ev = []; window.__frames = []")
                n_err = len(cerr)
                t0 = pg.evaluate("performance.now()")
                pg.keyboard.press("Escape")
                pg.keyboard.press("d")
                pg.keyboard.press(str(i + 1))
                try:
                    pg.wait_for_function("window.__scenescout.finished === true && window.__ev.length > 0",
                                         timeout=timeout_s * 1000)
                    timed_out = False
                except Exception:  # noqa: BLE001
                    timed_out = True
                pg.wait_for_timeout(1500)                               # answer paint and camera flight
                gate = close_gate(pg)
                evs = pg.evaluate("window.__ev")
                run = summarize_run([(x["t"] / 1000.0, x["e"]) for x in evs if x["e"].get("type") != "_open"],
                                    t0 / 1000.0)
                run.update(fps_stats(pg.evaluate("window.__frames")))
                run.update(beat=i + 1, rep=k + 1, task_id=demo[i]["task_id"], timed_out=timed_out, gate=gate,
                           console_errors=cerr[n_err:][:5],
                           verdict=pg.evaluate("(document.getElementById('verdict')||{}).innerText || ''")[:200])
                if shots is not None and k == 0:
                    shots.mkdir(parents=True, exist_ok=True)
                    pg.screenshot(path=str(shots / f"beat{i + 1}.png"))
                rows.append(run)
                print(_line(run), flush=True)
                pg.wait_for_timeout(1000)
        b.close()
    return rows, {"boot": boot, "idle": idle}


def _line(r: dict) -> str:
    f = (lambda x: "  -  " if x is None else f"{x:5.1f}")
    fps = f"  fps {r['fps_mean']}/{r['fps_p5']} (mean/p5)" if "fps_mean" in r else ""
    return (f"beat {r['beat']} #{r['rep']}  answer {f(r.get('t_answer_s'))} s  done {f(r.get('t_done_s'))} s  "
            f"{r.get('status', 'none'):8s} score {r.get('score')}  turns {r.get('turns')}  "
            f"errors {r.get('errors') or '-'}{fps}")


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 2)


def live(a) -> int:
    import httpx

    gw = a.gateway.rstrip("/")
    with httpx.Client() as c:
        meta = c.get(f"{gw}/api/meta", timeout=30).json()
        health = c.get(f"{gw}/api/health", timeout=30).json()
    demo = meta.get("demo") or []
    if not demo:
        raise SystemExit("the gateway lists no demo questions")
    beats = parse_beats(a.beats, len(demo))
    rows_local = {}
    f = DEMO_FILES.get(meta.get("demo_set") or "")
    if f is not None and f.exists():
        rows_local = {r["task_id"]: r for r in json.loads(f.read_text())}
    if a.mode == "model" and not (health.get("llm") or {}).get("ok"):
        raise SystemExit(f"model mode needs vLLM; gateway health: {health}")
    print(f"gateway {gw}  demo set {meta.get('demo_set')}  model {meta.get('model')}  mode {a.mode}  "
          f"beats {[i + 1 for i in beats]} x {a.repeats}{'  (browser, fps)' if a.fps else ''}", flush=True)
    extra = {}
    if a.fps:
        if a.mode != "model":
            raise SystemExit("--fps drives the console, which asks the model; use --mode model")
        rows, extra = browser_runs(gw, demo, beats, a.repeats, a.timeout_s, Path(a.shots) if a.shots else None)
        for r in rows:
            r["score"] = offline_score(rows_local.get(r["task_id"]), r.get("answer"))
    else:
        rows = []
        with httpx.Client() as c:
            for i in beats:
                for k in range(a.repeats):
                    r = http_run(c, gw, demo[i], a.mode, a.timeout_s)
                    r.update(beat=i + 1, rep=k + 1, task_id=demo[i]["task_id"])
                    r["score"] = offline_score(rows_local.get(demo[i]["task_id"]), r.get("answer"))
                    rows.append(r)
                    print(_line(r), flush=True)
    per_beat = {}
    for i in beats:
        rs = [r for r in rows if r["beat"] == i + 1]
        ok = [r for r in rs if r.get("complete") and not r.get("errors") and r.get("status") == "verified"
              and not r.get("timed_out") and not r.get("console_errors") and r.get("gate") != "not_recorded"]
        per_beat[i + 1] = {
            "task_id": demo[i]["task_id"], "question": demo[i]["question"], "runs": len(rs), "clean": len(ok),
            "answer_p50_s": pct([r.get("t_answer_s") for r in rs], 0.5),
            "answer_max_s": pct([r.get("t_answer_s") for r in rs], 1.0),
            "first_tool_p50_s": pct([r.get("t_first_tool_s") for r in rs], 0.5),
            "score_mean": round(statistics.mean(s), 3) if (s := [r["score"] for r in rs if r.get("score") is not None])
            else None,
            "errors": sorted({e for r in rs for e in (r.get("errors") or [])}),
            "flags": sorted({e for r in rs for e in (r.get("flags") or [])}),
            "fps_p5_min": min((r["fps_p5"] for r in rs if r.get("fps_p5")), default=None),
            "fps_mean_min": min((r["fps_mean"] for r in rs if r.get("fps_mean")), default=None),
            "console_errors": sum(len(r.get("console_errors") or []) for r in rs),
            "gates_recorded": sum(r.get("gate") == "recorded" for r in rs)}
    allr = [r.get("t_answer_s") for r in rows]
    summary = {"gateway": gw, "demo_set": meta.get("demo_set"), "model": meta.get("model"), "mode": a.mode,
               "repeats": a.repeats, "runs": len(rows),
               "clean_runs": sum(v["clean"] for v in per_beat.values()),
               "answer_p50_s": pct(allr, 0.5), "answer_p95_s": pct(allr, 0.95), "answer_max_s": pct(allr, 1.0),
               "per_beat": per_beat, **extra}
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "per_beat"}, indent=1))
    for b, v in per_beat.items():
        print(f"  beat {b}: {v['clean']}/{v['runs']} clean  answer p50 {v['answer_p50_s']} s max {v['answer_max_s']} s  "
              f"score {v['score_mean']}  errors {v['errors'] or '-'}  flags {v['flags'] or '-'}  "
              f"fps p5 min {v['fps_p5_min']}  fps mean min {v['fps_mean_min']}  console errors {v['console_errors']}  "
              f"gates recorded {v['gates_recorded']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1, default=str))
    print("->", out)
    return 0 if summary["clean_runs"] == summary["runs"] else 1


# ---------------------------------------------------------------------------------------------- paraphrase
def rewordings(client, row: dict, n: int, rng: random.Random) -> list[str]:
    from scripts.paraphrase import STYLES, ok, protected

    must = protected(row)
    outs: list[str] = []
    for _ in range(3):
        prompt = (f"Rewrite this question from a counter-drone operator at a port, {rng.choice(STYLES)}. Keep the meaning "
                  f"exactly. Copy these exactly as written: {', '.join(must) if must else '(nothing special)'}. "
                  f"Reply with the rewritten question only, one line.\n\nQuestion: {row['question']}")
        r = client.chat.completions.create(model="scenescout", messages=[{"role": "user", "content": prompt}],
                                           temperature=0.9, max_tokens=120, n=n + 2)
        for c in r.choices:
            t = (c.message.content or "").strip().strip('"').strip()
            if ok(row, t) and t != row["question"] and t not in outs:
                outs.append(t)
        if len(outs) >= n:
            break
    return outs[:n]


_LOCAL = __import__("threading").local()


def score(row: dict, question: str, model: str) -> dict:
    from harness.env import AkshaEnv
    from serve.agent import run_task
    from tasks.base import Task

    r = dict(row)
    r["question"] = question
    t = Task.from_row(r)
    env = getattr(_LOCAL, "env", None)                  # one environment per worker thread (RAM)
    if env is None or getattr(_LOCAL, "model", None) != model:
        env = _LOCAL.env = AkshaEnv(REPO / t.site, model_id=model)
        _LOCAL.model = model
    for _ in run_task(env, t, model=model):
        pass
    return {"score": round(float(getattr(env, "answer_score", 0.0) or 0.0), 3) if env.done else 0.0,
            "end": getattr(env, "end_reason", None), "turns": env.turn}


def paraphrase(a) -> int:
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="local", timeout=120)
    rows = json.loads(DEMO_FILES[a.obs].read_text())
    rng = random.Random(5)
    cache = REPO / "runs" / "eval" / ("demo_stress_questions.json" if a.obs == "v1" else f"demo_stress_questions_{a.obs}.json")
    if cache.exists():
        qs = json.loads(cache.read_text())
    else:
        qs = {r["task_id"]: [r["question"]] + rewordings(client, r, a.n, rng) for r in rows}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(qs, indent=1))
    out = {}
    for m in a.models.split(","):
        jobs = [(r, q) for r in rows for q in qs[r["task_id"]]]
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(lambda rq: score(rq[0], rq[1], m), jobs))
        per = {}
        for (r, q), s in zip(jobs, res):
            per.setdefault(r["task_id"], {"template": r["template"], "runs": []})["runs"].append({"q": q, **s})
        out[m] = per
        tot = [s["score"] for s in res]
        print(f"\n== {m}: mean {sum(tot) / len(tot):.3f} over {len(tot)} runs, "
              f"{sum(x >= 0.95 for x in tot)}/{len(tot)} fully correct")
        for tid, p in per.items():
            sc = [x["score"] for x in p["runs"]]
            print(f"  {p['template']:18s} as written {sc[0]:.2f} | reworded {' '.join(f'{x:.2f}' for x in sc[1:])}")
    name = "demo_stress.json" if a.obs == "v1" else f"demo_stress_{a.obs}.json"
    (REPO / "runs" / "eval" / name).write_text(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default=None, help="live mode: gateway base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--beats", default="1-6", help="1-based demo entries: '1-6', '2,4' or 'all'")
    ap.add_argument("--mode", choices=["model", "scripted"], default="model")
    ap.add_argument("--fps", action="store_true", help="drive the console in a browser and record fps")
    ap.add_argument("--shots", default=None, help="with --fps: directory for one screenshot per beat")
    ap.add_argument("--timeout-s", type=float, default=90.0)
    ap.add_argument("--out", default=str(REPO / "runs" / "eval" / "demo_stress_live.json"))
    ap.add_argument("--models", default="sft", help="paraphrase mode: served models")
    ap.add_argument("--n", type=int, default=3, help="paraphrase mode: rewordings per question")
    ap.add_argument("--obs", choices=sorted(DEMO_FILES), default="v1", help="paraphrase mode: demo set")
    a = ap.parse_args(argv)
    return live(a) if a.gateway else paraphrase(a)


if __name__ == "__main__":
    sys.exit(main())
