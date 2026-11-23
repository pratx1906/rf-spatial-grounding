"""Training watcher: progress table, health checks and a loss-curve PNG for an SFT run.

    python scripts/train_watch.py --run sft_v1 [--total 113] [--png]

Reads runs/sft/<run>/log.jsonl (only the latest run: entries after the last step-0 validation record), prints the
recent steps, smoothed loss, validation losses, ETA, and flags problems: NaN/inf loss, loss rising over the last
20 steps, validation loss rising twice in a row, no log line for > 10 minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def latest(rows: list[dict]) -> list[dict]:
    start = max((i for i, r in enumerate(rows) if r.get("step") == 0 and "val_loss" in r), default=0)
    return rows[start:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sft_v1")
    ap.add_argument("--total", type=int, default=113)
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    f = REPO / "runs" / "sft" / a.run / "log.jsonl"
    rows = latest([json.loads(x) for x in f.read_text().splitlines() if x.strip()]) if f.exists() else []
    steps = [r for r in rows if "loss" in r]
    vals = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    issues = []
    if not steps:
        print(f"{a.run}: no training step logged yet" + (f" (val@0 {vals[0][1]:.4f})" if vals else ""))
        return
    last = steps[-1]
    sm = [sum(s["loss"] for s in steps[max(0, i - 4): i + 1]) / len(steps[max(0, i - 4): i + 1]) for i in range(len(steps))]
    rate = last["elapsed_s"] / last["step"]
    eta = rate * (a.total - last["step"])
    age = time.time() - f.stat().st_mtime
    if any(not math.isfinite(s["loss"]) for s in steps):
        issues.append("non-finite loss")
    if len(sm) >= 25 and sm[-1] > sm[-21] * 1.05 and sm[-1] - sm[-21] > 0.02:   # relative AND absolute rise
        issues.append(f"loss rising over the last 20 steps ({sm[-21]:.3f} -> {sm[-1]:.3f})")
    if len(vals) >= 3 and vals[-1][1] > vals[-2][1] > vals[-3][1]:
        issues.append("validation loss rose twice in a row (overfitting?)")
    if age > 600:
        issues.append(f"no log line for {age / 60:.0f} min (stalled or finished)")
    print(f"{a.run}: step {last['step']}/{a.total} | loss {last['loss']:.4f} (smoothed {sm[-1]:.4f}, first {sm[0]:.4f}) | "
          f"val " + ", ".join(f"@{s}:{v:.4f}" for s, v in vals) + f" | lr {last['lr']:.2e} | vram {last['vram_gb']} GB | "
          f"{rate:.0f} s/step | ETA {eta / 60:.0f} min | " + ("OK" if not issues else "CHECK: " + "; ".join(issues)))
    if a.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
        ax.plot([s["step"] for s in steps], [s["loss"] for s in steps], color="#9fb4c0", lw=1, label="train loss")
        ax.plot([s["step"] for s in steps], sm, color="#1f6f8b", lw=2, label="train loss (5-step mean)")
        if vals:
            ax.plot([s for s, _ in vals], [v for _, v in vals], "o-", color="#d1495b", lw=2, label="validation loss")
        ax.set_xlabel("optimizer step (16 examples)")
        ax.set_ylabel("loss (assistant tokens)")
        ax.set_title(f"SceneScout {a.run}: step {last['step']}/{a.total}, ETA {eta / 60:.0f} min")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        out = REPO / "runs" / "sft" / a.run / "curve.png"
        fig.savefig(out)
        print("curve:", out)


if __name__ == "__main__":
    main()
