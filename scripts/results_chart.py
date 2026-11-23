"""Results chart and table for the README: per-area scores of each policy on frozen eval v3.

    python scripts/results_chart.py --summaries   # WSL: recompute summaries -> docs/media/results_v3.json, print table
    python scripts/results_chart.py               # any Python with matplotlib: draw docs/media/results_v3.png

Summaries are recomputed from the per-task rows with eval.run.summarise, so files written before a summary field
existed (e.g. `_generalisation`) are reported the same way as new ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SUMS = REPO / "docs" / "media" / "results_v3.json"

AREAS = ["operator_query", "df_ambiguity", "drone_search", "siting", "collateral", "replay", "unanswerable"]
LABELS = {"operator_query": "Operator\nquery", "df_ambiguity": "DF\nambiguity", "drone_search": "Drone\nsearch",
          "siting": "Siting", "collateral": "Collateral", "replay": "Replay", "unanswerable": "Unanswerable"}
POLICIES = [("Prompted Qwen3-4B", "prompted_v3_cap512.json", "#9aa5b1"),
            ("SceneScout SFT v1", "sft_v1_v3_cap512.json", "#0f8b8d"),
            ("Scripted expert", "expert_v3.json", "#f2a541")]


def summaries() -> None:
    from eval.run import summarise

    out = {label: summarise(json.loads((REPO / "runs" / "eval" / f).read_text())["rows"]) for label, f, _ in POLICIES}
    SUMS.parent.mkdir(parents=True, exist_ok=True)
    SUMS.write_text(json.dumps(out, indent=1))
    sums = [(label, out[label], c) for label, _, c in POLICIES]
    print("| Area | " + " | ".join(label for label, _, _ in sums) + " |")
    print("|---|" + "---|" * len(sums))
    for a in AREAS:
        print(f"| {a} | " + " | ".join(f"{s[a]['score']:.3f}" for _, s, _ in sums) + " |")
    print("| **macro** | " + " | ".join(f"**{s['_all']['macro_score']:.3f}**" for _, s, _ in sums) + " |")
    print("| seen families | " + " | ".join(f"{s['_generalisation']['seen_families']}" for _, s, _ in sums) + " |")
    print("| held-out families | " + " | ".join(f"{s['_generalisation']['held_out_families']}" for _, s, _ in sums) + " |")
    print("| no answer | " + " | ".join(f"{s['_all']['no_answer']}" for _, s, _ in sums) + " |")
    print("| latency p50 (s) | " + " | ".join(f"{s['_all']['latency_p50_s']}" for _, s, _ in sums) + " |")


def chart() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(SUMS.read_text())
    sums = [(label, data[label], c) for label, _, c in POLICIES]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    w = 0.26
    for i, (label, s, c) in enumerate(sums):
        xs = [k + (i - 1) * w for k in range(len(AREAS))]
        ax.bar(xs, [s[a]["score"] for a in AREAS], w, color=c, label=f"{label}  (macro {s['_all']['macro_score']:.2f})")
    ax.set_xticks(range(len(AREAS)), [LABELS[a] for a in AREAS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0-1)")
    ax.set_title("Frozen eval v3: 603 held-out-scenario questions, simulation only", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    out = REPO / "docs" / "media" / "results_v3.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("chart:", out.relative_to(REPO))


if __name__ == "__main__":
    summaries() if "--summaries" in sys.argv else chart()
