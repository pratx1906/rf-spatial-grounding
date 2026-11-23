"""Write the curated demo task list (offline; reads truth via the task generator) to data/demo_tasks.json.

    python scripts/make_demo_tasks.py

The gateway loads this file for its example questions and the scripted fallback; it never reads truth itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.m1_tasks import generate  # noqa: E402


def main() -> None:
    tasks = generate(REPO / "data" / "stub_site", n=40, seed=0)
    pick, seen = [], {}
    for t in tasks:                      # a few of each family, positives first
        k = t.template
        if seen.get(k, 0) >= (3 if k == "zone_incursion" else 2):
            continue
        if k == "zone_incursion" and not t.answer_key["boolean"] and seen.get(k, 0) < 2:
            continue
        seen[k] = seen.get(k, 0) + 1
        pick.append(t.to_row())
    out = REPO / "data" / "demo_tasks.json"
    out.write_text(json.dumps(pick, indent=1))
    print(f"{len(pick)} demo tasks -> {out}")


if __name__ == "__main__":
    main()
