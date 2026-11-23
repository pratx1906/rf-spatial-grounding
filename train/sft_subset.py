"""Balanced SFT subset (per-area caps, family round-robin) so a laptop epoch fits in hours, not days.

    python -m train.sft_subset --src data/sft/train.jsonl --out data/sft/train_bal.jsonl --total 2800
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# share of the subset per area: harder / rarer areas get more than their natural share
SHARE = {"operator_query": 0.22, "df_ambiguity": 0.24, "drone_search": 0.12, "replay": 0.14, "collateral": 0.10,
         "siting": 0.10, "unanswerable": 0.08}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sft/train.jsonl")
    ap.add_argument("--out", default="data/sft/train_bal.jsonl")
    ap.add_argument("--total", type=int, default=2800)
    a = ap.parse_args()
    rng = random.Random(7)
    fam = defaultdict(lambda: defaultdict(list))
    for line in (REPO / a.src).read_text().splitlines():
        r = json.loads(line)
        fam[r["area"]][r["template"]].append(line)
    out = []
    for area, share in SHARE.items():
        quota = int(a.total * share)
        pools = {f: rng.sample(v, len(v)) for f, v in fam[area].items()}
        got = []
        while len(got) < quota and any(pools.values()):
            for f in sorted(pools):
                if pools[f] and len(got) < quota:
                    got.append(pools[f].pop())
        out += got
        print(area, len(got), "of", sum(len(v) for v in fam[area].values()))
    rng.shuffle(out)
    (REPO / a.out).write_text("\n".join(out) + "\n")
    print("total", len(out))


if __name__ == "__main__":
    main()
