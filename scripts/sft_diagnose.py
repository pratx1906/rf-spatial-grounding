"""Why is the SFT loss near zero? Diagnose the training targets (CPU only).

    python scripts/sft_diagnose.py [--data data/sft/train_bal.jsonl] [--limit 1800]

Measures how predictable the assistant targets are:
- action skeletons: distinct sequences of tool names per episode (how many different "plans" exist);
- copy rate: share of argument values that appear verbatim in the question or earlier tool results
  (values the model can copy rather than infer);
- free choices: argument values that appear nowhere in the context (the genuinely hard tokens);
- phrasing diversity: distinct question templates per family (with numbers/refs masked).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from leaves(x)
    else:
        yield v


def mask(q: str) -> str:
    q = re.sub(r"(trk|hyp|evt|obj|zone|sensor|fp|ras|wp|nodecand|cand|fe):[\w\-@.]+", "<REF>", q)
    q = re.sub(r"\b(TF|J|S)-?\d+\b", "<ID>", q)
    return re.sub(r"\d+(\.\d+)?", "<N>", q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/train_bal.jsonl")
    ap.add_argument("--limit", type=int, default=1800)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (REPO / a.data).read_text().splitlines()[: a.limit]]
    skel = Counter()
    skel_by_family = defaultdict(Counter)
    phr = defaultdict(set)
    copied = novel = 0
    novel_kinds = Counter()
    turns = 0
    for r in rows:
        msgs = r["messages"]
        q = msgs[1]["content"]
        phr[r["template"]].add(mask(q))
        ctx = q
        names = []
        for m in msgs[2:]:
            if m["role"] == "tool":
                ctx += m["content"]
            elif m["role"] == "assistant":
                turns += 1
                for tc in m["tool_calls"]:
                    names.append(tc["function"]["name"])
                    for v in leaves(tc["function"]["arguments"]):
                        if isinstance(v, bool) or v is None:
                            continue
                        s = str(v)
                        if s in ctx or (isinstance(v, float) and f"{v:g}" in ctx):
                            copied += 1
                        else:
                            novel += 1
                            novel_kinds[tc["function"]["name"]] += 1
        skel[tuple(names)] += 1
        skel_by_family[r["template"]][tuple(names)] += 1
    print(f"episodes {len(rows)}, assistant turns {turns} ({turns / len(rows):.1f}/episode)")
    print(f"distinct tool-call skeletons: {len(skel)} (top 5 cover {sum(c for _, c in skel.most_common(5)) / len(rows):.0%})")
    fam_skel = [len(v) for v in skel_by_family.values()]
    print(f"families {len(skel_by_family)}; skeletons per family: median {sorted(fam_skel)[len(fam_skel) // 2]}, "
          f"max {max(fam_skel)}; families with exactly one plan: {sum(1 for x in fam_skel if x == 1)}")
    tot = copied + novel
    print(f"argument values: {tot}; copied from question/tool results {copied / tot:.0%}; not in context {novel / tot:.0%}")
    print("tools with most non-copied values:", novel_kinds.most_common(6))
    ph = sorted((len(v), k) for k, v in phr.items())
    print(f"question phrasings per family (numbers/refs masked): median {ph[len(ph) // 2][0]}, min {ph[0]}, max {ph[-1]}")


if __name__ == "__main__":
    main()
