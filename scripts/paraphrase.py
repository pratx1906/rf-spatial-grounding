"""Paraphrase operator questions with the local model, keeping every fact verbatim (SPEC 8.1 paraphrase step).

    python scripts/paraphrase.py --src eval/frozen/v3/tasks.jsonl --out eval/frozen/v3p/tasks.jsonl --k 1
    python scripts/paraphrase.py --src data/tasks/train.jsonl --out data/tasks/train_para.jsonl --k 4 --limit 6000

Needs the vLLM server (python -m serve.llm). Each paraphrase must keep, verbatim: every ref (trk:/hyp:/evt:...),
every number, every sensor id (S1..S5), and the place/object names from the task's slots; otherwise it is rejected.
The answer key is unchanged (the question means the same thing), so keys and expert trajectories stay valid.
Provenance: each output row carries `paraphrase_of` (original task_id) and `paraphrase_group` (same as original).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STYLES = ["casual, like a busy operator typing quickly", "terse radio-call style", "formal written request",
          "a question from a shift supervisor", "rephrased with a different sentence structure",
          "slightly informal with a short preamble"]
REF = re.compile(r"\b(?:trk|hyp|evt|obj|zone|sensor|fp|wp|nodecand|cand|fe):[\w\-@.]*\w")   # never ends on punctuation
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
SENS = re.compile(r"\bS[1-5]\b")


def protected(task: dict) -> list[str]:
    q = task["question"]
    spans = REF.findall(q) + SENS.findall(q) + NUM.findall(q)
    for k in ("zone_phrase", "obj_phrase", "phrase"):
        v = (task.get("params") or {}).get(k)
        if v and v.lower() in q.lower():
            spans.append(v)
    return list(dict.fromkeys(spans))


def echoed_list(text: str, spans: list[str]) -> bool:
    """The model sometimes repeats the prompt's 'copy these exactly' list as a prefix ('S2, 5800, 33, 80. Would...').
    True when the first clause is nothing but two or more protected spans and separators."""
    head = re.split(r"[.:;?](?:\s|$)|\s[—–-]\s", text.strip(), maxsplit=1)[0]   # also 'list — question'
    rest = head
    for s in sorted(spans, key=len, reverse=True):
        rest = rest.replace(s, " ")
    if sum(s in head for s in spans) >= 2 and re.sub(r"[\s,\-]+", "", rest) == "":
        return True
    # paper P10 fix: a fact list with a few glue words ('S2, trk:2060, 2 bearings, t=360. Which is direct?',
    # 'Trk:0783, 0783, 290. Where ...', '150, jetty zone, count drone tracks at t=150 s.'): the first clause is two
    # or more short comma items and the first item carries a number, ref or sensor id
    items = [x.strip() for x in head.split(",")]
    short = [x for x in items if x and len(x.split()) <= 4]
    return (len(items) >= 2 and len(short) >= len(items) - 1
            and bool(re.fullmatch(r"(?i)(?:trk:\w+|S[1-5]|\d[\w.=\- ]*|t=\d+(?: s)?)", items[0])))


MANGLED_REF = re.compile(r"\b(?:trk|hyp|evt|obj|zone|sensor):[\w\-]*\w\.\d", re.I)   # 'trk:5448.5448'


# Event verbs carry the meaning of replay questions: "entered" (an entry event) is not "was within" (presence).
# A rewording must keep the verb or a same-meaning form (runs/s78/training_targets.md #9-#10).
EVENT_VERBS = {
    r"\benter(?:ed|s)?\b": r"\b(?:enter(?:ed|s|ing)?|came into|come into|got into|crossed into|entr(?:y|ies))\b",
    r"\bleft\b|\bexit(?:ed)?\b": r"\b(?:left|leave|exit(?:ed|s)?|departed)\b",
    r"\bstart(?:ed)? loitering\b": r"\b(?:start(?:ed)?|began|begin) (?:to )?loiter",
}


# Other meaning-critical words (paper P10): dropping one changes the question ("first" vs any time, "direct" vs any
# path, "only" these sensors, "nearest", "last", a negation). Each must survive, or a same-meaning form must appear.
CRITICAL_WORDS = {
    r"\bfirst\b": r"\b(?:first|earliest|initial(?:ly)?)\b",
    r"\blast\b": r"\b(?:last|latest|final(?:ly)?|most recent(?:ly)?)\b",
    r"\bdirect\b": r"\bdirect\b|\bline[- ]of[- ]sight\b",
    r"\breflections?\b": r"\b(?:reflect\w*|ghosts?|multipath|bounce\w*|indirect)\b",
    r"\bnearest\b|\bclosest\b": r"\b(?:nearest|closest)\b",
    r"\bonly\b": r"\b(?:only|just|alone|solely)\b",
    r"\bbetween\b": r"\bbetween\b|\bfrom\b.*\bto\b",
    r"\bnot\b|n't\b": r"\bnot\b|n't\b|\bno\b|\bwithout\b",
    r"\bown\b": r"\bown\b|\bour\b",
}


def keeps_event_verbs(question: str, text: str) -> bool:
    table = {**EVENT_VERBS, **CRITICAL_WORDS}
    return all(re.search(alt, text, re.I) for src, alt in table.items() if re.search(src, question, re.I))


def ok(task: dict, text: str) -> bool:
    if not text or len(text) > 400 or "\n" in text.strip():
        return False
    spans = protected(task)
    if echoed_list(text, spans) or MANGLED_REF.search(text):
        return False
    if not keeps_event_verbs(task["question"], text):
        return False
    low = text.lower()
    return all(s.lower() in low for s in spans)


def stratified(rows: list[dict], per_family: int, rng: random.Random) -> list[dict]:
    """Up to `per_family` rows of each template (rng order), in source order."""
    by: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["template"], []).append(i)
    pick = sorted(i for v in by.values() for i in rng.sample(v, min(per_family, len(v))))
    return [rows[i] for i in pick]


def rules_main(a, rows: list[dict], rng: random.Random) -> int:
    """CPU backend: rule-based rewording (scripts/paraphrase_rules.py), same acceptance filter as the LLM path."""
    sys.path.insert(0, str(REPO))
    from scripts.paraphrase_rules import reword

    out = Path(REPO / a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, no_rule, rejected = 0, 0, 0
    with open(out, "w") as f:
        for t in rows:
            if a.target and kept >= a.target:
                break
            txt = None
            for _ in range(4):                       # a few draws over the rule's alternatives / prefixes
                c = reword(t["question"], rng)
                if c is None:
                    break
                if ok(t, c) and c != t["question"]:
                    txt = c
                    break
            if txt is None:
                no_rule += reword(t["question"], rng) is None
                rejected += reword(t["question"], rng) is not None
                continue
            r = dict(t)
            r.update({"question": txt, "task_id": f"{t['task_id']}-p0", "paraphrase_of": t["task_id"],
                      "paraphrase_group": t["task_id"], "paraphrase_method": "rules_v1 (scripts/paraphrase_rules.py)",
                      "human_written": False, "categories": list(t.get("categories") or []) + ["paraphrase"]})
            f.write(json.dumps(r) + "\n")
            kept += 1
    print(json.dumps({"backend": "rules", "tasks": len(rows), "paraphrases": kept, "no_rule": no_rule,
                      "rejected_by_filter": rejected, "out": str(out)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["llm", "rules"], default="llm",
                    help="llm: local vLLM server (GPU); rules: CPU rule-based rewording (scripts/paraphrase_rules.py)")
    ap.add_argument("--per-family", type=int, default=0, help="stratify: at most this many source rows per template")
    ap.add_argument("--target", type=int, default=0, help="rules backend: stop after this many paraphrases")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (REPO / a.src).read_text().splitlines()]
    rng = random.Random(a.seed)
    if a.per_family:
        rows = stratified(rows, a.per_family, rng)
        rng.shuffle(rows)
    if a.limit:
        rng.shuffle(rows)
        rows = rows[: a.limit]
    if a.backend == "rules":
        return rules_main(a, rows, rng)
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="local", timeout=120)

    def one(t):
        must = protected(t)
        style = rng.choice(STYLES)
        prompt = (f"Rewrite this question from a counter-drone operator at a port, {style}. Keep the meaning exactly. "
                  f"Copy these exactly as written: {', '.join(must) if must else '(nothing special)'}. "
                  f"Reply with the rewritten question only, one line.\n\nQuestion: {t['question']}")
        try:
            r = client.chat.completions.create(model="scenescout", messages=[{"role": "user", "content": prompt}],
                                               temperature=0.9, max_tokens=120, n=a.k + 2)
        except Exception as e:  # noqa: BLE001
            return t, [], str(e)
        outs = []
        for c in r.choices:
            txt = (c.message.content or "").strip().strip('"').strip()
            if ok(t, txt) and txt != t["question"] and txt not in outs:
                outs.append(txt)
        return t, outs[: a.k], None

    out = Path(REPO / a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex, open(out, "w") as f:
        for t, ps, err in ex.map(one, rows):
            if not ps:
                rejected += 1
                continue
            for i, p in enumerate(ps):
                r = dict(t)
                r["question"] = p
                r["task_id"] = f"{t['task_id']}-p{i}"
                r["paraphrase_of"] = t["task_id"]
                r["paraphrase_group"] = t["task_id"]
                r["categories"] = list(t.get("categories") or []) + ["paraphrase"]
                f.write(json.dumps(r) + "\n")
                kept += 1
    print(json.dumps({"tasks": len(rows), "paraphrases": kept, "tasks_without_valid_paraphrase": rejected,
                      "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
