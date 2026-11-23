"""Operator-style surface noise for training questions: how people actually type into a console under pressure.

    from scripts.augment_text import noisy;  noisy(question, random.Random(seed)) -> str

Every number and identifier survives (track and sensor ids, times, bands, thresholds); only the surface changes:
lowercase, missing question mark, loose refs ("track 4773", "trk 4773", "s3"), time written several ways, band
units swapped (5800 MHz <-> 5.8 GHz), a short prefix, and one adjacent-letter typo in an ordinary word. The answer
key and the expert trajectory stay valid because the meaning is unchanged (tools accept the loose ref forms and the
expert's calls use canonical refs, which is what the model should learn to emit).
"""

from __future__ import annotations

import random
import re

PREFIX = ["quick one: ", "ops: ", "pls check - ", "need this now: ", "", "", ""]
BAND = [(r"\b5800 MHz\b", "5.8 GHz"), (r"\b2400 MHz\b", "2.4 GHz"), (r"\b5\.8 GHz\b", "5800 MHz"), (r"\b2\.4 GHz\b", "2400 MHz")]


def _refs(q: str, rng: random.Random) -> str:
    def trk(m):
        return rng.choice([m.group(0), f"track {m.group(1)}", f"trk {m.group(1)}", f"trk:{m.group(1)}"])
    q = re.sub(r"\btrk:(\d+)\b", trk, q)
    q = re.sub(r"\btrack track (\d+)", r"track \1", q)
    q = re.sub(r"\bsensor (S[1-5])\b", lambda m: rng.choice([m.group(0), m.group(1), m.group(1).lower()]), q)
    return q


def _time(q: str, rng: random.Random) -> str:
    def t(m):
        n = m.group(1)
        return rng.choice([f"t={n} s", f"t={n}s", f"at {n} s", f"time {n}", f"t {n}s"])
    q = re.sub(r"\bat t=(\d+(?:\.\d+)?) ?s\b", lambda m: rng.choice([f"at t={m.group(1)} s", f"at {m.group(1)}s", f"at time {m.group(1)}"]), q)
    return re.sub(r"(?<!at )\bt=(\d+(?:\.\d+)?) ?s\b", t, q)


def _band(q: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        for pat, rep in BAND:
            if re.search(pat, q):
                return re.sub(pat, rep, q, count=1)
    return q


def _typo(q: str, rng: random.Random) -> str:
    words = [(m.start(), m.group(0)) for m in re.finditer(r"\b[a-z]{5,}\b", q)]
    if not words:
        return q
    i, w = rng.choice(words)
    k = rng.randrange(1, len(w) - 2)
    w2 = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return q[:i] + w2 + q[i + len(w):]


def noisy(question: str, rng: random.Random) -> str:
    q = _refs(question, rng)
    q = _time(q, rng)
    q = _band(q, rng)
    if rng.random() < 0.5:
        q = _typo(q, rng)
    if rng.random() < 0.45:
        q = q.lower()
    if rng.random() < 0.4:
        q = q.rstrip("?").rstrip()
    p = rng.choice(PREFIX)
    return (p + (q[0].lower() + q[1:] if p else q)).strip()
