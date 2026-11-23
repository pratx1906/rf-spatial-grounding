"""Rule-based operator-style noise must keep the facts (numbers, ids) while changing the surface form."""

import random
import re

from scripts.augment_text import noisy

Q = "At t=380 s, sensor S3 reports 2 bearings for track trk:4773 on 5800 MHz. Which one is direct?"


def digits(s):
    return sorted(re.findall(r"\d+(?:\.\d+)?", s))


def test_noise_changes_the_text_but_keeps_every_number_and_id():
    seen = set()
    for k in range(40):
        out = noisy(Q, random.Random(k))
        seen.add(out)
        assert "4773" in out and "380" in out and re.search(r"\b[sS]3\b", out)
        assert "5800" in out or "5.8" in out                     # band written as MHz or GHz
        rest = digits(out.replace("5.8", "5800"))
        assert rest == digits(Q), out                            # no number added, dropped or altered
    assert len(seen) >= 20                                       # genuinely varied


def test_noise_is_deterministic_for_a_seed():
    assert noisy(Q, random.Random(3)) == noisy(Q, random.Random(3))
