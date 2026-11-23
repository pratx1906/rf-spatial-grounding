"""Paraphrases must keep every ref, number, sensor id and named place verbatim, or be rejected."""

from scripts.paraphrase import ok, protected

T = {"question": "At t=380 s, sensor S3 reports 2 bearings for track trk:4773. Which one is direct?",
     "params": {"track": "trk:4773", "t": 380.0}}
Z = {"question": "Is any drone inside the TF-1 restricted zone at t=570 s?", "params": {"zone_phrase": "TF-1 restricted zone"}}


def test_protected_spans_cover_refs_numbers_sensors_and_places():
    assert {"trk:4773", "S3", "380", "2"} <= set(protected(T))
    assert "TF-1 restricted zone" in protected(Z)


def test_a_faithful_rewrite_passes_and_a_lossy_one_fails():
    assert ok(T, "S3 has 2 bearings on trk:4773 at t=380 s; which is the direct one?")
    assert not ok(T, "S3 has two bearings on trk:4773 at t=380 s; which is direct?")       # '2' lost
    assert not ok(Z, "Any drones in the tank farm zone at t=570 s?")                         # place renamed
    assert not ok(T, "line one\nline two trk:4773 S3 380 2")                                 # multi-line


J = {"question": "Would jamming 5800 MHz at 33 dBm from sensor S2 disrupt any of our own links or civil comms (threshold -80 dBm)?",
     "params": {}}


def test_an_echoed_fact_list_prefix_is_rejected():
    # the model sometimes echoes the "copy these exactly" list in front of the question (frozen v3p had 65 such rows)
    assert not ok(J, "S2, 5800, 33, 80. Would jamming 5800 MHz at 33 dBm from sensor S2 disrupt our links (threshold -80 dBm)?")
    assert not ok(T, "trk:4773, S3, 380, 2: S3 has 2 bearings on trk:4773 at t=380 s, which is direct?")
    assert ok(J, "S2 jamming at 5800 MHz, 33 dBm: would it hit our own links or civil comms above -80 dBm?")


B = {"question": "Which drones on 5800 MHz are inside the TF-6 restricted zone at t=70 s?",
     "params": {"zone_phrase": "TF-6 restricted zone"}}


def test_an_echoed_list_with_a_dash_separator_is_rejected():
    assert not ok(B, "5800, 6, 70, TF-6 restricted zone \u2014 which drones are inside?")
    assert not ok(B, "5800, 6, 70, TF-6 restricted zone - which drones are inside?")
    assert ok(B, "At t=70 s, which 5800 MHz drones sit inside the TF-6 restricted zone?")


E = {"question": "Which tracks entered the TF-3 restricted zone between t=240 s and t=300 s?",
     "params": {"zone_phrase": "TF-3 restricted zone", "t0": 240.0, "t1": 300.0}}


def test_a_rewording_that_drops_the_event_verb_is_rejected():
    # runs/s78/training_targets.md #9/#10: "entered" became "were within" (presence), which changed the key
    assert not ok(E, "Which tracks were within the TF-3 restricted zone from t=240 s to t=300 s?")
    assert not ok(E, "Tracks in TF-3 restricted zone between t=240 s and t=300 s?")
    assert ok(E, "Between t=240 s and t=300 s, which tracks came into the TF-3 restricted zone?")
    assert ok(E, "Which tracks entered the TF-3 restricted zone from t=240 s to t=300 s?")


def test_glue_word_fact_list_prefix_rejected():
    """Paper P10: fact lists with glue words in front of the question (v3p had these) and mangled refs."""
    from scripts.paraphrase import ok

    t = {"question": "Sensor S2 has 2 bearings on trk:2060 at t=360 s. Which is direct?", "params": {}}
    assert not ok(t, "S2, trk:2060, 2 bearings, t=360. Which is direct? Sensor S2 has 2 bearings on trk:2060 at t=360 s")
    t2 = {"question": "When was trk:5448 first detected?", "params": {}}
    assert not ok(t2, "When was trk:5448.5448 first detected?")
    assert ok(t2, "At what time did we first pick up trk:5448?")


def test_meaning_critical_words_kept():
    from scripts.paraphrase import ok

    t = {"question": "When was trk:5448 first detected?", "params": {}}
    assert not ok(t, "When was trk:5448 detected?")
    t = {"question": "Locate trk:1 at t=5 s with the bearings from S3 and S2 only.", "params": {}}
    assert not ok(t, "Locate trk:1 at t=5 s with the bearings from S3 and S2.")
    assert ok(t, "Using just S3 and S2, locate trk:1 at t=5 s.")


def test_rules_backend_rewords_and_passes_filter():
    import random

    from scripts.paraphrase import ok
    from scripts.paraphrase_rules import reword

    t = {"question": "Is trk:1234 a genuine drone?", "params": {}}
    out = reword(t["question"], random.Random(0))
    assert out and out != t["question"] and ok(t, out)
    assert reword("Something no rule covers.", random.Random(0)) is None
