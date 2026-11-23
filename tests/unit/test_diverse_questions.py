"""SFT v1.4 diverse question generator (PLAN 2026-10-23 rf-convergence Task 7): the same intent in many wordings,
with and without height / threshold / units; documented defaults when a value is omitted; weak_low_alt held out."""

import random
import re

import pytest

from tasks.kandla import diverse as dv


def _row(template, area, question, params, key=None, payload=None, tid="k01X3-x-00001"):
    return {"task_id": tid, "area": area, "template": template, "question": question, "answer_key": key or {},
            "split": "train", "singular_reference": False, "clarification": None, "categories": [],
            "scenario_seed": 1, "site": "data/scenarios_v3/kandla_s1_X", "drone_payload": payload, "params": params}


# --- siting: values omitted -> documented defaults ----------------------------------------------------------------
def test_omitted_values_take_the_documented_defaults_and_are_recorded_as_assumptions():
    rng = random.Random(0)
    for _ in range(50):
        q, p = dv.siting_question("weak_df", "2.4 GHz", "hero zone", None, None, "train", rng)
        assert p["threshold"] == dv.DEFAULTS["df_err_p90_deg"] == 10.0
        assert p["height_m"] == dv.DEFAULTS["height_m"] == 40.0
        assert not re.search(r"\d", q.replace("2.4", "")), q          # no number other than the band in the text
        assert any("10 deg" in s for s in p["assumed"]) and any("40 m" in s for s in p["assumed"])
        q, p = dv.siting_question("weak_detect", "5.8 GHz", "hero zone", None, None, "train", rng)
        assert p["threshold"] == 0.5 and p["height_m"] == 40.0 and p["metric"] == "detect_prob"


def test_given_values_are_rendered_in_some_unit_and_kept_in_params():
    rng = random.Random(1)
    seen_units = set()
    for _ in range(200):
        q, p = dv.siting_question("weak_df", "900 MHz", "jetty zone", 80.0, 15.0, "train", rng)
        assert p["threshold"] == 15.0 and p["height_m"] == 80.0 and p["assumed"] == []
        assert "15" in q and ("80" in q or "262 ft" in q), q
        seen_units |= {u for u in ("ft", "metres", "°", "degrees", " deg", " m") if u in q}
    assert {"ft", "metres", "°", "degrees"} <= seen_units


def test_feet_convert_back_to_a_cache_height():
    assert dv.fmt_height(40.0, "ft") == "131 ft" and dv.fmt_height(5.0, "ft") == "16 ft"
    assert dv.fmt_height(80.0, "ft") == "262 ft"
    assert all(abs(round(int(dv.fmt_height(h, "ft").split()[0]) * 0.3048) - h) < 1 for h in dv.HEIGHTS)


def test_weak_low_alt_stays_held_out():
    assert 15.0 not in dv.HEIGHTS                                # the held-out family asks at 15 m
    rng = random.Random(2)
    for split in ("train", "val"):
        for fam in ("weak_df", "weak_detect"):
            for h in (None,) + dv.HEIGHTS:
                for _ in range(30):
                    q, _ = dv.siting_question(fam, "2.4 GHz", "hero zone", h, None, split, rng)
                    assert not dv.is_low_alt_wording(q), q
    assert dv.is_low_alt_wording("Show the 2.4 GHz DF blind spots in the hero zone for low drones at 15 m.")
    assert "weak_low_alt" not in dv.WORDINGS


def test_train_and_val_wordings_are_disjoint():
    for fam, bank in dv.WORDINGS.items():
        assert bank["train"] and bank["val"], fam
        assert not set(bank["train"]) & set(bank["val"]), fam


# --- rewordings of existing tasks (key unchanged) -----------------------------------------------------------------
def test_drone_search_rewordings_keep_the_budget_and_track():
    pay = {"budget_moves": 3, "target_track": "trk:6387"}
    r = _row("ds_loiter", "drone_search", "Fly the DF drone to refine the position of trk:6387. You have 3 moves.",
             {"track": "trk:6387"}, {"position": {"x_m": 1, "y_m": 2}}, pay)
    qs = {dv.reword(r, "train", random.Random(s))["question"] for s in range(80)}
    assert len(qs) >= 6
    assert all("trk:6387" in q and "3" in q for q in qs)
    assert any("allotted" in q for q in qs) and any(q.startswith("DF drone to trk:6387") for q in qs)
    out = dv.reword(r, "train", random.Random(0))
    assert out["answer_key"] == r["answer_key"] and out["drone_payload"] == pay and out["params"] == r["params"]
    assert out["diverse_of"] == r["task_id"] and out["task_id"] != r["task_id"] and "diverse" in out["categories"]


def test_default_budget_may_be_omitted_but_a_short_budget_never_is():
    r4 = _row("ds_hover", "drone_search", "q", {"track": "trk:1"}, payload={"budget_moves": 4})
    r2 = _row("ds_hover_budget2", "drone_search", "q", {"track": "trk:1"}, payload={"budget_moves": 2})
    assert any("4" not in dv.reword(r4, "train", random.Random(s))["question"].replace("trk:1", "")
               for s in range(60))
    assert all("2" in dv.reword(r2, "train", random.Random(s))["question"] for s in range(60))


def test_collateral_rewordings_are_hypothetical_or_passive_never_commands():
    r = _row("jam_fratricide", "collateral", "q", {"effector": "sensor:S2", "band": "ism_5800", "eirp_dbm": 37.0,
                                                   "threshold_dbm": -90.0, "kinds": ["own_sensor"]})
    qs = {dv.reword(r, "train", random.Random(s))["question"] for s in range(60)}
    assert any("exposed to" in q for q in qs)
    for q in qs:
        assert "S2" in q and "37" in q and q.endswith("?"), q
        assert not re.match(r"(?i)^(jam|switch on|disable)\b", q), q    # an order would be an actuation (abstain)
    r2 = _row("jam_links", "collateral", "q", {"effector": "sensor:S1", "band": "ism_2400", "eirp_dbm": 33.0,
                                               "threshold_dbm": -80.0, "kinds": ["friendly_emitter", "civil_comms"]})
    assert all("-80" in dv.reword(r2, "train", random.Random(s))["question"] for s in range(30))


def test_los_rewordings_keep_yes_means_clear_and_contrast_with_direct_path():
    r = _row("los_check", "operator_query", "q", {"sensor": "S2", "track": "trk:2263", "t": 260.0},
             {"boolean": True})
    qs = {dv.reword(r, "train", random.Random(s))["question"] for s in range(60)}
    assert any("in the way" in q for q in qs)
    for q in qs:
        assert "S2" in q and "trk:2263" in q and "260" in q
        assert not re.search(r"(?i)\bblocked\?|obstructed\?$", q), q     # inverted polarity is never asked
        assert "bearing" not in q and "reflection" not in q
    d = _row("direct_path", "df_ambiguity", "q", {"sensor": "S2", "track": "trk:5433", "t": 470.0},
             {"all_hyp_refs": ["hyp:a@470", "hyp:b@470"]})
    dq = {dv.reword(d, "train", random.Random(s))["question"] for s in range(60)}
    assert all("2" in q and "S2" in q and "trk:5433" in q and "470" in q for q in dq)
    assert all(re.search(r"(?i)direct|true bearing", q) for q in dq)


def test_entered_between_keeps_an_entry_verb():
    r = _row("entered_between", "replay", "q", {"zone_phrase": "TF-3 restricted zone", "t0": 240.0, "t1": 300.0,
                                                "type": "zone_enter"})
    for s in range(60):
        q = dv.reword(r, "train", random.Random(s))["question"]
        assert re.search(r"(?i)\b(entered|came into|got into|crossed into|made an entry|entry)\b", q), q
        assert "TF-3 restricted zone" in q and "240" in q and "300" in q


def test_unsupported_and_held_out_families_are_not_reworded():
    assert dv.reword(_row("weak_low_alt", "siting", "q", {}), "train", random.Random(0)) is None
    assert dv.reword(_row("near_object", "operator_query", "q", {}), "train", random.Random(0)) is None
    assert dv.reword(_row("zone_incursion", "operator_query", "q", {}), "train", random.Random(0)) is None


def test_no_generated_family_is_held_out():
    from scripts.make_tasks import HELD_OUT

    assert not set(dv.WORDINGS) & HELD_OUT
    assert not set(dv.SITING_FAMILIES) & HELD_OUT


@pytest.mark.parametrize("split", ["train", "val"])
def test_every_template_formats(split):
    rng = random.Random(3)
    rows = {
        "ds_hover": _row("ds_hover", "drone_search", "q", {"track": "trk:9"}, payload={"budget_moves": 4}),
        "los_check": _row("los_check", "operator_query", "q", {"sensor": "S1", "track": "trk:9", "t": 10.0}),
        "direct_path": _row("direct_path", "df_ambiguity", "q", {"sensor": "S1", "track": "trk:9", "t": 10.0},
                            {"all_hyp_refs": ["a", "b", "c"]}),
        "entered_between": _row("entered_between", "replay", "q", {"zone_phrase": "jetty zone", "t0": 0.0,
                                                                   "t1": 60.0}),
    }
    for fam in ("jam_all", "jam_protected", "jam_links", "jam_aviation", "jam_fratricide"):
        rows[fam] = _row(fam, "collateral", "q", {"effector": "sensor:S3", "band": "sub_ghz_868_915",
                                                  "eirp_dbm": 27.0, "threshold_dbm": -90.0})
    for fam, r in rows.items():
        for _ in range(40):
            q = dv.reword({**r, "template": fam}, split, rng)["question"]
            assert "{" not in q and "}" not in q and "  " not in q, (fam, q)
            assert q[0].isupper() or q.startswith("trk:"), (fam, q)


def test_no_doubled_articles_in_rewordings():
    e = _row("entered_between", "replay", "q", {"zone_phrase": "the creek", "t0": 0.0, "t1": 60.0})
    j = _row("jam_aviation", "collateral", "q", {"effector": "sensor:S1", "band": "ism_5800", "eirp_dbm": 40.0,
                                                 "threshold_dbm": -90.0})
    for s in range(60):
        for r in (e, j):
            for split in ("train", "val"):
                q = dv.reword(r, split, random.Random(s))["question"]
                assert not re.search(r"(?i)\bthe the\b|dBm the\b", q), q
