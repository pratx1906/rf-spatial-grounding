"""Multipath web layer (scripts/multipath_layer.py): per-cell scoring, hot-spot clusters and outlines."""

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("multipath_layer", REPO / "scripts" / "multipath_layer.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def paths(amps, azs, ptypes):
    """One cell with the given path amplitudes / azimuths / ptypes, padded to K=6 with empty paths."""
    k = 6
    a = np.zeros(k, "f4"); az = np.full(k, np.nan, "f4"); pt = np.full(k, 255, "u1")
    a[:len(amps)], az[:len(azs)], pt[:len(ptypes)] = amps, azs, ptypes
    return a, np.zeros(k, "f4"), az, pt


def test_single_direct_path_scores_zero():
    m = mp.cell_metrics(*paths([1.0], [40.0], [0]))
    assert m["n_echo"] == 0 and m["not_direct"] == 0
    assert abs(float(m["spread_deg"])) < 1e-3 and abs(float(m["score"])) < 1e-3


def test_echoes_counted_within_10_db_only():
    # 1.0 (0 dB), 0.5 (-6 dB), 0.2 (-14 dB): one echo within 10 dB
    m = mp.cell_metrics(*paths([1.0, 0.5, 0.2], [40.0, 40.0, 40.0], [0, 1, 1]))
    assert m["n_echo"] == 1


def test_strongest_reflection_and_wide_spread_score_high():
    m = mp.cell_metrics(*paths([1.0, 0.9, 0.9, 0.9, 0.9], [0.0, 90.0, 180.0, 270.0, 45.0], [1, 0, 1, 8, 1]))
    assert m["not_direct"] == 1 and m["n_echo"] == 4
    assert float(m["spread_deg"]) > 30.0
    assert abs(float(m["score"]) - 1.0) < 1e-6


def test_no_path_is_nan_and_undetectable():
    m = mp.cell_metrics(*paths([], [], []))
    assert np.isnan(m["score"]) and np.isneginf(m["p0_db"])


def test_hot_clusters_pick_the_top_blob_and_ignore_specks():
    s = np.zeros((20, 20))
    s[2:5, 2:6] = 0.9          # 12-cell blob
    s[15, 15] = 0.95           # single speck (below min_cells)
    s[10:12, 10:12] = 0.8      # 4-cell blob
    s[0, :] = np.nan           # undetectable row is ignored
    cl = mp.hot_clusters(s, frac=0.02, min_hot=0.35, min_cells=3)
    assert [c["n"] for c in cl] == [12]          # top 2 % of 380 cells: the quantile (0.9) drops the 0.8 blob
    cl = mp.hot_clusters(s, frac=0.05, min_hot=0.35, min_cells=3)
    assert [c["n"] for c in cl] == [12, 4]       # quantile below min_hot: the floor decides; ranked by summed score
    assert mp.hot_clusters(np.full((4, 4), np.nan)) == []
    c = cl[0]
    assert abs(c["cy"] - 3.0) < 1e-9 and abs(c["cx"] - 3.5) < 1e-9


def test_outline_is_closed_rectangle_with_four_corners():
    m = np.zeros((6, 6), bool)
    m[1:3, 2:5] = True
    rings = mp.outline(m)
    assert len(rings) == 1
    assert sorted(rings[0]) == sorted([(2, 1), (5, 1), (5, 3), (2, 3)])


def test_outline_of_ring_shape_has_outer_and_hole_until_filled():
    m = np.ones((3, 3), bool)
    m[1, 1] = False
    assert len(mp.outline(m)) == 2
    assert mp.fill_holes(m).all() and len(mp.outline(mp.fill_holes(m))) == 1


def test_fill_holes_keeps_open_bays():
    m = np.zeros((5, 5), bool)
    m[1:4, 1] = m[1, 1:4] = m[3, 1:4] = True      # a C shape open to the east: the bay is outside, not a hole
    assert (mp.fill_holes(m) == m).all()


def test_smooth3_ignores_nan_and_keeps_nan_cells():
    s = np.array([[1.0, np.nan, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = mp.smooth3(s)
    assert np.isnan(out[0, 1])
    assert abs(out[0, 0] - 1.0 / 3.0) < 1e-12          # (1 + 0 + 0) over its 3 finite neighbours


def test_compass_words():
    assert mp.compass(0, 1) == "N" and mp.compass(1, 1) == "NE" and mp.compass(-1, 0) == "W"


def test_severity_words():
    assert mp.severity(0.9) == "heavy" and mp.severity(0.6) == "moderate" and mp.severity(0.4) == "light"


# ------------------------------------------------------------------ echo map inside DF answers (console3.js ECHO)
UI = REPO / "ui" / "kandla"


def _console():
    return (UI / "console3.js").read_text(encoding="utf-8")


def test_echo_map_files_cover_every_answer_pick():
    """Every band and echo height the answer can pick has a built file with all five sensors."""
    import json
    import re
    heights = [float(x) for x in re.search(r"ECHO_HEIGHTS = \[([\d., ]+)\]", _console()).group(1).split(",")]
    idx = json.loads((UI / "data" / "multipath" / "index.json").read_text())
    assert sorted(heights) == sorted(idx["heights_m"])
    for b in idx["bands"]:
        for h in heights:
            f = next(x["file"] for x in idx["files"] if x["band"] == b["id"] and x["height_m"] == h)
            d = json.loads((UI / "data" / "multipath" / f).read_text())
            assert {s["sensor"] for s in d["sensors"]} >= {"S1", "S2", "S3", "S4", "S5"}


def test_echo_map_wired_into_answers():
    js = _console()
    assert "paintAnswer(a, V); showEcho(a);" in js                    # verified, non-abstain answers only
    assert "function clearOverlays() {\n  hideEcho();" in js         # fades out when the answer is cleared
    assert 'if (n === "analysis" || n === "echo") echoRefresh();' in js
    assert 'data-layer="echo"' in (UI / "index.html").read_text(encoding="utf-8")
    assert "echo:" in (UI / "marks.js").read_text(encoding="utf-8")


def test_echo_pick_in_node():
    """echoPick (sensor of the rejected bearing, band of the work, height nearest the track), run in node."""
    import json
    import re
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not on PATH")
    js = _console()
    body = js[js.index("const ECHO = {"):js.index("const echoAllowed")]
    body = re.sub(r"^const echoColour[\s\S]*?^};\n", "", body, flags=re.M)
    prog = """
const S = {recs: new Map(), W: {simT: 0}};
const baseRef = (r) => String(r).split("@")[0];
const trackPos = (tr, t) => [0, 0, tr.z];
S.recs.set("trk:1", {data: {ts: [0], z: 12, band: "ism_2400"}});
S.recs.set("trk:2", {data: {ts: [0], z: 30, band: "ism_5800"}});
""" + body + """
const out = [
  echoPick({intent: "df_ambiguity", chosen_hyp_refs: ["hyp:S2_1_0470_887@470"], rejected_hyp_refs: ["hyp:S5_1_0470_888@470"]},
           {track: "trk:1", fitBand: "sub_ghz_868_915"}),
  echoPick({intent: "df_ambiguity", chosen_hyp_refs: ["hyp:S3_2_0100_1@100"]}, {track: "trk:2"}),
  echoPick({intent: "operator_query", evidence_refs: ["trk:1"]}, {track: "trk:1", aoa: [{sensor: "sensor:S4", band: "ism_5800", z: 60}]}),
  echoPick({intent: "collateral", evidence_refs: []}, {}),
];
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", prog], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got[0] == {"sensor": "S5", "band": "sub_ghz_868_915", "h": 5}      # rejected bearing's sensor, fit band, 12 m -> 5 m
    assert got[1] == {"sensor": "S3", "band": "ism_5800", "h": 40}            # no rejection: chosen sensor; track band; 30 m -> 40 m
    assert got[2] == {"sensor": "S4", "band": "ism_5800", "h": 5}             # predict_aoa at a sensor in any answer
    assert got[3] is None
