"""Mismatch worlds M1-M4 (plan 2026-10-28 T6, audit B1 / MINOR 2): hardware overrides deep-merge one key into the
world-side config; M3 is a use-time-only phase error; M4 makes the calibration drift act; mismatch builds refuse an
existing dir, pass world_variant=None, keep the nominal belief side, and record the realised per-node errors."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.obs_model import observe_v2 as ov
from sim.obs_model.observe_air7311 import HW_FILE

REPO = Path(__file__).resolve().parents[2]
MDIR = REPO / "configs" / "hardware" / "mismatch"
SIDS = ("S1", "S2", "S3", "S4", "S5")
EXPECT = {"M1": ("heading_datum", "bias_deg_1sigma", 3.0), "M2": ("heading_datum", "bias_deg_1sigma", 10.0),
          "M3": ("impairments", "use_phase_sigma_deg", 15.0), "M4": ("df_calibration", "drift_rel_db", -20.0)}


def _flat(d, pre=()):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, pre + (k,)) if isinstance(v, dict) else {pre + (k,): v})
    return out


def _ov(tag):
    return yaml.safe_load((MDIR / f"{tag}.yaml").read_text())


@pytest.mark.parametrize("tag", sorted(EXPECT))
def test_override_deep_merges_exactly_one_key(tag):
    base, merged = _flat(yaml.safe_load(HW_FILE.read_text())), _flat(ov.load_hw(_ov(tag)))
    sec, key, val = EXPECT[tag]
    diff = {k for k in set(base) | set(merged) if base.get(k, "absent") != merged.get(k, "absent")}
    assert diff == {(sec, key)} and merged[(sec, key)] == val
    assert ov.load_hw(None) == yaml.safe_load(HW_FILE.read_text())


def _old_cal_residuals(hw, seed, sid):        # the pre-T6 implementation, verbatim
    imp = hw.get("impairments", {})
    r = np.random.default_rng(ov.stable_seed(seed, "cal", sid))
    g = r.normal(0, imp.get("channel_gain_sigma_db", 0.0), 4)
    p = r.normal(0, imp.get("channel_phase_sigma_deg", 0.0), 4)
    return torch.tensor(g, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)


def test_nominal_residual_draw_unchanged_when_key_absent():
    hw = ov.load_hw()
    assert "use_phase_sigma_deg" not in hw["impairments"]
    for seed in (0, 15, 16, 51, 60):
        for sid in SIDS:
            a, b = ov.cal_residuals(hw, seed, sid), _old_cal_residuals(hw, seed, sid)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            z = ov.cal_residuals(ov.deep_merge(hw, {"impairments": {"use_phase_sigma_deg": 0.0}}), seed, sid)
            assert torch.equal(z[1], b[1])


def test_m3_perturbs_use_time_phase_only_never_the_cal_tones():
    hw, m3 = ov.load_hw(), ov.load_hw(_ov("M3"))
    d = []
    for seed in range(40):
        for sid in SIDS:
            (g0, p0), (g3, p3) = ov.cal_residuals(hw, seed, sid), ov.cal_residuals(m3, seed, sid)
            assert torch.equal(g0, g3)
            d.append((p3 - p0).numpy())
    assert abs(np.concatenate(d).std() - 15.0) < 1.5
    pos = ov.square_positions(0.05)
    for sid in ("S1", "S4"):
        a = ov.node_coupling_cal(hw, 15, sid, "ism_2400", pos, 2.44e9)
        b = ov.node_coupling_cal(m3, 15, sid, "ism_2400", pos, 2.44e9)
        assert np.array_equal(a, b)                    # C_hat cannot absorb the use-time error


def test_m4_drift_reaches_the_calibration():
    from sim.df.calibration import cal_config

    hw, m4 = ov.load_hw(), ov.load_hw(_ov("M4"))
    assert cal_config(m4)["drift_rel_db"] == -20.0 and cal_config(hw)["drift_rel_db"] == -40.0
    assert m4["df_calibration"]["coupling_cal"] is True
    pos = ov.square_positions(0.05)
    e0, e4 = [], []
    for seed in (15, 16):
        for sid in SIDS:
            c = ov.coupling(seed, sid)
            e0.append(np.linalg.norm(ov.node_coupling_cal(hw, seed, sid, "ism_2400", pos, 2.44e9) - c))
            e4.append(np.linalg.norm(ov.node_coupling_cal(m4, seed, sid, "ism_2400", pos, 2.44e9) - c))
    assert np.median(e4) > 3 * np.median(e0)         # the calibration no longer matches use-time coupling


def test_realised_biases_recorded_and_m2_is_m1_scaled():
    m1, m2 = ov.load_hw(_ov("M1")), ov.load_hw(_ov("M2"))
    for seed in (15, 16, 51):
        r1, r2 = ov.realised_hw_errors(m1, seed, SIDS), ov.realised_hw_errors(m2, seed, SIDS)
        for sid in SIDS:
            assert r1[sid]["heading_bias_deg"] == pytest.approx(ov.hw_errors(m1, seed, sid)[0], abs=1e-6)
            assert r2[sid]["heading_bias_deg"] == pytest.approx(r1[sid]["heading_bias_deg"] * 10 / 3, abs=1e-5)


def test_hw_variant_desc_records_tag_override_and_nodes():
    from envbuild.site_kandla import hw_variant_desc

    m2 = ov.load_hw(_ov("M2"))
    d = hw_variant_desc("M2", _ov("M2"), m2, 15, SIDS, ov.realised_hw_errors)
    assert d["id"] == "M2" and d["override"] == _ov("M2") and d["heading_bias_deg_1sigma"] == 10.0
    assert set(d["nodes"]) == set(SIDS)
    yaml.safe_load(yaml.safe_dump(d))                  # plain YAML types


# ---- scripts/build_scenarios.py (no GPU: write_site and the lock are stubbed) ----

@pytest.fixture
def bs(tmp_path, monkeypatch):
    import scripts.build_scenarios as m

    calls = []
    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    monkeypatch.setattr(m, "v2_dir", lambda s, v, version=None: tmp_path / (f"kandla_s{s}_X" + (f"_{v}" if v else "")))
    monkeypatch.setattr(m, "gpu_lock", lambda *a, **k: nullcontext())
    import envbuild.site_kandla as sk

    def fake_write_site(seed, out, **kw):
        calls.append((seed, Path(out), kw))
        return {"counts": {}, "version_hash": "x"}
    monkeypatch.setattr(sk, "write_site", fake_write_site)
    return m, calls, tmp_path


def test_mismatch_build_passes_none_world_variant_and_the_override(bs):
    m, calls, tmp = bs
    assert m.main(["--obs", "v2", "--seeds", "15-16", "--hw-override", str(MDIR / "M1.yaml"), "--tag", "M1"]) == 0
    assert [(s, o.name) for s, o, _ in calls] == [(15, "kandla_s15_X_M1"), (16, "kandla_s16_X_M1")]
    for _, _, kw in calls:
        assert kw["world_variant"] is None and kw["hw_override"] == _ov("M1") and kw["variant_tag"] == "M1"


def test_mismatch_build_refuses_an_existing_dir(bs):
    m, calls, tmp = bs
    (tmp / "kandla_s16_X_M2").mkdir()
    with pytest.raises(SystemExit, match="exist"):
        m.main(["--obs", "v2", "--seeds", "15-16", "--hw-override", str(MDIR / "M2.yaml"), "--tag", "M2"])
    assert calls == [] and not (tmp / "kandla_s15_X_M2").exists()       # refused before any work
    with pytest.raises(SystemExit):
        m.main(["--obs", "v2", "--seeds", "16", "--hw-override", str(MDIR / "M2.yaml"), "--tag", "M2",
                "--allow-existing"])


@pytest.mark.parametrize("args", [["--tag", "M1"], ["--hw-override", str(MDIR / "M1.yaml")],
                                  ["--tag", "M1", "--hw-override", str(MDIR / "M1.yaml"), "--world-variant", "W4"],
                                  ["--tag", "W9", "--hw-override", str(MDIR / "M1.yaml")]])
def test_mismatch_build_argument_guards(bs, args):
    m, calls, _ = bs
    with pytest.raises(SystemExit):
        m.main(["--obs", "v2", "--seeds", "15"] + args)
    assert calls == []


def test_untagged_build_refuses_existing_dir_unless_allowed(bs):
    m, calls, tmp = bs
    (tmp / "kandla_s15_X").mkdir()
    with pytest.raises(SystemExit, match="exist"):
        m.main(["--obs", "v2", "--seeds", "15"])
    assert m.main(["--obs", "v2", "--seeds", "15", "--allow-existing"]) == 0 and len(calls) == 1


# ---- envbuild.site_kandla.write_site (observation model stubbed; CPU) ----

def test_write_site_mismatch_keeps_belief_side_nominal(tmp_path, monkeypatch):
    import envbuild.site_kandla as sk
    import sim.events
    import sim.obs_model.grouping
    import sim.stub_site
    from envbuild.variant_roots import data_roots, rel

    monkeypatch.setenv("SCENESCOUT_DATA_VERSION", "v3")
    seen = {}

    def fake_observe(seed, emitters, sens, w_root, env, device="cuda", hw=None, jammers=None, links=None):
        seen.update(hw=hw, w_root=Path(w_root), sids=[s.id for s in sens if s.type == "fixed_df"])
        return []

    def fake_write_products(out, objs, zones, sensors, emitters, obs, events, seed, meta, **kw):
        seen["meta"] = meta
        (Path(out) / "truth").mkdir(parents=True, exist_ok=True)
        return {"counts": {}, "version_hash": "x"}

    monkeypatch.setattr(sk, "write_build_files", lambda b: None)          # never touch data/site_kandla
    monkeypatch.setattr(ov, "observe_v2", fake_observe)
    monkeypatch.setattr(ov, "products_v2", lambda *a, **k: {"points": []})
    monkeypatch.setattr(sim.obs_model.grouping, "group_tracks", lambda pts: [])
    monkeypatch.setattr(sim.events, "extract_events", lambda pts, zones: [])
    monkeypatch.setattr(sim.stub_site, "write_products", fake_write_products)
    out = tmp_path / "kandla_s15_X_M2"
    out.mkdir()
    sk.write_site(15, out, obs_version="v2", device="cpu", hw_override=_ov("M2"), variant_tag="M2")
    assert seen["hw"]["heading_datum"]["bias_deg_1sigma"] == 10.0
    assert seen["w_root"] == data_roots("v3")["rf_cache"]                  # nominal world RF, no variant cache
    assert seen["meta"]["rf_cache_dir"] == rel(data_roots("v3")["rf_cache"])
    assert "world_variant_id" not in seen["meta"] and "M2" not in yaml.safe_dump(seen["meta"])
    wv = yaml.safe_load((out / "truth" / "world_variant.yaml").read_text())
    assert wv["id"] == "M2" and wv["override"] == _ov("M2")
    assert set(wv["nodes"]) == set(seen["sids"])
    for sid in seen["sids"]:
        assert wv["nodes"][sid]["heading_bias_deg"] == pytest.approx(ov.hw_errors(seen["hw"], 15, sid)[0], abs=1e-6)
    with pytest.raises(SystemExit):
        sk.write_site(15, out, obs_version="v2", hw_override=_ov("M2"), variant_tag="M2", world_variant="W4")
