"""RF convergence plan Task 4 (rulings P1-1, P1-2): shifted ray lattices in the production precompute, the v3
convergence-scene roots, and the convergence_sweep grid (3 shifted patterns per setting, shift-vs-shift agreement,
K 24/48, N->W3 world-effect / lattice-noise ratio, the gate and the cheapest passing setting).
Numpy only: the Sionna side (the lattice patch itself) is exercised by the GPU sweep."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from envbuild import variant_roots as vr

REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "data" / "site_kandla"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (vr.ENV_VERSION, vr.ENV_ALLOW, vr.ENV_RF_GEOM):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def cs():
    return _load("cs", "scripts/convergence_sweep.py")


def _inside(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or a.resolve().is_relative_to(b.resolve())


# ------------------------------------------------------------------ lattice shift (P1-2)
def test_lattice_shift_is_seeded_uniform_rotation():
    from envbuild.air7311_precompute import lattice_shift_uv, rotate_unit

    assert lattice_shift_uv(None) is None
    u1, u2 = lattice_shift_uv(1), lattice_shift_uv(2)
    assert np.array_equal(u1, lattice_shift_uv(1))                  # deterministic
    assert not np.allclose(u1, u2) and np.all((u1 >= 0) & (u1 < 1))
    x = np.linspace(0, 0.999, 50)
    rx, ry = rotate_unit(x, x[::-1], u1)
    assert np.all((rx >= 0) & (rx < 1)) and np.all((ry >= 0) & (ry < 1))
    assert np.allclose((rx - x) % 1, u1[0]) and np.allclose((ry - x[::-1]) % 1, u1[1])


def test_convergence_sweep_reuses_the_production_shift():
    """S7.4's shifted patterns and the production option must be the same rotation (one implementation)."""
    src = (REPO / "scripts" / "convergence_sweep.py").read_text()
    assert "from envbuild.air7311_precompute import" in src and "shifted_lattice" in src
    assert "def shifted_lattice" not in src


def test_lattice_root_is_a_sub_dir_of_the_cache():
    root = vr.cache_root("N", "nominal", "v2", "v3")
    assert vr.lattice_root(root, None) == root
    p = vr.lattice_root(root, 2)
    assert p == root / "lattice" / "p2" and _inside(p, root)


def test_precompute_command_passes_the_lattice_shift(monkeypatch):
    monkeypatch.setenv(vr.ENV_VERSION, "v3")
    pv = _load("pv", "scripts/precompute_variants.py")
    v = yaml.safe_load((REPO / "configs" / "rf_variants.yaml").read_text())["variants_v2"]
    plain = pv.command("W3", v, "v2", 25000)
    assert "--lattice-shift" not in plain
    cmd = pv.command("W3", v, "v2", 25000, solver={"lattice_shift": 2})
    assert cmd[cmd.index("--lattice-shift") + 1] == "2"
    out = Path(cmd[cmd.index("--out-root") + 1])
    assert out == vr.cache_root("W3", "world", "v2") / "lattice" / "p2"
    assert Path(plain[plain.index("--out-root") + 1]) == vr.cache_root("W3", "world", "v2")


# ------------------------------------------------------------------ v3 convergence scenes
def test_convergence_scene_dirs_are_v3_and_side_separated():
    for g in ("rf", "rfr"):
        n = vr.convergence_scene_dir("N", "nominal", g, "v3")
        w = vr.convergence_scene_dir("W3", "world", g, "v3")
        assert _inside(n, vr.data_roots("v3")["rf_cache"]) and n.parent.name == g
        assert _inside(w, vr.data_roots("v3")["world_cache"])          # hidden worlds stay under world_cache*
        for d in vr.demo_roots():
            assert not _inside(n, d) and not _inside(w, d)
    assert vr.convergence_scene_dir("N", "nominal", "rf", "v3") != vr.convergence_scene_dir("N", "nominal", "rfr", "v3")
    with pytest.raises(ValueError):
        vr.convergence_scene_dir("N", "nominal", "rf", "v2")


def test_sweep_full_geometry_reads_the_demo_scenes_read_only(cs):
    assert cs.sweep_scene_dir("full", "N") == vr.scene_dir("N", "nominal", "v2", "v2")
    assert cs.sweep_scene_dir("full", "W3") == vr.scene_dir("W3", "world", "v2", "v2")
    assert cs.sweep_scene_dir("rfr", "W3") == vr.convergence_scene_dir("W3", "world", "rfr", "v3")


# ------------------------------------------------------------------ grid (P1-1)
def test_grid_has_three_shifted_patterns_per_setting(cs):
    rs = cs.grid_runs([1_000_000, 3_000_000], [2, 3], 3)
    assert len(rs) == 12 and len({r["tag"] for r in rs}) == 12
    for s in (1_000_000, 3_000_000):
        for d in (2, 3):
            pats = [r for r in rs if r["samples"] == s and r["depth"] == d]
            assert len(pats) == 3
            assert all(r["shift"] is not None for r in pats)          # never the unshifted (reference) lattice
            assert len({r["shift"] for r in pats}) == 3
    assert cs.setting_tag(1_000_000, 2) == "s1000k_d2"


def _fake(n=400, k=64, seed=0, flip=0.0, dpow_db=0.0, rng_flip=None):
    r = np.random.default_rng(seed)
    d = {"ptype": np.where(r.random((n, k)) < 0.5, 1, 2).astype("u1"),
         "az_deg": r.uniform(0, 360, (n, k)).astype("f4"),
         "h_re": (r.random((n, k)) * np.exp(-np.arange(k) / 4)).astype("f4"),
         "h_im": np.zeros((n, k), "f4"), "kept_energy": np.ones(n, "f4")}
    d["ptype"][:, 0] = 0
    if flip:
        m = (rng_flip or np.random.default_rng(seed + 99)).random(n) < flip
        d["ptype"][m, 0] = 1
        d["az_deg"][m, 0] = (d["az_deg"][m, 0] + 30) % 360
    if dpow_db:
        d["h_re"] *= 10 ** (dpow_db / 20)
    return d


def test_shift_vs_shift_identical_patterns_agree_fully(cs):
    a = _fake()
    m = cs.shift_vs_shift([a, dict(a), dict(a)], 24)
    assert m["type_agree"] == 1.0 and m["az_agree_1deg"] == 1.0 and m["power_err_db_p95"] == 0.0
    assert m["pairs"] == 3


def test_shift_vs_shift_sees_pattern_noise(cs):
    pats = [_fake(), _fake(flip=0.2, seed=0, rng_flip=np.random.default_rng(1)),
            _fake(flip=0.2, seed=0, rng_flip=np.random.default_rng(2))]
    m = cs.shift_vs_shift(pats, 24)
    assert 0.6 < m["type_agree"] < 0.95


def test_effect_noise_ratio(cs):
    base = _fake()
    n_pats = [base, dict(base), dict(base)]
    w_pats = [_fake(flip=0.3, seed=0, rng_flip=np.random.default_rng(7))] * 3
    r = cs.effect_noise(n_pats, w_pats)
    assert r["noise_changed"] == 0.0 and r["world_changed"] > 0.2 and r["ratio"] == float("inf")
    noisy = [_fake(), _fake(flip=0.2, rng_flip=np.random.default_rng(1)), _fake(flip=0.2, rng_flip=np.random.default_rng(2))]
    r2 = cs.effect_noise(noisy, noisy)                                   # W3 == N: no world effect at all
    assert r2["world_changed"] == 0.0 and r2["ratio"] == 0.0 and r2["noise_changed"] > 0.1
    assert 0 <= r2["jaccard_mean"] <= 1


def test_gate_and_cheapest_passing_setting(cs):
    ok = {"type_agree": 0.98, "az_agree_1deg": 0.96, "power_err_db_p95": 2.0}
    rows = [{"geom": "full", "setting": "s1000k_d2", "K": 24, "seconds": 6, **ok, "ratio": 2.0},
            {"geom": "rfr", "setting": "s3000k_d2", "K": 24, "seconds": 15, **ok, "ratio": 4.0},
            {"geom": "rfr", "setting": "s10000k_d2", "K": 48, "seconds": 40, **ok, "ratio": 9.0},
            {"geom": "rf", "setting": "s1000k_d2", "K": 24, "seconds": 5, **ok, "type_agree": 0.96, "ratio": 5.0}]
    g = [cs.gate(r) for r in rows]
    assert [x["pass"] for x in g] == [False, True, True, False]
    assert g[0]["fails"] == ["effect_ratio"] and g[3]["fails"] == ["type"]
    best = cs.cheapest_passing(rows)
    assert best["geom"] == "rfr" and best["setting"] == "s3000k_d2"
    assert cs.cheapest_passing(rows[:1] + rows[3:]) is None
    assert cs.gate({**ok, "ratio": None})["pass"] is False             # no W3 trace: cannot pass


def test_ensemble_union_strongest_and_spread(cs):
    a, b = _fake(), _fake(seed=1)
    b["h_re"][:, 0] = a["h_re"][:, 0] * 2                               # pattern b finds the stronger path everywhere
    e = cs.ensemble([a, b], 24)
    assert np.array_equal(e["ptype"], b["ptype"][:, 0]) and np.array_equal(e["az_deg"], b["az_deg"][:, 0])
    assert np.all(e["spread_db"] >= 0)
    same = cs.ensemble([a, dict(a), dict(a)], 24)
    assert np.allclose(same["spread_db"], 0)


def test_ensemble_split_noise_and_effect(cs):
    base = _fake()
    n = [base] * 6
    w = [_fake(flip=0.3, rng_flip=np.random.default_rng(7))] * 6
    r = cs.ensemble_split(n, w)
    assert r["patterns_per_ensemble"] == 3 and r["type_agree"] == 1.0 and r["noise_changed"] == 0.0
    assert r["world_changed"] > 0.2 and r["ratio"] == float("inf")
    assert "world_changed" not in cs.ensemble_split(n, None)


def test_task5_estimate_scales_production_time(cs):
    # 1.9 h for 18 caches at the production setting; a 3x slower setting -> 5.7 h; fallback x3 patterns
    assert cs.task5_hours(1.0, 1.0) == pytest.approx(cs.PROD_HOURS_18)
    assert cs.task5_hours(3.0, 1.0, patterns=3) == pytest.approx(9 * cs.PROD_HOURS_18)


# ------------------------------------------------------------------ corrected metrics (Task 4 audit)
def _cells(gains_db, az, ptype, n_paths=24):
    """Synthetic cells: per cell a list of (gain dB, az, type) paths, strongest first; padded with no-path."""
    n = len(gains_db)
    d = {"ptype": np.full((n, n_paths), 255, "u1"), "az_deg": np.full((n, n_paths), np.nan, "f4"),
         "h_re": np.zeros((n, n_paths), "f4"), "h_im": np.zeros((n, n_paths), "f4"), "kept_energy": np.ones(n, "f4")}
    for i, (g, a, t) in enumerate(zip(gains_db, az, ptype)):
        for j, (gj, aj, tj) in enumerate(zip(g, a, t)):
            d["h_re"][i, j] = 10 ** (gj / 20)
            d["az_deg"][i, j] = aj
            d["ptype"][i, j] = tj
    return d


def test_link_budget_floor_matches_the_hardware_model(cs):
    # 2.44 GHz: -174 + 10 log10(20 MHz) + NF 3 = -98.0 dBm; detect SNR 6 -> -92 dBm; EIRP 20, cable 2 -> -110 dB
    assert cs.link_floor_db() == pytest.approx(-110.0, abs=0.05)
    assert cs.FLOOR_DB == -115.0 and cs.FLOOR_DB <= cs.link_floor_db()


def test_floor_drops_weak_cells_only(cs):
    strong, weak = [[-90.0]], [[-140.0]]
    ref = _cells(strong + weak, [[10.0], [10.0]], [[0], [1]])
    test = _cells(strong + weak, [[10.0], [200.0]], [[0], [2]])                # the weak cell disagrees
    assert cs.compare(test, ref, 24, 24)["type_agree"] == 0.5                    # legacy: counted
    m = cs.compare_floor(test, ref, 24)
    assert m["cells"] == 1 and m["type_agree"] == 1.0 and m["az_agree_1deg"] == 1.0 and m["match"] == 1.0
    assert cs.compare_floor(test, ref, 24, floor_db=-150.0)["type_agree"] == 0.5


def test_strongest_path_tie_tolerance(cs):
    # reference: two near-equal paths (1 dB apart); the test's strongest is the reference's second one
    ref = _cells([[-90.0, -91.0]], [[10.0, 50.0]], [[1, 1]])
    test = _cells([[-90.5, -90.6]], [[50.0, 10.0]], [[1, 1]])
    m = cs.compare_floor(test, ref, 24)
    assert m["match"] == 1.0 and m["az_agree_1deg"] == 1.0 and m["type_agree"] == 1.0
    assert cs.compare(test, ref, 24, 24)["az_agree_1deg"] == 0.0                 # legacy: strict strongest only
    far = _cells([[-90.0, -95.0]], [[10.0, 50.0]], [[1, 1]])                     # second path 5 dB down: no tie
    assert cs.compare_floor(test, far, 24)["match"] == 0.0
    other_type = _cells([[-90.0, -91.0]], [[10.0, 50.0]], [[1, 2]])              # az matches, type does not
    assert cs.compare_floor(test, other_type, 24)["match"] == 0.0
    assert cs.compare_floor(test, other_type, 24)["az_agree_1deg"] == 1.0


def test_effect_reproducibility_across_patterns(cs):
    n = 200
    base = _cells([[-90.0]] * n, [[10.0]] * n, [[0]] * n)
    w = _cells([[-88.0]] * 40 + [[-90.0]] * (n - 40),                              # +2 dB and a new strongest path
               [[120.0]] * 40 + [[10.0]] * (n - 40), [[1]] * 40 + [[0]] * (n - 40))
    r = cs.effect_repro([base, dict(base), dict(base)], [w, dict(w), dict(w)], 24)
    assert r["jaccard"] == 1.0 and r["dp_diff_db_median"] == 0.0 and r["world_changed"] == pytest.approx(0.2)
    assert r["dp_world_db_median"] == pytest.approx(2.0, abs=0.01) and r["affected"] == pytest.approx(0.2)
    # a different 40 cells change on each pattern: not reproducible
    ws = []
    for p in range(3):
        idx = np.arange(n)
        sel = (idx >= 40 * p) & (idx < 40 * (p + 1))
        ws.append(_cells([[-88.0] if s else [-90.0] for s in sel],
                         [[120.0] if s else [10.0] for s in sel], [[1] if s else [0] for s in sel]))
    r2 = cs.effect_repro([base, dict(base), dict(base)], ws, 24)
    assert r2["jaccard"] == 0.0 and r2["dp_diff_db_median"] > 1.0
    none = cs.effect_repro([base] * 3, [base] * 3, 24)                         # no effect: not applicable
    assert none["world_changed"] == 0.0 and none["jaccard"] is None


def test_effect_noise_zero_over_zero_is_nan(cs):
    base = _fake()
    r = cs.effect_noise([base, dict(base), dict(base)], [base, dict(base), dict(base)])
    assert r["world_changed"] == 0.0 and r["noise_changed"] == 0.0 and np.isnan(r["ratio"])


def test_gate_v2_per_case_effect_only_when_applicable(cs):
    ok = {"type_agree": 0.98, "az_agree_1deg": 0.96, "power_err_db_p95": 2.0}
    assert cs.gate_v2({**ok, "eff_jaccard": None})["pass"] is True              # no W3 effect in this case
    assert cs.gate_v2({**ok, "eff_jaccard": 0.3})["fails"] == ["effect_repro"]
    assert cs.gate_v2({**ok, "eff_jaccard": 0.8})["pass"] is True
    assert cs.gate_v2({**ok, "type_agree": 0.9, "eff_jaccard": None})["fails"] == ["type"]
    assert cs.gate_v2({**ok, "type_agree": None})["pass"] is False             # no cells above the floor
    # world power change reproduced across lattices to a median 1 dB (only where there is a power effect)
    assert cs.gate_v2({**ok, "eff_jaccard": None, "eff_dp_diff_db_median": 1.4})["fails"] == ["effect_dp"]
    assert cs.gate_v2({**ok, "eff_jaccard": 0.7, "eff_dp_diff_db_median": 0.4})["pass"] is True


def test_effect_jaccard_is_structural_not_power_threshold(cs):
    # a 0.6 dB power change on every cell, no strongest-path change: no structural effect, power effect reproduced
    n = 100
    base = _cells([[-90.0]] * n, [[10.0]] * n, [[0]] * n)
    w = _cells([[-89.4]] * n, [[10.0]] * n, [[0]] * n)
    r = cs.effect_repro([base] * 3, [w] * 3, 24)
    assert r["jaccard"] is None and r["world_changed"] == 0.0 and r["affected"] == 1.0
    assert r["dp_world_db_median"] == pytest.approx(0.6, abs=0.01) and r["dp_diff_db_median"] == 0.0
