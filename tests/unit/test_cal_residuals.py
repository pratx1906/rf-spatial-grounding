"""Audit Q18/Q21: v2 applies the configured per-channel calibration residuals (gain/phase), per node per scenario."""

import numpy as np
import yaml

from sim.obs_model.observe_air7311 import HW_FILE
from sim.obs_model.observe_v2 import cal_residuals


def test_residuals_follow_the_configured_sigmas():
    hw = yaml.safe_load(HW_FILE.read_text())
    imp = hw["impairments"]
    g, p = [], []
    for seed in range(60):
        for sid in ("S1", "S2", "S3", "S4", "S5"):
            ge, pe = cal_residuals(hw, seed, sid)
            g.append(ge.numpy())
            p.append(pe.numpy())
    g, p = np.concatenate(g), np.concatenate(p)
    assert abs(g.std() - imp["channel_gain_sigma_db"]) < 0.05
    assert abs(p.std() - imp["channel_phase_sigma_deg"]) < 0.15


def test_residuals_are_fixed_per_node_and_scenario_and_differ_between_them():
    hw = yaml.safe_load(HW_FILE.read_text())
    a, b = cal_residuals(hw, 15, "S1"), cal_residuals(hw, 15, "S1")
    assert np.array_equal(a[0].numpy(), b[0].numpy()) and np.array_equal(a[1].numpy(), b[1].numpy())
    assert not np.array_equal(cal_residuals(hw, 15, "S2")[0].numpy(), a[0].numpy())
    assert not np.array_equal(cal_residuals(hw, 16, "S1")[0].numpy(), a[0].numpy())


def test_observe_v2_passes_the_residuals_to_the_receiver():
    import inspect

    from sim.obs_model import observe_v2 as ov

    src = inspect.getsource(ov.observe_v2)
    assert "gain_err_db=" in src and "phase_err_deg=" in src and "f_lo_hz=" in src
