"""Model-error floor for the DF order estimate (validation finding, docs/RESULTS/rf_validation.md): with forward-
backward averaging, residual channel-gain mismatch adds a spurious eigenvalue ~-29 dB below the signal; at high SNR
MDL calls it a second source. Eigenvalues more than `model_error_floor_db` below the largest count as noise.
Off by default so v1 observations stay byte-identical."""

import torch

from sim.df.air7311 import DFEngine


def _ev(*vals):
    return torch.tensor([sorted(vals)], dtype=torch.float64)


def test_default_is_plain_mdl():
    ev = _ev(1.0, 1.0, 1e3, 1e6)                       # spurious -30 dB eigenvalue at high SNR
    assert int(DFEngine.mdl_order(ev, 256)) == 2
    assert int(DFEngine.mdl_order(ev, 256, rel_floor_db=None)) == 2


def test_floor_removes_a_calibration_eigenvalue_but_keeps_a_real_second_path():
    assert int(DFEngine.mdl_order(_ev(1.0, 1.0, 1e3, 1e6), 256, rel_floor_db=-20.0)) == 1
    assert int(DFEngine.mdl_order(_ev(1.0, 1.0, 1e5, 1e6), 256, rel_floor_db=-20.0)) == 2
