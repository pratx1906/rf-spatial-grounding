"""Burst -> track tagging must not collide across carriers (v3 audit 2026-10-26: det_track keyed on
(sensor, epoch, t0) only; continuous carriers all start at t0 ~ 0, so CCTV / Wi-Fi bursts took a drone track's id)."""

from types import SimpleNamespace

from sim.obs_model.observe_v2 import burst_key, group_keys


def _det(f_mhz, t0=0.0):
    return SimpleNamespace(sensor="S1", t=100.0, t0_s=t0, f_mhz=f_mhz)


def test_same_start_different_carrier_gives_different_keys():
    assert burst_key(_det(5785.0)) != burst_key(_det(5745.0))


def test_group_keys_cover_exactly_its_members():
    g = {"sensor": "S1", "t0s": [0.0, 0.0], "fs": [5785.0, 5786.5]}
    keys = group_keys(g, 100.0)
    assert burst_key(_det(5785.0)) in keys and burst_key(_det(5786.5)) in keys
    assert burst_key(_det(5745.0)) not in keys          # a co-timed CCTV link on another carrier


def test_group_without_member_carriers_keeps_old_behaviour():
    g = {"sensor": "S1", "t0": 0.0}                     # groups from older pickles have no fs / t0s
    assert len(group_keys(g, 100.0)) == 1
