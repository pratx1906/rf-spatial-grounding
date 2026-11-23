import numpy as np

from envbuild.geo import ENU


def test_roundtrip_is_sub_millimetre():
    enu = ENU(23.02, 70.21)
    lat = np.array([23.0, 23.03, 23.05])
    lon = np.array([70.19, 70.22, 70.25])
    e, n, u = enu.to_enu(lat, lon, 0.0)
    la, lo, h = enu.to_geodetic(e, n, u)
    assert np.max(np.abs(la - lat)) < 1e-9 and np.max(np.abs(lo - lon)) < 1e-9 and np.max(np.abs(h)) < 1e-3


def test_known_distances():
    enu = ENU(23.02, 70.21)
    e, n, _ = enu.to_enu(23.02 + 1 / 60, 70.21)      # one arc-minute north is ~1843 m at this latitude
    assert abs(n - 1843.0) < 3 and abs(e) < 0.01
    e, n, _ = enu.to_enu(23.02, 70.21 + 1 / 60)      # one arc-minute east is ~1708 m at 23 deg
    assert abs(e - 1708.0) < 3 and abs(n) < 1.0
