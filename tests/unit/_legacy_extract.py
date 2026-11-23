"""Frozen copy of envbuild.air7311_precompute.extract before the pull/reduce split (equivalence test)."""

import numpy as np

from envbuild.air7311_precompute import TOP_K, _rx_paths


def extract_legacy(paths, names: dict[int, str], n_rx: int, f_hz: float, dual_pol: bool = False) -> dict[str, np.ndarray]:
    """Top-K paths per receiver with the FULL narrowband coefficient h = a * exp(-j 2 pi f tau) (Sionna's `a`
    excludes the propagation phase; validated: with it, analytic steering matches per-element tracing)."""
    ar, ai = (np.array(x) for x in paths.a)
    n_p = ar.shape[-1]
    phi, theta, tau = (_rx_paths(np.array(v), n_rx, n_p) for v in (paths.phi_t, paths.theta_t, paths.tau))
    a = (_rx_paths(ar, n_rx, n_p) + 1j * _rx_paths(ai, n_rx, n_p)) * np.exp(-1j * 2 * np.pi * f_hz * tau)
    if dual_pol:      # receive antenna 1 = H: the VH coefficient of the same paths
        a_vh = (_rx_paths(ar, n_rx, n_p, 1) + 1j * _rx_paths(ai, n_rx, n_p, 1)) * np.exp(-1j * 2 * np.pi * f_hz * tau)
        phi_r, theta_r = (_rx_paths(np.array(v), n_rx, n_p) for v in (paths.phi_r, paths.theta_r))
    depth = max(np.array(paths.interactions).shape[0], 1)
    inter = np.array(paths.interactions).reshape(depth, n_rx, -1, n_p)[:, :, 0, :]
    objs = np.array(paths.objects).reshape(depth, n_rx, -1, n_p)[:, :, 0, :]
    power = np.abs(a) ** 2
    k = min(TOP_K, n_p)
    order = np.argsort(-power, axis=1)[:, :k]
    tk = lambda x: np.take_along_axis(x, order, 1)  # noqa: E731
    ok = tk(power) > 0
    first = np.take_along_axis(inter[0], order, 1)
    ptype = np.where(ok, np.where(first == 0, 0, first), 255).astype("u1")
    fobj = np.take_along_axis(objs[0], order, 1)
    ids = sorted(names)
    lut = {oid: i for i, oid in enumerate(ids)}
    via = np.full(fobj.shape, -1, "i4")
    for oid, i in lut.items():
        via[(fobj == oid) & ok & (ptype != 0)] = i
    total = power.sum(1)
    out = {
        "h_re": np.where(ok, tk(a).real, 0).astype("f4"), "h_im": np.where(ok, tk(a).imag, 0).astype("f4"),
        "kept_energy": np.where(total > 0, tk(power).sum(1) / np.maximum(total, 1e-40), 1.0).astype("f4"),
        "tau_ns": np.where(ok, tk(tau) * 1e9, np.nan).astype("f4"),
        "az_deg": np.where(ok, (90.0 - np.degrees(tk(phi))) % 360.0, np.nan).astype("f4"),
        "el_deg": np.where(ok, 90.0 - np.degrees(tk(theta)), np.nan).astype("f4"),
        "ptype": ptype, "via": via,
    }
    if dual_pol:
        out["h_vh_re"] = np.where(ok, tk(a_vh).real, 0).astype("f4")
        out["h_vh_im"] = np.where(ok, tk(a_vh).imag, 0).astype("f4")
        # drone-end direction toward the incoming wave (compass az, elevation); LOS: bearing drone -> node
        out["az_r_deg"] = np.where(ok, (90.0 - np.degrees(tk(phi_r))) % 360.0, np.nan).astype("f4")
        out["el_r_deg"] = np.where(ok, 90.0 - np.degrees(tk(theta_r)), np.nan).astype("f4")
    if k < TOP_K:
        fill = {"ptype": 255, "via": -1, "h_re": 0.0, "h_im": 0.0, "h_vh_re": 0.0, "h_vh_im": 0.0}
        for key, arr in out.items():
            if arr.ndim == 2:
                out[key] = np.concatenate([arr, np.full((n_rx, TOP_K - k), fill.get(key, np.nan), arr.dtype)], 1)
    out["_names"] = np.array([names[i] for i in ids])
    return out


