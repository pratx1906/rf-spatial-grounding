"""Task 2b: the GPU-batched paths of the v2 build against the per-item reference implementations.

Runs on the CPU (make test hides the GPU) and, when CUDA is present, on the GPU as well. Exact where the batched
path repeats the reference's float operations (detection, percentile, line mask); float-rounding close elsewhere
(Bluestein FFTs, matrix-product path sums, batched eigendecomposition).
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sim.df.air7311 import DFEngine, square_positions
from sim.obs_model.observe_v2 import burst_df, burst_df_batch
from sim.rf import bursts as B
from sim.rf import channel as ch
from sim.rf import classify as C
from sim.rf import pathcache
from sim.rf import receiver as rx
from sim.rf import toa as T
from sim.rf import waveforms as W
from sim.rf.fastfft import fft_any, ifft_any

REPO = Path(__file__).resolve().parents[2]
HW = yaml.safe_load((REPO / "configs" / "hardware" / "air7311.yaml").read_text())
HW = {**HW, "processing": {**HW["processing"], "model_error_floor_db": -20.0}}
FS, DWELL = 125e6, 4e-3
POS = square_positions(0.054)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def dwell():
    """One 4 ms dwell with an OcuSync-like burst train, a GFSK hopper, a LoRa packet and analogue FM video."""
    rng = np.random.default_rng(3)
    n = int(FS * DWELL)
    pre = ch.pre_roll(n)
    p_n = 10 ** ((-174 + 10 * math.log10(FS) + 3 - 30) / 10)
    y = torch.zeros((4, n), dtype=torch.complex64)
    srcs = [(W.ofdm_ocusync_like(10.0, 2 * DWELL, rng), 12e6, 60.0, 25.0),
            (W.gfsk_hop(1e6, 0.5e-3, 1.4e-3, [0.0], 2 * DWELL, rng), -20e6, 200.0, 20.0),
            (W.lora_elrs(500e3, 6, 400.0, [0.0], 2 * DWELL, rng, data_symbols=8), 35e6, 300.0, 22.0),
            (W.fm_fpv(2 * DWELL, rng), -40e6, 120.0, 18.0)]
    for w, f_off, az, snr in srcs:
        s = ch.resample_shift(w.iq[: int(round((n + pre) / FS * w.fs))], w.fs, FS, f_off, n + pre, 0.0, ibw_hz=100e6)
        amp = math.sqrt(p_n * 10 ** (snr / 10) / 30)
        y += ch.propagate(s * amp, FS, np.array([1.0 + 0j, 0.3 + 0.2j]), np.array([2e-6, 2.4e-6]),
                          np.array([az, az + 40.0]), np.array([5.0, 2.0]), POS, 2.44e9 + f_off, 2.44e9, n_pre=pre)
    x, info = rx.receive(y, FS, HW, torch.Generator().manual_seed(1), interchip_deg=2.0)
    return x


def _close(a, b, rel=1e-5, abs_=1e-9):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


@pytest.mark.parametrize("L", [7, 97, 1000, 4097, 30011])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_bluestein_fft_matches_the_direct_fft(L, dtype):
    g = torch.Generator().manual_seed(L)
    x = torch.complex(torch.randn(3, L, generator=g, dtype=torch.float64),
                      torch.randn(3, L, generator=g, dtype=torch.float64)).to(dtype)
    tol = 2e-5 if dtype == torch.complex64 else 1e-11
    ref = torch.fft.fft(x.to(torch.complex128))
    got = fft_any(x, bluestein=True).to(torch.complex128)
    assert ((got - ref).abs().max() / ref.abs().max()).item() < tol
    back = ifft_any(fft_any(x, bluestein=True), bluestein=True).to(torch.complex128)
    assert ((back - x.to(torch.complex128)).abs().max() / x.abs().max()).item() < tol


@pytest.mark.parametrize("n", [1000, 77777, 2_000_003])
def test_percentile_is_numpy_bit_identical(n):
    a = (np.random.default_rng(n).exponential(size=n) * 1e-9).astype(np.float32)
    got = B.percentile_np(torch.as_tensor(a), 20) + 1e-30
    ref = np.percentile(a, 20) + 1e-30
    assert got == ref and type(got) is type(ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_detect_gpu_finds_the_same_bursts(dwell, dev):
    x = dwell.to(dev)
    ref, got = B.detect(x, FS), B.detect_gpu(x, FS)
    assert len(ref) >= 4
    assert [(b.t0, b.dur, b.f_lo, b.f_hi, b.snr_db) for b in ref] == [(b.t0, b.dur, b.f_lo, b.f_hi, b.snr_db)
                                                                        for b in got]


@pytest.mark.parametrize("dev", DEVICES)
def test_band_pass_cov_power_match_the_reference(dwell, dev):
    x = dwell.to(dev)
    for b in [b for b in B.detect(x, FS) if b.dur >= 150e-6]:
        ref = B.burst_samples(x, FS, b)
        got = B.band_pass(x, FS, b)
        assert ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item() < 1e-4
        Rr, Rg = B.burst_cov(x, FS, b), B.cov_from(got)
        assert ((Rg - Rr).abs().max() / Rr.abs().max()).item() < 1e-4
        assert _close(float(got.abs().pow(2).mean()), float(ref.abs().pow(2).mean()), 1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_features_batch_matches_features(dwell, dev):
    x = dwell.to(dev)
    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
    got = C.features_batch([B.band_pass(x, FS, b)[0] for b in bs], FS, bs)
    protos = set()
    for b, g in zip(bs, got):
        r = C.features(x, FS, b)
        assert set(r) == set(g)
        for k in r:
            assert _close(float(r[k]), float(g[k]), rel=1e-3, abs_=1e-6), (k, r[k], g[k])
        assert C.classify(r)[0] == C.classify(g)[0]
        protos.add(C.classify(g)[0])
    assert len(protos) >= 3                      # the fixture exercises several feature branches


@pytest.mark.parametrize("dev", DEVICES)
def test_toa_batch_matches_toa_s_with_the_same_draws(dwell, dev):
    x = dwell.to(dev)
    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    ref = [T.toa_s(x, FS, b, r1, 20.0) for b in bs]
    got = T.toa_batch(x, FS, bs, r2, 20.0)
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1 / FS
    assert any(math.isfinite(a) for a in ref)


@pytest.mark.parametrize("dev", DEVICES)
def test_burst_df_batch_matches_burst_df(dwell, dev):
    x = dwell.to(dev)
    bs = [b for b in B.detect(x, FS) if b.dur >= 150e-6]
    eng = DFEngine(HW, dev)
    Rs = torch.stack([B.burst_cov(x, FS, b) for b in bs])
    got = burst_df_batch(eng, Rs, bs, POS, 2.44e9, HW, 3.0)
    for b, R, g in zip(bs, Rs, got):
        ref = burst_df(eng, R, b, POS, 2.44e9, HW, 3.0)
        assert len(ref) == len(g)
        for (a1, e1, s1), (a2, e2, s2) in zip(ref, g):
            assert abs((a1 - a2 + 180) % 360 - 180) < 1e-6 and e1 == e2 and abs(s1 - s2) < 1e-3


def test_mdl_order_n_equals_mdl_order():
    g = torch.Generator().manual_seed(2)
    ev = torch.sort(torch.rand(50, 4, generator=g) ** 4, dim=1).values
    for n_snap in (16, 300, 4096):
        a = DFEngine.mdl_order(ev, n_snap, rel_floor_db=-20.0)
        b = DFEngine.mdl_order_n(ev, torch.full((50,), float(n_snap), dtype=torch.float64), rel_floor_db=-20.0)
        assert torch.equal(a, b)


@pytest.mark.parametrize("dev", DEVICES)
def test_resample_and_propagate_fast_match_the_reference(dev):
    rng = np.random.default_rng(9)
    w = W.wifi_ofdm(2e-3, rng)
    n, pre = 200_000, ch.pre_roll(200_000)
    a = ch.resample_shift(w.iq[:30011], w.fs, FS, 7e6, n + pre, 0.0, device=dev, ibw_hz=100e6)
    b = ch.resample_shift_fast(w.iq[:30011], w.fs, FS, 7e6, n + pre, 0.0, device=dev, ibw_hz=100e6)
    assert ((a - b).abs().max() / a.abs().max()).item() < 1e-4
    P = 13
    h = (rng.normal(size=P) + 1j * rng.normal(size=P)) * 1e-4
    h[3] = 0
    tau = rng.uniform(1e-7, 5e-6, P)
    az, el = rng.uniform(0, 360, P), rng.uniform(-10, 30, P)
    dop = np.round(rng.uniform(-600, 600, P), -1)
    dop[:4] = dop[4]                                    # one group with several paths
    rip = 1 + 0.01 * rng.normal(size=(P, 4))
    ref = ch.propagate(a, FS, h, tau, az, el, POS, 2.45e9, 2.44e9, dop, elem_gain=rip, f_lo_hz=2.443e9, n_pre=pre)
    got = ch.propagate_fast(a, FS, h, tau, az, el, POS, 2.45e9, 2.44e9, dop, elem_gain=rip, f_lo_hz=2.443e9,
                            n_pre=pre, S=torch.fft.fft(a))
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 1e-5


def test_pathcache_loads_once_per_root(tmp_path):
    def mk(root, v):
        f = root / "air7311_S1" / "spec_diff" / "ism_2400.npz"
        f.parent.mkdir(parents=True)
        np.savez(f, xs=np.arange(3.0) * v)
        return f

    f1, f2 = mk(tmp_path / "a", 1.0), mk(tmp_path / "b", 2.0)
    pathcache.clear()
    pathcache.prefetch([f1])
    z1 = pathcache.load(f1)
    assert pathcache.load(f1) is z1 and z1["xs"][1] == 1.0
    z2 = pathcache.load(f2)                              # another root: the first root's arrays are dropped
    assert z2["xs"][1] == 2.0 and pathcache.load(f1) is not z1
    pathcache.clear()
