"""Strong-signal front end of the AIR7311 node in the IQ path (S7.2, audit C Q18/Q23, B Q16; world side, torch).

Two stages, both referred to the antenna port (after cable loss), values from configs/hardware/air7311.yaml
front_end (all assumptions [HUMAN]):

  1. Preselector + LNA. The preselector passes the tuning window (ibw_mhz around the window centre) and rejects
     everything else by `out_of_window_rejection_db`. Out-of-window transmitters cannot be sampled at 125 MSPS, so
     they enter as blocker powers: P_b = P_rx - rejection. The LNA is a memoryless Rapp AM/AM (smoothness p = 1,
     no AM/PM) with input P1dB `p1db_lna_dbm`. Under a strong blocker the small-signal gain of an in-window
     signal is that of a Rapp p = 1 stage linearised around a CW blocker of power P (blocker phase averaged):
         g_ss(P) = (1 + P / (2 Ps)) / (1 + P / Ps)^(3/2),   Ps = P1dB / (10^0.1 - 1)
     (desensitisation; the companion term at the blocker's mirror frequency lies outside the window).
  2. In-window RF/IF chain: Rapp p = 1 on the sampled complex envelope with input P1dB `p1db_in_dbm`:
         y = x / sqrt(1 + |x|^2 / Ps)
     so compression, cross-modulation and IM3 between in-window signals appear in the IQ. At low level this is the
     cubic y ~ x (1 - |x|^2 / (2 Ps)): for two equal tones of power P each the IM3 tone sits at
     20 log10(P / (2 Ps)) dBc.

Blocking (same thresholds as sim.rf.spectrum.SpectrumWorld.epoch): a dwell is `blocked` when the in-window input
power exceeds p1db_in_dbm or any out-of-window blocker after the preselector exceeds p1db_lna_dbm. A blocked dwell's
observations are dropped (the receiver is in compression); its IQ still carries the desensitised signals.
Not modelled: IM3 of two out-of-window blockers falling in-window, reciprocal mixing, AM/PM, memory effects.
"""

from __future__ import annotations

import math

import torch

P1DB_FACTOR = 10 ** 0.1 - 1          # Rapp p=1: 1 dB compression at P = Ps * (10^0.1 - 1)


def psat_w(p1db_dbm: float) -> float:
    """Rapp (p = 1) saturation power (W) for an input 1 dB compression point (dBm)."""
    return 10 ** ((p1db_dbm - 30) / 10) / P1DB_FACTOR


def rapp(x: torch.Tensor, p1db_dbm: float) -> torch.Tensor:
    """Memoryless Rapp p = 1 AM/AM on a complex envelope in sqrt(W) units (unit small-signal gain)."""
    ps = psat_w(p1db_dbm)
    return x * torch.rsqrt(1 + x.real.pow(2).add(x.imag.pow(2)) / ps).to(x.dtype)


def blocker_gain(p_blocker_w: float, p1db_dbm: float) -> float:
    """Small-signal amplitude gain of an in-window signal under a CW blocker of power p_blocker_w at the stage."""
    u = p_blocker_w / psat_w(p1db_dbm)
    return (1 + u / 2) / (1 + u) ** 1.5


def preselector_db(df_hz: float, ibw_hz: float, rejection_db: float) -> float:
    """Attenuation (dB) of a carrier df_hz from the window centre: 0 inside +-ibw/2, rejection_db outside."""
    return 0.0 if abs(df_hz) <= ibw_hz / 2 else rejection_db


def front_end(y: torch.Tensor, fe: dict, blockers_dbm: list[tuple[str, float]] | None = None
              ) -> tuple[torch.Tensor, dict]:
    """Apply the front end to the in-window antenna-port signal y [M, N] (sqrt(W)).

    blockers_dbm: (id, received power at the antenna port in dBm BEFORE the preselector) of transmitters outside
    the window. Returns (y', info) with info = {blocked, reasons, gain_db (blocker desense, <= 0),
    in_window_dbm, blocker_dbm (strongest, after the preselector)}."""
    rej = fe.get("out_of_window_rejection_db", 40.0)
    p1_lna = fe.get("p1db_lna_dbm", -10.0)
    p1_in = fe.get("p1db_in_dbm", -20.0)
    reasons = []
    pb = [(i, p - rej) for i, p in (blockers_dbm or [])]
    p_tot_w = sum(10 ** ((p - 30) / 10) for _, p in pb)
    g = blocker_gain(p_tot_w, p1_lna) if p_tot_w > 0 else 1.0
    for i, p in pb:
        if p > p1_lna:
            reasons.append(f"lna_compression:{i}")
    p_in_w = float(y.real.pow(2).add(y.imag.pow(2)).mean(1).max()) if y.numel() else 0.0
    p_in_dbm = 10 * math.log10(max(p_in_w, 1e-30)) + 30
    if p_in_dbm > p1_in:
        reasons.append(f"in_window_compression:{p_in_dbm:.1f}dBm")
    out = rapp(y * g if g != 1.0 else y, p1_in)
    return out, {"blocked": bool(reasons), "reasons": reasons, "gain_db": 20 * math.log10(g),
                 "in_window_dbm": p_in_dbm,
                 "blocker_dbm": max((p for _, p in pb), default=-math.inf)}
