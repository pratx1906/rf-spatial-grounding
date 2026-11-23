"""AIR7311 receive chain on a 4-channel dwell (RF phase 2, Task 6; world side, torch).

Order: thermal noise (kTB F at the dwell rate) -> residual calibration (per-channel gain/phase, inter-chip phase
on channels 2-3) -> common LO phase noise per chip (random walk, `lo_pn_deg_rms` over the dwell) -> IQ imbalance
(y' = mu y + nu conj(y)) and DC offset -> AGC to `agc_backoff_db` below full scale on the strongest channel ->
14-bit quantisation with clipping. Values: configs/hardware/air7311.yaml (receiver/impairments/front_end/iq);
the IQ-path values are assumptions [HUMAN].
"""

from __future__ import annotations

import math

import torch

IQ_DEFAULTS = {"iq_gain_db": 0.2, "iq_phase_deg": 1.0, "dc_dbfs": -50.0, "lo_pn_deg_rms": 1.0, "agc_backoff_db": 12.0}


def iq_imbalance(iq: dict | None = None) -> tuple[complex, complex]:
    """(mu, nu) of the IQ imbalance y' = mu y + nu conj(y) for gain error iq_gain_db and phase error iq_phase_deg."""
    iq = {**IQ_DEFAULTS, **(iq or {})}
    eps, phi = 10 ** (iq["iq_gain_db"] / 20) - 1, math.radians(iq["iq_phase_deg"])
    return (complex(math.cos(phi / 2), eps * math.sin(phi / 2)),
            complex(eps * math.cos(phi / 2), -math.sin(phi / 2)) / 2)


def image_rejection_db(iq: dict | None = None) -> float:
    """Image rejection ratio |mu|^2 / |nu|^2 (dB): a signal at +f leaves an image at -f (mirror about the LO) this
    far below it. Defaults (0.2 dB, 1 deg): 38.1 dB."""
    mu, nu = iq_imbalance(iq)
    return 10 * math.log10(abs(mu) ** 2 / abs(nu) ** 2)


def receive(y: torch.Tensor, fs: float, hw: dict, gen: torch.Generator, interchip_deg: float = 2.0,
            gain_err_db: torch.Tensor | None = None, phase_err_deg: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """y: [4, N] complex at the antenna ports (W^0.5 units). Returns ADC samples [4, N] (full scale = 1) and the
    scale (volts per ADC unit) so powers can be referred back to dBm."""
    dev = y.device
    m, n = y.shape
    iq = {**IQ_DEFAULTS, **hw.get("iq", {})}
    nf = hw["receiver"]["noise_figure_db"]
    p_n = 10 ** ((-174 + 10 * math.log10(fs) + nf - 30) / 10)                          # W over the dwell band
    noise = torch.complex(torch.randn((m, n), generator=gen, device=dev),
                          torch.randn((m, n), generator=gen, device=dev)) * math.sqrt(p_n / 2)
    x = y + noise
    g = gain_err_db if gain_err_db is not None else torch.zeros(m, device=dev)
    ph = phase_err_deg if phase_err_deg is not None else torch.zeros(m, device=dev)
    ph = ph + torch.tensor([0.0, 0.0, interchip_deg, interchip_deg], device=dev)[:m]
    cal = (10 ** (g / 20)) * torch.exp(1j * torch.deg2rad(ph))
    x = x * cal[:, None].to(torch.complex64)
    steps = torch.randn((2, n), generator=gen, device=dev) * math.radians(iq["lo_pn_deg_rms"]) / math.sqrt(n)
    pn = torch.cumsum(steps, 1)
    chip = torch.tensor([0, 0, 1, 1], device=dev)[:m]
    x = x * torch.exp(1j * pn[chip]).to(torch.complex64)
    mu, nu = iq_imbalance(iq)
    x = mu * x + nu * torch.conj(x)
    rms = x.abs().pow(2).mean(1).max().sqrt().clamp_min(1e-30)
    x = x + (10 ** (iq["dc_dbfs"] / 20)) * rms * (1 + 1j)      # constant DC: -47 dB below the strongest channel
    scale = rms / (10 ** (-iq["agc_backoff_db"] / 20))                                   # volts per full scale
    q = 2 ** (hw["receiver"]["adc_bits"] - 1)
    xr = torch.clamp(torch.round(x.real / scale * q), -q, q - 1) / q
    xi = torch.clamp(torch.round(x.imag / scale * q), -q, q - 1) / q
    clipped = float(((xr.abs() >= 1 - 1 / q) | (xi.abs() >= 1 - 1 / q)).float().mean())
    return torch.complex(xr, xi), {"scale": float(scale), "noise_w": p_n, "clipped_frac": clipped}
