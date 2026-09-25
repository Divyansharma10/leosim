"""Free-space path loss link budget, Shannon capacity, propagation latency."""

from __future__ import annotations

import numpy as np

from .config import BOLTZMANN_J_K, SPEED_OF_LIGHT_M_S, LinkBudgetConfig

# 20*log10(4*pi/c) with distance in metres and frequency in Hz => -147.55 dB.
# Equivalent to the familiar +92.45 dB form when using km and GHz.
_FSPL_CONST_DB = 20.0 * np.log10(4.0 * np.pi / SPEED_OF_LIGHT_M_S)


def free_space_path_loss_db(distance_km, cfg: LinkBudgetConfig):
    """FSPL in dB. Zero-distance entries are returned as +inf loss."""
    d_m = np.asarray(distance_km, dtype=np.float64) * 1000.0
    with np.errstate(divide="ignore"):
        loss = (
            20.0 * np.log10(d_m)
            + 20.0 * np.log10(cfg.carrier_hz)
            + _FSPL_CONST_DB
        )
    return np.where(d_m > 0.0, loss, np.inf)


def noise_power_dbw(cfg: LinkBudgetConfig) -> float:
    """Thermal noise floor N = k*T*B over the channel bandwidth."""
    return 10.0 * np.log10(BOLTZMANN_J_K * cfg.noise_temperature_k * cfg.bandwidth_hz)


def snr_db(distance_km, cfg: LinkBudgetConfig):
    """Receive SNR in dB.

    Pr = Pt + Gt + Gr - FSPL - L_pol - L_misalign
    SNR = Pr - N
    """
    received_dbw = (
        cfg.tx_power_dbw
        + cfg.total_gain_dbi
        - free_space_path_loss_db(distance_km, cfg)
        - cfg.total_fixed_loss_db
    )
    return received_dbw - noise_power_dbw(cfg)


def capacity_bps(distance_km, cfg: LinkBudgetConfig):
    """Shannon capacity C = B * log2(1 + SNR) in bits/s.

    Strictly decreasing in distance for all distance > 0.
    """
    snr_linear = np.power(10.0, snr_db(distance_km, cfg) / 10.0)
    return cfg.bandwidth_hz * np.log2(1.0 + snr_linear)


def latency_s(distance_km):
    """One-way propagation delay. Vacuum ISLs, so no medium slowdown factor."""
    return np.asarray(distance_km, dtype=np.float64) * 1000.0 / SPEED_OF_LIGHT_M_S


def link_metrics(
    distance_km: np.ndarray,
    visible: np.ndarray,
    cfg: LinkBudgetConfig,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Capacity and latency for one block, zeroed where no link exists."""
    safe = np.where(visible, distance_km, 1.0)  # avoid log10(0) on masked entries
    capacity = np.where(visible, capacity_bps(safe, cfg), 0.0).astype(dtype)
    latency = np.where(visible, latency_s(safe), 0.0).astype(dtype)
    return capacity, latency
