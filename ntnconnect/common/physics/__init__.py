"""ntnconnect.common.physics - single-sourced physical-layer math.

Pure, stateless functions shared by every radio stage so the SINR / rate /
energy-efficiency formulas live in exactly one place. Stages differ only in how
they compute `signal_w` (full-power, allocated, MU-MIMO-beamformed); the metric
math is unified here.
"""
from ntnconnect.common.physics.link_metrics import (
    sinr_db_from_powers, snr_db_from_powers, spectral_efficiency_bps_hz,
    mimo_spectral_efficiency, spatial_streams,
    served_mask_from_assignments, served_link_metrics, energy_kpis,
    channel_power_gain, propagation_delay_ns,
)

__all__ = [
    "sinr_db_from_powers", "snr_db_from_powers", "spectral_efficiency_bps_hz",
    "mimo_spectral_efficiency", "spatial_streams",
    "served_mask_from_assignments", "served_link_metrics", "energy_kpis",
    "channel_power_gain", "propagation_delay_ns",
]
