# Steering lift: scalar h -> per-antenna Rician (cfg.steering_strategy).
# DEPRECATED no-op: all registered models already produce per-antenna h.
from __future__ import annotations
import logging
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent
from ntnconnect.registry.channels._steering import (
    off_nadir_rad,
    get_steering_strategy,
    list_steering_strategies,
)

_log = logging.getLogger(__name__)


class SteeringUPAComponent(AbstractChannelComponent):
    name  = "steering_upa"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_steering_upa", False)):
            return
        if data.h is None:
            return
        h_in = np.asarray(data.h)
        # Idempotent for already per-antenna input (models that declare
        # `produces_per_antenna_h = True` or return ndim > 1).
        model_per_ant = bool(
            state and state.data.get("model_produces_per_antenna_h", False))
        if h_in.ndim > 1 or model_per_ant:
            return

        n_links = int(h_in.size)
        if n_links == 0:
            return

        # Pick the strategy.
        strat_name = str(getattr(cfg, "steering_strategy", "upa_kron")).lower()
        try:
            strat_fn = get_steering_strategy(strat_name)
        except KeyError:
            _log.warning("steering_strategy='%s' not recognised; "
                         "falling back to upa_kron. Available: %s",
                         strat_name, list_steering_strategies())
            strat_fn = get_steering_strategy("upa_kron")
            strat_name = "upa_kron"

        # Geometry + K + array parameters
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(n_links)), dtype=float)
        az   = np.asarray(data.extras.get("azimuth_deg",
                                          np.zeros(n_links)), dtype=float)
        k_db = np.asarray(data.extras.get("k_factor_db",
                                          np.full(n_links, 10.0)), dtype=float)
        dist = np.asarray(data.extras.get("dist_m",
                                          np.full(n_links, 1.0)), dtype=float)

        n_x = int(getattr(cfg, "n_tx_x", 8))
        n_y = int(getattr(cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_over_lambda = float(getattr(cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m  = float(getattr(cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(cfg, "carrier_freq_hz", 20.0e9))

        # Per-link Rician decomposition.
        K_lin  = 10.0 ** (k_db / 10.0)
        los_w  = np.sqrt(K_lin / (K_lin + 1.0))
        nlos_w = np.sqrt(1.0   / (K_lin + 1.0))
        rng = getattr(state, "rng", None) or np.random
        nlos_re = rng.standard_normal((n_links, n_tx))
        nlos_im = rng.standard_normal((n_links, n_tx))
        nlos    = (nlos_re + 1j * nlos_im) / np.sqrt(2.0 * n_tx)

        # Vectorised LOS steering for all links at once (was a per-link
        # Python loop calling strat_fn n_links times). The steering phases are
        # closed-form, so this is BIT-EXACT - the NLOS draws above are already
        # vectorised and the LOS part is deterministic.
        theta = np.asarray(off_nadir_rad(elev, sat_alt_m), dtype=float)  # (n_links,)
        phi   = np.radians(az)                                           # (n_links,)
        if strat_name in ("upa_kron", "upa_geometric_los"):
            KX, KY = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
            KXf = KX.ravel()[None, :]; KYf = KY.ravel()[None, :]        # (1, n_tx)
            phase = (2.0*np.pi*d_over_lambda) * np.sin(theta)[:, None] * (
                KXf*np.cos(phi)[:, None] + KYf*np.sin(phi)[:, None])     # (n_links,n_tx)
            A = np.exp(1j*phase)
            # upa_geometric_los == upa_kron; distance phase owned by
            # PropagationDelayComponent (no bulk phase here).
        elif strat_name == "ula":
            phase = (2.0*np.pi*d_over_lambda) * np.sin(theta)[:, None] * \
                    np.arange(n_tx)[None, :]
            A = np.exp(1j*phase)
        else:
            A = np.empty((n_links, n_tx), dtype=complex)                 # rare path
            for r in range(n_links):
                A[r, :] = strat_fn(float(theta[r]), float(phi[r]),
                                   n_x, n_y, d_over_lambda)
        h_per_ant = los_w[:, None] * A + nlos_w[:, None] * nlos
        data.h = h_per_ant
        data.extras["steering_strategy"] = strat_name
