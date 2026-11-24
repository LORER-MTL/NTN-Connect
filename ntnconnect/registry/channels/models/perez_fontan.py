# Perez-Fontan three-state Markov LMS: per-state Loo (lognormal LoS + Rayleigh).
# pf_mode: iid | correlated (slow AR-1 shadow + Clarke multipath). Ref: ITU-R P.681.

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


@register_model("Perez_Fontan")
class PerezFontan(AbstractChannelModel):
    """Three-state Markov LMS with a per-state Loo process; selectable
    intra-state temporal fading (iid / correlated)."""

    has_intrinsic_spatial_structure = False
    consumes_external_k_factor    = False
    produces_per_antenna_h = True
    channel_class = "complete_lms"
    # Perez-Fontan models log-normal shadowing inside h (the per-state log-normal
    # Loo amplitude), so the pipeline must not also add an external shadowing
    # component.
    handles_internally = frozenset({"shadowing", "shadowing_38811",
                                    "shadowing_gudmundson"})

    def __init__(self, cfg):
        super().__init__(cfg)
        self._P  = np.asarray(cfg.pf_P)
        self._mu = np.asarray(cfg.pf_mu)
        self._sg = np.asarray(cfg.pf_sigma)
        self._mv = np.asarray(cfg.pf_multipath_var)

    def _transition(self, n, state, rng):
        prev = state.data.get("pf_state") if state is not None else None
        if prev is None or len(prev) != n:
            prev = np.zeros(n, dtype=int)
        new = prev.copy()
        for i in range(n):
            row = self._P[prev[i]]
            new[i] = int(np.searchsorted(np.cumsum(row), rng.random()))
        if state is not None:
            state.data["pf_state"] = new
        return new

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        from ntnconnect.registry.channels._steering import (get_steering_strategy, steering_matrix_batch,
                                 off_nadir_rad)
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        rng = getattr(state, "rng", None) or np.random
        new = self._transition(n, state, rng)

        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_lambda = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        strategy = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()
        sqrt_mp = np.sqrt(self._mv[new])

        if str(getattr(self.cfg, "pf_mode", "iid")).lower() == "correlated" \
                and state is not None:
            from ntnconnect.registry.channels._fading import diffuse, slow_lognormal_amp, doppler_spread_hz
            A = steering_matrix_batch(off_nadir_rad(elev, sat_alt_m), np.radians(az),
                                      n_x, n_y, strategy=strategy, dist_m=d,
                                      carrier_hz=carrier_hz, d_over_lambda=d_lambda)
            f_d = doppler_spread_hz(self.cfg, carrier_hz)
            g = diffuse("ar1", n, n_tx, f_d, float(dt_sec), state, "pf", rng)
            if bool(getattr(self.cfg, "rician_unit_mean_fading", True)):
                g = g * np.sqrt(n_tx)          # per-element unit variance
            amp = slow_lognormal_amp(self._mu[new], self._sg[new], n,
                                     float(getattr(self.cfg, "pf_shadow_rho", 0.99)),
                                     state, "pf_los", rng)
            h = amp[:, None] * A + sqrt_mp[:, None] * g
            state.data["model_did_temporal_fading"] = True
            return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                               extras={"pf_state": new, "pf_mode": "correlated"})

        # "iid" (default): white intra-state Loo process.
        s_db = self._mu[new] + self._sg[new] * rng.standard_normal(n)
        amp = 10.0 ** (s_db / 20.0)
        strat_fn = get_steering_strategy(strategy)
        nlos_denom = (np.sqrt(2.0)
                      if bool(getattr(self.cfg, "rician_unit_mean_fading", True))
                      else np.sqrt(2.0 * n_tx))
        h = np.zeros((n, n_tx), dtype=complex)
        for i in range(n):
            theta = float(off_nadir_rad(elev[i], sat_alt_m))
            phi = float(np.radians(az[i]))
            if strategy == "upa_kron":
                a_vec = strat_fn(theta, phi, n_x, n_y, d_lambda)
            elif strategy == "ula":
                a_vec = strat_fn(theta, n_tx, d_lambda)
            else:
                a_vec = strat_fn(theta, phi, n_x, n_y, float(d[i]), carrier_hz, d_lambda)
            nlos = (rng.standard_normal(n_tx)
                    + 1j*rng.standard_normal(n_tx)) / nlos_denom
            h[i] = amp[i] * a_vec + sqrt_mp[i] * nlos
        if state is not None:
            state.data["model_did_temporal_fading"] = False
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"pf_state": new, "pf_mode": "iid"})
