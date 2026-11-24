# Lutz (1991) two-state Markov LMS: good=Rician, bad=lognormal-shadowed Rayleigh.
# lutz_mode: iid | correlated (Clarke diffuse + slow AR-1 bad-state shadow).

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


@register_model("Lutz")
class Lutz(AbstractChannelModel):
    """Two-state Markov: 'good' = Rician LOS, 'bad' = log-normal shadowed
    Rayleigh; selectable per-state temporal fading (iid / correlated)."""

    has_intrinsic_spatial_structure = False
    consumes_external_k_factor    = False
    produces_per_antenna_h = True
    channel_class = "complete_lms"
    # Lutz models log-normal shadowing inside h (the bad-state log-normal Rayleigh),
    # so the pipeline must not also add an external shadowing component.
    handles_internally = frozenset({"shadowing", "shadowing_38811",
                                    "shadowing_gudmundson"})

    def __init__(self, cfg):
        super().__init__(cfg)
        self._P = np.asarray(cfg.lutz_P)
        self._k_db = float(cfg.lutz_k_factor_db)
        self._mu_bad = float(cfg.lutz_mu_bad)
        self._sigma_bad = float(cfg.lutz_sigma_bad)

    def _transition(self, n, state, rng):
        prev = state.data.get("lutz_state") if state is not None else None
        if prev is None or len(prev) != n:
            prev = np.zeros(n, dtype=int)
        rand = rng.random(n)
        new = np.where(prev == 0,
                       np.where(rand < self._P[0, 1], 1, 0),
                       np.where(rand < self._P[1, 0], 0, 1))
        if state is not None:
            state.data["lutz_state"] = new
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
        k_lin = 10.0 ** (self._k_db / 10.0)
        los_w_good = float(np.sqrt(k_lin / (k_lin + 1.0)))
        nlos_w_good = float(np.sqrt(1.0 / (k_lin + 1.0)))

        if str(getattr(self.cfg, "lutz_mode", "iid")).lower() == "correlated" \
                and state is not None:
            from ntnconnect.registry.channels._fading import diffuse, slow_lognormal_amp, doppler_spread_hz
            A = steering_matrix_batch(off_nadir_rad(elev, sat_alt_m), np.radians(az),
                                      n_x, n_y, strategy=strategy, dist_m=d,
                                      carrier_hz=carrier_hz, d_over_lambda=d_lambda)
            f_d = doppler_spread_hz(self.cfg, carrier_hz)
            g = diffuse("ar1", n, n_tx, f_d, float(dt_sec), state, "lutz", rng)
            if bool(getattr(self.cfg, "rician_unit_mean_fading", True)):
                g = g * np.sqrt(n_tx)          # per-element unit variance
            amp_bad = slow_lognormal_amp(self._mu_bad, self._sigma_bad, n,
                                         float(getattr(self.cfg, "lutz_shadow_rho", 0.99)),
                                         state, "lutz_bad", rng)
            good = (new == 0)[:, None]
            h = np.where(good, los_w_good * A + nlos_w_good * g,
                         amp_bad[:, None] * g)
            state.data["model_did_temporal_fading"] = True
            return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                               extras={"lutz_state": new, "lutz_mode": "correlated"})

        # "iid" (default): per-state white fading.
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
            if new[i] == 0:
                h[i] = los_w_good * a_vec + nlos_w_good * nlos
            else:
                s_db = self._mu_bad + self._sigma_bad * rng.standard_normal()
                h[i] = (10.0 ** (s_db / 20.0)) * nlos
        if state is not None:
            state.data["model_did_temporal_fading"] = False
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"lutz_state": new, "lutz_mode": "iid"})
