# Loo (1985) shadowed-Rician LMS: lognormal LoS + Rayleigh multipath.
# loo_mode: iid | correlated (slow AR-1 shadow + Clarke multipath). Ref: ITU-R P.681.
from __future__ import annotations
import numpy as np
from scipy.special import j0
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


@register_model("Loo")
class Loo(AbstractChannelModel):
    """Per-antenna shadowed Rician (Loo 1985), with a selectable temporal model
    (iid simplified / correlated full)."""

    has_intrinsic_spatial_structure = False
    consumes_external_k_factor    = False
    produces_per_antenna_h = True
    channel_class = "complete_lms"
    # Loo models log-normal shadowing inside h (the log-normal LoS amplitude), so
    # the pipeline must not also add an external shadowing component.
    handles_internally = frozenset({"shadowing", "shadowing_38811",
                                    "shadowing_gudmundson"})

    def _los_amp_iid(self, mu, sigma, n, rng):
        s_db = mu + sigma * rng.standard_normal(n)
        return 10.0 ** (s_db / 20.0)

    def _los_amp_correlated(self, mu, sigma, n, state, rng):
        """Slowly-varying lognormal LoS amplitude via an AR-1 log-shadow process."""
        rho = float(np.clip(getattr(self.cfg, "loo_los_rho", 0.995), 0.0, 0.99999))
        key = f"loo_los_g_n{n}"
        g_prev = state.data.get(key) if state is not None else None
        z = rng.standard_normal(n)
        if g_prev is None or np.asarray(g_prev).shape != (n,):
            g = z                                       # cold start at the marginal
        else:
            g = rho * g_prev + np.sqrt(1.0 - rho * rho) * z
        if state is not None:
            state.data[key] = g
        s_db = mu + sigma * g                           # 20 log10 z ~ N(mu, sigma^2)
        return 10.0 ** (s_db / 20.0)

    def _multipath_correlated(self, n, n_tx, f_d, dt_sec, state, rng):
        """Doppler-correlated Rayleigh multipath, E[||w||^2] = 1 (before mp scale)."""
        key = f"loo_mp_prev_n{n}_{n_tx}"
        w = (rng.standard_normal((n, n_tx))
             + 1j * rng.standard_normal((n, n_tx))) / np.sqrt(2.0 * n_tx)
        prev = state.data.get(key) if state is not None else None
        if prev is None or getattr(prev, "shape", None) != w.shape:
            g = w
        else:
            rho = j0(2.0 * np.pi * f_d * dt_sec) if (f_d > 0.0 and dt_sec > 0.0) else 1.0
            rho = float(np.clip(rho, -0.9999, 0.9999))
            g = rho * prev + np.sqrt(1.0 - rho * rho) * w
        if state is not None:
            state.data[key] = g
        return g

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        from ntnconnect.registry.channels._steering import (get_steering_strategy, steering_matrix_batch,
                                 off_nadir_rad)
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        mu     = float(self.cfg.loo_mu)
        sigma  = float(self.cfg.loo_sigma)
        mp_var = float(self.cfg.loo_multipath_var)
        sqrt_mp = float(np.sqrt(mp_var))
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_lambda = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        strategy = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()
        theta = off_nadir_rad(elev, sat_alt_m)
        phi = np.radians(az)

        mode = str(getattr(self.cfg, "loo_mode", "iid")).lower()
        rng = getattr(state, "rng", None) or np.random
        # Per-element unit-variance multipath (E[||w||^2]=n_tx) so the shadowed-
        # Rician LoS/multipath ratio is not inflated by n_tx. False = legacy.
        unit_mean = bool(getattr(self.cfg, "rician_unit_mean_fading", True))
        nlos_denom = np.sqrt(2.0) if unit_mean else np.sqrt(2.0 * n_tx)

        if mode == "correlated" and state is not None:
            a_los = self._los_amp_correlated(mu, sigma, n, state, rng)
            A = steering_matrix_batch(theta, phi, n_x, n_y, strategy=strategy,
                                      dist_m=d, carrier_hz=carrier_hz,
                                      d_over_lambda=d_lambda)
            v_ms = float(getattr(self.cfg, "ue_speed_mps", 0.0))
            f_d = abs(v_ms) * carrier_hz / C
            w = self._multipath_correlated(n, n_tx, f_d, float(dt_sec), state, rng)
            if unit_mean:
                w = w * np.sqrt(n_tx)          # E[||w||^2]: 1 -> n_tx
            h = a_los[:, None] * A + sqrt_mp * w
            state.data["model_did_temporal_fading"] = True   # skip pipeline Jakes
            return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                               extras={"loo_mode": mode, "fading_doppler_hz": f_d})

        # "iid" (default): direct construction, white in time.
        strat_fn = get_steering_strategy(strategy)
        a_los = self._los_amp_iid(mu, sigma, n, rng)
        h = np.zeros((n, n_tx), dtype=complex)
        for i in range(n):
            th = float(off_nadir_rad(elev[i], sat_alt_m))
            ph = float(np.radians(az[i]))
            if strategy == "upa_kron":
                a_vec = strat_fn(th, ph, n_x, n_y, d_lambda)
            elif strategy == "ula":
                a_vec = strat_fn(th, n_tx, d_lambda)
            else:
                a_vec = strat_fn(th, ph, n_x, n_y, float(d[i]), carrier_hz, d_lambda)
            nlos = (rng.standard_normal(n_tx)
                    + 1j*rng.standard_normal(n_tx)) / nlos_denom
            h[i] = a_los[i] * a_vec + sqrt_mp * nlos
        if state is not None:
            state.data["model_did_temporal_fading"] = False   # white -> pipeline may filter
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"loo_mode": "iid"})
