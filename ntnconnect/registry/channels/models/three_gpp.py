# 3GPP TR 38.811 §6.7 Rician fast fading: h = sqrt(K/(K+1))a + sqrt(1/(K+1))g.
# fading mode (threegpp_fading_mode): iid | ar1 | sos. f_D = ue_speed*f_c/c.
# K: threegpp_k_mode constant|elevation, or external k_factor_db_pending.
from __future__ import annotations
import numpy as np
from scipy.special import j0
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model
from ntnconnect.registry.channels.components.k_factor_38811 import _EL_REF, _MU_REF   # TR 38.811 mean-K table

C = 299_792_458.0


@register_model("3GPP")
class ThreeGPP_TR38811(AbstractChannelModel):
    """Per-antenna Rician fading with a selectable temporal model
    (iid / ar1 / sos). FSPL & K-factor come from pre-model components when
    enabled; steering vectors from the shared helpers."""

    produces_per_antenna_h = True

    def _k_db(self, elev, n, state):
        """Resolve the per-link K-factor (dB): external component > k_mode."""
        k_pending = (state.data.get("k_factor_db_pending")
                     if state is not None else None)
        if k_pending is not None and np.asarray(k_pending).size == n:
            return np.asarray(k_pending, dtype=float)
        if str(getattr(self.cfg, "threegpp_k_mode", "constant")).lower() == "elevation":
            return np.interp(elev, _EL_REF, _MU_REF)          # TR 38.811 Tbl 6.7.2
        return np.full(n, float(getattr(self.cfg, "threegpp_k_factor_los_db", 10.0)))

    def _white_nlos(self, n, n_tx, rng):
        """White diffuse draw with E[||g||^2] = 1 (same convention as the
        vectorised Rician helper's default)."""
        buf = (rng.standard_normal((n, 2 * n_tx)) if hasattr(rng, "standard_normal")
               else rng.randn(n, 2 * n_tx))
        return (buf[:, :n_tx] + 1j * buf[:, n_tx:]) / np.sqrt(2.0 * n_tx)

    def _diffuse(self, mode, n, n_tx, f_d, dt_sec, state, rng):
        """Time-correlated diffuse term g of shape (n, n_tx), E[||g||^2] = 1.
        State is keyed by (n, n_tx) so a change in the visible-link set resets
        the correlation (same convention as the K-factor / Markov models)."""
        if mode == "ar1":
            key = f"threegpp_nlos_ar1_n{n}_{n_tx}"
            w = self._white_nlos(n, n_tx, rng)
            prev = state.data.get(key) if state is not None else None
            if prev is None or getattr(prev, "shape", None) != w.shape:
                g = w                                          # cold start
            else:
                rho = j0(2.0 * np.pi * f_d * dt_sec) if (f_d > 0.0 and dt_sec > 0.0) else 1.0
                rho = float(np.clip(rho, -0.9999, 0.9999))
                g = rho * prev + np.sqrt(1.0 - rho * rho) * w
            if state is not None:
                state.data[key] = g
            return g
        # "sos" - Clarke/Jakes sum-of-sinusoids
        M = max(4, int(getattr(self.cfg, "threegpp_sos_M", 16)))
        key_ph = f"threegpp_sos_phi_n{n}_{n_tx}_M{M}"
        key_t = f"threegpp_sos_t_n{n}_{n_tx}_M{M}"
        phi = state.data.get(key_ph) if state is not None else None
        if phi is None or getattr(phi, "shape", None) != (n, n_tx, M):
            phi = (rng.uniform(-np.pi, np.pi, size=(n, n_tx, M))
                   if hasattr(rng, "uniform")
                   else np.random.uniform(-np.pi, np.pi, size=(n, n_tx, M)))
            t = 0.0
        else:
            t = float(state.data.get(key_t, 0.0)) + float(dt_sec)
        alpha = 2.0 * np.pi * (np.arange(M) + 0.5) / M          # incident angles
        theta_t = (2.0 * np.pi * f_d * t) * np.cos(alpha)       # (M,)
        osc = np.exp(1j * (theta_t[None, None, :] + phi))       # (n, n_tx, M)
        g = osc.sum(axis=2) / np.sqrt(M)                        # E[|elem|^2] = 1
        g = g / np.sqrt(n_tx)                                   # -> E[||g||^2] = 1
        if state is not None:
            state.data[key_ph] = phi
            state.data[key_t] = t
        return g

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        from ntnconnect.registry.channels._steering import (per_antenna_rician_batch, steering_matrix_batch,
                                 off_nadir_rad)
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        k_db = self._k_db(elev, n, state)

        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_lambda = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        strategy = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()
        theta = off_nadir_rad(elev, sat_alt_m)
        phi = np.radians(az)

        mode = str(getattr(self.cfg, "threegpp_fading_mode", "iid")).lower()
        v_ms = float(getattr(self.cfg, "ue_speed_mps", 0.0))
        f_d = abs(v_ms) * carrier_hz / C                        # diffuse Doppler spread
        # Per-element unit-variance diffuse (E[||g||^2]=n_tx) -> realized K == K.
        # rician_unit_mean_fading=False restores the legacy inflated-K behaviour.
        unit_mean = bool(getattr(self.cfg, "rician_unit_mean_fading", True))
        rng = getattr(state, "rng", None) or np.random

        if mode in ("ar1", "sos") and state is not None:
            # Rician with a time-correlated diffuse term.
            A = steering_matrix_batch(theta, phi, n_x, n_y, strategy=strategy,
                                      dist_m=d, carrier_hz=carrier_hz,
                                      d_over_lambda=d_lambda)
            k_lin = 10.0 ** (k_db / 10.0)
            los_w = np.sqrt(k_lin / (k_lin + 1.0))[:, None]
            nlos_w = np.sqrt(1.0 / (k_lin + 1.0))[:, None]
            g = self._diffuse(mode, n, n_tx, f_d, float(dt_sec), state, rng)
            if unit_mean:
                g = g * np.sqrt(n_tx)          # E[||g||^2]: 1 -> n_tx (per-element var 1)
            h = los_w * A + nlos_w * g
        else:
            # "iid" block fading (default).
            mode = "iid"
            h = per_antenna_rician_batch(k_db, theta, phi, n_x, n_y,
                                         strategy=strategy, dist_m=d,
                                         carrier_hz=carrier_hz, d_over_lambda=d_lambda,
                                         unit_mean=unit_mean, rng=rng)
        # Tell the pipeline whether temporal fading was already applied here, so
        # the Jakes component does not filter h a second time (no double calc).
        if state is not None:
            state.data["model_did_temporal_fading"] = (mode in ("ar1", "sos"))
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"k_factor_db": k_db,
                                   "fading_mode": mode, "fading_doppler_hz": f_d})
