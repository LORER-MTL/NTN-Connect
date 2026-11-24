# Spatial Geometric Rician: per-antenna Rician (like 3GPP). Distance phase owned
# by PropagationDelayComponent. sgr_fading_mode: iid | ar1 | sos.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


@register_model("Spatial_Geometric_Rician")
class SpatialGeometricRician(AbstractChannelModel):
    """Per-antenna Rician with a geometric LoS phase exp(j 2π f d/c) and a
    selectable temporal fading model (iid / ar1 / sos)."""

    produces_per_antenna_h = True

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        from ntnconnect.registry.channels._steering import (per_antenna_rician_batch, steering_matrix_batch,
                                 off_nadir_rad)
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        k_db = (state.data.get("k_factor_db_pending") if state is not None else None)
        if k_db is None or np.asarray(k_db).size != n:
            k_db = np.full(n, float(getattr(self.cfg, "threegpp_k_factor_los_db", 10.0)))
        else:
            k_db = np.asarray(k_db, dtype=float)
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_lambda = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        unit_mean = bool(getattr(self.cfg, "rician_unit_mean_fading", True))
        rng = getattr(state, "rng", None) or np.random
        theta = off_nadir_rad(elev, sat_alt_m)
        phi = np.radians(az)

        mode = str(getattr(self.cfg, "sgr_fading_mode", "iid")).lower()
        if mode in ("ar1", "sos") and state is not None:
            from ntnconnect.registry.channels._fading import diffuse, doppler_spread_hz
            A = steering_matrix_batch(theta, phi, n_x, n_y,
                                      strategy="upa_geometric_los", dist_m=d,
                                      carrier_hz=carrier_hz, d_over_lambda=d_lambda)
            k_lin = 10.0 ** (k_db / 10.0)
            los_w = np.sqrt(k_lin / (k_lin + 1.0))[:, None]
            nlos_w = np.sqrt(1.0 / (k_lin + 1.0))[:, None]
            f_d = doppler_spread_hz(self.cfg, carrier_hz)
            g = diffuse(mode, n, n_tx, f_d, float(dt_sec), state, "sgr", rng,
                        sos_M=int(getattr(self.cfg, "threegpp_sos_M", 16)))
            if unit_mean:
                g = g * np.sqrt(n_tx)          # E[||g||^2] = n_tx (unit-mean fading)
            h = los_w * A + nlos_w * g
            state.data["model_did_temporal_fading"] = True
        else:
            mode = "iid"
            h = per_antenna_rician_batch(
                k_db, theta, phi, n_x, n_y, strategy="upa_geometric_los",
                dist_m=d, carrier_hz=carrier_hz, d_over_lambda=d_lambda,
                unit_mean=unit_mean, rng=rng)
            if state is not None:
                state.data["model_did_temporal_fading"] = False
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"k_factor_db": k_db, "fading_mode": mode})
