# Geometric_Nominal: DT nominal channel, pure-LoS h from (aged) geometry.
# Aged geometry: AR(1) elev/az error, rho=exp(-dt/tau). std=0 => off (deterministic).
# Knobs: geo_nom_other_loss_db, geo_nom_aged_elev_std_deg, geo_nom_aged_az_std_deg,
#  geo_nom_aged_corr_time_s.
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model


@register_model("Geometric_Nominal")
class GeometricNominal(AbstractChannelModel):
    """Pure-geometry nominal channel for the digital-twin pipeline.

    Per-antenna LoS-only: h = a_steering(elev, az). No Rician fading.
    """

    # No overlap with the auto-attached components.
    handles_internally = frozenset()
    produces_per_antenna_h = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.other_loss_db = float(getattr(cfg, "geo_nom_other_loss_db", 0.5))
        # Aged/noisy-geometry model. Steady-state std of the position-estimate
        # error in elevation/azimuth [deg], and its correlation time [s].
        # Defaults 0.0 => aging OFF (clean geometric reference, deterministic).
        self.el_std   = float(getattr(cfg, "geo_nom_aged_elev_std_deg", 0.0))
        self.az_std   = float(getattr(cfg, "geo_nom_aged_az_std_deg",   0.0))
        self.aged_tau = float(getattr(cfg, "geo_nom_aged_corr_time_s",  0.0))

    def _ar1_error(self, key, n, rho, std, state, rng):
        """Zero-mean AR(1) Gauss-Markov geometry-estimate error, steady-state std
        `std` [deg], correlation `rho`. Per-link state keyed by n so a change in the
        visible-link set resets it. rho=0 => iid per tick; rho->1 => frozen bias."""
        z = (rng.standard_normal(n) if hasattr(rng, "standard_normal")
             else rng.randn(n))
        kk = f"{key}_prev_n{n}"
        prev = state.data.get(kk) if state is not None else None
        if prev is None or np.asarray(prev).shape != (n,):
            e = std * z                                   # cold start at the marginal
        else:
            e = rho * prev + np.sqrt(max(1.0 - rho * rho, 0.0)) * std * z
        if state is not None:
            state.data[kk] = e
        return e

    def compute_channel(self, dist_m, elev_deg, dt_sec,
                        state=None, pos_m=None, range_rate_m_s=None,
                        azimuth_deg=None):
        from ntnconnect.registry.channels._steering import per_antenna_los_only, off_nadir_rad
        d    = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n    = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        # Aged/noisy geometry: the nominal is what the twin computes from an
        # IMPERFECT position estimate. Add a zero-mean AR(1) error to elevation/
        # azimuth (correlation rho = exp(-dt/tau); tau=0 => iid; std=0 => off).
        aged = (self.el_std > 0.0) or (self.az_std > 0.0)
        if aged:
            dt  = float(dt_sec) if dt_sec is not None else 0.0
            rho = float(np.exp(-dt / self.aged_tau)) if self.aged_tau > 0.0 else 0.0
            rng = getattr(state, "rng", None) or np.random.default_rng()
            if self.el_std > 0.0:
                elev = np.clip(elev + self._ar1_error("geo_el", n, rho, self.el_std,
                                                       state, rng), 0.0, 90.0)
            if self.az_std > 0.0:
                az = az + self._ar1_error("geo_az", n, rho, self.az_std, state, rng)
        # FSPL owned by FSPLComponent. We add only the fixed "other loss".
        pl_db = np.full(n, self.other_loss_db)
        # Per-antenna LoS direction only (no fading).
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        d_lambda  = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz= float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        strategy  = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()
        h = np.zeros((n, n_x * n_y), dtype=complex)
        for i in range(n):
            theta = float(off_nadir_rad(elev[i], sat_alt_m))
            phi   = float(np.radians(az[i]))
            h[i]  = per_antenna_los_only(theta, phi, n_x, n_y,
                                          strategy=strategy,
                                          dist_m=float(d[i]),
                                          carrier_hz=carrier_hz,
                                          d_over_lambda=d_lambda)
        return ChannelData(
            loss_db=pl_db, h=h, doppler_hz=np.zeros(n),
            extras={"geo_nominal": True, "geo_nom_aged": bool(aged)},
        )
