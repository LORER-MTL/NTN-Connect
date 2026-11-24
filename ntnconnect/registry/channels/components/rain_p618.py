# Rain attenuation (ITU-R P.618/P.838, simplified): A = k*R_p^alpha*L_eff [dB].
# Simplified slant path (not full P.618).
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent

# k,alpha (P.838, circular polarization). Sparse table - linear-log interpolation.
_F_GHZ = np.array([4.0, 6.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
_K     = np.array([0.000591, 0.00175, 0.0101, 0.0188, 0.0367, 0.0751, 0.124, 0.187, 0.350, 0.536])
_ALPHA = np.array([1.121,    1.308,   1.276,  1.217,  1.154,  1.099,  1.061, 1.021, 0.939, 0.873])


def _k_alpha(f_ghz: float) -> tuple[float, float]:
    f_ghz = float(np.clip(f_ghz, _F_GHZ[0], _F_GHZ[-1]))
    # log-frequency interpolation in k (k spans many orders of magnitude)
    log_f = np.log10(_F_GHZ)
    log_k = np.log10(_K)
    k     = 10.0 ** np.interp(np.log10(f_ghz), log_f, log_k)
    alpha = float(np.interp(f_ghz, _F_GHZ, _ALPHA))
    return float(k), alpha


class RainP618Component(AbstractChannelComponent):
    name  = "rain_p618"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_rain_p618", False)):
            return
        if data.loss_db is None:
            return
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(data.n_links())), dtype=float)
        elev = np.clip(elev, 1e-3, 90.0)                  # avoid div-by-zero

        f_ghz = float(cfg.carrier_freq_hz) / 1.0e9
        k, alpha = _k_alpha(f_ghz)

        R_p   = float(getattr(cfg, "p618_rain_rate_mmph", 25.0))
        h_R   = float(getattr(cfg, "p618_rain_height_km", 3.0))   # effective rain height
        # Slant-path length through the rain layer
        L_s   = h_R / np.sin(np.radians(elev))                    # km
        # P.618 path-length adjustment (§2.2.1.3, simplified)
        L_0   = 35.0 * np.exp(-0.015 * R_p)                       # km
        nu    = 1.0 / (1.0 + L_s / L_0)
        L_eff = L_s * nu

        rain_db = k * (R_p ** alpha) * L_eff                      # dB
        data.loss_db = np.asarray(data.loss_db, dtype=float) + rain_db
        data.extras["rain_db"] = rain_db
        data.extras["rain_model"] = "p618"
