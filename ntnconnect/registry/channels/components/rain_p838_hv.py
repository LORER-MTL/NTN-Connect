# Rain (ITU-R P.838) with H/V/circular polarization. rain_polarization: H|V|circular.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

_F = np.array([4., 6., 10., 12., 15., 20., 25., 30., 40., 50.])
# k, alpha for horizontal (H) - ITU-R P.838-3 Table 3.
_KH = np.array([0.000650, 0.00175, 0.0101, 0.0188, 0.0367, 0.0751, 0.124, 0.187, 0.350, 0.536])
_AH = np.array([1.121,    1.308,   1.276,  1.217,  1.154,  1.099,  1.061, 1.021, 0.939, 0.873])
# k, alpha for vertical (V).
_KV = np.array([0.000591, 0.00155, 0.00887,0.0168, 0.0335, 0.0691, 0.113, 0.167, 0.310, 0.479])
_AV = np.array([1.075,    1.265,   1.264,  1.200,  1.128,  1.065,  1.030, 1.000, 0.929, 0.868])


def _interp_k_alpha(f_ghz, K_table, A_table):
    f_clip = float(np.clip(f_ghz, _F[0], _F[-1]))
    k = 10.0 ** np.interp(np.log10(f_clip), np.log10(_F), np.log10(K_table))
    a = float(np.interp(f_clip, _F, A_table))
    return float(k), a


class RainP838HVComponent(AbstractChannelComponent):
    name  = "rain_p838_hv"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_rain_p838_hv", False)):
            return
        if data.loss_db is None:
            return
        f_ghz = float(cfg.carrier_freq_hz) / 1.0e9
        pol = str(getattr(cfg, "rain_polarization", "circular")).lower()
        if pol == "h":
            k, a = _interp_k_alpha(f_ghz, _KH, _AH)
        elif pol == "v":
            k, a = _interp_k_alpha(f_ghz, _KV, _AV)
        else:                                           # circular (tilt 45°), P.838
            kh, ah = _interp_k_alpha(f_ghz, _KH, _AH)
            kv, av = _interp_k_alpha(f_ghz, _KV, _AV)
            # ITU-R P.838: k = (kH+kV)/2; alpha = (kH*aH + kV*aV)/(kH+kV)
            k = 0.5 * (kh + kv)
            a = (kh * ah + kv * av) / (kh + kv)
            pol = "circular"
        R_p = float(getattr(cfg, "p618_rain_rate_mmph", 25.0))
        h_R = float(getattr(cfg, "p618_rain_height_km", 3.0))
        elev = np.clip(np.asarray(data.extras.get("elev_deg",
                                                  np.zeros(data.n_links())), dtype=float),
                       1e-3, 90.0)
        L_s = h_R / np.sin(np.radians(elev))
        L_0 = 35.0 * np.exp(-0.015 * R_p)
        nu  = 1.0 / (1.0 + L_s / L_0)
        rain_db = k * (R_p ** a) * L_s * nu
        data.loss_db = np.asarray(data.loss_db, dtype=float) + rain_db
        data.extras["rain_db"] = rain_db
        data.extras["rain_model"] = f"p838_{pol}"
