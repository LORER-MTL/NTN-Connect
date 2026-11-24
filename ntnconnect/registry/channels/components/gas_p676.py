# Gas absorption (ITU-R P.676, simplified O2+H2O fit): A = gamma*h_eff/sin(el).
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


def _gamma_o2_dB_per_km(f_ghz):
    """Simplified O2 attenuation around 60 GHz (Liebig approximation)."""
    f = f_ghz
    return (7.27 / (f * f + 0.351)
            + (7.5 / ((f - 57.0) ** 2 + 2.44))
            + (0.0021 * f * f)) * 1.0e-3 * (1.0 + 0.0 * f)


def _gamma_h2o_dB_per_km(f_ghz, rho_w):
    """Simplified water-vapour attenuation. rho_w in g/m^3."""
    f = f_ghz
    # 22.235 GHz line + 183.31 GHz line + continuum
    term1 = (3.79e-7 * rho_w * f * f
             / ((f - 22.235) ** 2 + 9.81))
    term2 = (11.73e-12 * rho_w * f * f
             / ((f - 183.31) ** 2 + 11.85))
    cont  = 6.0e-5 * rho_w
    return (term1 + term2 + cont) * 1.0e-1


class GasP676Component(AbstractChannelComponent):
    name  = "gas_p676"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_gas_p676", False)):
            return
        if data.loss_db is None:
            return
        f_ghz = float(cfg.carrier_freq_hz) / 1.0e9
        rho_w = float(getattr(cfg, "p676_water_vapour_g_m3", 7.5))
        h_eff = float(getattr(cfg, "p676_effective_height_km", 6.0))
        gamma = _gamma_o2_dB_per_km(f_ghz) + _gamma_h2o_dB_per_km(f_ghz, rho_w)
        elev  = np.asarray(data.extras.get("elev_deg",
                                           np.zeros(data.n_links())),
                           dtype=float)
        elev  = np.clip(elev, 1e-3, 90.0)
        L_eff = h_eff / np.sin(np.radians(elev))
        gas_db = gamma * L_eff
        data.loss_db = np.asarray(data.loss_db, dtype=float) + gas_db
        data.extras["gas_db"] = gas_db
        data.extras["gas_model"] = "p676"
