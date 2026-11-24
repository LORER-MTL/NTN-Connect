# K-factor: distance decay K_dB = k0 + 10*gamma*log10(d_ref/d).
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


class KFactorDistanceComponent(AbstractChannelComponent):
    name  = "k_factor_distance"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_distance", False)):
            return
        dist_m = np.asarray(data.extras.get("dist_m",
                                            np.ones(data.n_links())), dtype=float)
        dist_m = np.maximum(dist_m, 1.0)
        k0_db   = float(getattr(cfg, "k_distance_k0_db",  12.0))
        d_ref   = float(getattr(cfg, "k_distance_ref_m",  6.0e5))
        gamma   = float(getattr(cfg, "k_distance_gamma",  0.5))
        k_db = k0_db + 10.0 * gamma * np.log10(d_ref / dist_m)
        if state is not None:
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
