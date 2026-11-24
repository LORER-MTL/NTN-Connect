# K-factor: sigmoid LoS-prob blend K = p*k_los + (1-p)*k_nlos, p=sigmoid(el).
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class KFactorSigmoidComponent(AbstractChannelComponent):
    name  = "k_factor_sigmoid"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_sigmoid", False)):
            return
        elev = np.asarray(data.extras.get("elev_deg", np.zeros(data.n_links())),
                          dtype=float)
        elev = np.clip(elev, 0.0, 90.0)
        k_los  = float(getattr(cfg, "threegpp_k_factor_los_db", 10.0))
        k_nlos = float(getattr(cfg, "threegpp_k_factor_nlos_db", 0.0))
        scale  = float(getattr(cfg, "threegpp_los_prob_scale", 4.0))
        offset = float(getattr(cfg, "threegpp_los_prob_offset", 0.5))
        p_los  = 1.0 / (1.0 + np.exp(-(elev / 90.0 - offset) * scale))
        k_db   = p_los * k_los + (1.0 - p_los) * k_nlos
        if state is not None:
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
