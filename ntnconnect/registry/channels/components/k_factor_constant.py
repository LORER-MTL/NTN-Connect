# K-factor: constant threegpp_k_factor_los_db (default source).
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class KFactorConstantComponent(AbstractChannelComponent):
    name  = "k_factor_constant"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_constant", False)):
            return
        n = data.n_links() or len(data.extras.get("elev_deg", []))
        if n == 0: return
        k_db_value = float(getattr(cfg, "threegpp_k_factor_los_db", 10.0))
        k_db = np.full(n, k_db_value)
        if state is not None:
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
