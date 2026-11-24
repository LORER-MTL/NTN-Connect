# Gas absorption: cosec(el) A = pro_ntn_gas_factor_db / sin(el).
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class GasCosecComponent(AbstractChannelComponent):
    name = "gas_absorption"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_gas_absorption", False)):
            return
        if data.loss_db is None:
            return
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(data.n_links())),
                          dtype=float)
        factor = float(getattr(cfg, "pro_ntn_gas_factor_db", 0.5))
        gas_db = factor / np.maximum(np.sin(np.radians(elev)), 0.1)
        data.loss_db = np.asarray(data.loss_db, dtype=float) + gas_db
        data.extras["gas_db"] = gas_db
