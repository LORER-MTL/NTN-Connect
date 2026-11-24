# Refraction (ITU-R P.834): apparent-elevation correction Δel≈1/tan(el+7.31/(el+4.4)) arcmin.
# Pre-model; corrects extras["elev_deg"].
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


class RefractionP834Component(AbstractChannelComponent):
    name  = "refraction_p834"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_refraction_p834", False)):
            return
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(data.n_links())), dtype=float)
        elev_safe = np.maximum(elev, 0.1)
        delta_arcmin = 1.0 / np.tan(np.radians(elev_safe + 7.31 / (elev_safe + 4.4)))
        delta_deg = delta_arcmin / 60.0
        corrected = np.clip(elev + delta_deg, 0.0, 90.0)
        data.extras["elev_deg"]           = corrected
        data.extras["refraction_arcmin"]  = delta_arcmin
