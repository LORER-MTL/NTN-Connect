# Cloud (ITU-R P.840, simplified): A = K_l*L/sin(el), K_l~1e-3*f^1.9.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


class CloudP840Component(AbstractChannelComponent):
    name  = "cloud_p840"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_cloud_p840", False)):
            return
        if data.loss_db is None:
            return
        f_ghz = float(cfg.carrier_freq_hz) / 1.0e9
        L_kg  = float(getattr(cfg, "p840_cloud_liquid_kg_m2", 0.5))
        K_l   = 0.001 * (f_ghz ** 1.9)               # simplified P.840
        elev  = np.asarray(data.extras.get("elev_deg",
                                           np.zeros(data.n_links())),
                           dtype=float)
        elev  = np.clip(elev, 1e-3, 90.0)
        cloud_db = K_l * L_kg / np.sin(np.radians(elev))
        data.loss_db = np.asarray(data.loss_db, dtype=float) + cloud_db
        data.extras["cloud_db"] = cloud_db
        data.extras["cloud_model"] = "p840"
