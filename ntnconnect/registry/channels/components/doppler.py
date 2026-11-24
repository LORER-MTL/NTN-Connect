# Doppler shift from range rate.
# f_d = -range_rate * f_c / c (positive when approaching)

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

C = 299_792_458.0


class DopplerComponent(AbstractChannelComponent):
    name = "doppler"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(cfg.enable_doppler): return
        f_c = float(cfg.carrier_freq_hz)
        # range_rate stashed in extras by the model or stage (m/s)
        rate = data.extras.get("range_rate_m_s")
        if rate is None: return
        data.doppler_hz = -np.asarray(rate, dtype=float) * f_c / C
