# Per-link propagation delay tau = d / c, applied as a phase rotation
# on the complex channel: h <- h * exp(-j * 2*pi*f_c*tau).

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

C = 299_792_458.0


class PropagationDelayComponent(AbstractChannelComponent):
    name = "propagation_delay"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(cfg.enable_propagation_delay): return
        dist_m = data.extras.get("dist_m")
        if dist_m is None: return
        f_c = float(cfg.carrier_freq_hz)
        d = np.asarray(dist_m, dtype=float)
        tau = d / C
        data.delay_s = tau
        phase = np.exp(-1j * 2 * np.pi * f_c * tau)
        if data.h is not None:
            # broadcast across any antenna dims
            shp = [-1] + [1] * (data.h.ndim - 1) if data.h.ndim > 1 else [-1]
            data.h = data.h * phase.reshape(shp)
