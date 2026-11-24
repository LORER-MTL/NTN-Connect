# Rain: AR-1 Gauss-Markov scalar (0 dB dry, ~6 dB in bursts). rho from pro_ntn_rain_corr.
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class RainAR1Component(AbstractChannelComponent):
    name = "rain_ar1"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_rain_ar1", False)):
            return
        if data.loss_db is None:
            return
        beta = float(getattr(cfg, "pro_ntn_rain_corr", 2e-4))
        # Persistent AR-1 scalar lives on ChannelState.data, keyed so
        # multiple operators share their own rain trajectory.
        key = "rain_ar1_z"
        z_prev = 0.0 if state is None else float(state.data.get(key, 0.0))
        rho = float(np.exp(-beta * 1.0))
        rng = getattr(state, "rng", None) or np.random
        innov = float(rng.standard_normal())
        z = rho * z_prev + np.sqrt(max(1.0 - rho * rho, 0.0)) * innov
        if state is not None:
            state.data[key] = z
        # Squelch + linear ramp above z = 0.5 (matches Colab ProfNTN).
        rain_db_scalar = 6.0 * (z - 0.5) if z > 0.5 else 0.0
        n = data.n_links()
        rain_db = np.full(n, rain_db_scalar)
        data.loss_db = np.asarray(data.loss_db, dtype=float) + rain_db
        data.extras["rain_db"] = rain_db
