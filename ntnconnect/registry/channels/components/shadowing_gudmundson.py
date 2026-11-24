# Shadowing: Gudmundson (1991) AR-1 lognormal, rho=exp(-v*dt/d_corr).
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


class ShadowingGudmundsonComponent(AbstractChannelComponent):
    name  = "shadowing_gudmundson"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_shadowing_gudmundson", False)):
            return
        if data.loss_db is None:
            return
        sigma = float(getattr(cfg, "shadowing_std_db", 2.0))
        d_corr = max(float(getattr(cfg, "shadowing_gudmundson_corr_m", 50.0)), 1e-3)
        v_ms   = float(getattr(cfg, "shadowing_gudmundson_user_speed_m_s", 0.0))
        delta_d = max(v_ms * float(dt_sec), 0.1)        # treat static as 0.1m drift
        rho_g = float(np.exp(-delta_d / d_corr))
        n = data.n_links()
        rng = getattr(state, "rng", None) or np.random
        key = f"gudmundson_prev_n{n}"
        prev = None if state is None else state.data.get(key)
        innov = sigma * rng.standard_normal(n)
        if prev is None or len(prev) != n:
            new = innov
        else:
            new = rho_g * prev + np.sqrt(max(1.0 - rho_g * rho_g, 0.0)) * innov
        if state is not None:
            state.data[key] = new
        data.loss_db = np.asarray(data.loss_db, dtype=float) + new
        data.extras["shadow_db"] = new
        data.extras["shadow_corr_rho"] = rho_g
