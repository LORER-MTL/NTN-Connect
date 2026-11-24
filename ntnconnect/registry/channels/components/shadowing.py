# Log-normal shadowing with optional spatial correlation.

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent, ChannelData


class ShadowingComponent(AbstractChannelComponent):
    name = "shadowing"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(cfg.enable_shadowing) or data.h is None:
            return
        sigma_db = float(cfg.shadowing_std_db)
        corr_dist_m = float(cfg.shadowing_correlation_distance_m)
        n = data.n_links()
        rng = getattr(state, "rng", None) or np.random
        # New iid sample
        new = sigma_db * rng.standard_normal(n)
        prev = None if state is None else state.data.get("shadow_db_prev")
        # corr_dist_m > 0 opts into correlated shadowing (0 => iid). When on,
        # derive rho from the decorrelation distance and per-tick user drift
        # (rho = exp(-v*dt/corr_dist), Gudmundson-style) if a speed is known;
        # otherwise use a configurable constant (shadowing_corr_rho, default 0.8).
        if prev is not None and len(prev) == n and corr_dist_m > 0.0:
            v_ms = float(getattr(cfg, "shadowing_user_speed_m_s", 0.0))
            if v_ms > 0.0 and float(dt_sec) > 0.0:
                rho = float(np.exp(-(v_ms * float(dt_sec)) / corr_dist_m))
            else:
                rho = float(getattr(cfg, "shadowing_corr_rho", 0.8))
            new = rho * prev + np.sqrt(max(1.0 - rho * rho, 0.0)) * new
        if state is not None:
            state.data["shadow_db_prev"] = new
        data.shadow_db = new
        # Apply to loss
        if data.loss_db is not None:
            data.loss_db = np.asarray(data.loss_db, dtype=float) + new
        # Consistency: expose as a per-link diagnostic so downstream
        # KPIs / hooks can decompose loss_db = fspl + rain + gas + shadow.
        data.extras["shadow_db"] = new
