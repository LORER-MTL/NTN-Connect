# DT observation noise: h += CN(0, sigma^2 I/N_t). DT pipeline only, off by default.
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class DTObservationNoiseComponent(AbstractChannelComponent):
    name  = "dt_obs_noise"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_dt_obs_noise", False)):
            return
        if data.h is None:
            return
        sigma = float(getattr(cfg, "dt_obs_noise_sigma", 0.0))
        if sigma <= 0.0:
            return
        h_shape = data.h.shape
        # Normalize the per-antenna noise the same way the steering
        # component normalizes its NLoS draw (variance 1/N_t per antenna).
        n_tx = h_shape[-1] if data.h.ndim >= 1 else 1
        scale = sigma / np.sqrt(2.0 * max(n_tx, 1))
        rng = getattr(state, "rng", None) or np.random
        re = rng.standard_normal(size=h_shape)
        im = rng.standard_normal(size=h_shape)
        data.h = data.h + scale * (re + 1j * im)
        data.extras["dt_obs_noise_sigma"] = sigma
