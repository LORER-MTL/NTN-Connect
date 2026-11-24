# Jakes/Clarke temporal correlation: h = rho*h_prev + sqrt(1-rho^2)*h, rho=J0(2pi f_D dt).
# f_D = fading SPREAD (v*f_c/c), not the LEO Doppler shift. Runs first post-model;
# skips models that already did in-model temporal fading. Fallback for iid models.
from __future__ import annotations
import numpy as np
from scipy.special import j0
from ntnconnect.registry.channels.base import AbstractChannelComponent

C = 299_792_458.0


class JakesCorrelationComponent(AbstractChannelComponent):
    name = "jakes"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(cfg.enable_jakes_corr):
            return
        if data.h is None or state is None:
            return
        # Do not filter twice: models with ar1/sos fading already own the temporal
        # correlation (and correlate only the diffuse term, which is more correct).
        if state.data.get("model_did_temporal_fading"):
            return
        if dt_sec is None or float(dt_sec) <= 0.0:
            return

        # Fading Doppler SPREAD (terminal motion), not the bulk range-rate shift.
        f_d = data.extras.get("fading_doppler_hz")
        if f_d is None:
            v_ms = float(getattr(cfg, "ue_speed_mps", 0.0))
            f_c  = float(getattr(cfg, "carrier_freq_hz", 20.0e9))
            f_d  = abs(v_ms) * f_c / C
        f_d = float(np.max(np.abs(np.asarray(f_d, dtype=float)))) \
            if np.ndim(f_d) else float(f_d)
        if f_d <= 0.0:
            # Quasi-static terminal: no fading decorrelation to impose. Leave the
            # model's per-tick draw as-is (consistent with the ar1/sos f_D=0 limit).
            return

        rho = float(j0(2.0 * np.pi * f_d * float(dt_sec)))
        rho = max(min(rho, 0.9999), -0.9999)

        prev = state.data.get("jakes_h_prev")
        if prev is not None and getattr(prev, "shape", None) == data.h.shape:
            data.h = rho * prev + np.sqrt(1.0 - rho * rho) * data.h
        state.data["jakes_h_prev"] = data.h.copy()
        data.extras["jakes_rho"] = rho
        data.extras["jakes_fading_doppler_hz"] = f_d
