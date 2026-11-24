# Free-space path loss - reusable PRE-MODEL component.
#
# Computes
#  FSPL [dB] = intercept + 20·log10(f_MHz) + 20·log10(d_km)
# with default intercept = 32.45 dB (the standard 32.45 + 20log10(f_MHz)
# + 20log10(d_km) convention, f in MHz, d in km).
# note: the code uses f in MHz. Do not set threegpp_fspl_intercept=92.45
# (the GHz-convention constant) - that would double-count ~60 dB. Use 32.45.
#
# Runs before the channel model so models no longer need to compute
# FSPL themselves - they just provide the small-scale fading.
#
# Reads: cfg.carrier_freq_hz
#  cfg.threegpp_fspl_intercept (default 32.45, MHz convention)
#  data.extras["dist_m"] stashed by ChannelPipeline
# Writes: data.loss_db initial fill (additive - components
#  further down the chain ADD to it)
#  data.extras["fspl_db"] diagnostic
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent


class FSPLComponent(AbstractChannelComponent):
    name  = "fspl"
    phase = "pre_model"        # runs before the channel model

    def apply(self, data, cfg, dt_sec, state=None):
        # Always-on when this component is in the pipeline. Toggle via
        # cfg.enable_fspl (defaults True so existing configs work).
        if not bool(getattr(cfg, "enable_fspl", True)):
            return
        dist_m = data.extras.get("dist_m")
        if dist_m is None:
            return
        d_m   = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        d_km  = d_m / 1.0e3
        # The 3GPP TR 38.811 convention is:
        #  FSPL = 32.45 + 20·log10(f_MHz) + 20·log10(d_km)
        # `cfg.threegpp_fspl_intercept` defaults to 32.45 (MHz convention).
        # We honour that by using MHz here. Users who configured 92.45
        # (GHz convention) get the same numerical answer because
        # 92.45 + 20·log(f_GHz) = 32.45 + 20·log(f_MHz).
        f_mhz = float(cfg.carrier_freq_hz) / 1.0e6
        intercept = float(getattr(cfg, "threegpp_fspl_intercept", 32.45))
        fspl_db = intercept + 20.0 * np.log10(f_mhz) + 20.0 * np.log10(d_km)
        # Initialize loss_db OR add (later components can add rain/gas/...)
        if data.loss_db is None:
            data.loss_db = fspl_db
        else:
            data.loss_db = np.asarray(data.loss_db, dtype=float) + fspl_db
        data.extras["fspl_db"] = fspl_db
