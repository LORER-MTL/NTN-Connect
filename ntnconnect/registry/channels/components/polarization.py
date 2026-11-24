# Cross-polarization ratio: a fraction (1/XPR_lin) of co-pol energy leaks
# to the cross-pol path. Implementation: scale |h| by an effective gain
# computed from the XPR. Dual-polarized arrays would need a 2x2 channel
# per link; here we apply a scalar attenuation factor consistent with the
# legacy package's narrowband mode.

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent


class PolarizationComponent(AbstractChannelComponent):
    name  = "polarization"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_polarization", False)): return
        if data.h is None: return
        xpr_db  = float(getattr(cfg, "xpr_db", 10.0))
        xpr_lin = 10.0 ** (xpr_db / 10.0)
        # Effective co-pol fraction of received energy at the linearly-
        # polarized receiver. (1/XPR) of energy goes to the cross-pol
        # path which we treat as lost in this single-polarisation view.
        eff_lin = xpr_lin / (1.0 + xpr_lin)
        eff_amp = float(np.sqrt(eff_lin))
        # Per-link diagnostics
        data.xpr_db = np.full(data.n_links(), xpr_db)
        data.extras["xpr_db"]     = data.xpr_db
        data.extras["pol_eff_db"] = float(10.0 * np.log10(eff_lin))
        # Broadcast-safe scaling: works for h shapes (n_links,),
        # (n_links, n_tx), (n_links, n_rx, n_tx).
        data.h = data.h * eff_amp
