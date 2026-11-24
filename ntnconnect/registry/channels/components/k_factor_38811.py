# K-factor: TR 38.811 elevation table, AR-1 evolution (rho=pro_ntn_kappa_rho).
from __future__ import annotations
import numpy as np

from ntnconnect.registry.channels.base import AbstractChannelComponent

# TR 38.811-style smooth-interpolated marginal table
_EL_REF  = np.array([0.0, 10.0, 30.0, 60.0, 90.0])
_MU_REF  = np.array([4.0,  6.0,  9.0, 14.0, 16.0])   # mean K (dB)
_SIG_REF = np.array([4.0,  3.5,  2.5,  1.5,  1.0])   # std K (dB)


class KFactor38811Component(AbstractChannelComponent):
    name  = "k_factor_38811"
    phase = "pre_model"        # runs before the channel model so the
                                # base Rician model can use this K value

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_38811", False)):
            return
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(data.n_links())),
                          dtype=float)
        n = elev.size
        mu = np.interp(elev, _EL_REF, _MU_REF)
        sd = np.interp(elev, _EL_REF, _SIG_REF)
        rho = float(getattr(cfg, "pro_ntn_kappa_rho", 0.95))

        # Persistent AR-1 state, keyed by length so a constellation
        # change (sat reassignment, HO) doesn't carry stale state.
        rng = getattr(state, "rng", None) or np.random
        key = f"k_factor_38811_prev_n{n}"
        prev = None if state is None else state.data.get(key)
        if prev is None or len(prev) != n:
            # Cold-start at the marginal
            prev = mu + sd * rng.standard_normal(n)
        z = rng.standard_normal(n)
        k_db = mu + rho * (prev - mu) + sd * np.sqrt(1.0 - rho * rho) * z
        if state is not None:
            state.data[key] = k_db
            # Make the pre-model K available to the channel model on the
            # same tick. Any Rician-style model can pick this up via
            # state.data["k_factor_db_pending"].
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
