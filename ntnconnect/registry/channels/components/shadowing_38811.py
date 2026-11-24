# Shadowing: TR 38.811 elevation-dependent sigma(el), iid (approximate tables).
# shadowing_38811_scenario: dense_urban|urban|suburban|rural.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

_EL_REF = np.array([0.0, 10.0, 30.0, 60.0, 90.0])
_SIG_BY_SCENARIO = {
    "dense_urban": [6.0, 4.5, 3.0, 1.8, 1.0],
    "urban":       [5.5, 4.0, 2.5, 1.5, 0.8],
    "suburban":    [4.0, 3.0, 2.0, 1.2, 0.6],
    "rural":       [3.0, 2.5, 1.5, 0.8, 0.4],
}


class Shadowing38811Component(AbstractChannelComponent):
    name  = "shadowing_38811"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_shadowing_38811", False)):
            return
        if data.loss_db is None:
            return
        scenario = str(getattr(cfg, "shadowing_38811_scenario", "rural")).lower()
        if scenario not in _SIG_BY_SCENARIO:
            scenario = "rural"
        sig_ref = np.array(_SIG_BY_SCENARIO[scenario])
        elev = np.asarray(data.extras.get("elev_deg", np.zeros(data.n_links())),
                          dtype=float)
        sig = np.interp(elev, _EL_REF, sig_ref)
        n   = elev.size
        rng = getattr(state, "rng", None) or np.random
        shadow = sig * rng.standard_normal(n)
        data.loss_db = np.asarray(data.loss_db, dtype=float) + shadow
        data.extras["shadow_db"] = shadow
        data.extras["shadowing_38811_scenario"] = scenario
