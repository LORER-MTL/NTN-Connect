# K-factor: ITU-R P.681 LMS by environment class (approximate).
# p681_environment: open|rural_wooded|suburban|urban.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

# K-factor table indexed by environment, with mild elevation dependence.
# Values approximate ITU-R P.681 §4.
_ENV_K = {
    "open":         {"k_los": 12.0, "k_nlos": 8.0},
    "rural_wooded": {"k_los":  8.0, "k_nlos": 4.0},
    "suburban":     {"k_los":  5.0, "k_nlos": 0.0},
    "urban":        {"k_los":  2.0, "k_nlos":-3.0},
}


class KFactorP681Component(AbstractChannelComponent):
    name  = "k_factor_p681"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_p681", False)):
            return
        env = str(getattr(cfg, "p681_environment", "rural_wooded")).lower()
        if env not in _ENV_K:
            env = "rural_wooded"
        elev = np.asarray(data.extras.get("elev_deg", np.zeros(data.n_links())),
                          dtype=float)
        elev = np.clip(elev, 0.0, 90.0)
        k_los  = _ENV_K[env]["k_los"]
        k_nlos = _ENV_K[env]["k_nlos"]
        # Elevation-dependent linear blend from k_nlos (el=0) to k_los (el=90).
        k_db = k_nlos + (k_los - k_nlos) * (elev / 90.0)
        if state is not None:
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
        data.extras["p681_environment"] = env
