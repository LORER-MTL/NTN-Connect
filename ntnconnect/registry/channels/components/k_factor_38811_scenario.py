# K-factor: TR 38.811 per-scenario elevation tables (approximate), AR-1.
# k_factor_scenario_name: dense_urban|urban|suburban|rural.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

_EL_REF = np.array([0.0, 10.0, 30.0, 60.0, 90.0])
# Approximations of TR 38.811 §6.7.2 per-scenario marginal tables.
_SCENARIOS = {
    "dense_urban": {"mu":  [4.0, 6.0, 9.0, 14.0, 16.0], "sig": [4.0, 3.5, 2.5, 1.5, 1.0]},
    "urban":       {"mu":  [5.0, 7.0,10.0, 14.5, 16.5], "sig": [3.5, 3.0, 2.0, 1.3, 0.9]},
    "suburban":    {"mu":  [7.0, 9.0,12.0, 16.0, 18.0], "sig": [3.0, 2.5, 1.8, 1.2, 0.8]},
    "rural":       {"mu":  [9.0,12.0,14.0, 17.0, 19.0], "sig": [2.5, 2.0, 1.5, 1.0, 0.7]},
}


class KFactor38811ScenarioComponent(AbstractChannelComponent):
    name  = "k_factor_38811_scenario"
    phase = "pre_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_k_factor_38811_scenario", False)):
            return
        scenario = str(getattr(cfg, "k_factor_scenario_name", "rural")).lower()
        if scenario not in _SCENARIOS:
            scenario = "rural"
        mu_ref  = np.array(_SCENARIOS[scenario]["mu"])
        sig_ref = np.array(_SCENARIOS[scenario]["sig"])
        elev = np.asarray(data.extras.get("elev_deg", np.zeros(data.n_links())),
                          dtype=float)
        n = elev.size
        mu = np.interp(elev, _EL_REF, mu_ref)
        sd = np.interp(elev, _EL_REF, sig_ref)
        rho = float(getattr(cfg, "pro_ntn_kappa_rho", 0.95))
        rng = getattr(state, "rng", None) or np.random
        key = f"k_38811_scenario_prev_n{n}"
        prev = None if state is None else state.data.get(key)
        if prev is None or len(prev) != n:
            prev = mu + sd * rng.standard_normal(n)
        z = rng.standard_normal(n)
        k_db = mu + rho * (prev - mu) + sd * np.sqrt(1.0 - rho * rho) * z
        if state is not None:
            state.data[key] = k_db
            state.data["k_factor_db_pending"] = k_db
        data.extras["k_factor_db"] = k_db
        data.extras["k_factor_scenario_name"] = scenario
