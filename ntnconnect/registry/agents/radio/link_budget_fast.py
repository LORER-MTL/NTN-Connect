# Radio algorithm: link_budget_fast - full-power pass-through link budget (no SOCP).
# signal=P_tx*G*10^(-PL/10); noise=kB*T_sys*BW; rate=BW*log2(1+SINR). I_ext added later.
#
# Everything is fully vectorised - no Python loops, no CVX solves.
# A subsequent `paper_sinr` stage overwrites sinr_db with the proper
# (signal / (noise + i_ext)) formula once external_interference_const
# has populated i_ext_w.
from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.registry.agents.radio_algorithm import BaseRadioAlgorithm, register_radio_algorithm

_KB = 1.380649e-23


@register_radio_algorithm("link_budget_fast")
class LinkBudgetFast(BaseRadioAlgorithm):
    """Vectorised pass-through link-budget computer.

    Useful when downstream stages compute the SINR (paper_sinr) and you
    just want the geometric/budget columns populated cheaply.
    """

    def compute(self, links: pd.DataFrame, assigns: dict,
                state=None) -> pd.DataFrame:
        if not isinstance(links, pd.DataFrame) or links.empty:
            return links

        n = len(links)
        # Per-operator P_tx (vectorised lookup, no Python loop)
        op_p = {int(op.id): float(getattr(op.radio, "power_max_w", 1.0))
                for op in self.cfg.operators}
        op_ids = links.get("operator_id",
                            pd.Series(np.ones(n, dtype=int))
                          ).astype(int).to_numpy()
        # vectorised map: build a sentinel array indexed by op_id range
        max_op = int(op_ids.max(initial=0)) + 1
        op_lut = np.ones(max_op + 1, dtype=float)
        for k, v in op_p.items():
            if 0 <= k <= max_op: op_lut[k] = v
        p_tx = op_lut[np.clip(op_ids, 0, max_op)]

        g_db  = links.get("antenna_gain_db",
                          pd.Series(np.zeros(n))).astype(float).to_numpy()
        pl_db = links.get("path_loss_db",
                          pd.Series(np.zeros(n))).astype(float).to_numpy()
        bw    = links.get("bandwidth_hz",
                          pd.Series(np.full(n, 20e6))
                         ).astype(float).to_numpy()

        # System temperature & noise figure
        T_sys = (float(self.cfg.defaults_radio.system_temperature_k)
                 * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        noise_w = _KB * T_sys * np.maximum(bw, 1.0)

        g_lin  = 10.0 ** (g_db / 10.0)
        pl_lin = 10.0 ** (-pl_db / 10.0)
        signal_w = p_tx * g_lin * pl_lin

        # SNR only (paper_sinr will divide by noise + i_ext)
        snr_lin = signal_w / np.maximum(noise_w, 1e-30)
        sinr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-30))
        se      = np.log2(1.0 + snr_lin)

        out = links.copy()
        out["tx_power_w"]                  = p_tx
        out["signal_w"]                    = signal_w
        out["noise_w"]                     = noise_w
        out["interference_w"]              = 0.0
        out["interference_intra_w"]        = 0.0
        out["interference_ext_w"]          = 0.0
        out["snr_db"]                      = sinr_db
        out["sinr_db"]                     = sinr_db   # paper_sinr will fix
        out["spectral_efficiency_bps_hz"]  = se
        out["link_rate_bps"]               = se * bw
        out["radio_status"]                = "link_budget_fast"
        out["bf_gain_db"]                  = 0.0
        return out
