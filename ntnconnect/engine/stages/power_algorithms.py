# RRM power algorithms (write tx_power_w): rrm_power_equal | waterfilling | utility.
from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry   import register_stage


def _per_sat_iter(out: pd.DataFrame, assigns: dict):
    """Yield (sat_id, row_index_array) for every sat that has >=1 served user."""
    n_arr = out["node_id"].astype(int).to_numpy()
    u_arr = out["user_id"].astype(int).to_numpy()
    served_by = {}
    for u, s in assigns.items():
        if int(s) >= 0:
            served_by.setdefault(int(s), []).append(int(u))
    for s, uids in served_by.items():
        mask = (n_arr == s) & np.isin(u_arr, uids)
        if mask.any(): yield s, np.where(mask)[0]


def _g_eff(links: pd.DataFrame) -> np.ndarray:
    """Effective channel gain per link = |h|^2 * 10^(-PL/10)."""
    g  = links.get("chan_gain_lin", pd.Series(np.ones(len(links)))).astype(float).to_numpy()
    pl = links.get("path_loss_db",  pd.Series(np.zeros(len(links)))).astype(float).to_numpy()
    return np.maximum(g * 10**(-pl / 10.0), 1e-30)


@register_stage("rrm_power_equal", internal=True)
class RRMPowerEqual(Stage):
    """Equal split of Pmax across the users a sat serves."""
    reads, writes, thread_safe = ["links", "assignments"], ["links"], False
    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        assigns = state.data.get("assignments", {}) or {}
        if not assigns: return
        out = links.copy()
        pmax = float(self.cfg.operators[0].radio.power_max_w)
        tx = np.zeros(len(out))
        for _, idx in _per_sat_iter(out, assigns):
            tx[idx] = pmax / len(idx)
        out["tx_power_w"] = tx
        if "antenna_gain_db" not in out.columns: out["antenna_gain_db"] = 0.0
        state.data["links"] = out


@register_stage("rrm_power_waterfilling", internal=True)
class RRMPowerWaterfilling(Stage):
    """Per-sat waterfilling on the effective channel gain.

    Knob: cfg.handover.waterfilling_noise_floor_w  (default 1e-13)
    """
    reads, writes, thread_safe = ["links", "assignments"], ["links"], False
    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        assigns = state.data.get("assignments", {}) or {}
        if not assigns: return
        out = links.copy()
        pmax = float(self.cfg.operators[0].radio.power_max_w)
        n0   = float(getattr(self.cfg.handover, "waterfilling_noise_floor_w", 1e-13))
        g_eff = _g_eff(out)
        tx = np.zeros(len(out))
        for _, idx in _per_sat_iter(out, assigns):
            n_g = n0 / g_eff[idx]
            lo, hi = n_g.min(), n_g.min() + pmax + n_g.max()
            for _it in range(50):
                mu = 0.5 * (lo + hi)
                if np.maximum(mu - n_g, 0.0).sum() > pmax: hi = mu
                else: lo = mu
            p = np.maximum(mu - n_g, 0.0)
            if p.sum() > 0: p *= pmax / p.sum()
            tx[idx] = p
        out["tx_power_w"] = tx
        if "antenna_gain_db" not in out.columns: out["antenna_gain_db"] = 0.0
        state.data["links"] = out


@register_stage("rrm_power_utility", internal=True)
class RRMPowerUtility(Stage):
    """Proportional-to-gain allocation with a floor per user.

    Knob: cfg.handover.utility_min_share  (default 0.05)
    """
    reads, writes, thread_safe = ["links", "assignments"], ["links"], False
    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        assigns = state.data.get("assignments", {}) or {}
        if not assigns: return
        out = links.copy()
        pmax  = float(self.cfg.operators[0].radio.power_max_w)
        floor = float(getattr(self.cfg.handover, "utility_min_share", 0.05))
        g_eff = _g_eff(out)
        tx = np.zeros(len(out))
        for _, idx in _per_sat_iter(out, assigns):
            w = g_eff[idx] / g_eff[idx].sum()
            w = np.maximum(w, floor)
            w = w / w.sum()
            tx[idx] = w * pmax
        out["tx_power_w"] = tx
        if "antenna_gain_db" not in out.columns: out["antenna_gain_db"] = 0.0
        state.data["links"] = out
