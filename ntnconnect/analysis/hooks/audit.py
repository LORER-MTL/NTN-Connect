# Audit hook - dumps the per-link inputs the SINR formula consumes
# so a reviewer can recompute KPIs by hand from disk.
#
# Activated by cfg.outputs.tables.link_audit/power_alloc/beamforming_audit.
# All four audit tables default OFF in master.yaml because they can be
# large; turn them ON for validation runs.

from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook
from ntnconnect.common.schemas import LINK_AUDIT, POWER_ALLOC, BEAMFORMING_AUDIT


# Constants
kB = 1.380649e-23


def _intra_inter_split(links: pd.DataFrame) -> tuple:
    """If the stages already split intra/inter, use those columns; else
    return total interference in `inter_interf_w` and zero in intra."""
    intra = links.get("intra_interf_w")
    inter = links.get("inter_interf_w")
    total = links.get("interference_w")
    n = len(links)
    if intra is None and inter is None and total is not None:
        return np.zeros(n), total.values
    if intra is None: intra = pd.Series(np.zeros(n))
    if inter is None: inter = pd.Series(np.zeros(n))
    return intra.values, inter.values


class LinkAuditHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = LINK_AUDIT.collector(engine)

    def on_step_end(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        if not LINK_AUDIT.is_enabled(self.engine.cfg):
            return
        assigns = state.data.get("assignments", {}) or {}
        serving = {int(u): int(s) for u, s in assigns.items()}

        intra, inter = _intra_inter_split(links)
        n = len(links)

        # Vectorized assembly: one DataFrame, single insert.
        df = pd.DataFrame({
            "t_idx":                     np.full(n, state.t_idx, dtype="int32"),
            "user_id":                   links["user_id"].astype(int).values,
            "node_id":                   links["node_id"].astype(int).values,
            "is_serving":                np.array([
                int(serving.get(int(u), -1) == int(n_))
                for u, n_ in zip(links["user_id"], links["node_id"])
            ], dtype="int8"),
            "dist_m":                    links["dist_m"].astype(float).values,
            "elev_deg":                  links.get("elev_deg",
                                                    pd.Series([np.nan]*n)).astype(float).values,
            "azimuth_deg":               links.get("azimuth",
                                                    pd.Series([np.nan]*n)).astype(float).values,
            "range_rate_mps":            (links.get("range_rate_km_s",
                                                     pd.Series([0.0]*n)).astype(float).values
                                          * 1000.0),
            "doppler_hz":                links.get("doppler_hz",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "tx_power_w":                links.get("tx_power_w",
                                                    links.get("signal_w",
                                                              pd.Series([0.0]*n))).astype(float).values,
            "antenna_gain_tx_db":        (links.get("antenna_gain_db",
                                                     pd.Series([0.0]*n)).astype(float).values / 2.0),
            "antenna_gain_rx_db":        (links.get("antenna_gain_db",
                                                     pd.Series([0.0]*n)).astype(float).values / 2.0),
            "beamforming_gain_db":       (links["bf_gain_db"].astype(float).values
                                          if "bf_gain_db" in links.columns
                                          else np.full(n, np.nan)),
            "path_loss_db":              links.get("path_loss_db",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "channel_gain_lin":          links.get("chan_gain_lin",
                                                    pd.Series([1.0]*n)).astype(float).values,
            "carrier_hz":                links.get("carrier_hz",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "bandwidth_hz":              links.get("bandwidth_hz",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "received_power_w":          links.get("signal_w",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "noise_power_w":             links.get("noise_w",
                                                    pd.Series([0.0]*n)).astype(float).values,
            "intra_interf_w":            intra.astype(float),
            "inter_interf_w":            inter.astype(float),
            "total_interf_w":            (intra + inter).astype(float),
            "snr_db":                    links.get("snr_db",
                                                    pd.Series([np.nan]*n)).astype(float).values,
            "sinr_db":                   links.get("sinr_db",
                                                    pd.Series([np.nan]*n)).astype(float).values,
            "spectral_efficiency_bps_hz":links.get("spectral_efficiency_bps_hz",
                                                    pd.Series([np.nan]*n)).astype(float).values,
            "link_rate_bps":             links.get("link_rate_bps",
                                                    pd.Series([np.nan]*n)).astype(float).values,
        })
        self.c.add_df(df)

    def on_simulation_end(self, state):
        self.c.flush()


class PowerAllocHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = POWER_ALLOC.collector(engine)

    def on_step_end(self, state):
        links = state.data.get("links")
        assigns = state.data.get("assignments", {}) or {}
        if not isinstance(links, pd.DataFrame) or links.empty or not assigns:
            return
        if not POWER_ALLOC.is_enabled(self.engine.cfg):
            return

        # Build {node_id: [user_ids served]}
        served = {}
        for u, s in assigns.items():
            if int(s) < 0: continue
            served.setdefault(int(s), []).append(int(u))

        # Per-op power_max via operators_dim available from cfg
        op_pmax = {op.id: float(op.radio.power_max_w) for op in self.engine.cfg.operators}
        nid_to_op = (links[["node_id","operator_id"]].drop_duplicates()
                          .set_index("node_id")["operator_id"].to_dict())

        for nid, users in served.items():
            op_id = int(nid_to_op.get(nid, -1))
            p_max = op_pmax.get(op_id, 0.0)
            n_users = len(users)
            self.c.add({
                "t_idx":            state.t_idx,
                "node_id":          int(nid),
                "total_tx_power_w": p_max,
                "n_served_users":   int(n_users),
                "power_per_user_w": float(p_max / n_users) if n_users else 0.0,
                "equal_split":      1,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class BeamformingAuditHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = BEAMFORMING_AUDIT.collector(engine)

    def on_step_end(self, state):
        links = state.data.get("links")
        assigns = state.data.get("assignments", {}) or {}
        if not isinstance(links, pd.DataFrame) or links.empty or not assigns:
            return
        if not BEAMFORMING_AUDIT.is_enabled(self.engine.cfg):
            return
        # Per-operator MIMO config
        op_mimo = {op.id: op for op in self.engine.cfg.operators}
        for u, s in assigns.items():
            if int(s) < 0: continue
            mask = (links["user_id"] == int(u)) & (links["node_id"] == int(s))
            if not mask.any(): continue
            r = links.loc[mask].iloc[0]
            op = op_mimo.get(int(r["operator_id"]))
            mimo = op.channel_model if op is not None else None
            self.c.add({
                "t_idx":           state.t_idx,
                "node_id":         int(s),
                "user_id":         int(u),
                "bf_mode":         op.radio.beamforming_mode if op is not None else "none",
                "bf_gain_db":      float(r.get("antenna_gain_db", 0.0)),
                "n_rx":            int(getattr(mimo, "n_rx", 0)) if mimo else 0,
                "n_tx":            int(getattr(mimo, "n_tx_x", 1)
                                       * getattr(mimo, "n_tx_y", 1)) if mimo else 0,
                "precoder_norm":   None,
                "steering_az_deg": float(r.get("azimuth", np.nan)),
                "steering_el_deg": float(r.get("elev_deg", np.nan)),
            })

    def on_simulation_end(self, state):
        self.c.flush()


@register_hook("link_audit")
def _b1(engine): return LinkAuditHook(engine)

@register_hook("power_alloc")
def _b2(engine): return PowerAllocHook(engine)

@register_hook("beamforming_audit")
def _b3(engine): return BeamformingAuditHook(engine)
