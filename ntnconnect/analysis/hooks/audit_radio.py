# Radio resource audit: RB allocation, beam budget, spectrum usage.
# Populates whatever the current stage stack provides; columns that depend
# on stages not yet wired (true rrm_allocate_rb / rrm_mcs) are NaN today,
# guaranteeing schema-stable disk files when those stages land later.

from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook
from ntnconnect.common.schemas import RB_ALLOCATION, BEAM_BUDGET, SPECTRUM_USAGE


def _shannon_mcs_from_se(se: float) -> int | None:
    """Coarse SE -> MCS-like index mapping used as placeholder until a
    real 3GPP MCS table is wired (rrm_mcs stage)."""
    if se is None or np.isnan(se): return None
    bins = [0.15, 0.38, 0.88, 1.48, 1.91, 2.41, 2.73, 3.32, 3.90,
            4.52, 5.12, 5.55, 6.23, 6.91, 7.41]
    for i, b in enumerate(bins):
        if se < b: return i
    return len(bins)


class RBAllocationHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = RB_ALLOCATION.collector(engine)

    def on_step_end(self, state):
        if not RB_ALLOCATION.is_enabled(self.engine.cfg): return
        links = state.data.get("links")
        assigns = state.data.get("assignments", {}) or {}
        if not isinstance(links, pd.DataFrame) or links.empty or not assigns: return
        # Resource-block model: 1 RB = 180 kHz (LTE-like). With Shannon SE
        # proxy we allocate proportionally; later, rrm_allocate_rb stage
        # will overwrite this with the real grid.
        RB_HZ = 180_000.0
        for u, s in assigns.items():
            if int(s) < 0: continue
            mask = (links["user_id"] == int(u)) & (links["node_id"] == int(s))
            if not mask.any(): continue
            r = links.loc[mask].iloc[0]
            bw_node = float(r.get("bandwidth_hz", 0.0))
            n_rbs_total = max(1, int(bw_node / RB_HZ))
            # Equal-split among the users this node serves
            n_users_on_node = sum(1 for su in assigns.values() if int(su) == int(s))
            n_rbs = max(1, n_rbs_total // max(1, n_users_on_node))
            bw_alloc = n_rbs * RB_HZ
            se = float(r.get("spectral_efficiency_bps_hz", np.nan))
            self.c.add({
                "t_idx":               state.t_idx,
                "user_id":             int(u),
                "node_id":             int(s),
                "n_rbs_allocated":     n_rbs,
                "n_rbs_total":         n_rbs_total,
                "bw_allocated_hz":     bw_alloc,
                "mcs_index":           _shannon_mcs_from_se(se),
                "modulation":          None,
                "coding_rate":         None,
                "target_bler":         None,
                "achievable_rate_bps": se * bw_alloc if not np.isnan(se) else 0.0,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class BeamBudgetHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = BEAM_BUDGET.collector(engine)

    def on_step_end(self, state):
        if not BEAM_BUDGET.is_enabled(self.engine.cfg): return
        assigns = state.data.get("assignments", {}) or {}
        if not assigns: return
        # Per-node serving count = active beam count (one beam per UE in this
        # placeholder model). When chan_beamforming stage lands, the real
        # multi-beam scheduler replaces this row content.
        per_node: dict = {}
        for u, s in assigns.items():
            if int(s) < 0: continue
            per_node.setdefault(int(s), []).append(int(u))
        op_pmax = {op.id: float(op.radio.power_max_w) for op in self.engine.cfg.operators}
        links = state.data.get("links")
        nid_to_op = ({int(r): int(o) for r, o in
                      links[["node_id","operator_id"]].drop_duplicates().values}
                     if isinstance(links, pd.DataFrame) and not links.empty else {})
        for nid, users in per_node.items():
            n_beams = len(users)
            p_total = op_pmax.get(nid_to_op.get(nid, -1), 0.0)
            p_per_beam = p_total / n_beams if n_beams else 0.0
            for i, uid in enumerate(users):
                self.c.add({
                    "t_idx":             state.t_idx,
                    "node_id":           nid,
                    "beam_idx":          i,
                    "n_beams_total":     n_beams,
                    "power_per_beam_w":  p_per_beam,
                    "steering_az_deg":   None,
                    "steering_el_deg":   None,
                    "n_users_on_beam":   1,
                    "active":            1,
                })

    def on_simulation_end(self, state):
        self.c.flush()


class SpectrumUsageHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = SPECTRUM_USAGE.collector(engine)

    def on_step_end(self, state):
        if not SPECTRUM_USAGE.is_enabled(self.engine.cfg): return
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        op_cfg = {op.id: op for op in self.engine.cfg.operators}
        for op_id, g in links.groupby("operator_id", observed=False):
            op = op_cfg.get(int(op_id))
            if op is None: continue
            bw_total = float(op.radio.bandwidth_hz)
            # Used = bandwidth attributed to served links (estimate)
            n_served = int(g["node_id"].nunique())
            bw_used = min(bw_total, n_served * bw_total / max(1, n_served))
            self.c.add({
                "t_idx":                  state.t_idx,
                "operator_id":            int(op_id),
                "bw_total_hz":            bw_total,
                "bw_used_hz":             bw_used,
                "bw_used_pct":            100.0 * bw_used / bw_total if bw_total else 0.0,
                "frequency_reuse_factor": float(getattr(op.interference,
                                                        "frequency_reuse_factor", 1.0)),
                "activity_factor":        float(getattr(op.interference,
                                                        "activity_factor", 1.0)),
            })

    def on_simulation_end(self, state):
        self.c.flush()


@register_hook("rb_allocation")
def _b1(engine): return RBAllocationHook(engine)

@register_hook("beam_budget")
def _b2(engine): return BeamBudgetHook(engine)

@register_hook("spectrum_usage")
def _b3(engine): return SpectrumUsageHook(engine)
