# Consolidated radio (RRM) stage: one stage, three branches keyed by cfg.radio.csi_mode.
# Every branch emits the same decision schema (admission + power + beamforming); only
# how power is sized differs. Grading is done downstream by sinr_calc, so outage
# differences emerge from the sizing.
#   stochastic : statistics + outage reserve -> min_power_miso (handover study, unchanged).
#   csi        : perfect CSI on the true channel, no reserve (genie bound).
#   dt         : sized on the digital-twin predicted channel; prediction error -> outage.
# Output is decisions only; unified inputs are published as state.data["radio_context"].

from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.engine.stages.radio_context import build_radio_context


# RRM algorithm registry (cfg.radio.power -> method). Each method is a
# callable(stage, state) that writes the power+admission decision onto
# state.data["links"]; add one via @register_radio_method("name"). min_power
# further dispatches on cfg.radio.csi_mode (which channel it sizes power on).


# --- csi_mode sub-branches of min_power ---
def _branch_stochastic(stage, state):
    """Statistics + reserve (handover study). Delegates unchanged to min_power_miso."""
    from ntnconnect.engine.stages.min_power_miso import MinPowerMISO
    mp = MinPowerMISO(stage.cfg)
    mp.engine = getattr(stage, "engine", None)
    mp.run(state)


def _branch_csi(stage, state):
    """Perfect CSI on the true channel, no reserve (min_power_miso genie path)."""
    from ntnconnect.engine.stages.min_power_miso import MinPowerMISO
    prev = state.data.get("genie_power")
    state.data["genie_power"] = True
    try:
        mp = MinPowerMISO(stage.cfg)
        mp.engine = getattr(stage, "engine", None)
        mp.run(state)
    finally:
        if prev is None:
            state.data.pop("genie_power", None)
        else:
            state.data["genie_power"] = prev


def _branch_dt(stage, state):
    """Size power on the predicted (aged) twin channel, no reserve; the true channel
    grades it downstream. Falls back to stochastic when no twin channel is present."""
    ctx = state.data.get("radio_context")
    if ctx is None or not ctx.has_twin:
        _branch_stochastic(stage, state)
        return
    links = state.data.get("links")
    if not isinstance(links, pd.DataFrame) or links.empty:
        return
    n = len(ctx)
    # required power on the twin channel: p = gamma*(N + I)/g_twin
    p_req = ctx.gamma_th * (ctx.noise_w + ctx.interference_w) \
        / np.maximum(ctx.twin_gain, 1e-30)
    # per-satellite budget, served-only, cheapest-first (as in min_power_miso)
    admitted = np.zeros(n, dtype=bool)
    cum: dict = {}
    srv = np.where(ctx.served)[0]
    for k in srv[np.argsort(p_req[srv], kind="stable")]:
        s = int(ctx.node_id[k])
        psat = float(ctx.power_budget_w.get(int(ctx.operator_id[k]), 12.6))
        c = cum.get(s, 0.0)
        if c + p_req[k] <= psat + 1e-12:
            admitted[k] = True
            cum[s] = c + p_req[k]
    tx = np.where(ctx.served & admitted, p_req, 0.0)
    out = links.copy()
    out["tx_power_w"]       = tx
    out["min_power_w"]      = p_req
    out["slot_admitted"]    = ctx.served & admitted
    out["min_power_capped"] = ctx.served & ~admitted
    state.data["links"] = out


RADIO_CSI_BRANCHES = {
    "stochastic": _branch_stochastic,
    "csi":        _branch_csi,
    "dt":         _branch_dt,
}


def _delegate_stage(cls_path):
    """Method that delegates to an existing power-allocation Stage."""
    def _method(stage, state):
        mod, _, cls = cls_path.rpartition(".")
        import importlib
        klass = getattr(importlib.import_module(mod), cls)
        s = klass(stage.cfg)
        s.engine = getattr(stage, "engine", None)
        s.run(state)
    return _method


def _apply_beamforming(stage, state):
    """Beamforming decision; rebuilds radio_context so power is sized on it."""
    from ntnconnect.engine.stages.rrm import ChanBeamforming
    bf = ChanBeamforming(stage.cfg)
    bf.engine = getattr(stage, "engine", None)
    bf.run(state)
    state.data["radio_context"] = build_radio_context(state, stage.cfg)


def _method_min_power(stage, state):
    """Closed-form MISO min-power: optional beamforming, then power+admission
    sized under the csi_mode channel assumption."""
    if bool(getattr(stage.cfg.radio, "run_beamforming", False)):
        _apply_beamforming(stage, state)
    mode = str(stage.cfg.radio.csi_mode).lower()
    if mode not in RADIO_CSI_BRANCHES:
        raise ValueError(f"Unknown radio csi_mode '{mode}'. "
                         f"Registered: {sorted(RADIO_CSI_BRANCHES)}")
    RADIO_CSI_BRANCHES[mode](stage, state)


def _method_max_min_rate(stage, state):
    """Full-power, max-min-fair RRM. Each satellite transmits its whole budget P_sat
    and splits it across served users to equalize SINR (max-min fair):
        p_u = gamma* * (N0_u + beta_u) / g_u,  gamma* = P_sat / sum_u (N0_u+beta_u)/g_u.
    Power is sized on the large-scale (mean) channel; sinr_calc grades the realized
    SINR on the full channel incl. fast fading. No QoS floor, no blocking."""
    import numpy as np
    import pandas as pd
    from ntnconnect.common.physics import channel_power_gain
    _KB = 1.380649e-23
    links = state.data.get("links")
    if not isinstance(links, pd.DataFrame) or links.empty:
        return
    n = len(links)
    cfg = stage.cfg
    bw = links.get("bandwidth_hz",
                   pd.Series(np.full(n, float(getattr(cfg.defaults_radio,
                                                      "bandwidth_hz", 5e6))))
                  ).astype(float).to_numpy()
    # large-scale (mean) channel gain = antenna gain x mean path loss (fading out);
    # fall back to the full channel gain when the mean column is absent
    gd  = links.get("antenna_gain_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
    plm = links.get("path_loss_mean_db", links.get("path_loss_db")).astype(float).to_numpy() \
        if ("path_loss_mean_db" in links.columns or "path_loss_db" in links.columns) else None
    g = (10.0 ** (gd / 10.0) * 10.0 ** (-plm / 10.0)) if plm is not None \
        else channel_power_gain(links, cfg.operators)
    g = np.maximum(np.asarray(g, dtype=float), 1e-30)
    T_sys = (float(cfg.defaults_radio.system_temperature_k)
             * 10.0 ** (cfg.defaults_radio.noise_figure_db / 10.0))
    noise_w = _KB * T_sys * np.maximum(bw, 1.0)
    beta_w = links.get("i_ext_w", pd.Series(np.zeros(n))).astype(float).to_numpy()

    import os as _os
    op_ids = links.get("operator_id", pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
    node = links.get("node_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    user = links.get("user_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    _op_psat = {int(op.id): float(getattr(op.radio, "power_max_w", 12.6))
                for op in cfg.operators}
    _env_psat = _os.environ.get("WISEE_PMAX_W")
    assigns = state.data.get("assignments", {}) or {}
    served = np.array([assigns.get(int(user[k]), -2) == int(node[k])
                       for k in range(n)], dtype=bool)

    # per satellite: equal-SINR (max-min) split of P_sat over its served users
    tx = np.zeros(n, dtype=float)
    by_sat: dict = {}
    for k in np.where(served)[0]:
        by_sat.setdefault(int(node[k]), []).append(int(k))
    for s, ks in by_sat.items():
        psat = float(_env_psat) if _env_psat else _op_psat.get(int(op_ids[ks[0]]), 12.6)
        w = (noise_w[ks] + beta_w[ks]) / g[ks]        # inverse-channel weight
        denom = float(np.sum(w))
        if denom <= 0.0:
            continue
        gamma_star = psat / denom                     # common max-min SINR
        for j, k in enumerate(ks):
            tx[k] = gamma_star * w[j]                 # p_u ; sum_u p_u = P_sat

    out = links.copy()
    out["tx_power_w"]       = tx
    out["min_power_w"]      = tx
    out["slot_admitted"]    = served
    out["min_power_capped"] = served & (tx <= 0.0)     # only if channel unusable
    out["assoc_guaranteed"] = served.astype(np.int8)
    state.data["links"] = out


def _method_fixed_power(stage, state):
    """Every served beam transmits the same fixed power P_max; sinr_calc grades."""
    import numpy as np
    import pandas as pd
    links = state.data.get("links")
    if not isinstance(links, pd.DataFrame) or links.empty:
        return
    n = len(links)
    cfg = stage.cfg
    import os as _os
    op_ids = links.get("operator_id", pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
    node = links.get("node_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    user = links.get("user_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    _op_pmax = {int(op.id): float(getattr(op.radio, "power_max_w", 12.6))
                for op in cfg.operators}
    _env_pmax = _os.environ.get("WISEE_PMAX_W")
    assigns = state.data.get("assignments", {}) or {}
    served = np.array([assigns.get(int(user[k]), -2) == int(node[k])
                       for k in range(n)], dtype=bool)
    pmax = np.array([float(_env_pmax) if _env_pmax else _op_pmax.get(int(op_ids[k]), 12.6)
                     for k in range(n)], dtype=float)
    tx = np.where(served, pmax, 0.0)

    out = links.copy()
    out["tx_power_w"]       = tx
    out["min_power_w"]      = tx
    out["slot_admitted"]    = served
    out["min_power_capped"] = np.zeros(n, dtype=bool)
    out["assoc_guaranteed"] = served.astype(np.int8)
    state.data["links"] = out


# RRM method registry: cfg.radio.power -> callable.
RADIO_ALGORITHMS = {
    "min_power":    _method_min_power,
    "max_min_rate": _method_max_min_rate,
    "fixed_power":  _method_fixed_power,
    "equal":        _delegate_stage("ntnconnect.engine.stages.power_algorithms.RRMPowerEqual"),
    "waterfilling": _delegate_stage("ntnconnect.engine.stages.power_algorithms.RRMPowerWaterfilling"),
    "utility":      _delegate_stage("ntnconnect.engine.stages.power_algorithms.RRMPowerUtility"),
}


def register_radio_method(name):
    """Decorator: register a callable(stage, state) as an RRM/power method."""
    def deco(fn):
        RADIO_ALGORITHMS[str(name).lower()] = fn
        return fn
    return deco


def list_radio_methods():
    return dict(RADIO_ALGORITHMS)


def get_radio_method(cfg):
    name = str(cfg.radio.power).lower()
    if name not in RADIO_ALGORITHMS:
        raise ValueError(f"Unknown radio power method '{name}'. "
                         f"Registered: {sorted(RADIO_ALGORITHMS)}")
    return RADIO_ALGORITHMS[name]


@register_stage("radio")
class Radio(Stage):
    """Build the unified context, then run the selected RRM method (decisions only)."""
    reads  = ["links", "assignments", "twin_channel"]
    writes = ["links", "radio_context"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        from ntnconnect.engine.stages.csi_buffer import CSIBuffer
        self._csi_buf = CSIBuffer(cfg)      # online CSI-update timing (large-scale)

    def run(self, state):
        # age the large-scale CSI by the feedback delay; realized path_loss still
        # grades. At coarse slots the delay rounds to 0 -> aged == current.
        links = state.data.get("links")
        if isinstance(links, pd.DataFrame) and not links.empty \
                and "path_loss_mean_db" in links.columns:
            links = links.copy()
            links["path_loss_mean_db"] = self._csi_buf.aged(
                links, int(getattr(state, "t_idx", 0)), "path_loss_mean_db")
            state.data["links"] = links

        # unified inputs, published for downstream/DT
        ctx = build_radio_context(state, self.cfg)
        if ctx is None:
            return
        state.data["radio_context"] = ctx

        # run the selected RRM method (min_power further dispatches on csi_mode)
        get_radio_method(self.cfg)(self, state)

        # normalize the decision schema
        links = state.data.get("links")
        if isinstance(links, pd.DataFrame) and not links.empty:
            if "slot_admitted" in links.columns:
                links["served_admitted"] = links["slot_admitted"].astype(bool)
            if "min_power_capped" in links.columns:
                links["served_blocked"] = links["min_power_capped"].astype(bool)
            state.data["links"] = links
