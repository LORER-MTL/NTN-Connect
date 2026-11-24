# Unified radio-input context: the SINGLE input every RRM/radio branch reads.
#
# The radio (RRM) stage has three branches keyed by cfg.radio.csi_mode -
# csi (perfect), stochastic (the handover study: statistics + reserve), dt (twin).
# They differ ONLY in HOW they act on the channel knowledge, not in what they
# receive. So we build ONE context here and hand the same object to every
# branch. Nothing is decided here; this is inputs only.
#
# Conventions (kept identical to min_power_miso so a branch that reuses these
# numbers reproduces the handover-study results exactly):
#   * one row per VISIBLE link (the working set), aligned arrays.
#   * `served` marks the link that is the user's association (assignments).
#   * gains are linear power gains per unit tx power (||h||^2), so
#     SINR = tx_power * gain / (interference + noise).
#
# Two notes the study cares about:
#   * INTERFERENCE is an input only. Association already used it; an RRM
#     algorithm MAY or MAY NOT use it (equal-power ignores it, min-power uses
#     it). It is provided, never forced.
#   * TARGETS differ study to study, so they are explicit (RadioTargets) and
#     come from config, not hard-coded.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

_KB = 1.380649e-23


@dataclass
class RadioTargets:
    """Per-study service targets. Explicit so a study states exactly what it
    asks for. gamma_th (per link) is derived from the rate target and bandwidth
    by the builder; the scalars below are the study-level knobs."""
    qos_rate_bps: float = 1.0e6      # R0: min per-user rate -> gamma_th = 2^(R0/B)-1
    outage_eps: float = 0.05         # outage level for the power reserve


@dataclass
class RadioContext:
    """Unified per-link input for all radio (RRM) branches. Read-only view built
    from the channel + interference + assignments; branches consume it and only
    differ in how they size power and grade SINR.

    All arrays are length N (visible links) and index-aligned."""
    # link identity
    user_id: np.ndarray
    node_id: np.ndarray
    operator_id: np.ndarray
    served: np.ndarray                 # bool: this link is the user's association
    bandwidth_hz: np.ndarray

    # target (per link + study scalars)
    gamma_th: np.ndarray               # SINR target = 2^(R0/B) - 1
    targets: RadioTargets

    # channel knowledge
    true_gain: np.ndarray              # full channel incl. fast fading (grades SINR)
    large_scale_gain: np.ndarray       # antenna gain x MEAN path loss only (reserve sizing)
    k_factor_db: np.ndarray            # Rician K (fading distribution, for the reserve)
    sigma_sh_db: np.ndarray            # shadowing std  (composite reserve)
    sigma_wx_db: np.ndarray            # weather std    (composite reserve)

    # interference (INPUT only - a branch may or may not use it) + noise
    interference_w: np.ndarray         # beta: aggregate co-channel interference [W]
    noise_w: np.ndarray                # kB * T_sys * 10^(NF/10) * B

    # per-operator power budget P_sat [W]
    power_budget_w: dict = field(default_factory=dict)

    # twin channel (only when the DT lane is active; None otherwise)
    twin_gain: Optional[np.ndarray] = None      # DT-predicted gain (aged)
    twin_sigma: Optional[np.ndarray] = None     # prediction uncertainty

    def __len__(self):
        return len(self.user_id)

    @property
    def has_twin(self) -> bool:
        return self.twin_gain is not None


def build_radio_context(state, cfg) -> Optional[RadioContext]:
    """Assemble the unified radio-input context from state.data["links"] +
    assignments + config. Read-only: does not mutate links or state, so the
    handover-study path is untouched. Returns None when there are no links."""
    links = state.data.get("links")
    if not isinstance(links, pd.DataFrame) or links.empty:
        return None
    n = len(links)
    ho = cfg.handover
    dr = cfg.defaults_radio

    # --- identity ---
    user = links.get("user_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    node = links.get("node_id", pd.Series(np.arange(n))).astype(int).to_numpy()
    op_ids = links.get("operator_id",
                       pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
    assigns = state.data.get("assignments", {}) or {}
    served = np.array([assigns.get(int(user[k]), -2) == int(node[k])
                       for k in range(n)], dtype=bool)

    # --- target: gamma_th from rate/bandwidth (same as min_power_miso) ---
    bw = links.get("bandwidth_hz",
                   pd.Series(np.full(n, float(getattr(dr, "bandwidth_hz", 20e6))))
                   ).astype(float).to_numpy()
    R0 = float(getattr(ho, "qos_rate_bps", 1.0e6))
    eps = float(getattr(ho, "dro_epsilon", 0.05))
    targets = RadioTargets(qos_rate_bps=R0, outage_eps=eps)
    gamma_th = 2.0 ** (R0 / np.maximum(bw, 1.0)) - 1.0

    # --- channel: true (full, incl. fading) and large-scale (mean) gains ---
    from ntnconnect.common.physics import channel_power_gain
    true_gain = channel_power_gain(links, cfg.operators)
    gd = links.get("antenna_gain_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
    pl_mean = links.get("path_loss_mean_db",
                        links.get("path_loss_db", pd.Series(np.zeros(n)))
                        ).astype(float).to_numpy()
    large_scale_gain = 10.0 ** (gd / 10.0) * 10.0 ** (-pl_mean / 10.0)

    # --- fading statistics (for reserve sizing) ---
    kdb = links.get("k_factor_db", pd.Series(np.full(n, 10.0))).astype(float).to_numpy()
    kdb = np.where(np.isfinite(kdb), kdb, 10.0)
    ssh = links.get("sigma_sh_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
    swx = links.get("sigma_wx_db", pd.Series(np.zeros(n))).astype(float).to_numpy()

    # --- interference (input) + noise floor ---
    interference_w = links.get("i_ext_w", pd.Series(np.zeros(n))).astype(float).to_numpy()
    T_sys = float(dr.system_temperature_k) * 10.0 ** (float(dr.noise_figure_db) / 10.0)
    noise_w = _KB * T_sys * np.maximum(bw, 1.0)

    # --- per-operator power budget ---
    power_budget_w = {int(op.id): float(getattr(op.radio, "power_max_w", 12.6))
                      for op in cfg.operators}

    # --- twin channel (only if the DT lane published one) ---
    twin_gain = twin_sigma = None
    tw = state.data.get("twin_channel")
    if isinstance(tw, dict):
        g = tw.get("gain"); s = tw.get("sigma")
        if g is not None and len(g) == n:
            twin_gain = np.asarray(g, dtype=float)
        if s is not None and len(s) == n:
            twin_sigma = np.asarray(s, dtype=float)

    return RadioContext(
        user_id=user, node_id=node, operator_id=op_ids, served=served,
        bandwidth_hz=bw, gamma_th=gamma_th, targets=targets,
        true_gain=true_gain, large_scale_gain=large_scale_gain,
        k_factor_db=kdb, sigma_sh_db=ssh, sigma_wx_db=swx,
        interference_w=interference_w, noise_w=noise_w,
        power_budget_w=power_budget_w,
        twin_gain=twin_gain, twin_sigma=twin_sigma,
    )
