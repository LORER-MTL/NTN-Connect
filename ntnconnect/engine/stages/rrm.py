# RRM stages: RB allocation + 3GPP MCS lookup.

from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage


# 3GPP TS 38.214 Table 5.1.3.1-1 (CQI table 1) - condensed
# (sinr_threshold_db, modulation, code_rate, spectral_efficiency_bps_hz)
MCS_TABLE_3GPP = [
    (-9.5,  "BPSK",   0.07,  0.1523),  # 0
    (-6.0,  "QPSK",   0.12,  0.2344),
    (-4.0,  "QPSK",   0.19,  0.3770),
    (-2.0,  "QPSK",   0.30,  0.6016),
    ( 0.0,  "QPSK",   0.44,  0.8770),
    ( 2.0,  "QPSK",   0.59,  1.1758),
    ( 4.0,  "16QAM",  0.37,  1.4766),
    ( 6.0,  "16QAM",  0.48,  1.9141),
    ( 8.0,  "16QAM",  0.60,  2.4063),
    (10.0,  "64QAM",  0.46,  2.7305),
    (12.0,  "64QAM",  0.55,  3.3223),
    (14.0,  "64QAM",  0.65,  3.9023),
    (16.0,  "64QAM",  0.75,  4.5234),
    (18.0,  "64QAM",  0.85,  5.1152),
    (20.0,  "64QAM",  0.93,  5.5547),
]


def _mcs_from_sinr(sinr_db: float):
    """Return (mcs_idx, modulation, coding_rate, spectral_eff)."""
    if sinr_db is None or np.isnan(sinr_db):
        return None, None, None, None
    last = (0, MCS_TABLE_3GPP[0][1], MCS_TABLE_3GPP[0][2], MCS_TABLE_3GPP[0][3])
    for i, row in enumerate(MCS_TABLE_3GPP):
        if sinr_db >= row[0]:
            last = (i, row[1], row[2], row[3])
        else:
            break
    return last


@register_stage("rrm_mcs", internal=True)
class RRMMcs(Stage):
    """Annotate every link with the 3GPP MCS chosen from its SINR."""
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        if "sinr_db" not in links.columns: return
        out = links.copy()
        mcs_idx, mod, coderate, se = [], [], [], []
        for s in out["sinr_db"].values:
            mi, mo, cr, e = _mcs_from_sinr(float(s) if s is not None else None)
            mcs_idx.append(mi); mod.append(mo); coderate.append(cr); se.append(e)
        out["mcs_index"]            = mcs_idx
        out["modulation"]           = mod
        out["coding_rate"]          = coderate
        out["se_3gpp_bps_hz"]       = se
        state.data["links"] = out


@register_stage("rrm_allocate_rb", internal=True)
class RRMAllocateRB(Stage):
    """Proportional-fair RB allocation per serving node.

    Resource block = 180 kHz (LTE/NR-FR1 numerology 0).
    """
    reads  = ["links", "assignments"]
    writes = ["rb_alloc_map"]
    thread_safe = False

    RB_HZ = 180_000.0

    def run(self, state):
        links = state.data.get("links")
        assigns = state.data.get("assignments", {}) or {}
        if not isinstance(links, pd.DataFrame) or links.empty or not assigns:
            state.data["rb_alloc_map"] = {}
            return
        alloc: dict = {}
        # Group users by serving node
        per_node: dict = {}
        for u, s in assigns.items():
            if int(s) < 0: continue
            per_node.setdefault(int(s), []).append(int(u))
        for nid, users in per_node.items():
            sub = links[links["node_id"] == nid]
            if sub.empty: continue
            bw = float(sub["bandwidth_hz"].iloc[0])
            n_rb_total = max(1, int(bw / self.RB_HZ))
            # Proportional fair: weight by serving SINR linear
            weights = []
            for u in users:
                row = sub[sub["user_id"] == u]
                if row.empty: weights.append(1.0); continue
                sinr_db = float(row["sinr_db"].iloc[0])
                weights.append(10 ** (sinr_db / 10.0))
            wsum = sum(weights) or 1.0
            for u, w in zip(users, weights):
                rb_u = max(1, int(round(n_rb_total * w / wsum)))
                alloc[(int(nid), int(u))] = {
                    "n_rbs": rb_u,
                    "n_rbs_total": n_rb_total,
                    "bw_alloc_hz": rb_u * self.RB_HZ,
                }
        state.data["rb_alloc_map"] = alloc


@register_stage("acc_admission", internal=True)
class AccAdmission(Stage):
    """Per-node admission cap. Marks rejected users with assignment = -2
    so downstream stages can distinguish from outage (-1)."""
    reads  = ["assignments", "links"]
    writes = ["assignments"]
    thread_safe = False

    def run(self, state):
        assigns = state.data.get("assignments", {}) or {}
        if not assigns: return
        cap = int(self.cfg.handover.max_users_per_sat)
        per_node_count: dict = {}
        kept: dict = {}
        # Keep in user-id order, deterministic
        for u in sorted(assigns):
            s = int(assigns[u])
            if s < 0: kept[u] = s; continue
            if per_node_count.get(s, 0) >= cap:
                kept[u] = -2     # rejected by admission
                continue
            kept[u] = s
            per_node_count[s] = per_node_count.get(s, 0) + 1
        state.data["assignments"] = kept


@register_stage("chan_beamforming", internal=True)
class ChanBeamforming(Stage):
    """Apply physical MIMO array-factor + mode-aware efficiency + hybrid
    multiplexing penalty. Ported from the original skyntn.mimo.engine.MIMOEngine.

    Per operator op:
        n_tx = op.channel_model.n_tx_x * op.channel_model.n_tx_y
        n_rx = op.channel_model.n_rx
        base_gain_lin = n_tx * n_rx                       (array factor)
        mode = op.radio.beamforming_mode  in {digital, analog, hybrid, none}
        eff   = bf_eff_<mode>                              (efficiency)
        mux   = digital -> 1
                analog  -> 1 / n_users_op                  (single-stream)
                hybrid  -> min(1, n_rf / n_users_op)       (RF chain limit)
        bf_gain_lin = base_gain_lin * eff * mux
        bf_gain_db  = 10*log10(bf_gain_lin)

    The dB gain is *added to antenna_gain_db* so downstream signal-power
    computation (InterferenceStage) automatically picks it up via the same
    10^(G/10) factor it already applies.
    """
    reads  = ["links", "assignments"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        assigns = state.data.get("assignments", {}) or {}

        # Per-operator served-user count (drives the multiplexing penalty)
        op_n_users: dict = {}
        for u, s in assigns.items():
            if int(s) < 0: continue
            row = links[links["node_id"] == int(s)].head(1)
            if row.empty: continue
            oid = int(row["operator_id"].iloc[0])
            op_n_users[oid] = op_n_users.get(oid, 0) + 1

        # Per-operator BF-gain-dB lookup
        op_bf_db: dict = {}
        for op in self.cfg.operators:
            cm = op.channel_model
            n_tx = max(1, int(cm.n_tx_x) * int(cm.n_tx_y))
            n_rx = max(1, int(cm.n_rx))
            base = n_tx * n_rx
            mode = (op.radio.beamforming_mode or "none").lower()
            n_u  = max(1, op_n_users.get(op.id, 1))
            if mode == "digital":
                eff = float(cm.bf_eff_digital); mux = 1.0
            elif mode == "analog":
                eff = float(cm.bf_eff_analog);  mux = 1.0 / n_u
            elif mode == "hybrid":
                eff = float(cm.bf_eff_hybrid)
                mux = min(1.0, float(cm.num_rf_chains) / n_u)
            else:
                eff = 0.0; mux = 0.0   # "none" -> 0 dB BF gain
            bf_lin = max(1e-12, base * eff * mux) if eff > 0 else 1.0
            op_bf_db[op.id] = 10.0 * np.log10(bf_lin) if eff > 0 else 0.0

        out = links.copy()
        bf_gain = out["operator_id"].astype(int).map(op_bf_db).fillna(0.0).values
        # F4 - channel-aware precoder
        # On top of the geometric array gain, MRT-style precoding boosts
        # links whose instantaneous channel gain is above the operator's
        # mean and attenuates those below it. We normalize each operator
        # to unit average so total transmitted power is preserved (this
        # is exactly the effect of conjugate beamforming with equal-power
        # allocation across users on the same node).
        if "chan_gain_lin" in out.columns:
            mrt_db = np.zeros(len(out), dtype=float)
            for op_id in out["operator_id"].astype(int).unique():
                m = (out["operator_id"].astype(int).values == int(op_id))
                cg = np.maximum(out.loc[m, "chan_gain_lin"].astype(float).values, 1e-30)
                ratio = cg / cg.mean()
                mrt_db[m] = 10.0 * np.log10(np.maximum(ratio, 1e-12))
        else:
            mrt_db = np.zeros(len(out), dtype=float)
        bf_total = bf_gain + mrt_db
        out["bf_gain_db"]      = bf_total
        out["bf_array_db"]     = bf_gain          # geometric component
        out["bf_precoder_db"]  = mrt_db           # channel-aware component
        out["antenna_gain_db"] = (out["antenna_gain_db"].astype(float).values
                                  + bf_total)
        state.data["links"] = out


@register_stage("chan_power_control", internal=True)
class ChanPowerControl(Stage):
    """Per-link tx-power assignment. Default: equal split of node's P_max
    across served users (matches POWER_ALLOC hook)."""
    reads  = ["links", "assignments"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        assigns = state.data.get("assignments", {}) or {}
        if not isinstance(links, pd.DataFrame) or links.empty: return
        # Per-node # served
        per_node: dict = {}
        for u, s in assigns.items():
            if int(s) < 0: continue
            per_node[int(s)] = per_node.get(int(s), 0) + 1
        op_pmax = {op.id: float(op.radio.power_max_w) for op in self.cfg.operators}
        out = links.copy()
        nid_arr = out["node_id"].astype(int).to_numpy()
        oid_arr = out["operator_id"].astype(int).to_numpy()
        pmax_arr = np.array([op_pmax.get(o, 1.0) for o in oid_arr])
        n_arr    = np.array([max(1, per_node.get(n, 1)) for n in nid_arr])
        out["tx_power_w"] = pmax_arr / n_arr
        state.data["links"] = out
