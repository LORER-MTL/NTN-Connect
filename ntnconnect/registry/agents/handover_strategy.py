# SkyNTNPro v1 - Handover strategy
# Three concrete implementations behind one interface.
# Switch via cfg.handover.mode.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

# HO classification constants used in per_user_kpis.ho_type
HO_NONE          = "none"
HO_INITIAL       = "initial"
HO_INTRA_OP_TYPE = "intra-operator-intra-type"
HO_INTRA_OP_INTERTYPE = "intra-operator-inter-type"
HO_INTER_OP      = "inter-operator"
HO_OUTAGE_ENTER  = "outage-enter"
HO_OUTAGE_RECOV  = "outage-recover"


@dataclass
class HandoverContext:
    t_idx: int
    timestamp: Any
    users:        np.ndarray      # (n_user,)
    node_ids:     np.ndarray      # (n_node,)
    node_types:   np.ndarray      # (n_node,) str
    node_op_ids:  np.ndarray      # (n_node,) int
    vis_mat:      np.ndarray      # (n_node, n_user)
    dist_mat:     np.ndarray
    elev_mat:     np.ndarray
    rate_mat:     np.ndarray
    prev_assign:  Dict[int, int]  # user_id -> previous node_idx (-1 = outage)
    cfg:          Any
    # Earth-cell id per user; None when clustering is off.
    user_cluster_ids: np.ndarray | None = None      # shape (n_user,) int32
    # Write-back slot for beam_id when sat_cell_capacity_mode == "beam_tagged".
    out_beam_ids: Dict[int, int] | None = None      # user_id -> beam_id
    out_assoc_phase: Dict[int, int] | None = None   # user_id -> 1 (guaranteed) | 0 (best-effort)
    # Predicted remaining visibility [s]; NaN = unknown / rising sat.
    remain_mat: np.ndarray | None = None            # (n_node, n_user) float
    # Aggregate co-channel interference beta_{u,s} [W]; None -> noise-only QoS.
    beta_mat: np.ndarray | None = None              # (n_node, n_user) float
    # Large-scale channel power gain g_bar_{u,s} = 10^((G-PL)/10); None -> use FSPL proxy.
    gain_mat: np.ndarray | None = None              # (n_node, n_user) float


class BaseHandoverStrategy(ABC):
    name = "base"
    @abstractmethod
    def assign(self, ctx: HandoverContext) -> np.ndarray: ...


def _ho_period_s(cfg) -> float:
    """Real time [s] advanced per handover decision (run_every_n_ticks * step_s); env/cfg override."""
    import os as _os
    env = _os.environ.get("WISEE_HO_PERIOD_S")
    if env:
        return float(env)
    ho = getattr(cfg, "handover", None)
    p = getattr(ho, "ho_decision_period_s", None) if ho else None
    if p:
        return float(p)
    ren = float(getattr(ho, "run_every_n_ticks", 1) or 1) if ho else 1.0
    st  = float(getattr(getattr(cfg, "env", None), "step_size_s", 1.0) or 1.0)
    return ren * st


def classify_ho(prev_node_type, prev_op_id, new_node_type, new_op_id) -> str:
    if prev_node_type is None:
        return HO_INITIAL
    if new_op_id == -1 and prev_op_id != -1:
        return HO_OUTAGE_ENTER
    if new_op_id != -1 and prev_op_id == -1:
        return HO_OUTAGE_RECOV
    if prev_op_id == new_op_id and prev_node_type == new_node_type:
        return HO_INTRA_OP_TYPE
    if prev_op_id == new_op_id:
        return HO_INTRA_OP_INTERTYPE
    return HO_INTER_OP


class CentralizedHO(BaseHandoverStrategy):
    """Global pass: hysteresis + load cap + operator preference + inter-op penalty."""
    name = "centralized"

    def __init__(self, cfg):
        ho = cfg.handover
        self.hyst = ho.hysteresis_deg
        self.cap  = ho.max_users_per_sat
        self.inter_op_pen   = ho.inter_operator_penalty
        self.inter_type_pen = ho.inter_node_type_penalty
        self.min_elev = ho.min_elevation_deg
        self.pref_op  = cfg.users.preferences.preferred_operator_id
        self.fb_cost  = cfg.users.preferences.fallback_cost

    def assign(self, ctx):
        n_user = len(ctx.users)
        out = np.full(n_user, -1, dtype=int)
        load = np.zeros(len(ctx.node_ids), dtype=int)

        order = np.argsort(-ctx.elev_mat.max(axis=0))    # best-elev users first (greedy)
        for u in order:
            cand = np.where(
                (ctx.vis_mat[:, u] > 0) &
                (ctx.elev_mat[:, u] >= self.min_elev) &
                (load < self.cap)
            )[0]
            if len(cand) == 0:
                continue
            prev = ctx.prev_assign.get(int(ctx.users[u]), -1)
            if prev in cand:
                if (ctx.elev_mat[cand, u].max() - ctx.elev_mat[prev, u]) < self.hyst:
                    out[u] = prev; load[prev] += 1; continue

            score = ctx.elev_mat[cand, u].astype(float)
            if prev != -1:
                same_op   = (ctx.node_op_ids[cand]   == ctx.node_op_ids[prev])
                same_type = (ctx.node_types[cand]    == ctx.node_types[prev])
                score -= np.where(~same_op,   self.inter_op_pen,   0.0)
                score -= np.where(~same_type, self.inter_type_pen, 0.0)
            if self.pref_op is not None:
                score -= np.where(ctx.node_op_ids[cand] != self.pref_op, self.fb_cost, 0.0)
            score -= 0.1 * load[cand]
            best = cand[int(np.argmax(score))]
            out[u] = best; load[best] += 1
        return out


class DistributedUEHO(BaseHandoverStrategy):
    """Each UE picks best by elevation from its own visible set. No coordination."""
    name = "distributed_ue"

    def __init__(self, cfg):
        ho = cfg.handover
        self.hyst = ho.hysteresis_deg
        self.min_elev = ho.min_elevation_deg

    def assign(self, ctx):
        n_user = len(ctx.users)
        out = np.full(n_user, -1, dtype=int)
        for u in range(n_user):
            cand = np.where(
                (ctx.vis_mat[:, u] > 0) &
                (ctx.elev_mat[:, u] >= self.min_elev)
            )[0]
            if len(cand) == 0: continue
            prev = ctx.prev_assign.get(int(ctx.users[u]), -1)
            if prev in cand:
                if (ctx.elev_mat[cand, u].max() - ctx.elev_mat[prev, u]) < self.hyst:
                    out[u] = prev; continue
            out[u] = cand[int(np.argmax(ctx.elev_mat[cand, u]))]
        return out


class DistributedSatelliteHO(BaseHandoverStrategy):
    """Per-satellite admission control: each node admits up to its cap."""
    name = "distributed_satellite"

    def __init__(self, cfg):
        ho = cfg.handover
        self.cap = ho.satellite_admission_cap
        self.K = ho.top_k_local
        self.min_elev = ho.min_elevation_deg

    def assign(self, ctx):
        n_user = len(ctx.users)
        out = np.full(n_user, -1, dtype=int)
        load = np.zeros(len(ctx.node_ids), dtype=int)
        for u in range(n_user):
            cand_full = np.where(
                (ctx.vis_mat[:, u] > 0) &
                (ctx.elev_mat[:, u] >= self.min_elev)
            )[0]
            if len(cand_full) == 0: continue
            ranked = cand_full[np.argsort(-ctx.elev_mat[cand_full, u])][:self.K]
            for s in ranked:
                if load[s] < self.cap:
                    out[u] = int(s); load[s] += 1; break
        return out


class HybridPerOperatorHO(BaseHandoverStrategy):
    """Centralized per operator, then outage users fall back to other operators."""
    name = "hybrid"

    def __init__(self, cfg):
        self.cfg = cfg
        self._central = CentralizedHO(cfg)

    def assign(self, ctx):
        cfg = self.cfg
        op_ids = sorted(set(int(x) for x in ctx.node_op_ids))
        out = np.full(len(ctx.users), -1, dtype=int)

        for op_id in op_ids:
            mask = ctx.node_op_ids == op_id
            if not mask.any(): continue
            ctx_op = HandoverContext(               # this operator's nodes only
                t_idx=ctx.t_idx, timestamp=ctx.timestamp,
                users=ctx.users,
                node_ids=ctx.node_ids[mask],
                node_types=ctx.node_types[mask],
                node_op_ids=ctx.node_op_ids[mask],
                vis_mat=ctx.vis_mat[mask, :],
                dist_mat=ctx.dist_mat[mask, :],
                elev_mat=ctx.elev_mat[mask, :],
                rate_mat=ctx.rate_mat[mask, :],
                prev_assign={u: (-1) for u in ctx.prev_assign},   # local view
                cfg=cfg,
            )
            local = self._central.assign(ctx_op)
            global_idx = np.where(mask)[0]       # local -> global index
            for u in range(len(ctx.users)):
                if out[u] != -1: continue       # already served by previous operator
                if local[u] != -1:
                    out[u] = int(global_idx[local[u]])
        return out


class NearestSatHO(BaseHandoverStrategy):
    """Centralized: nearest visible sat, with 10% hysteresis."""
    name = "nearest_sat"
    HYSTERESIS_FACTOR = 0.90

    def __init__(self, cfg):
        self.min_elev = cfg.handover.min_elevation_deg

    def assign(self, ctx):
        n_user = len(ctx.users)
        out = np.full(n_user, -1, dtype=int)
        masked = np.where(
            (ctx.vis_mat > 0) & (ctx.elev_mat >= self.min_elev),
            ctx.dist_mat, np.inf,
        )
        new_best = np.argmin(masked, axis=0)
        no_vis = np.all(masked == np.inf, axis=0)
        new_best[no_vis] = -1

        # 10% hysteresis: switch only if d_cand < HF * d_prev
        for u in range(n_user):
            uid = int(ctx.users[u])
            prev = ctx.prev_assign.get(uid, -1)
            if prev < 0:
                out[u] = int(new_best[u]); continue
            if ctx.vis_mat[prev, u] == 0:
                out[u] = int(new_best[u]); continue
            cand = int(new_best[u])
            if cand < 0 or cand == prev:
                out[u] = prev; continue
            d_prev = ctx.dist_mat[prev, u]
            d_cand = ctx.dist_mat[cand, u]
            out[u] = cand if d_cand < d_prev * self.HYSTERESIS_FACTOR else prev
        return out


def _enforce_user_cap(out, score_su, vis_su, users, max_cells: int):
    """Enforce the per-satellite beam cap at USER level: keep the top-score
    max_cells users per sat, demote the rest to a next-best free beam or block."""
    if max_cells is None or max_cells <= 0 or max_cells >= 1000:
        return out
    n_user = len(users)
    S = score_su.shape[0]
    load = np.zeros(S, dtype=int)
    for uc in range(n_user):
        s = int(out[uc])
        if 0 <= s < S:
            load[s] += 1
    # iterate until no sat exceeds cap (bounded: each demotion frees a beam)
    changed = True
    while changed:
        changed = False
        for s in range(S):
            if load[s] <= max_cells:
                continue
            members = [uc for uc in range(n_user) if int(out[uc]) == s]
            members.sort(key=lambda uc: score_su[s, uc], reverse=True)   # keep best
            for uc in members[max_cells:]:
                order = np.argsort(score_su[:, uc])[::-1]                 # next-best sats
                newsat = -1
                for cand in order:
                    cand = int(cand)
                    if cand == s or not vis_su[cand, uc]:
                        continue
                    if not np.isfinite(score_su[cand, uc]):
                        continue
                    if load[cand] < max_cells:
                        newsat = cand
                        break
                out[uc] = newsat
                load[s] -= 1
                if newsat >= 0:
                    load[newsat] += 1
                changed = True
    return out


def _setup_antichurn(strategy, cfg):
    """Attach persistent TTT/MTS anti-churn state + tick thresholds (0 = disabled)."""
    ho = getattr(cfg, "handover", None)
    strategy._ttt_ticks = int(getattr(ho, "ttt_ticks", 0) or 0)
    strategy._mts_ticks = int(getattr(ho, "mts_ticks", 0) or 0)
    strategy._ttt_state = {}          # cluster_id -> (candidate_sat_idx, t_first)
    strategy._mts_state = {}          # cluster_id -> t_idx of last cluster HO


def _cluster_or_user_assign(score, ctx, hyst, cap_mode, max_cells, owner=None):
    """Route a score matrix through _cluster_argmax when cell ids exist, else the
    per-user _argmax_with_hysteresis. Honours the beam cap and owner's TTT/MTS state."""
    tts = getattr(owner, "_ttt_state", None)
    mts = getattr(owner, "_mts_state", None)
    ttt_ticks = int(getattr(owner, "_ttt_ticks", 0) or 0)
    mts_ticks = int(getattr(owner, "_mts_ticks", 0) or 0)
    t_idx = int(getattr(ctx, "t_idx", 0) or 0)
    # shared knob cfg.users.cluster_score_aggregation (a strategy may override self.agg)
    _cfg = getattr(ctx, "cfg", None)
    agg = "mean"
    if _cfg is not None and getattr(_cfg, "users", None) is not None:
        agg = str(getattr(_cfg.users, "cluster_score_aggregation", "mean")).lower()
    cids = ctx.user_cluster_ids
    if cids is None or np.all(np.asarray(cids) < 0):
        out = _argmax_with_hysteresis(score, ctx.vis_mat, ctx.prev_assign,
                                      hyst, ctx.users, ttt_state=tts,
                                      mts_state=mts, t_idx=t_idx,
                                      ttt_ticks=ttt_ticks, mts_ticks=mts_ticks)
        if str(cap_mode).lower() in ("capped", "strict_one", "beam_tagged"):   # beam cap
            out = _enforce_user_cap(out, score, ctx.vis_mat, ctx.users, int(max_cells))
        return out
    out, _beams = _cluster_argmax(score, ctx.vis_mat, np.asarray(cids),
                                  ctx.prev_assign, hyst, ctx.users, agg,
                                  capacity_mode=cap_mode,
                                  max_cells_per_sat=max_cells,
                                  ttt_state=tts, mts_state=mts, t_idx=t_idx,
                                  ttt_ticks=ttt_ticks, mts_ticks=mts_ticks,
                                  protect_incumbents=bool(
                                      getattr(owner, "_protect_incumbents", False)))
    return out


class FarthestApproachingHO(BaseHandoverStrategy):
    """MST: cluster-aware, beam-capped. Associate each cell to the longest-visible
    sat (max remaining_visibility_s) and STAY until it sets -> fewest handovers."""
    name = "farthest_approaching"

    def __init__(self, cfg):
        ho = cfg.handover
        self.min_elev   = float(ho.min_elevation_deg)
        self._hyst      = 1e9     # STAY: ride the whole pass
        self._cap_mode  = str(getattr(ho, "sat_cell_capacity_mode", "unlimited")).lower()
        self._max_cells = int(getattr(ho, "max_cells_per_sat", 1000))
        self._protect_incumbents = True   # reject-don't-bump: no above-mask capacity HOs
        _setup_antichurn(self, cfg)

    def _score(self, ctx):
        valid = (ctx.vis_mat > 0) & (ctx.elev_mat >= self.min_elev)
        if ctx.remain_mat is not None:
            remain = np.asarray(ctx.remain_mat, dtype=float)
            # remaining_visibility_s is quantised to whole seconds, so ties are
            # common; subtract a tiny elevation term to break them toward the
            # lowest-elevation (still-rising) sat, which outlasts its tie-mates.
            score = np.where(np.isfinite(remain), remain, 0.0) \
                - 1e-3 * np.nan_to_num(ctx.elev_mat, nan=0.0)
        else:
            score = ctx.elev_mat    # elev tie-break when remaining unknown
        return np.where(valid, score, -np.inf)

    def assign(self, ctx):
        out = _cluster_or_user_assign(self._score(ctx), ctx, self._hyst,
                                      self._cap_mode, self._max_cells,
                                      owner=self)
        # PURE served/unserved (no power admission): served -> 1, else -1.
        ctx.out_assoc_phase = {int(ctx.users[i]): (1 if int(out[i]) >= 0 else -1)
                               for i in range(len(ctx.users))}
        return out


class PersistentNearestHO(BaseHandoverStrategy):
    """Cluster-aware, beam-capped STAY policy: keep the serving sat until it sets,
    then re-acquire the highest-elevation visible sat."""
    name = "persistent_nearest"

    def __init__(self, cfg):
        ho = cfg.handover
        self.min_elev   = float(ho.min_elevation_deg)
        self._hyst      = 1e9     # STAY: never leave a still-visible serving sat
        self._cap_mode  = str(getattr(ho, "sat_cell_capacity_mode", "unlimited")).lower()
        self._max_cells = int(getattr(ho, "max_cells_per_sat", 1000))
        _setup_antichurn(self, cfg)

    def _score(self, ctx):
        valid = (ctx.vis_mat > 0) & (ctx.elev_mat >= self.min_elev)   # new entry: highest elev
        return np.where(valid, ctx.elev_mat, -np.inf)

    def assign(self, ctx):
        return _cluster_or_user_assign(self._score(ctx), ctx, self._hyst,
                                       self._cap_mode, self._max_cells,
                                       owner=self)


# Benchmark strategies: pick the visible sat maximising a score, with dB hysteresis.
def _argmax_with_hysteresis(score_su: np.ndarray, vis_su: np.ndarray,
                            prev_idx: Dict[int, int], hyst_db: float,
                            users: np.ndarray,
                            ttt_state: Dict | None = None,
                            mts_state: Dict | None = None,
                            t_idx: int = 0,
                            ttt_ticks: int = 0,
                            mts_ticks: int = 0) -> np.ndarray:
    """argmax(score) with dB hysteresis plus optional 3GPP TTT/MTS anti-churn.
       ttt_state[uid] = (candidate_sat_idx, t_first_seen); mts_state[uid] = t_idx of last HO."""
    masked = np.where(vis_su, score_su, -np.inf)
    new = np.argmax(masked, axis=0)
    valid = np.isfinite(masked.max(axis=0))
    out = np.where(valid, new, -1).astype(int)
    for u_col, uid in enumerate(users):
        uid = int(uid)
        p = int(prev_idx.get(uid, -1))
        if p < 0 or p >= vis_su.shape[0] or not vis_su[p, u_col]:
            # no valid previous sat -> forced argmax; record HO time
            if mts_state is not None and out[u_col] >= 0:
                mts_state[uid] = int(t_idx)
            continue
        if masked[p, u_col] >= masked[out[u_col], u_col] - hyst_db:   # dB hysteresis
            out[u_col] = p
            if ttt_state is not None: ttt_state.pop(uid, None)   # candidate stream broken
            continue
        cand = int(out[u_col])
        if mts_state is not None and mts_ticks > 0:   # min-time-of-stay: block HO in window
            last_ho = int(mts_state.get(uid, -10**9))
            if (t_idx - last_ho) < mts_ticks:
                out[u_col] = p          # forced to stay
                if ttt_state is not None: ttt_state.pop(uid, None)
                continue
        if ttt_state is not None and ttt_ticks > 0:   # time-to-trigger: N consecutive ticks
            entry = ttt_state.get(uid)
            if entry is None or int(entry[0]) != cand:
                ttt_state[uid] = (cand, int(t_idx))
                out[u_col] = p          # not enough TTT yet -> stay
                continue
            t_first = int(entry[1])
            if (t_idx - t_first) < ttt_ticks:
                out[u_col] = p
                continue
            ttt_state.pop(uid, None)    # TTT cleared -> HO this tick
        if mts_state is not None:       # HO accepted -> record MTS time
            mts_state[uid] = int(t_idx)
    return out


class MaxSINRHO(BaseHandoverStrategy):
    """MAX-RSRP baseline: greedy strongest-signal (RSRP proxy -20log10 d), near-zero
    hysteresis. The ideal RSRP ceiling; GPPA3HO adds the standards offset+TTT."""
    name = "max_sinr"

    def __init__(self, cfg):
        ho = cfg.handover
        self.min_elev   = float(getattr(ho, "min_elevation_deg", 20.0))
        self._hyst      = float(getattr(ho, "hysteresis_deg", 3.0))
        self._cap_mode  = str(getattr(ho, "sat_cell_capacity_mode", "unlimited")).lower()
        self._max_cells = int(getattr(ho, "max_cells_per_sat", 1000))
        _setup_antichurn(self, cfg)

    def _score(self, ctx):
        valid = (ctx.vis_mat > 0) & (ctx.elev_mat >= self.min_elev)
        with np.errstate(divide="ignore"):
            base = -20.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))   # RSRP proxy [dB]
        return np.where(valid, base, -np.inf)

    def assign(self, ctx):
        out = _cluster_or_user_assign(self._score(ctx), ctx, self._hyst,
                                      self._cap_mode, self._max_cells,
                                      owner=self)
        # PURE served/unserved (no power admission): served -> 1, else -1.
        ctx.out_assoc_phase = {int(ctx.users[i]): (1 if int(out[i]) >= 0 else -1)
                               for i in range(len(ctx.users))}
        return out


class GPPA3HO(BaseHandoverStrategy):
    """3GPP-A3 measurement handover (TS 38.331): HO only when neighbour RSRP beats
    serving by a3_offset_db, sustained a3_ttt_ticks. Standards-faithful, damped."""
    name = "gpp_a3"

    def __init__(self, cfg):
        ho = cfg.handover
        self.min_elev   = float(getattr(ho, "min_elevation_deg", 20.0))
        self._cap_mode  = str(getattr(ho, "sat_cell_capacity_mode", "unlimited")).lower()
        self._max_cells = int(getattr(ho, "max_cells_per_sat", 1000))
        # A3 event (TS 38.331 5.5.4.4: Mn - Hys > Mp + Off). Switch margin = Off + Hys.
        self._off       = float(getattr(ho, "a3_offset_db", 3.0))      # A3 Off (dB)
        self._a3_hys    = float(getattr(ho, "a3_hysteresis_db", 0.0))  # A3 Hys (dB)
        self._hyst      = self._off + self._a3_hys                     # switch margin
        _setup_antichurn(self, cfg)
        self._ttt_ticks = int(getattr(ho, "a3_ttt_ticks", 2))       # time-to-trigger
        self._mts_ticks = 0
        # Optional L3 filter (TS 38.331 5.5.3.2): F_n = (1-a)F_{n-1} + a M_n, a = 1/2^(k/4).
        self._l3_on = bool(getattr(ho, "a3_l3_filter", False))
        self._l3_k  = int(getattr(ho, "a3_l3_filter_k", 4))
        self._l3_a  = 1.0 / (2.0 ** (self._l3_k / 4.0))
        self._l3    = {}                        # (node_id, user_id) -> filtered RSRP

    def _score(self, ctx):
        valid = (ctx.vis_mat > 0) & (ctx.elev_mat >= self.min_elev)
        # RSRP M_n = 10*log10(g_bar) = G-PL when gain_mat present, else -20log10(d) proxy.
        gm = getattr(ctx, "gain_mat", None)
        with np.errstate(divide="ignore"):
            if gm is not None:
                rsrp = 10.0 * np.log10(np.maximum(np.asarray(gm, dtype=float), 1e-30))
            else:
                rsrp = -20.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))
        m = np.where(valid, rsrp, -np.inf)
        if not self._l3_on:
            return m
        # L3 filter per (satellite, user), keyed by stable node_id
        a = self._l3_a
        node_ids = ctx.node_ids; users = ctx.users
        F = np.full_like(m, -np.inf)
        seen = set()
        vi, vu = np.where(valid)
        for i, uc in zip(vi.tolist(), vu.tolist()):
            key = (int(node_ids[i]), int(users[uc])); seen.add(key)
            prev = self._l3.get(key)
            f = m[i, uc] if prev is None else (1.0 - a) * prev + a * m[i, uc]
            self._l3[key] = f
            F[i, uc] = f
        # drop out-of-view links so a re-acquired sat restarts its filter
        for k in [k for k in self._l3 if k not in seen]:
            self._l3.pop(k, None)
        return F

    def assign(self, ctx):
        return _cluster_or_user_assign(self._score(ctx), ctx, self._hyst,
                                       self._cap_mode, self._max_cells,
                                       owner=self)


class MaxRxPowerHO(BaseHandoverStrategy):
    """Highest received-power proxy = G_tot - PL (dB); same order as MaxSINRHO at constant noise."""
    name = "max_rx_power"
    def __init__(self, cfg):
        self.hyst_db = float(getattr(cfg.handover, "hysteresis_deg", 1.5))
    def assign(self, ctx):
        score = -20.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))  # ~ -PL_FSPL
        return _argmax_with_hysteresis(score, ctx.vis_mat,
                                       ctx.prev_assign, self.hyst_db,
                                       ctx.users)


class LoadAwareHO(BaseHandoverStrategy):
    """SINR proxy penalised by previously-attached load on each candidate."""
    name = "load_aware"
    def __init__(self, cfg):
        self.hyst_db = float(getattr(cfg.handover, "hysteresis_deg", 1.5))
        self.alpha   = float(getattr(cfg.handover, "load_penalty_db", 0.5))
    def assign(self, ctx):
        S = ctx.vis_mat.shape[0]
        load = np.zeros(S, dtype=float)
        for s in ctx.prev_assign.values():
            if 0 <= int(s) < S: load[int(s)] += 1.0
        base = -10.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))
        score = base - self.alpha * load[:, None]
        return _argmax_with_hysteresis(score, ctx.vis_mat,
                                       ctx.prev_assign, self.hyst_db,
                                       ctx.users)


class UtilityHO(BaseHandoverStrategy):
    """alpha*log(rate_proxy) + (1-alpha)*log(1+SINR_lin)."""
    name = "utility_based"
    def __init__(self, cfg):
        self.hyst_db = float(getattr(cfg.handover, "hysteresis_deg", 1.5))
        self.alpha   = float(getattr(cfg.handover, "utility_alpha", 0.5))
    def assign(self, ctx):
        sinr_db = -10.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))
        sinr_lin = 10**(sinr_db/10.0)
        rate = np.log2(1.0 + sinr_lin)
        score = (self.alpha * np.log(np.maximum(rate, 1e-6))
                 + (1.0 - self.alpha) * np.log(1.0 + sinr_lin))
        return _argmax_with_hysteresis(score, ctx.vis_mat,
                                       ctx.prev_assign, self.hyst_db,
                                       ctx.users)


# CLUSTER-aware variants: decide per Earth-cell (cluster_id), broadcast to its users.
# Aggregation via cfg.users.cluster_score_aggregation {"min","mean","sum_rate"}.
def _cluster_argmax(score_su: np.ndarray, vis_su: np.ndarray,
                    cluster_ids: np.ndarray, prev_idx: Dict[int, int],
                    hyst_db: float, users: np.ndarray,
                    aggregation: str,
                    capacity_mode: str = "unlimited",
                    max_cells_per_sat: int = 1000,
                    ttt_state: Dict | None = None,
                    mts_state: Dict | None = None,
                    t_idx: int = 0,
                    ttt_ticks: int = 0,
                    mts_ticks: int = 0,
                    protect_incumbents: bool = False):
    """Pick one sat per cluster, broadcast to its users. capacity_mode (unlimited |
    capped | strict_one | beam_tagged) bounds cells per sat. protect_incumbents=True
    keeps incumbents first (reject-don't-bump, used by MST); False lets a high-score
    arrival bump a lower-score incumbent. Returns (assignments, beam_ids), -1 = outage."""
    from collections import Counter
    S, U = score_su.shape
    out  = np.full(U, -1, dtype=int)
    beam = np.full(U, -1, dtype=int)
    masked   = np.where(vis_su, score_su, -np.inf)
    sinr_lin = 10**(masked / 10.0)

    # Stage 1: per-cluster aggregated score over sats
    cluster_users:  dict = {}     # c -> column indices of its users
    cluster_agg:    dict = {}     # c -> per-sat aggregated score (S,)
    for c in np.unique(cluster_ids):
        if int(c) < 0: continue
        cu = np.where(cluster_ids == c)[0]
        cluster_users[int(c)] = cu
        all_vis = np.all(vis_su[:, cu], axis=1)
        sub     = masked[:, cu]
        if aggregation == "min":
            agg = np.where(all_vis, sub.min(axis=1), -np.inf)
        elif aggregation == "sum_rate":
            r = np.log2(1.0 + sinr_lin[:, cu])
            agg = np.where(all_vis, r.sum(axis=1), -np.inf)
        else:
            agg = np.where(all_vis, sub.mean(axis=1), -np.inf)
        if np.isfinite(agg.max()):
            cluster_agg[int(c)] = agg

    # Stage 2: initial argmax per cluster + cluster-level hysteresis
    cluster_choice: dict = {}     # c -> sat idx
    cluster_score:  dict = {}     # c -> aggregated score at the chosen sat
    for c, agg in cluster_agg.items():
        best = int(np.argmax(agg))
        prev_in_c = [int(prev_idx.get(int(users[k]), -1))
                     for k in cluster_users[c]]
        prev_in_c = [p for p in prev_in_c if p >= 0]
        prev_maj = Counter(prev_in_c).most_common(1)[0][0] if prev_in_c else -1
        incumbent_ok = (0 <= prev_maj < S and np.isfinite(agg[prev_maj]))
        if incumbent_ok and agg[prev_maj] >= agg[best] - hyst_db:
            best = prev_maj                       # dB hysteresis: hold incumbent
        # cluster-level TTT/MTS anti-churn: only for a switch away from a valid
        # incumbent (a forced switch is never blocked)
        if incumbent_ok and best != prev_maj:
            blocked = False
            if mts_state is not None and mts_ticks > 0:
                if (t_idx - int(mts_state.get(c, -10**9))) < mts_ticks:
                    best = prev_maj; blocked = True          # min-time-of-stay
                    if ttt_state is not None: ttt_state.pop(c, None)
            if not blocked and ttt_state is not None and ttt_ticks > 0:
                e = ttt_state.get(c)
                if e is None or int(e[0]) != best:
                    ttt_state[c] = (best, int(t_idx))         # start TTT window
                    best = prev_maj
                elif (t_idx - int(e[1])) < ttt_ticks:
                    best = prev_maj                           # TTT not met yet
                else:
                    ttt_state.pop(c, None)                    # TTT met -> switch
        # reset MTS clock whenever the cluster changes serving sat
        if mts_state is not None and best != prev_maj:
            mts_state[c] = int(t_idx)
        cluster_choice[c] = best
        cluster_score[c]  = float(agg[best])

    # Stage 3: capacity enforcement
    mode = (capacity_mode or "unlimited").lower()
    K_eff = (1 if mode == "strict_one"
             else max(1, int(max_cells_per_sat)) if mode in ("capped", "beam_tagged")
             else None)            # None means unlimited
    by_sat: dict = {}              # sat -> [cluster ids in score order]
    if K_eff is None:
        for c, s in cluster_choice.items():
            by_sat.setdefault(int(s), []).append(int(c))
    else:
        # Group, evict over-cap losers, give them a second pick.
        seed: dict = {}
        for c, s in cluster_choice.items():
            seed.setdefault(int(s), []).append(int(c))
        # Pass 1 fills every sat with its top-K_eff clusters and collects overflow;
        # this must finish for all sats before any re-placement (else a later sat's
        # slice would overwrite an earlier eviction's placement -> spurious outage).
        def _incumbent_sat(cc):   # majority previous sat of the cluster's users
            prev_in = [int(prev_idx.get(int(users[k]), -1))
                       for k in cluster_users[cc]]
            prev_in = [p for p in prev_in if p >= 0]
            return Counter(prev_in).most_common(1)[0][0] if prev_in else -1

        overflow = []
        for s, clist in seed.items():
            if protect_incumbents:      # incumbents first, then by score; new arrivals overflow
                clist.sort(key=lambda cc: (0 if _incumbent_sat(cc) == int(s) else 1,
                                           -cluster_score[cc]))
            else:
                clist.sort(key=lambda cc: -cluster_score[cc])
            by_sat[int(s)] = clist[:K_eff]
            overflow.extend(clist[K_eff:])
        # Pass 2: re-place each evicted cluster on its best visible sat with a free beam
        for evicted in overflow:
            agg = cluster_agg[evicted]
            placed = False
            for s2 in np.argsort(-agg).tolist():
                if not np.isfinite(agg[int(s2)]): break
                if len(by_sat.get(int(s2), [])) < K_eff:
                    by_sat.setdefault(int(s2), []).append(evicted)
                    cluster_choice[evicted] = int(s2)
                    cluster_score[evicted]  = float(agg[int(s2)])
                    placed = True
                    break
            if not placed:
                cluster_choice[evicted] = -1     # all visible sats full -> outage

    # Stage 4: write assignments + beam ids
    for s, clist in by_sat.items():
        for beam_idx, c in enumerate(clist):
            cu = cluster_users[c]
            out[cu] = s
            if mode == "beam_tagged":
                beam[cu] = beam_idx
    for c, s in cluster_choice.items():
        if s < 0:
            out[cluster_users[c]] = -1
    return out, beam


class _BaseClusterHO(BaseHandoverStrategy):
    def __init__(self, cfg):
        self.cfg      = cfg                                    # needed by _ho_period_s (age update)
        self.hyst_db  = float(getattr(cfg.handover, "hysteresis_deg", 1.5))
        self.agg      = str(getattr(cfg.users, "cluster_score_aggregation",
                                    "mean")).lower()
        self.cap_mode = str(getattr(cfg.handover, "sat_cell_capacity_mode",
                                    "unlimited")).lower()
        self.max_cells = int(getattr(cfg.handover, "max_cells_per_sat", 1000))
        _setup_antichurn(self, cfg)

    def _score(self, ctx):
        raise NotImplementedError

    def assign(self, ctx):
        score = self._score(ctx)
        t_idx = int(getattr(ctx, "t_idx", 0) or 0)
        if ctx.user_cluster_ids is None or (ctx.user_cluster_ids < 0).all():
            return _argmax_with_hysteresis(score, ctx.vis_mat,      # per-user fallback
                                           ctx.prev_assign, self.hyst_db,
                                           ctx.users, ttt_state=self._ttt_state,
                                           mts_state=self._mts_state, t_idx=t_idx,
                                           ttt_ticks=self._ttt_ticks,
                                           mts_ticks=self._mts_ticks)
        out, beams = _cluster_argmax(score, ctx.vis_mat, ctx.user_cluster_ids,
                                     ctx.prev_assign, self.hyst_db, ctx.users,
                                     self.agg,
                                     capacity_mode=self.cap_mode,
                                     max_cells_per_sat=self.max_cells,
                                     ttt_state=self._ttt_state,
                                     mts_state=self._mts_state, t_idx=t_idx,
                                     ttt_ticks=self._ttt_ticks,
                                     mts_ticks=self._mts_ticks)
        if self.cap_mode == "beam_tagged":      # stash beam_ids for HandoverStage
            ctx.out_beam_ids = {int(ctx.users[k]): int(beams[k])
                                for k in range(len(out)) if beams[k] >= 0}
        return out


class MaxSINRClusterHO(_BaseClusterHO):
    name = "max_sinr_cluster"
    def _score(self, ctx):
        return -10.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))


class MaxRxPowerClusterHO(_BaseClusterHO):
    name = "max_rx_power_cluster"
    def _score(self, ctx):
        return -20.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))


class LoadAwareClusterHO(_BaseClusterHO):
    name = "load_aware_cluster"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha = float(getattr(cfg.handover, "load_penalty_db", 0.5))
    def _score(self, ctx):
        S = ctx.vis_mat.shape[0]
        load = np.zeros(S, dtype=float)
        for s in ctx.prev_assign.values():
            if 0 <= int(s) < S: load[int(s)] += 1.0
        base = -10.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))
        return base - self.alpha * load[:, None]


class UtilityClusterHO(_BaseClusterHO):
    name = "utility_based_cluster"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.alpha = float(getattr(cfg.handover, "utility_alpha", 0.5))
    def _score(self, ctx):
        sinr_db  = -10.0 * np.log10(np.maximum(ctx.dist_mat, 1.0))
        sinr_lin = 10**(sinr_db/10.0)
        rate     = np.log2(1.0 + sinr_lin)
        return (self.alpha * np.log(np.maximum(rate, 1e-6))
                + (1.0 - self.alpha) * np.log(1.0 + sinr_lin))




# Backwards-compatible name maps (old CLI/YAML strings -> new)
_ALIAS = {
    "Persistent_Nearest": "persistent_nearest",
    "Nearest_Sat":        "nearest_sat",
    "persistent":         "persistent_nearest",
    "nearest":            "nearest_sat",
    "farthest":           "farthest_approaching",
}


# Handover strategy registry: name -> class, selected by cfg.handover.mode.
HANDOVER_STRATEGIES = {
        # Centralized family
        "centralized":            CentralizedHO,       # generic load-balanced
        "persistent_nearest":     PersistentNearestHO,  # default in the old code
        "nearest_sat":            NearestSatHO,
        "farthest_approaching":   FarthestApproachingHO,
        # Distributed family
        "distributed_ue":         DistributedUEHO,
        "distributed_satellite":  DistributedSatelliteHO,
        # Hybrid
        "hybrid":                 HybridPerOperatorHO,
        # Benchmark families (used by scenarios/risk_based_rrm_ho/)
        "max_sinr":               MaxSINRHO,
        "gpp_a3":                 GPPA3HO,
        "max_rx_power":           MaxRxPowerHO,
        "load_aware":             LoadAwareHO,
        "utility_based":          UtilityHO,
        # Cluster-aware variants (one sat decision per Earth-cell)
        "max_sinr_cluster":       MaxSINRClusterHO,
        "max_rx_power_cluster":   MaxRxPowerClusterHO,
        "load_aware_cluster":     LoadAwareClusterHO,
        "utility_based_cluster":  UtilityClusterHO,
}


def register_handover_strategy(name):
    def deco(cls):
        HANDOVER_STRATEGIES[str(name).lower()] = cls
        return cls
    return deco


def list_handover_strategies():
    return dict(HANDOVER_STRATEGIES)


def get_handover_strategy(cfg) -> BaseHandoverStrategy:
    raw  = (cfg.handover.mode or "centralized")
    mode = _ALIAS.get(raw, raw).lower()
    cls = HANDOVER_STRATEGIES.get(mode, PersistentNearestHO)
    return cls(cfg)
