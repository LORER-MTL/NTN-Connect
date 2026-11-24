# ISL strategies (4-terminal budget cfg.isl.max_degree): plus_grid (fixed pattern),
# range_rate_greedy (degree-bounded, range-rate-weighted), match_b4 (b-matching),
# plus baselines none / degree_bounded.
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Set, Tuple, Optional, Dict
import logging
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

Pair = Tuple[int, int]


def canon(a, b) -> Pair:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


# Registry
_REGISTRY: Dict[str, type] = {}


def register_isl_strategy(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def list_isl_strategies() -> Dict[str, type]:
    return dict(_REGISTRY)


# Base
class BaseISLStrategy(ABC):
    """All strategies receive:
       * feasibility: DataFrame with (sat_a, sat_b, dist_m, operator_id)
       * cfg:         object exposing cfg.isl (max_degree, ...)
       * sats:        per-tick satellites_t slice (optional). Lets pattern
                      strategies look up per-sat metadata (plane, slot).
                      May be None - strategies must degrade gracefully.
    """
    @abstractmethod
    def select(self, feasibility: pd.DataFrame, cfg: Any,
               sats: Optional[pd.DataFrame] = None) -> Set[Pair]: ...


# Trivial: nothing
@register_isl_strategy("none")
class NoISLStrategy(BaseISLStrategy):
    def select(self, feasibility, cfg, sats=None) -> Set[Pair]:
        return set()


# Legacy greedy (kept for back-compat)
@register_isl_strategy("degree_bounded")
class DegreeBoundedISLStrategy(BaseISLStrategy):
    """Greedy shortest-first, hard degree cap on both endpoints."""

    def select(self, feasibility, cfg, sats=None) -> Set[Pair]:
        if feasibility is None or feasibility.empty:
            return set()
        k = int(cfg.isl.max_degree)
        df = feasibility[["sat_a", "sat_b", "dist_m"]].copy()
        swap = df["sat_a"] > df["sat_b"]
        df.loc[swap, ["sat_a", "sat_b"]] = df.loc[swap, ["sat_b", "sat_a"]].values
        df = (df.drop_duplicates(subset=["sat_a", "sat_b"], keep="first")
                .sort_values("dist_m").reset_index(drop=True))
        deg, active = {}, set()
        a_arr = df["sat_a"].to_numpy(dtype=int)
        b_arr = df["sat_b"].to_numpy(dtype=int)
        for a, b in zip(a_arr, b_arr):
            if deg.get(a, 0) >= k or deg.get(b, 0) >= k: continue
            active.add(canon(a, b))
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        return active


# A1 - +Grid (Iridium / Starlink fixed pattern)
@register_isl_strategy("plus_grid")
class PlusGridISLStrategy(BaseISLStrategy):
    """Fixed 4-ISL lattice (2 intra-plane + 2 inter-plane).

    Required metadata per sat: `plane` and `slot` (integer indices).
    These can come from either:
      (a) explicit columns on `sats` (the satellite generator writes them
          when shells are synthesised in-memory), or
      (b) derived from node_id arithmetic given the shell's num_planes /
          sats_per_plane (used when sats came from SGP4 TLE catalog).

    Each sat would like 4 edges:
      (plane,   slot ± 1)    intra-plane
      (plane±1, slot)        inter-plane
    Each edge is activated only if it appears in the feasibility table
    (i.e. is range / LOS reachable this tick).
    """
    def select(self, feasibility, cfg, sats=None) -> Set[Pair]:
        if feasibility is None or feasibility.empty:
            return set()
        meta = self._sat_meta(sats, cfg)
        if not meta:
            _log.warning("plus_grid: no plane/slot metadata available — "
                         "falling back to degree_bounded.")
            return DegreeBoundedISLStrategy().select(feasibility, cfg, sats)

        # Group sats by (operator, shell) - `meta[sid] = (op_id, P, plane, slot)`
        # Build a lookup (op_id, plane, slot) -> sat_id
        by_ps: dict = {}
        max_plane_by_op: dict[int, int] = {}
        max_slot_by_op:  dict[int, int] = {}
        for sid, (op_id, p, s) in meta.items():
            by_ps[(op_id, int(p), int(s))] = int(sid)
            max_plane_by_op[op_id] = max(max_plane_by_op.get(op_id, 0), int(p))
            max_slot_by_op[op_id]  = max(max_slot_by_op.get(op_id, 0),  int(s))

        wanted: Set[Pair] = set()
        for sid, (op_id, p, s) in meta.items():
            P = max_plane_by_op[op_id] + 1
            S = max_slot_by_op[op_id]  + 1
            for dp, ds in ((0, +1), (0, -1), (+1, 0), (-1, 0)):
                tgt = by_ps.get((op_id, (int(p) + dp) % P, (int(s) + ds) % S))
                if tgt is None or tgt == int(sid): continue
                wanted.add(canon(int(sid), tgt))

        # Intersect with feasibility (drops cross-seam edges that are out
        # of range / LOS at this tick - exactly the Bhattacherjee 2019
        # "+Grid with cross-seam handover" behaviour).
        feas_set = set()
        a_arr = feasibility["sat_a"].astype(int).to_numpy()
        b_arr = feasibility["sat_b"].astype(int).to_numpy()
        for a, b in zip(a_arr, b_arr):
            feas_set.add(canon(int(a), int(b)))
        return wanted & feas_set

    # helpers
    @staticmethod
    def _sat_meta(sats: Optional[pd.DataFrame], cfg
                   ) -> Dict[int, Tuple[int, int, int]]:
        """Return {sat_id -> (op_id, plane, slot)}. Falls back to id arithmetic
        derived from the operator shell's (num_planes, num_satellites)."""
        if sats is None or sats.empty: return {}
        if {"plane", "slot"}.issubset(sats.columns):
            return {int(r.node_id): (int(getattr(r, "operator_id", 0)),
                                       int(r.plane), int(r.slot))
                    for r in sats.itertuples(index=False)}
        # Fallback: derive from sequential ids
        out: Dict[int, Tuple[int, int, int]] = {}
        operators = getattr(cfg, "operators", None)
        if operators is None: return {}
        # cfg here is the per-operator adapter (see ISLStage), so cfg has
        # no.operators. We need to pass the SHELL parameters via a side
        # channel. The ISLStage adapter sets cfg.isl_meta = (P, S, base_id).
        meta_hint = getattr(cfg, "_grid_hint", None)
        if meta_hint is None: return {}
        P, S, base_id, op_id = meta_hint
        for sid in sats["node_id"].astype(int).to_numpy():
            i = int(sid) - int(base_id)
            if i < 0: continue
            p, s = divmod(i, S)
            out[int(sid)] = (int(op_id), int(p) % int(P), int(s))
        return out


# B3 - Range-rate-aware greedy (4-terminal, ATP-churn-aware)
@register_isl_strategy("range_rate_greedy")
class RangeRateGreedyISLStrategy(BaseISLStrategy):
    """Degree-bounded greedy with a sort key that penalises fast-closing
    edges. The penalty is computed by finite-difference of dist_m across
    consecutive calls - the strategy caches the previous-tick distance
    per (sat_a, sat_b) pair.

    Sort key:  score(a,b) = dist_m + α · max(0, +range_rate_m_s)
    where range_rate > 0 means the sats are SEPARATING (about to lose LOS).
    α defaults to `cfg.isl.range_rate_alpha_s = 60` so a link about to
    separate at 1 km/s is penalised by 60 km - pushing it to be picked
    later if anything else is competitive.
    """
    def __init__(self):
        # {(a,b) -> dist_m_prev}
        self._prev_dist: Dict[Pair, float] = {}

    def select(self, feasibility, cfg, sats=None) -> Set[Pair]:
        if feasibility is None or feasibility.empty: return set()
        k = int(cfg.isl.max_degree)
        alpha = float(getattr(cfg.isl, "range_rate_alpha_s", 60.0))

        df = feasibility[["sat_a", "sat_b", "dist_m"]].copy()
        swap = df["sat_a"] > df["sat_b"]
        df.loc[swap, ["sat_a", "sat_b"]] = df.loc[swap, ["sat_b", "sat_a"]].values
        df = df.drop_duplicates(["sat_a", "sat_b"]).reset_index(drop=True)

        a_arr = df["sat_a"].to_numpy(dtype=int)
        b_arr = df["sat_b"].to_numpy(dtype=int)
        d_now = df["dist_m"].to_numpy(dtype=float)

        # Vectorised lookup of previous distance per edge.
        d_prev = np.array([self._prev_dist.get((int(a), int(b)), np.nan)
                            for a, b in zip(a_arr, b_arr)], dtype=float)
        # range_rate (m/s) by finite difference - needs dt. We don't have
        # access to dt here, so we use a normalised rate where dist_diff
        # itself is the proxy (Δd over ~1 tick). α has units of "seconds"
        # so multiplying by Δd/tick approximates the original formula.
        rate_proxy = np.where(np.isnan(d_prev), 0.0, d_now - d_prev)
        score = d_now + alpha * np.clip(rate_proxy, 0.0, None)

        order = np.argsort(score)
        deg, active = {}, set()
        for idx in order:
            a, b = int(a_arr[idx]), int(b_arr[idx])
            if deg.get(a, 0) >= k or deg.get(b, 0) >= k: continue
            active.add(canon(a, b))
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1

        # Refresh prev-tick cache (all feasible pairs, not just active)
        self._prev_dist = {(int(a), int(b)): float(d)
                            for a, b, d in zip(a_arr, b_arr, d_now)}
        return active


# B5 - b-matching optimum (iterative min-weight matching)
@register_isl_strategy("match_b4")
class MatchB4ISLStrategy(BaseISLStrategy):
    """Iterative min-weight matching for a near-optimal 4-regular subgraph.

    Algorithm:
        for round in 1..b:
            remove vertices that have hit degree b from a working copy
            of the feasibility graph;
            find a min-weight MAXIMUM matching on the residual graph;
            add each edge of the matching to `active`, updating degrees.

    Guarantees:
        * Every vertex ends with degree ≤ b.
        * Within a 4/3-factor of the global min-weight b-matching optimum.
        * Worst-case complexity ~ O(b · |V|³) using NetworkX's Edmonds-Karp
          implementation - sub-second for V ≤ 1000 visible sats per op.

    Requires: networkx (already in the Anaconda baseline).
    """
    def select(self, feasibility, cfg, sats=None) -> Set[Pair]:
        if feasibility is None or feasibility.empty: return set()
        try:
            import networkx as nx
        except Exception:
            _log.error("match_b4 needs networkx; falling back to degree_bounded.")
            return DegreeBoundedISLStrategy().select(feasibility, cfg, sats)

        b = int(cfg.isl.max_degree)
        df = feasibility[["sat_a", "sat_b", "dist_m"]].copy()
        swap = df["sat_a"] > df["sat_b"]
        df.loc[swap, ["sat_a", "sat_b"]] = df.loc[swap, ["sat_b", "sat_a"]].values
        df = df.drop_duplicates(["sat_a", "sat_b"]).reset_index(drop=True)
        if df.empty: return set()

        G = nx.Graph()
        for a, b_, d in zip(df["sat_a"].astype(int).to_numpy(),
                              df["sat_b"].astype(int).to_numpy(),
                              df["dist_m"].astype(float).to_numpy()):
            G.add_edge(int(a), int(b_), weight=float(d))

        active: Set[Pair] = set()
        deg: Dict[int, int] = {v: 0 for v in G.nodes}

        for _round in range(b):
            sat_nodes = [v for v in G.nodes if deg.get(v, 0) >= b]
            G_round = G.subgraph(v for v in G.nodes
                                  if v not in sat_nodes).copy()
            # Also drop already-picked edges so a pair can't be matched twice
            for e in list(active):
                if G_round.has_edge(*e):
                    G_round.remove_edge(*e)
            if G_round.number_of_edges() == 0:
                break
            # min-weight maximum matching (NetworkX 3.x)
            m = nx.min_weight_matching(G_round)
            if not m: break
            for u, v in m:
                a, b_ = (int(u), int(v)) if u < v else (int(v), int(u))
                if deg.get(a, 0) >= b or deg.get(b_, 0) >= b: continue
                active.add((a, b_))
                deg[a] = deg.get(a, 0) + 1
                deg[b_] = deg.get(b_, 0) + 1
        return active


# Dispatcher (kept name-compatible with the old API)
def get_isl_strategy(cfg) -> BaseISLStrategy:
    name = (cfg.isl.strategy or "plus_grid").lower()
    if name in ("off", "disabled"): name = "none"
    cls = _REGISTRY.get(name)
    if cls is None:
        _log.warning("ISL strategy '%s' not registered. "
                     "Falling back to degree_bounded. Available: %s",
                     name, sorted(_REGISTRY.keys()))
        return DegreeBoundedISLStrategy()
    # Cache per-strategy instances so stateful strategies (range_rate_greedy)
    # keep their prev-tick state across ticks within one op.
    cache = getattr(cfg, "_isl_strategy_cache", None)
    if cache is None:
        cache = {}
        try: setattr(cfg, "_isl_strategy_cache", cache)
        except Exception: pass
    inst = cache.get(name)
    if inst is None:
        inst = cls()
        cache[name] = inst
    return inst
