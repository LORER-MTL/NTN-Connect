# Shared per-tick "serving-link view" - the single hot data structure behind
# the per-user / per-tick KPIs.
#
# WHY: before this, ~10 KPIs (serving_sinr, serving_elev, user_throughput,
# system_jain and the six system_*_rate stats via _serving_rates) each rebuilt
# the serving-link subset INDEPENDENTLY, every tick, with a per-user Python loop:
#
#  for u, nid in assigns.items(): # O(U)
#  mask = (links.user_id == u) & (links.node_id == nid) # O(L) scan + 2 Series
#  val = links.loc[mask, col].iloc[0]
#
# i.e. O(K · U · L) per tick with K duplicate passes, dominated by pandas Series
# construction / boolean-compare overhead. Profiling a 900-tick cell showed the
# metrics stage at 65% of total runtime (90 s / 162 s).
#
# NOW: compute the serving view ONCE per tick via a single hash-merge (O(U + L))
# and memoise it on state.data (which the orchestrator resets every tick, so the
# cache is automatically per-tick). Every KPI reads the cached arrays.
from __future__ import annotations
import numpy as np
import pandas as pd

_CACHE_KEY = "_serving_view_cache"
# Columns the KPIs pull off the serving link. Extend here if a new serving-link
# KPI needs another column; the merge already carries the whole row cheaply.
_SERVING_COLS = ("sinr_db", "elev_deg", "link_rate_bps")


def serving_view(state, links) -> dict:
    """Per-tick, memoised. Returns a dict:
        user_ids   : (U,) int - every assigned user, in assignment order
        node_ids   : (U,) int - its serving node_id (<0 = outage)
        served     : (U,) bool - node_ids >= 0
        cols       : {name: (U,) float} for each present serving column;
                     outage users and unmatched links are NaN.
    Built with one merge instead of U boolean scans.
    """
    cached = state.data.get(_CACHE_KEY)
    if cached is not None:
        return cached

    assigns = state.data.get("assignments", {}) or {}
    if assigns:
        user_ids = np.fromiter((int(u) for u in assigns.keys()),
                               dtype=np.int64, count=len(assigns))
        node_ids = np.fromiter((int(s) for s in assigns.values()),
                               dtype=np.int64, count=len(assigns))
    else:
        user_ids = np.empty(0, dtype=np.int64)
        node_ids = np.empty(0, dtype=np.int64)
    served = node_ids >= 0

    present = ([c for c in _SERVING_COLS if c in links.columns]
               if isinstance(links, pd.DataFrame) and not links.empty else [])
    cols: dict[str, np.ndarray] = {c: np.full(user_ids.shape, np.nan) for c in present}

    if present and served.any():
        # Single hash-join: served (user_id, node_id) -> serving-link columns.
        adf = pd.DataFrame({"user_id": user_ids[served],
                            "node_id": node_ids[served]})
        merged = adf.merge(links[["user_id", "node_id", *present]],
                           on=["user_id", "node_id"], how="left",
                           sort=False)
        # merge(how="left", sort=False) preserves adf (served) row order, so the
        # merged columns align positionally with served-user slots.
        sidx = np.flatnonzero(served)
        # A served user can in principle match >1 link row (multi-conn); keep the
        # first per (user,node) to mirror the old `.iloc[0]` semantics.
        if len(merged) != len(sidx):
            merged = merged.drop_duplicates(subset=["user_id", "node_id"],
                                            keep="first")
        for c in present:
            cols[c][sidx] = merged[c].to_numpy()

    view = {"user_ids": user_ids, "node_ids": node_ids,
            "served": served, "cols": cols}
    state.data[_CACHE_KEY] = view
    return view


def serving_rates(state, links) -> np.ndarray:
    """Per-user serving-link rate array (bps); outage / unmatched -> 0.0.
    Same length & order as assignments - drop-in for the old _serving_rates."""
    v = serving_view(state, links)
    n = v["user_ids"].shape[0]
    if n == 0 or "link_rate_bps" not in v["cols"]:
        return np.zeros(n, dtype=float)
    return np.nan_to_num(v["cols"]["link_rate_bps"], nan=0.0)
