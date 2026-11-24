# RRM-side KPIs. Reads `rb_alloc_map` populated by rrm_allocate_rb.

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("mcs_distribution", granularity="per_link", units="index",
              default_plot="hist",
              description="Per-link MCS index assigned by rrm_mcs (3GPP TS 38.214).")
def mcs_distribution(state, links, history):
    if links is None or links.empty or "mcs_index" not in links.columns: return None
    df = links[["user_id","node_id","mcs_index"]].copy()
    df["timestamp"] = state.timestamp
    df["t_idx"]     = state.t_idx
    return df


@register_kpi("rb_utilization_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="Average RB utilization across serving nodes (allocated/total).")
def rb_util(state, links, history):
    alloc = state.data.get("rb_alloc_map", {}) or {}
    if not alloc:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "rb_utilization_pct": 0.0}])
    by_node: dict = {}
    for (nid, uid), e in alloc.items():
        slot = by_node.setdefault(int(nid),
                                  {"used": 0, "total": int(e.get("n_rbs_total", 0))})
        slot["used"] += int(e.get("n_rbs", 0))
    if not by_node:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "rb_utilization_pct": 0.0}])
    avg = sum(100.0 * s["used"] / max(s["total"], 1) for s in by_node.values()) \
          / len(by_node)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "rb_utilization_pct": float(avg)}])


@register_kpi("bandwidth_utilization_pct", granularity="per_operator", units="%",
              default_plot="line",
              description="Per-operator allocated bandwidth as fraction of total.")
def bw_util(state, links, history):
    alloc = state.data.get("rb_alloc_map", {}) or {}
    if links is None or links.empty or not alloc: return None
    nid_to_op = (links[["node_id","operator_id","bandwidth_hz"]]
                 .drop_duplicates(subset=["node_id"])
                 .set_index("node_id").to_dict("index"))
    per_op: dict = {}
    for (nid, uid), e in alloc.items():
        meta = nid_to_op.get(int(nid))
        if meta is None: continue
        op = int(meta["operator_id"]); bw_total = float(meta["bandwidth_hz"])
        slot = per_op.setdefault(op, {"used": 0.0, "total": 0.0, "nodes": set()})
        slot["used"] += float(e.get("bw_alloc_hz", 0.0))
        if nid not in slot["nodes"]:
            slot["total"] += bw_total
            slot["nodes"].add(nid)
    rows = []
    for op, s in per_op.items():
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "operator_id": int(op),
                     "bandwidth_utilization_pct":
                         100.0 * s["used"] / s["total"] if s["total"] > 0 else 0.0})
    return pd.DataFrame(rows)
