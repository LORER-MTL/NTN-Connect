# Network-level KPIs (ISL, routing, handover rates).

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("n_isl_active", granularity="per_tick", units="count",
              default_plot="line", description="Number of active ISL edges per tick.")
def n_isl_active(state, links, history):
    active = state.data.get("isl_active", set())
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "n_isl_active": len(active),
    }])


@register_kpi("isl_degree_distribution", granularity="per_tick", units="count",
              default_plot="hist", description="Per-sat ISL degree histogram.")
def isl_degree(state, links, history):
    active = state.data.get("isl_active", set())
    if not active: return None
    deg = {}
    for a, b in active:
        deg[int(a)] = deg.get(int(a), 0) + 1
        deg[int(b)] = deg.get(int(b), 0) + 1
    return pd.DataFrame([
        {"timestamp": state.timestamp, "t_idx": state.t_idx,
         "sat_id": sid, "isl_degree": d}
        for sid, d in deg.items()
    ])


@register_kpi("handover_count_per_tick", granularity="per_tick", units="count",
              default_plot="bar", description="Total HO events at this tick.")
def ho_count(state, links, history):
    cl = state.data.get("ho_classified", [])
    n = sum(1 for r in cl if r.get("ho_type") not in (None, "none", "initial"))
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "handover_count_per_tick": int(n),
    }])


@register_kpi("inter_operator_ho_fraction", granularity="per_tick", units="ratio",
              default_plot="line", description="Fraction of HOs that cross operators.")
def inter_op_ho(state, links, history):
    cl = state.data.get("ho_classified", [])
    real = [r for r in cl if r.get("ho_type") not in (None, "none", "initial")]
    if not real:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "inter_operator_ho_fraction": 0.0}])
    n_inter = sum(1 for r in real if r["ho_type"] == "inter-operator")
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "inter_operator_ho_fraction": float(n_inter) / float(len(real)),
    }])


@register_kpi("intra_operator_ho_fraction", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Fraction of HOs that stay within the same operator.")
def intra_op_ho(state, links, history):
    cl = state.data.get("ho_classified", [])
    real = [r for r in cl if r.get("ho_type") not in (None, "none", "initial")]
    if not real:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "intra_operator_ho_fraction": 0.0}])
    n_intra = sum(1 for r in real if r["ho_type"]
                  in ("intra-operator-intra-type", "intra-operator-inter-type"))
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "intra_operator_ho_fraction": float(n_intra) / float(len(real)),
    }])


@register_kpi("n_outage_users_per_tick", granularity="per_tick", units="count",
              default_plot="line",
              description="Number of users with no serving node at this tick.")
def n_outage(state, links, history):
    assigns = state.data.get("assignments", {})
    n = sum(1 for v in assigns.values() if int(v) < 0)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "n_outage_users_per_tick": int(n)}])


@register_kpi("multi_connectivity_degree", granularity="per_tick", units="count",
              default_plot="hist",
              description="Average # of access links per UE (placeholder = 1 today).")
def multi_conn(state, links, history):
    assigns = state.data.get("assignments", {})
    n_users = len(assigns)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "multi_connectivity_degree": 1.0 if n_users else 0.0}])


@register_kpi("isl_link_utilization_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="Average ISL utilization (active / feasible).")
def isl_util(state, links, history):
    active = state.data.get("isl_active", set())
    feas = state.data.get("isl_feasibility")
    if feas is None or not hasattr(feas, "empty") or feas.empty:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "isl_link_utilization_pct": 0.0}])
    util = 100.0 * len(active) / max(len(feas), 1)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "isl_link_utilization_pct": float(util)}])


# Module-level caches for KPIs that need history. MetricsStage clears
# them at the start of every run via reset_module_state() below.
_isl_prev: dict = {"set": set()}
_ho_history: dict = {}   # user_id -> [(t_idx, prev_serving_node_id)]


def reset_module_state() -> None:
    """Called by MetricsStage on simulation start to prevent cross-run leakage."""
    _isl_prev["set"] = set()
    _ho_history.clear()


@register_kpi("isl_topology_change_rate", granularity="per_tick", units="edges",
              default_plot="line",
              description="Edges added + removed compared to previous tick.")
def isl_change(state, links, history):
    cur = state.data.get("isl_active", set())
    prev = _isl_prev["set"]
    added = len(cur - prev); removed = len(prev - cur)
    _isl_prev["set"] = set(cur)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "isl_topology_change_rate": int(added + removed)}])


@register_kpi("ping_pong_ho_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Fraction of HOs this tick where user returned to the same "
                          "previous node within a short window.")
def ping_pong(state, links, history):
    cl = state.data.get("ho_classified", [])
    n_pp = n_total = 0
    for r in cl:
        if r.get("ho_type") in (None, "none", "initial"): continue
        u = int(r["user_id"])
        cur_serving = int(r.get("serving_node_idx", -1))
        hist_list = _ho_history.setdefault(u, [])
        # Ping-pong if we recently served this same node
        if any(prev_serving == cur_serving for (_, prev_serving) in hist_list[-3:]):
            n_pp += 1
        n_total += 1
        hist_list.append((state.t_idx, int(r.get("prev_node_idx", -1))))
        if len(hist_list) > 10: del hist_list[:-10]
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "ping_pong_ho_rate":
                              (n_pp / n_total) if n_total else 0.0}])


@register_kpi("ho_failure_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Fraction of attempted HOs ending in outage.")
def ho_failure(state, links, history):
    cl = state.data.get("ho_classified", [])
    real = [r for r in cl if r.get("ho_type") not in (None, "none", "initial")]
    if not real:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "ho_failure_rate": 0.0}])
    fails = sum(1 for r in real if r["ho_type"] == "outage-enter")
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "ho_failure_rate": fails / len(real)}])
