# Routing KPIs. Reads state.data["routes"] populated by net_route_e2e.

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("route_e2e_latency_s", granularity="per_user", units="s",
              default_plot="cdf",
              description="End-to-end propagation delay along chosen route.")
def route_latency(state, links, history):
    routes = state.data.get("routes", []) or []
    if not routes: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": int(r["user_id"]),
        "route_e2e_latency_s": float(r.get("total_delay_s", 0.0)),
    } for r in routes])


@register_kpi("route_hops", granularity="per_user", units="count",
              default_plot="hist",
              description="Number of hops on the user's route.")
def route_hops(state, links, history):
    routes = state.data.get("routes", []) or []
    if not routes: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": int(r["user_id"]),
        "route_hops": int(r.get("total_hops", 0)),
    } for r in routes])


@register_kpi("route_bottleneck_bps", granularity="per_user", units="bps",
              default_plot="cdf",
              description="Capacity of the limiting edge on the route.")
def route_bottleneck(state, links, history):
    routes = state.data.get("routes", []) or []
    if not routes: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": int(r["user_id"]),
        "route_bottleneck_bps": float(r.get("bottleneck_bps", 0.0)),
    } for r in routes])


@register_kpi("route_inter_op_hops", granularity="per_user", units="count",
              default_plot="hist",
              description="Number of operator transitions on the route.")
def route_inter_op(state, links, history):
    routes = state.data.get("routes", []) or []
    if not routes: return None
    rows = []
    for r in routes:
        crosses = r.get("crosses_operators") or []
        if isinstance(crosses, str):       # JSON-encoded fallback
            crosses = crosses.split(",") if crosses else []
        n_transitions = max(0, len(crosses) - 1) if crosses else 0
        rows.append({
            "timestamp": state.timestamp, "t_idx": state.t_idx,
            "user_id": int(r["user_id"]),
            "route_inter_op_hops": int(n_transitions),
        })
    return pd.DataFrame(rows)


@register_kpi("n_routes_failed", granularity="per_tick", units="count",
              default_plot="line",
              description="Users requesting a route that could not be computed.")
def n_routes_failed(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    routes = state.data.get("routes", []) or []
    expected = sum(1 for s in assigns.values() if int(s) >= 0)
    got = len(routes)
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "n_routes_failed": max(0, expected - got),
    }])
