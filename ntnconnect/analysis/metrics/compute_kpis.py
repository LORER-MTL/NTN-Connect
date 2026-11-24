# Compute / workload / energy KPIs.

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("n_tasks_generated", granularity="per_tick", units="count",
              default_plot="bar", description="Tasks arrived this tick.")
def n_tasks_generated(state, links, history):
    n = len(state.data.get("new_tasks", []))
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "n_tasks_generated": int(n),
    }])


@register_kpi("n_tasks_running", granularity="per_tick", units="count",
              default_plot="line", description="Tasks running this tick.")
def n_tasks_running(state, links, history):
    n = sum(1 for t in state.data.get("task_events", [])
            if t.get("status") == "running")
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "n_tasks_running": int(n),
    }])


@register_kpi("n_tasks_dropped", granularity="per_tick", units="count",
              default_plot="bar", description="Tasks dropped (no eligible node).")
def n_tasks_dropped(state, links, history):
    n = sum(1 for t in state.data.get("task_events", [])
            if t.get("status") == "dropped")
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "n_tasks_dropped": int(n),
    }])


@register_kpi("node_queue_length", granularity="per_link", units="count",
              default_plot="heatmap", description="Per-node queue length over time.")
def node_queue_length(state, links, history):
    df = state.data.get("node_state_now")
    if df is None or df.empty: return None
    return df[["timestamp","t_idx","node_idx","queue_length"]].copy() \
        if "node_idx" in df.columns else df[["timestamp","t_idx","node_id","queue_length"]].copy()


@register_kpi("task_done_count_per_tick", granularity="per_tick", units="count",
              default_plot="line",
              description="Number of tasks completed this tick.")
def task_done(state, links, history):
    n = sum(1 for t in state.data.get("task_events", [])
            if t.get("status") == "done")
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "task_done_count_per_tick": int(n)}])


@register_kpi("task_deadline_met_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Fraction of completed tasks meeting their deadline.")
def task_deadline_met(state, links, history):
    events = state.data.get("task_events", [])
    done = [t for t in events if t.get("status") == "done"]
    if not done:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "task_deadline_met_rate": 0.0}])
    met = sum(1 for t in done if t.get("deadline_met", 1))
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "task_deadline_met_rate": met / len(done)}])


@register_kpi("mec_admission_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Tasks admitted / generated this tick.")
def mec_admission_rate(state, links, history):
    new = state.data.get("new_tasks", []) or []
    events = state.data.get("task_events", []) or []
    n_admitted = sum(1 for t in events if t.get("status") in ("running", "done"))
    n_gen = len(new) if new else (len(events) or 0)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "mec_admission_rate":
                              (n_admitted / n_gen) if n_gen else 0.0}])


@register_kpi("cache_hit_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Aggregate cache hit rate this tick (avg over nodes).")
def cache_hit_rate(state, links, history):
    cache = state.data.get("cache_state", {}) or {}
    # Cache state today is {node_id: {content_id: size_mb}}; until the cache
    # stage writes hit/miss counters, hit rate is reported as 0.
    n_items = sum(len(v) if isinstance(v, dict) else 0 for v in cache.values())
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "cache_hit_rate": 0.0,
                          "cache_n_items": int(n_items)}])


@register_kpi("task_queue_latency_ms", granularity="per_user", units="ms",
              default_plot="cdf",
              description="Per-task queue latency from arrival to start.")
def task_queue_latency(state, links, history):
    events = state.data.get("task_events", []) or []
    step_ms = 60_000.0 # default tick = 60s; real step injected from cfg if available
    rows = []
    for t in events:
        if t.get("status") not in ("running", "done"): continue
        arrival = int(t.get("arrival_t_idx", state.t_idx))
        start   = int(t.get("start_t_idx",   arrival))
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "user_id": int(t.get("user_id", -1)),
                     "task_queue_latency_ms": max(0, start - arrival) * step_ms})
    return pd.DataFrame(rows) if rows else None
