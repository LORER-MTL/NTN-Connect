# Whole-system KPIs (one row per simulation run).

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("system_sum_rate_bps", granularity="per_tick", units="bps",
              default_plot="line", description="Sum of all access-link Shannon rates.")
def system_sum_rate(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "system_sum_rate_bps": float(links["link_rate_bps"].sum()),
    }])


@register_kpi("system_mean_sinr_db", granularity="per_tick", units="dB",
              default_plot="line", description="Mean SINR across all access links.")
def system_mean_sinr(state, links, history):
    if links is None or links.empty or "sinr_db" not in links.columns: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "system_mean_sinr_db": float(links["sinr_db"].mean()),
    }])


@register_kpi("system_outage_fraction", granularity="per_tick", units="ratio",
              default_plot="line", description="Fraction of users without a serving node.")
def system_outage(state, links, history):
    assigns = state.data.get("assignments", {})
    if not assigns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_outage_fraction": 0.0}])
    n = len(assigns); n_out = sum(1 for v in assigns.values() if int(v) < 0)
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "system_outage_fraction": float(n_out) / float(n),
    }])


@register_kpi("system_throughput_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="Sum of serving-link rates across the whole system.")
def system_throughput(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_throughput_bps": 0.0}])
    assigns = state.data.get("assignments", {}) or {}
    serving = set((int(u), int(s)) for u, s in assigns.items() if int(s) >= 0)
    if not serving:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_throughput_bps": 0.0}])
    m = links.apply(lambda r: (int(r["user_id"]), int(r["node_id"])) in serving, axis=1)
    sub = links[m]
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_throughput_bps": float(sub["link_rate_bps"].sum())}])


@register_kpi("system_jain_fairness", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Jain's fairness index over per-user serving rates.")
def system_jain(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    rates = _serving_rates(state, links)   # vectorised, per-tick cached
    if rates.size == 0:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_jain_fairness": 0.0}])
    s1 = float(rates.sum()); s2 = float((rates * rates).sum())
    j = (s1*s1) / (rates.size * s2) if s2 > 0 else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_jain_fairness": float(j)}])


@register_kpi("system_blocking_rate", granularity="per_tick", units="ratio",
              default_plot="line",
              description="Fraction of users blocked by admission (assigned == -2).")
def system_blocking(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    if not assigns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_blocking_rate": 0.0}])
    n_blocked = sum(1 for v in assigns.values() if int(v) == -2)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_blocking_rate": n_blocked / len(assigns)}])


@register_kpi("system_active_users_per_tick", granularity="per_tick", units="count",
              default_plot="line",
              description="Number of users with a non-outage serving node.")
def system_active(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    n_active = sum(1 for v in assigns.values() if int(v) >= 0)
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_active_users_per_tick": int(n_active)}])


# Rate statistics (across served users per tick)
def _serving_rates(state, links):
    """Return numpy array of per-user serving-link rates for this tick.

    Vectorised: delegates to the per-tick memoised serving view (one merge for
    the whole tick) instead of a per-user O(L) boolean scan. The six
    system_*_rate KPIs all call this; they now share the same cached view, so
    the serving subset is built once per tick, not six+ times."""
    from ntnconnect.analysis.metrics._serving import serving_rates as _sr
    return _sr(state, links)


@register_kpi("system_min_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="Minimum per-user rate this tick (worst-served UE).")
def sys_min_rate(state, links, history):
    r = _serving_rates(state, links)
    v = float(r.min()) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_min_rate_bps": v}])


@register_kpi("system_max_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="Maximum per-user rate this tick (best-served UE).")
def sys_max_rate(state, links, history):
    r = _serving_rates(state, links)
    v = float(r.max()) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_max_rate_bps": v}])


@register_kpi("system_mean_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="Mean per-user rate this tick.")
def sys_mean_rate(state, links, history):
    r = _serving_rates(state, links)
    v = float(r.mean()) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_mean_rate_bps": v}])


@register_kpi("system_median_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="Median per-user rate this tick.")
def sys_median_rate(state, links, history):
    import numpy as np
    r = _serving_rates(state, links)
    v = float(np.median(r)) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_median_rate_bps": v}])


@register_kpi("system_p5_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="5th-percentile per-user rate (cell-edge user).")
def sys_p5_rate(state, links, history):
    import numpy as np
    r = _serving_rates(state, links)
    v = float(np.percentile(r, 5)) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_p5_rate_bps": v}])


@register_kpi("system_p95_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="95th-percentile per-user rate (peak user).")
def sys_p95_rate(state, links, history):
    import numpy as np
    r = _serving_rates(state, links)
    v = float(np.percentile(r, 95)) if r.size else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_p95_rate_bps": v}])
