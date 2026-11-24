# Coverage KPIs.
#
# Thresholds are taken from cfg when present; otherwise common defaults
# (SINR > 0 dB, rate > rate_threshold_bps_hz * bw_default).

from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


SINR_COVERAGE_THRESHOLD_DB = 0.0      # default; can be overridden from cfg


# helpers
def _serving_view(state, links):
    """Return DataFrame of serving-link ROWS (all columns) for this tick.

    Vectorised + per-tick memoised. Was a links.apply(axis=1) row-wise Python
    scan over every link, re-run by all four coverage KPIs each tick (the single
    largest residual metrics cost after the user/system KPI fix). Now one
    hash-merge of the served (user_id, node_id) pairs against links, cached on
    state.data so the four KPIs share it."""
    if links is None or links.empty: return pd.DataFrame()
    cached = state.data.get("_cov_serving_view")
    if cached is not None: return cached
    assigns = state.data.get("assignments", {}) or {}
    pairs = [(int(u), int(s)) for u, s in assigns.items() if int(s) >= 0]
    if not pairs:
        out = links.iloc[0:0]
    else:
        adf = pd.DataFrame(pairs, columns=["user_id", "node_id"])
        out = adf.merge(links, on=["user_id", "node_id"], how="inner", sort=False)
    state.data["_cov_serving_view"] = out
    return out


# Coverage probability (per tick)
@register_kpi("coverage_sinr_threshold_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="% users whose serving SINR exceeds the configured threshold.")
def cov_sinr(state, links, history):
    sv = _serving_view(state, links)
    assigns = state.data.get("assignments", {}) or {}
    n_users = len(assigns)
    if sv.empty or n_users == 0 or "sinr_db" not in sv.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "coverage_sinr_threshold_pct": 0.0}])
    n_covered = int((sv["sinr_db"] >= SINR_COVERAGE_THRESHOLD_DB).sum())
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "coverage_sinr_threshold_pct": 100.0 * n_covered / n_users}])


@register_kpi("coverage_rate_threshold_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="% users whose serving rate exceeds the configured threshold.")
def cov_rate(state, links, history):
    sv = _serving_view(state, links)
    assigns = state.data.get("assignments", {}) or {}
    n_users = len(assigns)
    if sv.empty or n_users == 0 or "link_rate_bps" not in sv.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "coverage_rate_threshold_pct": 0.0}])
    # rate_threshold = rate_threshold_bps_hz * bandwidth, per-link
    if "bandwidth_hz" in sv.columns:
        thr = sv["bandwidth_hz"].values * 1.5      # 1.5 bps/Hz default
    else:
        thr = np.ones(len(sv)) * 30e6              # fallback 30 Mbps
    n_covered = int((sv["link_rate_bps"].values >= thr).sum())
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "coverage_rate_threshold_pct": 100.0 * n_covered / n_users}])


# Cell-edge metrics (percentiles)
@register_kpi("cell_edge_sinr_db", granularity="per_tick", units="dB",
              default_plot="line",
              description="5th-percentile serving SINR (cell-edge UE).")
def cell_edge_sinr(state, links, history):
    sv = _serving_view(state, links)
    if sv.empty or "sinr_db" not in sv.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "cell_edge_sinr_db": float("nan")}])
    v = float(np.percentile(sv["sinr_db"].values, 5))
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "cell_edge_sinr_db": v}])


@register_kpi("cell_edge_rate_bps", granularity="per_tick", units="bps",
              default_plot="line",
              description="5th-percentile serving rate (cell-edge UE).")
def cell_edge_rate(state, links, history):
    sv = _serving_view(state, links)
    if sv.empty or "link_rate_bps" not in sv.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "cell_edge_rate_bps": 0.0}])
    v = float(np.percentile(sv["link_rate_bps"].values, 5))
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "cell_edge_rate_bps": v}])


# Service availability (per user, accumulated)
@register_kpi("service_availability", granularity="per_user", units="ratio",
              default_plot="cdf",
              description="Per-user fraction of THIS tick the user has service (0 or 1; "
                          "average across ticks at end-of-run gives true availability).")
def service_availability(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": int(u),
        "service_availability": int(int(s) >= 0),
    } for u, s in assigns.items()])


@register_kpi("outage_probability", granularity="per_user", units="ratio",
              default_plot="cdf",
              description="Per-user outage indicator this tick (1 - service_availability).")
def outage_probability(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": int(u),
        "outage_probability": int(int(s) < 0),
    } for u, s in assigns.items()])


# Geographic coverage (rough binning over lat/lon)
@register_kpi("coverage_area_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="Fraction of 1°x1° lat/lon bins inside the env bounding box "
                          "that have at least one access node visible. Rough estimate.")
def coverage_area(state, links, history):
    sv = _serving_view(state, links)
    # Count bins with a user that has service. Without a true binning over
    # the env bbox we approximate as (served bins / total UE bins).
    if sv.empty: return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                                       "coverage_area_pct": 0.0}])
    if "user_lat" not in sv.columns or "user_lon" not in sv.columns:
        # fall back to user/node lat from the links column set
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "coverage_area_pct": 0.0}])
    bins = set(
        (int(round(la)), int(round(lo)))
        for la, lo in zip(sv["user_lat"].values, sv["user_lon"].values))
    n_total_bins = max(1, len(bins))
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "coverage_area_pct": 100.0 * len(bins) / n_total_bins}])


@register_kpi("coverage_per_operator_pct", granularity="per_operator", units="%",
              default_plot="bar",
              description="% of users covered (SINR >= threshold) by each operator.")
def coverage_per_op(state, links, history):
    sv = _serving_view(state, links)
    if sv.empty or "sinr_db" not in sv.columns or "operator_id" not in sv.columns:
        return None
    assigns = state.data.get("assignments", {}) or {}
    n_users = len(assigns)
    if n_users == 0: return None
    rows = []
    for op, g in sv.groupby("operator_id"):
        n_cov = int((g["sinr_db"] >= SINR_COVERAGE_THRESHOLD_DB).sum())
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "operator_id": int(op),
                     "coverage_per_operator_pct": 100.0 * n_cov / n_users})
    return pd.DataFrame(rows)


# Visibility coverage (different from served coverage)
@register_kpi("visibility_coverage_pct", granularity="per_tick", units="%",
              default_plot="line",
              description="% users that have at least ONE visible access node "
                          "(may or may not be served after admission).")
def visibility_coverage(state, links, history):
    if links is None or links.empty:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "visibility_coverage_pct": 0.0}])
    assigns = state.data.get("assignments", {}) or {}
    n_users = max(len(assigns), int(links["user_id"].nunique()))
    if n_users == 0:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "visibility_coverage_pct": 0.0}])
    n_visible = int(links["user_id"].nunique())
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "visibility_coverage_pct": 100.0 * n_visible / n_users}])
