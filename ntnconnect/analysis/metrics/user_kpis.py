# Per-user KPIs (one row per user per tick).

from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi
from ntnconnect.analysis.metrics._serving import serving_view


def _serving_frame(state, links, src_col, out_col, fill_outage=None):
    """Vectorised per-user serving-link KPI frame from the per-tick cached view.

    fill_outage is None  -> emit only served users with a matching link
                            (drops outage + unmatched; serving_sinr/elev).
    fill_outage is a num -> emit a row for every assigned user, with outage /
                            unmatched users set to that value (user_throughput)."""
    v = serving_view(state, links)
    uids = v["user_ids"]
    if uids.size == 0:
        return pd.DataFrame()
    vals = v["cols"].get(src_col)
    if vals is None:
        vals = np.full(uids.shape, np.nan)
    if fill_outage is None:
        keep = np.isfinite(vals)              # served + matched only
    else:
        vals = np.where(np.isfinite(vals), vals, float(fill_outage))
        keep = np.ones(uids.shape, dtype=bool)
    if not keep.any():
        return pd.DataFrame()
    return pd.DataFrame({
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "user_id": uids[keep].astype(int),
        out_col: vals[keep].astype(float),
    })


@register_kpi("serving_sinr_db", granularity="per_user", units="dB",
              default_plot="cdf", description="SINR of the user's serving link.")
def serving_sinr(state, links, history):
    if links is None or links.empty: return None
    return _serving_frame(state, links, "sinr_db", "serving_sinr_db")


@register_kpi("serving_elev_deg", granularity="per_user", units="deg",
              default_plot="hist", description="Elevation of serving link.")
def serving_elev(state, links, history):
    if links is None or links.empty: return None
    return _serving_frame(state, links, "elev_deg", "serving_elev_deg")


@register_kpi("num_visible_nodes", granularity="per_user", units="count",
              default_plot="line", description="Number of visible access nodes per user.")
def n_visible(state, links, history):
    if links is None or links.empty: return None
    g = links.groupby("user_id").size().reset_index(name="num_visible_nodes")
    g["timestamp"] = state.timestamp; g["t_idx"] = state.t_idx
    return g


@register_kpi("is_in_outage", granularity="per_user", units="bool",
              default_plot="bar", description="Whether user has no serving node.")
def in_outage(state, links, history):
    assigns = state.data.get("assignments", {})
    rows = [{"timestamp": state.timestamp, "t_idx": state.t_idx,
             "user_id": int(u), "is_in_outage": int(int(nid) < 0)}
            for u, nid in assigns.items()]
    return pd.DataFrame(rows)


@register_kpi("user_throughput_bps", granularity="per_user", units="bps",
              default_plot="cdf",
              description="Per-user achieved link rate (sum across multi-conn).")
def user_throughput(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    # One row per assigned user; outage / unmatched -> 0.0. (WISEE is 1
    # connection per user, so the old multi-conn.sum() == the serving value.)
    return _serving_frame(state, links, "link_rate_bps", "user_throughput_bps",
                          fill_outage=0.0)


@register_kpi("user_handover_event", granularity="per_user", units="bool",
              default_plot="bar",
              description="Per-user HO event flag at this tick (1 if HO != none/initial).")
def user_handover_event(state, links, history):
    cl = state.data.get("ho_classified", [])
    rows = []
    for r in cl:
        ho = r.get("ho_type")
        had = int(ho not in (None, "none", "initial"))
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "user_id": int(r["user_id"]), "user_handover_event": had})
    return pd.DataFrame(rows)


@register_kpi("user_serving_operator_id", granularity="per_user", units="",
              default_plot="bar",
              description="Per-user currently-serving operator id.")
def user_serving_op(state, links, history):
    cl = state.data.get("ho_classified", [])
    rows = [{"timestamp": state.timestamp, "t_idx": state.t_idx,
             "user_id": int(r["user_id"]),
             "user_serving_operator_id": int(r.get("serving_operator_id", -1))}
            for r in cl]
    return pd.DataFrame(rows)


@register_kpi("user_data_volume_mb_tick", granularity="per_user", units="MB",
              default_plot="cdf",
              description="Per-user data volume delivered this tick "
                          "(link_rate_bps * step_size_s / 8).")
def user_data_volume(state, links, history):
    df = user_throughput(state, links, history)
    if df is None or df.empty: return None
    step_s = float(state.metadata.get("step_size_s")) \
        if state.metadata.get("step_size_s") else 60.0
    df = df.copy()
    df["user_data_volume_mb_tick"] = df["user_throughput_bps"] * step_s / 8.0 / 1e6
    return df[["timestamp","t_idx","user_id","user_data_volume_mb_tick"]]
