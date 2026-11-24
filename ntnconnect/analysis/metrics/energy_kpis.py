# Energy KPIs. Reads node_state_now produced by state_energy stage.

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("node_energy_j_per_tick", granularity="per_link", units="J",
              default_plot="heatmap",
              description="Per-node energy consumed this tick (P_total * step_size_s).")
def node_energy(state, links, history):
    df = state.data.get("node_state_now")
    if df is None or df.empty: return None
    if "energy_j_tick" not in df.columns: return None
    out = df[["t_idx"]].copy() if "t_idx" in df.columns else pd.DataFrame()
    out["timestamp"] = state.timestamp
    out["t_idx"]     = state.t_idx
    out["node_id"]   = df["node_id"] if "node_id" in df.columns else df.get("node_idx", -1)
    out["node_energy_j_per_tick"] = df["energy_j_tick"].astype(float).values
    return out[["timestamp","t_idx","node_id","node_energy_j_per_tick"]]


@register_kpi("system_energy_j_total", granularity="per_tick", units="J",
              default_plot="line",
              description="System-wide energy consumed this tick.")
def system_energy(state, links, history):
    df = state.data.get("node_state_now")
    if df is None or df.empty or "energy_j_tick" not in df.columns:
        return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                              "system_energy_j_total": 0.0}])
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "system_energy_j_total": float(df["energy_j_tick"].sum())}])


@register_kpi("energy_efficiency_bps_per_joule", granularity="per_tick",
              units="bps/J", default_plot="line",
              description="System throughput / system energy = bits per joule.")
def energy_efficiency(state, links, history):
    df = state.data.get("node_state_now")
    rate_total = 0.0
    if isinstance(links, type(df)) and links is not None and not links.empty \
       and "link_rate_bps" in links.columns:
        rate_total = float(links["link_rate_bps"].sum())
    energy = 0.0
    if df is not None and not df.empty and "energy_j_tick" in df.columns:
        energy = float(df["energy_j_tick"].sum())
    eff = (rate_total / energy) if energy > 0 else 0.0
    return pd.DataFrame([{"timestamp": state.timestamp, "t_idx": state.t_idx,
                          "energy_efficiency_bps_per_joule": eff}])
