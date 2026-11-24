# Per-link KPIs.
#
# These KPIs read the per-link DataFrame produced by Phase-2 stages
# (geometry -> channel -> interference -> rrm_mcs...).
# Crucially, `links` does not carry a `timestamp` column - we attach
# state.timestamp here so the emitted KPI row carries it.

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


def _attach_time(df: pd.DataFrame, state) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = state.timestamp
    df["t_idx"] = state.t_idx
    return df


@register_kpi("snr_db", granularity="per_link", units="dB",
              default_plot="cdf", description="Per-link SNR.")
def snr_db(state, links, history):
    if links is None or links.empty or "snr_db" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","snr_db"]], state)


@register_kpi("sinr_db", granularity="per_link", units="dB",
              default_plot="cdf", description="Per-link SINR including co-band interference.")
def sinr_db(state, links, history):
    if links is None or links.empty or "sinr_db" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","sinr_db"]], state)


@register_kpi("spectral_efficiency_bps_hz", granularity="per_link", units="bps/Hz",
              default_plot="cdf", description="Shannon SE = log2(1+SINR).")
def spec_eff(state, links, history):
    if links is None or links.empty or "spectral_efficiency_bps_hz" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","spectral_efficiency_bps_hz"]], state)


@register_kpi("link_rate_bps", granularity="per_link", units="bps",
              default_plot="cdf", description="Per-link Shannon rate (SE * bandwidth).")
def link_rate(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","link_rate_bps"]], state)


@register_kpi("path_loss_db", granularity="per_link", units="dB",
              default_plot="cdf", description="Free-space path loss + extras.")
def path_loss(state, links, history):
    if links is None or links.empty or "path_loss_db" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","path_loss_db"]], state)


@register_kpi("doppler_hz", granularity="per_link", units="Hz",
              default_plot="hist", description="Doppler shift per link.")
def doppler(state, links, history):
    if links is None or links.empty or "doppler_hz" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","doppler_hz"]], state)


@register_kpi("range_rate_km_s", granularity="per_link", units="km/s",
              default_plot="hist",
              description="Per-link range rate (sat-to-UE radial velocity).")
def range_rate(state, links, history):
    if links is None or links.empty or "range_rate_km_s" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","range_rate_km_s"]], state)


@register_kpi("azimuth_deg", granularity="per_link", units="deg",
              default_plot="hist", description="Per-link azimuth angle.")
def azimuth(state, links, history):
    # Accept either 'azimuth_deg' or 'azimuth' (legacy name).
    col = "azimuth_deg" if "azimuth_deg" in (links.columns if links is not None else []) \
          else ("azimuth" if links is not None and "azimuth" in links.columns else None)
    if links is None or links.empty or col is None: return None
    sub = links[["user_id","node_id", col]].rename(columns={col: "azimuth_deg"})
    return _attach_time(sub, state)


@register_kpi("chan_gain_lin", granularity="per_link", units="",
              default_plot="hist",
              description="Per-link |H|_F^2 channel gain (linear).")
def chan_gain(state, links, history):
    if links is None or links.empty or "chan_gain_lin" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","chan_gain_lin"]], state)


@register_kpi("rician_k_factor_db", granularity="per_link", units="dB",
              default_plot="hist",
              description="Per-link Rician K-factor (when channel model exposes it).")
def k_factor(state, links, history):
    if links is None or links.empty or "k_factor_db" not in links.columns: return None
    return _attach_time(links[["user_id","node_id","k_factor_db"]], state)
