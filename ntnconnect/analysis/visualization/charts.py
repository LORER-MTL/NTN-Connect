"""
Enterprise Charts Component

Provides memoized, performant Plotly figure generators and geometry builders.
Decouples visual generation from Streamlit UI state.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

try:
    from ntnconnect.analysis.visualization import kpi_engine as E
except ImportError:
    try:
        import kpi_engine as E
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
        import kpi_engine as E

_R_EQ = 6378137.0  # Earth equatorial radius in meters

PRO_PALETTE = ["#2563EB", "#0EA5A4", "#7C3AED", "#F43F5E", "#F59E0B",
               "#0891B2", "#DB2777", "#16A34A", "#6366F1", "#EA580C"]

@st.cache_data(show_spinner=False)
def get_earth_surface_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0, np.pi, 60)
    theta = np.linspace(0, 2 * np.pi, 90)
    phi, theta = np.meshgrid(phi, theta)
    r = _R_EQ * 0.999
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return x, y, z, z 

def make_earth_sphere() -> go.Surface:
    x, y, z, surfacecolor = get_earth_surface_data()
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#02040a"], [0.45, "#060e20"],
                     [0.55, "#0a1a38"], [1, "#133068"]],
        surfacecolor=surfacecolor, showscale=False,
        opacity=0.98, hoverinfo="skip",
        lighting=dict(ambient=0.25, diffuse=0.9, specular=0.6, roughness=0.3, fresnel=0.8),
        lightposition=dict(x=2e7, y=0, z=2e7),
        name="Earth"
    )

@st.cache_data(show_spinner=False)
def _agg_over_time(df: pd.DataFrame, kpi: str, time_col: str, color_col: str = None) -> pd.DataFrame:
    gcols = [time_col] + ([color_col] if color_col else [])
    g = (df[gcols + [kpi]].apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna().astype(float)
         if not color_col else df[gcols + [kpi]])
    return g.groupby(gcols, dropna=True)[kpi].median().reset_index()


@st.cache_data(show_spinner=False)
def build_chart(df: pd.DataFrame, kind: str, kpi: str, info: dict, 
                time_col: str, group_col: str, color_col: str) -> go.Figure:
    """Factory method for generating performance-optimized Plotly figures."""
    u = info[kpi].unit
    ylab = f"{kpi} [{u}]" if u else kpi
    
    if kind == "line_trend" and time_col:
        d = _agg_over_time(df, kpi, time_col, color_col)
        fig = px.line(d, x=time_col, y=kpi, color=color_col, markers=False)
        fig.update_layout(title=f"{kpi} — trend over {time_col}", yaxis_title=ylab)
        return fig
        
    if kind == "histogram":
        fig = px.histogram(df, x=kpi, color=color_col, barmode="overlay",
                           nbins=50, opacity=0.7 if color_col else 1.0, marginal="box")
        fig.update_layout(title=f"{kpi} — distribution", xaxis_title=ylab)
        return fig
        
    if kind == "ccdf":
        fig = go.Figure()
        series_iter = (df.groupby(color_col) if color_col else [(None, df)])
        for name, sub in series_iter:
            # We compute CCDF mathematically for performance
            s = sub[kpi].dropna()
            if not s.empty:
                x_data = np.sort(s)
                y_data = 1.0 - np.arange(1, len(x_data) + 1) / len(x_data)
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode="lines",
                    name=str(name) if name is not None else kpi, line=dict(width=2.4)
                ))
        fig.update_layout(title=f"{kpi} — CCDF P(X > x)", xaxis_title=ylab, yaxis_title="P(X > x)")
        return fig
        
    if kind == "box_by_group" and group_col:
        fig = px.box(df, x=group_col, y=kpi, color=color_col, points="outliers")
        fig.update_layout(title=f"{kpi} — by {group_col}", yaxis_title=ylab)
        return fig
        
    if kind == "control_chart":
        d = (_agg_over_time(df, kpi, time_col, None) if time_col
             else df[[kpi]].reset_index().rename(columns={"index": "n"}))
        xcol = time_col if time_col else "n"
        cl = E.control_limits(d[kpi])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d[xcol], y=d[kpi], mode="lines+markers",
                      name=kpi, line=dict(color=PRO_PALETTE[0])))
        out = d[cl["outlier_mask"][:len(d)]] if cl["outlier_mask"].size else d.iloc[0:0]
        if not out.empty:
            fig.add_trace(go.Scatter(x=out[xcol], y=out[kpi], mode="markers",
                          name="anomaly", marker=dict(color="#D55E00", size=10, symbol="x")))
        
        for y, nm, dash in ((cl["mean"], "mean", "solid"), (cl["ucl"], "+3σ", "dash"), (cl["lcl"], "−3σ", "dash")):
            fig.add_hline(y=y, line_dash=dash, line_color="#888", annotation_text=nm, annotation_position="right")
        
        fig.update_layout(title=f"{kpi} — control chart ({cl.get('n_outliers', 0)} anomalies)",
                          xaxis_title=xcol, yaxis_title=ylab)
        return fig

    if kind == "violin" and group_col:
        fig = px.violin(df, x=group_col, y=kpi, color=color_col, box=True, points=False)
        fig.update_layout(title=f"{kpi} — distribution by {group_col}", yaxis_title=ylab)
        return fig
        
    if kind == "outliers":
        s = pd.to_numeric(df[kpi], errors="coerce").astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        xcol = time_col if (time_col and time_col in df.columns) else None
        x = df[xcol].to_numpy() if xcol else np.arange(len(s))
        norm = (s >= lo) & (s <= hi)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=x[norm.to_numpy()], y=s[norm].to_numpy(),
                      mode="markers", name="in-range",
                      marker=dict(size=4, color=PRO_PALETTE[0], opacity=0.45)))
        fig.add_trace(go.Scattergl(x=x[(~norm).to_numpy()], y=s[~norm].to_numpy(),
                      mode="markers", name="outlier",
                      marker=dict(size=9, color="#D55E00", symbol="x")))
        for yv in (hi, lo):
            fig.add_hline(y=yv, line_dash="dash", line_color="#888")
        fig.update_layout(title=f"{kpi} — IQR outlier scan ({int((~norm).sum())} flagged)",
                          yaxis_title=ylab, xaxis_title=xcol or "index")
        return fig
        
    if kind == "bar_counts":
        vc = df[kpi].astype(str).value_counts().reset_index()
        vc.columns = [kpi, "count"]
        fig = px.bar(vc, x=kpi, y="count", color=kpi)
        fig.update_layout(title=f"{kpi} — counts", showlegend=False)
        return fig

    # fallback
    fig = px.histogram(df, x=kpi)
    fig.update_layout(title=kpi)
    return fig


def render_chart_downloads(fig: go.Figure, key: str):
    """Renders export buttons (PNG, SVG, PDF) for the provided chart."""
    cols = st.columns(3)
    for c, fmt, mime in zip(cols, ("png", "svg", "pdf"),
                            ("image/png", "image/svg+xml", "application/pdf")):
        # To avoid blocking the page load for 30+ seconds generating high-res static images for every chart,
        # we generate the image lazily only when the user explicitly clicks the prepare button.
        btn_key = f"{key}_prep_{fmt}"
        if c.button(f"📷 Prep {fmt.upper()}", key=btn_key, use_container_width=True):
            try:
                with st.spinner(f"Rendering {fmt.upper()}..."):
                    data = fig.to_image(format=fmt, scale=3, width=1200, height=720)
                c.download_button(f"⬇ Download {fmt.upper()}", data, file_name=f"{key}.{fmt}",
                                  mime=mime, use_container_width=True, key=f"{key}_dl_{fmt}")
            except Exception as e:
                c.error(f"Failed to generate {fmt.upper()}: {e}")

