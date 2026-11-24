"""SkyNTNPro - KPI Analytics Studio.

A professional, publication-quality KPI dashboard (Tableau / Power-BI style)
for the Phase-2 outputs. Drag-and-drop one or many parquet/csv files (or pick
runs from v1/outputs/runs), and the app:

  * auto-discovers every KPI column (present or future) and categorises it,
  * auto-selects the most appropriate chart per KPI,
  * supports single / selected / all batch views, comparison, ranking,
    trend and anomaly detection,
  * renders clean, colour-blind-safe, publication-ready figures with
    high-resolution PNG / SVG / PDF export,
  * is fully data-driven - new KPIs appear automatically, no code changes.

Run:
    streamlit run v1/ntnconnect/visualization/kpi_dashboard.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
from layout import apply_enterprise_theme, render_app_header

# Pro colour palettes used by theme toggle + correlation heatmap
PRO_PALETTE = ["#0EA5A4", "#3b82f6", "#8b5cf6", "#f43f5e", "#f59e0b",
               "#10b981", "#db2777", "#6366f1", "#ea580c", "#2dd4bf"]
PRO_DIV = "RdBu"

# Apply Enterprise Theme natively (must BE first STREAMLIT COMMAND)
apply_enterprise_theme(page_title="SkyNTNPro · KPI Analytics Studio", page_icon="📡")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from ntnconnect.analysis.visualization import kpi_engine as E
except ImportError:
    import kpi_engine as E

# Project root = two levels up from ntnconnect/visualization/. The Phase-2/3
# artefacts live under <root>/outputs/ (runs/, ephemeris/, analysis/).
_V1 = _HERE.parents[1]
OUT_DIR = _V1 / "outputs"
RUNS_DIR = OUT_DIR / "runs"

import streamlit.components.v1 as components
import http.server as _hs, socketserver as _ss, threading as _th, functools as _ft


@st.cache_resource
def _static_server(root: str) -> int:
    """Serve `root` over http on a free port (once per process). The 3D map
    iframe points here so Cesium loads over http (its web workers refuse to
    run from Streamlit's null-origin srcdoc sandbox)."""
    handler = _ft.partial(_hs.SimpleHTTPRequestHandler, directory=root)
    httpd = _ss.TCPServer(("127.0.0.1", 0), handler)
    httpd.allow_reuse_address = True
    port = httpd.server_address[1]
    _th.Thread(target=httpd.serve_forever, daemon=True).start()
    return port


def _ensure_cesium_html(out_dir: Path, eph_dir: Path, scan_root: Path) -> bool:
    """(Re)generate cesium_view.html from the chosen ephemeris + runs folders."""
    import subprocess
    out_html = out_dir / "cesium_view.html"
    try:
        subprocess.run(
            [sys.executable, str(_HERE / "cesium_view.py"), "--step-s", "5",
             "--port", "0", "--out", str(out_html),
             "--eph-dir", str(eph_dir), "--scan-root", str(scan_root)],
            cwd=str(_V1.parent), timeout=900,
            env={**__import__("os").environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
    except Exception as e:
        st.error(f"3D view generation failed: {e}")
        return False
    return out_html.exists()


def _pick_folder() -> str | None:
    """Open a native OS folder-picker (runs on the machine hosting Streamlit)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        root.attributes("-topmost", True)
        d = filedialog.askdirectory(title="Select data folder")
        root.destroy()
        return d or None
    except Exception as e:
        st.sidebar.info(f"Native dialog unavailable ({e}). Type the path instead.")
        return None



# Data loading
def _read_any(path_or_buf, name: str) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(path_or_buf)
    return pd.read_parquet(path_or_buf)


@st.cache_data(show_spinner=False)
def _load_run_stream(run_dir: str, stream: str) -> pd.DataFrame:
    rd = Path(run_dir)
    shards = sorted(rd.glob(f"{stream}*.parquet"))
    if not shards:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def _discover_runs(runs_root: Path) -> dict[str, Path]:
    """Find run cells. Prefer `<cnst>__<strat>/run_NNN`; fall back to bare
    `<cnst>__<strat>` dirs only when they hold parquet directly (no run_ subdir),
    so we never double-count a cell and its parent."""
    found = {}
    for cell in (list(runs_root.glob("*__*/run_*")) +
                 list(runs_root.glob("*/*__*/run_*"))):
        if cell.is_dir() and cell.name.startswith("run_"):
            found[cell.parent.name] = cell          # key by cnst__strat (1 per cell)
    if not found:                                    # bare-dir fallback
        for cell in runs_root.glob("*__*"):
            if cell.is_dir() and "__" in cell.name and list(cell.glob("*.parquet")):
                found[cell.name] = cell
    return found


def _stream_names(run_dir: Path) -> list[str]:
    names = set()
    for p in run_dir.glob("*.parquet"):
        stem = p.stem
        for suf in ("_batch", "_batch0", "_batch00"):
            if suf in stem:
                stem = stem.split("_batch")[0]
                break
        names.add(stem)
    return sorted(names)


def _all_streams(run_dirs: list[Path]) -> list[str]:
    s: set[str] = set()
    for rd in run_dirs:
        s |= set(_stream_names(rd))
    return sorted(s)


# TAB MAP defined here so it is visible throughout the script
# Keys are the sidebar radio display labels; values are the internal tab identifiers.
_TAB_MAP = {
    "🌐": "🧠 Executive Summary",
    "📊": "📊 KPI Explorer",
    "⚖️": "🆚 Compare & Rank",
    "🌍": "🌍 3D Map",
    "🔔": "🔗 Correlations",
    "📋": "📦 Batch Analytics",
    "⚙️": "⚙ Data & Schema",
}

# Initialize configuration in session state if not already set
if "base_dir" not in st.session_state:
    st.session_state["base_dir"] = str(OUT_DIR)
if "auto_load_runs" not in st.session_state:
    st.session_state["auto_load_runs"] = True
if "selected_stream" not in st.session_state:
    st.session_state["selected_stream"] = "sinr_snapshot"
if "picked_runs" not in st.session_state:
    st.session_state["picked_runs"] = []

base = st.session_state["base_dir"]
base_p = Path(base)
if not base_p.is_dir():
    base_p = OUT_DIR

runs_dir = (base_p / "runs") if (base_p / "runs").exists() else base_p
eph_dir  = (base_p / "ephemeris") if (base_p / "ephemeris").exists() else base_p
out_dir  = base_p

runs = _discover_runs(runs_dir)
run_names = sorted(runs.keys())
if runs and not st.session_state["picked_runs"]:
    st.session_state["picked_runs"] = run_names

auto = st.session_state["auto_load_runs"]
picked = run_names if auto else st.session_state["picked_runs"]
stream = None
if runs:
    all_streams = _all_streams(list(runs.values()))
    default_stream = next((s for s in ("sinr_snapshot", "visibility_per_tick",
                           "handover_events", "link_visibility")
                           if s in all_streams), (all_streams[0] if all_streams else "sinr_snapshot"))
    if st.session_state["selected_stream"] not in all_streams:
        st.session_state["selected_stream"] = default_stream
    stream = st.session_state["selected_stream"]

uploads = []

with st.sidebar:
    # Left Navigation Rail
    selected_icon = st.radio(
        "nav", list(_TAB_MAP.keys()),
        index=3,               # default: 3D Map
        label_visibility="collapsed",
        key="nav_selected",
    )
    selected_tab = _TAB_MAP[selected_icon]

# Assemble the working dataframe (auto-loaded runs + any uploads)
@st.cache_data(show_spinner=False)
def _assemble_runs(items: tuple, stream: str) -> pd.DataFrame:
    """Concat the chosen run streams into one tagged DataFrame - cached on
    (runs, stream) so the heavy read+concat happens once, not every rerun."""
    frames = []
    for label, d in items:
        avail = set(_stream_names(Path(d)))
        use = stream if stream in avail else (sorted(avail)[0] if avail else None)
        if not use:
            continue
        x = _load_run_stream(d, use)
        if not x.empty:
            x = x.copy(); x["series"] = label; frames.append(x)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


parts: list[pd.DataFrame] = []
if runs and stream and picked:
    run_df = _assemble_runs(tuple(sorted((p, str(runs[p])) for p in picked)), stream)
    if not run_df.empty:
        parts.append(run_df)
for up in (uploads or []):
    try:
        d = _read_any(up, up.name)
        d["series"] = up.name.rsplit(".", 1)[0]
        parts.append(d)
    except Exception as e:
        st.sidebar.error(f"{up.name}: {e}")

if not parts:
    st.info("No data yet. Runs under `v1/outputs/runs` load automatically — "
            "or drop a parquet/csv in the sidebar.")
    st.stop()

# Align columns across heterogeneous sources via outer concat.
df = pd.concat(parts, ignore_index=True, sort=False)
info = E.discover(df)
time_col = E.time_column(info)
keys = E.key_columns(info)
multi_series = df["series"].nunique() > 1 if "series" in df.columns else False



# Initialize session state for batch filtering and theme
if "tpb" not in st.session_state:
    st.session_state["tpb"] = 900
if "batch_mode" not in st.session_state:
    st.session_state["batch_mode"] = "All"
if "selected_batch" not in st.session_state:
    st.session_state["selected_batch"] = 0
if "selected_batches" not in st.session_state:
    st.session_state["selected_batches"] = []
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"
if "selected_filters" not in st.session_state:
    st.session_state["selected_filters"] = {}

# Apply batch column and filtering based on state
if time_col:
    tpb = st.session_state["tpb"]
    df = E.add_batch_column(df, time_col, int(tpb))
    if "batch" in df.columns and df["batch"].notna().any():
        all_b = sorted(int(b) for b in df["batch"].dropna().unique())
        mode = st.session_state["batch_mode"]
        if mode == "Single":
            b = st.session_state["selected_batch"]
            if b not in all_b:
                b = all_b[0]
                st.session_state["selected_batch"] = b
            df = df[df["batch"] == b]
        elif mode == "Selected":
            bsel = st.session_state["selected_batches"]
            if bsel:
                df = df[df["batch"].isin(bsel)]

# Key filters
for k in keys:
    vals = sorted(df[k].dropna().unique().tolist())
    if 1 < len(vals) <= 200:
        chosen = st.session_state["selected_filters"].get(k, [])
        if chosen:
            df = df[df[k].isin(chosen)]

group_col = "batch" if "batch" in df.columns else ("series" if multi_series else None)
color_col = "series" if multi_series else None

# Dynamic Top Navigation Bar & Synchronized KPI Strip
_head_kpis = {}
curr_batch = st.session_state.get("selected_batch", 0)

if 'runs' in locals() and 'picked' in locals() and runs and picked:
    users_tot = 0
    handovers_tot = 0
    
    for p in picked:
        rp = Path(runs[p])
        
        # 1. Handovers for the current tick
        ho_p = rp / "ho_events.parquet"
        if ho_p.exists():
            try:
                hodf = pd.read_parquet(ho_p)
                if 't_idx' in hodf.columns:
                    handovers_tot += len(hodf[hodf['t_idx'] == curr_batch])
            except: pass

    # 2. Active Users for the current tick (from the active df if it has users)
    if 'df' in locals() and 'user_id' in df.columns:
        users_tot += df['user_id'].nunique()
        
    if users_tot > 0: _head_kpis["Active Users"] = str(int(users_tot))
    if handovers_tot > 0: _head_kpis["Handovers"] = str(int(handovers_tot))

if 'df' in locals() and not df.empty and info:
    # Use df (which is now filtered by time) to get throughput and other metrics
    num_kpis = [k for k in E.kpi_columns(info) if info[k].kind == "numeric"]
    priority_kpis = ['throughput_mbps', 'n_visible_sats', 'mean_sinr_db', 'sinr_db', 'outage_fraction']
    selected_kpis = [pk for pk in priority_kpis if pk in num_kpis]
    
    for k in num_kpis:
        if len(_head_kpis) + len(selected_kpis) >= 6: break
        if k not in selected_kpis and not any(x in k.lower() for x in ['user', 'handover', 'throughput', 'visible']):
            selected_kpis.append(k)
            
    for k in selected_kpis:
        if len(_head_kpis) >= 6: break
        stat = E.summary_stats(df[k])
        if stat:
            label = k
            if k == 'throughput_mbps': label = 'Avg Throughput'
            elif k == 'n_visible_sats': label = 'Visible Sats'
            _head_kpis[label] = E._fmt(stat['median'], info[k].unit)

from layout import render_mission_control_chrome
cnst_val = st.query_params.get("cnst", "kuiper")
strat_val = st.query_params.get("strat", "farthest_approaching")
show_all_val_str = st.query_params.get("show_all", "true")
show_all_val = show_all_val_str.lower() == "true"
render_mission_control_chrome(_head_kpis, cnst=cnst_val, strat=strat_val, show_all=show_all_val)

groups = E.grouped_by_category(info)

# Chart factory
from charts import build_chart, render_chart_downloads as _downloads

# Theme toggle
_theme = st.session_state["theme"]
_T = E.THEMES["dark" if _theme == "Dark" else "light"]
try:
    _lay = pio.templates["skyntn"].layout
    if _theme == "Dark":
        _lay.paper_bgcolor = _T["paper"]; _lay.plot_bgcolor = _T["plot"]
        _lay.font.color = _T["font"]; _lay.title.font.color = _T["title"]
    else:                                   # keep charts transparent on the cards
        _lay.paper_bgcolor = "rgba(0,0,0,0)"; _lay.plot_bgcolor = "rgba(0,0,0,0)"
        _lay.font.color = "#243044"; _lay.title.font.color = "#0b1f3a"
    _lay.xaxis.gridcolor = _T["grid"]; _lay.yaxis.gridcolor = _T["grid"]
    _lay.colorway = PRO_PALETTE             # attractive palette in both themes
    pio.templates.default = "skyntn"
except Exception:
    pass


# selected_tab is already set in the sidebar block above.
# Use a plain container so the main content fills the full width.
main_col = st.container()

# ---- 3D Map (embedded Cesium globe) ----
with main_col:
    if selected_tab == '🌍 3D Map':
        # Check both out_dir and the project root for cesium_view.html
        html_path = out_dir / "cesium_view.html"
        if not html_path.exists() and (_HERE / "cesium_view.html").exists():
            html_path = _HERE / "cesium_view.html"
        # Determine the directory to serve
        serve_dir = str(html_path.parent)
        html_name = html_path.name

        # Auto-generate the Cesium HTML if it doesn't exist yet
        if not html_path.exists():
            with st.spinner(f"Generating 3D view from {eph_dir} + {runs_dir}…"):
                success = _ensure_cesium_html(out_dir, eph_dir, runs_dir)
            if success:
                html_path = out_dir / "cesium_view.html"
                serve_dir = str(out_dir)
                st.session_state["cesium_port"] = _static_server(serve_dir)
            else:
                st.error("3D globe generation failed – see logs for details.")

        # Render the globe
        if html_path.exists():
            port = st.session_state.get("cesium_port") or _static_server(serve_dir)
            st.session_state["cesium_port"] = port

            placeholder = st.empty()
            with placeholder:
                st.components.v1.iframe(
                    f"http://localhost:{port}/{html_name}?cnst={cnst_val}&strat={strat_val}&show_all={show_all_val_str}",
                    height=760,
                    scrolling=False
                )
        else:
            st.info("Interactive constellation globe is currently unavailable. "
                    "Click ↻ Regenerate Globe above to build it.")

_CTX = {"time_col": time_col, "group_col": group_col}

# ---- Executive Summary ----
with main_col:
    if selected_tab == '🧠 Executive Summary':
        st.subheader("Executive summary")
        
        # Headline KPI tiles (median of the top SINR/Throughput/Quality KPIs)
        head = [k for k in E.kpi_columns(info) if info[k].kind == "numeric"
                and E.categorize2(k) in ("SINR & Signal", "Throughput", "Quality",
                                          "Handover", "Delay & RTT")][:6]
        if head:
            st.markdown("##### Headline KPIs")
            cols = st.columns(len(head))
            for c, k in zip(cols, head):
                sstat = E.summary_stats(df[k])
                if sstat:
                    m_str = E._fmt(sstat['median'], info[k].unit)
                    p5_str = E._fmt(sstat['p05'])
                    p95_str = E._fmt(sstat['p95'])
                    
                    # Create custom metric + sparkline container
                    c.markdown(f'''
                        <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">{k}</div>
                            <div style="font-size: 1.5rem; font-weight: 600; color: #f8fafc; margin-bottom: 4px;">{m_str}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Generate Sparkline
                    import plotly.graph_objects as go
                    trend_data = df[k].dropna().tail(50) # Last 50 points
                    color = "#10b981" if sstat['median'] > sstat['mean'] else "#f59e0b"
                    fig = go.Figure(go.Scatter(y=trend_data, mode='lines', line=dict(color=color, width=2, shape='spline')))
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0), height=30,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        showlegend=False, hovermode=False
                    )
                    c.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        insights = E.executive_summary(df, info, group_col, time_col)
        if not insights:
            st.info("No numeric KPIs to summarise.")
        for s in insights:
            # Convert **text** to <b>text</b> so it renders correctly inside the raw HTML div
            import re
            s_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
            st.markdown(f'<div class="insight">📌 {s_html}</div>', unsafe_allow_html=True)

# ---- KPI Explorer ----
with main_col:
    if selected_tab == '📊 KPI Explorer':
        st.subheader("Auto-rendered KPI Explorer")
        cat_names = list(groups.keys())
        if not cat_names:
            st.info("No categories discovered.")
        else:
            # LAZY RENDERING: pick one category + one KPI and render only that
            # KPI's charts. The previous design built every KPI × every chart
            # across all category tabs on every rerun (~80+ Plotly figures →
            # multi-second hangs / timeouts). Streamlit can't tell which tab is
            # client-side active, so eager tab content is the anti-pattern; a
            # selector that renders a single KPI is the scalable fix.
            csel, ksel = st.columns([1, 2])
            cat = csel.selectbox("Category", cat_names, key="exp_cat")
            kpi = ksel.selectbox("KPI", groups[cat], key="exp_kpi")

            recs = E.recommend_registry(df, info, kpi, _CTX) or ["histogram"]
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.markdown(f"### {kpi}")

            if info[kpi].kind == "numeric":
                s = E.summary_stats(df[kpi])
                if s:
                    an = E.detect_anomalies(df[kpi])
                    st.markdown(f'''<div class="metric-row">
                        <div class="kpi-metric-card"><div class="kpi-metric-label">Median</div>
                            <div class="kpi-metric-value">{s['median']:.2f}</div></div>
                        <div class="kpi-metric-card"><div class="kpi-metric-label">Mean</div>
                            <div class="kpi-metric-value">{s['mean']:.2f}</div></div>
                        <div class="kpi-metric-card"><div class="kpi-metric-label">5–95% Range</div>
                            <div class="kpi-metric-value">{s['p05']:.1f} - {s['p95']:.1f}</div></div>
                        <div class="kpi-metric-card"><div class="kpi-metric-label">Anomalies</div>
                            <div class="kpi-metric-value">{an['z_outliers']}</div></div>
                    </div>''', unsafe_allow_html=True)

            st.markdown(f'<div class="insight">💡 {E.auto_insight(df, info, kpi, time_col)}</div>',
                        unsafe_allow_html=True)

            labels = [E._CHART_REGISTRY[r].label if r in E._CHART_REGISTRY else r for r in recs]
            chart_tabs = st.tabs(labels)
            for ctab, kind in zip(chart_tabs, recs):
                with ctab:
                    fig = build_chart(df, kind, kpi, info, time_col, group_col, color_col)
                    st.plotly_chart(fig, use_container_width=True)
                    _downloads(fig, f"{cat}_{kpi}_{kind}")

            st.markdown("</div>", unsafe_allow_html=True)

# ---- Batch Analytics ----
with main_col:
    if selected_tab == '📦 Batch Analytics':
        num_kpis = [k for k in E.kpi_columns(info) if info[k].kind == "numeric"]
        if not group_col or group_col not in df.columns:
            st.info("Batch analytics needs a grouping column. Set a tick/batch size "
                    "in the sidebar, or load multiple files/runs (series).")
        elif not num_kpis:
            st.info("No numeric KPIs available.")
        else:
            bk = st.selectbox("KPI", num_kpis, key="batch_kpi")
            stat = st.radio("Statistic", ["median", "mean", "p95", "p05"],
                            horizontal=True, key="batch_stat")
            bw = E.best_worst(df, bk, group_col, stat)
            if bw:
                hib = bw["higher_is_better"]
                arrow = "↑ higher is better" if hib else "↓ lower is better" if hib is False else "neutral"
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🏆 Best {group_col}", str(bw['best'][group_col]),
                          f"{bw['best'][f'{stat}_{bk}']:.2f}")
                c2.metric(f"⚠ Worst {group_col}", str(bw['worst'][group_col]),
                          f"{bw['worst'][f'{stat}_{bk}']:.2f}")
                c3.metric("Polarity", arrow)
            # Trend evolution
            ev = E.batch_evolution(df, bk, group_col, stat)
            if not ev.empty:
                fige = px.line(ev, x=group_col, y=bk, markers=True)
                fige.update_layout(title=f"{bk} — {stat} evolution across {group_col}")
                st.plotly_chart(fige, use_container_width=True)
                _downloads(fige, f"evolution_{bk}")
            # Benchmark table
            st.markdown("##### Benchmark table")
            st.dataframe(E.benchmark(df, bk, group_col), use_container_width=True)
            # Waterfall of batch-to-batch deltas
            if not ev.empty and len(ev) > 1:
                delta = ev[bk].diff().fillna(ev[bk].iloc[0])
                figw = go.Figure(go.Waterfall(
                    x=[str(v) for v in ev[group_col]], y=delta.tolist(),
                    connector={"line": {"color": "#9fb4d8"}}))
                figw.update_layout(title=f"{bk} — {stat} change between {group_col}s")
                st.plotly_chart(figw, use_container_width=True)
                _downloads(figw, f"waterfall_{bk}")
            # Radar across batches (multi-KPI fingerprint, min-max normalised)
            radar_kpis = num_kpis[:8]
            if len(radar_kpis) >= 3:
                st.markdown("##### Multi-KPI radar (per-batch fingerprint, normalised)")
                piv = (df.groupby(group_col)[radar_kpis].median())
                piv_range = piv.max() - piv.min()
                piv_range = piv_range.replace(0, np.nan)
                norm = (piv - piv.min()) / piv_range
                norm = norm.fillna(0.5)
                figr = go.Figure()
                for g, row in norm.iterrows():
                    figr.add_trace(go.Scatterpolar(
                        r=row.tolist() + [row.tolist()[0]],
                        theta=radar_kpis + [radar_kpis[0]],
                        fill="toself", name=str(g), opacity=0.45))
                figr.update_layout(title="Per-batch KPI fingerprint",
                                   polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                st.plotly_chart(figr, use_container_width=True)
                _downloads(figr, "radar_batches")
            # Significance between two batches
            gv = sorted(df[group_col].dropna().unique().tolist())
            if len(gv) >= 2:
                st.markdown("##### Statistical significance")
                s1, s2 = st.columns(2)
                ga = s1.selectbox(f"{group_col} A", gv, index=0, key="sigA")
                gb = s2.selectbox(f"{group_col} B", gv, index=len(gv) - 1, key="sigB")
                sig = E.significance(df[df[group_col] == ga][bk],
                                     df[df[group_col] == gb][bk])
                if sig["p_value"] is not None:
                    verdict = "**significant**" if sig["significant"] else "not significant"
                    st.markdown(f'<div class="insight">{sig["method"]}: '
                                f'Δmedian = {sig.get("delta_median", 0):+.2f}{info[bk].unit}, '
                                f'p = {sig["p_value"]:.2e} → {verdict} (α=0.05).</div>',
                                unsafe_allow_html=True)
                else:
                    st.caption(f"Significance unavailable ({sig['method']}).")

# ---- Compare & Rank ----
with main_col:
    if selected_tab == '🆚 Compare & Rank':
        num_kpis = [k for k in E.kpi_columns(info) if info[k].kind == "numeric"]
        if not num_kpis:
            st.info("No numeric KPIs to rank.")
        else:
            kpi = st.selectbox("Rank by KPI", num_kpis, key="rank_kpi")
            stat = st.radio("Statistic", ["median", "mean", "p95", "p05"], horizontal=True)
            by = st.selectbox("Group by", [c for c in ("series", "batch") + tuple(keys)
                                            if c in df.columns] or ["series"])
            agg = (df.groupby(by)[kpi].median() if stat == "median" else
                   df.groupby(by)[kpi].mean() if stat == "mean" else
                   df.groupby(by)[kpi].quantile(0.95) if stat == "p95" else
                   df.groupby(by)[kpi].quantile(0.05))
            agg = agg.sort_values(ascending=False).reset_index()
            agg.columns = [by, f"{stat}_{kpi}"]
            figr = px.bar(agg, x=by, y=f"{stat}_{kpi}", color=by, text_auto=".2f")
            figr.update_layout(title=f"Ranking — {stat} of {kpi} by {by}", showlegend=False)
            st.plotly_chart(figr, use_container_width=True)
            _downloads(figr, f"rank_{kpi}_{by}")
            st.dataframe(agg, use_container_width=True)

# ---- Correlations ----
with main_col:
    if selected_tab == '🔗 Correlations':
        num_kpis = [k for k in E.kpi_columns(info) if info[k].kind == "numeric"]
        if len(num_kpis) < 2:
            st.info("Need ≥2 numeric KPIs for correlation analysis.")
        else:
            c1, c2 = st.columns(2)
            xk = c1.selectbox("X", num_kpis, key="cx")
            yk = c2.selectbox("Y", num_kpis, index=min(1, len(num_kpis) - 1), key="cy")
            samp = df[[xk, yk] + ([color_col] if color_col else [])].dropna()
            if len(samp) > 20000:
                samp = samp.sample(20000, random_state=0)
            try:
                import statsmodels  # noqa: F401  (required by plotly OLS trendline)
                _trend = "ols"
            except Exception:
                _trend = None
            figs = px.scatter(samp, x=xk, y=yk, color=color_col, opacity=0.5,
                              trendline=_trend)
            figs.update_layout(title=f"{yk} vs {xk}")
            st.plotly_chart(figs, use_container_width=True)
            _downloads(figs, f"scatter_{xk}_{yk}")
            corr = df[num_kpis].corr(numeric_only=True)
            figh = px.imshow(corr, color_continuous_scale=PRO_DIV, zmin=-1, zmax=1,
                             aspect="auto", text_auto=".2f")
            figh.update_layout(title="Correlation matrix")
            st.plotly_chart(figh, use_container_width=True)
            _downloads(figh, "corr_matrix")

# ---- Data & schema ----
with main_col:
    if selected_tab == '⚙ Data & Schema':
        st.subheader("Data & Constellation Configuration")
        
        # Folder Path Config
        cpath, cbtn = st.columns([6, 1])
        new_base = cpath.text_input("Folder path", st.session_state["base_dir"])
        if cbtn.button("📁", help="Browse for a folder"):
            chosen = _pick_folder()
            if chosen:
                st.session_state["base_dir"] = chosen
                st.rerun()
        if new_base != st.session_state["base_dir"]:
            st.session_state["base_dir"] = new_base
            st.rerun()
            
        # Auto-load Runs
        new_auto = st.checkbox("Auto-load all runs", value=st.session_state["auto_load_runs"],
                               help="Automatically load all runs in the folder.")
        if new_auto != st.session_state["auto_load_runs"]:
            st.session_state["auto_load_runs"] = new_auto
            st.rerun()
            
        # Selected Runs / Stream
        if runs:
            all_streams = _all_streams(list(runs.values()))
            stream_sel = st.selectbox("Analysis Stream", all_streams or ["sinr_snapshot"],
                                      index=(all_streams.index(st.session_state["selected_stream"])
                                             if st.session_state["selected_stream"] in all_streams else 0))
            if stream_sel != st.session_state["selected_stream"]:
                st.session_state["selected_stream"] = stream_sel
                st.rerun()
                
            if not new_auto:
                picked_sel = st.multiselect("Select Run(s)", run_names, default=st.session_state["picked_runs"])
                if picked_sel != st.session_state["picked_runs"]:
                    st.session_state["picked_runs"] = picked_sel
                    st.rerun()
            st.caption(f"✓ Active runs folder: `{runs_dir.name}`  ·  ephemeris folder: `{eph_dir.name}`")
        
        # Regenerate globe
        rebuild = st.button("↻ Regenerate 3D Globe View", use_container_width=True)
        if rebuild:
            with st.spinner("Regenerating standalone 3D Cesium view..."):
                success = _ensure_cesium_html(out_dir, eph_dir, runs_dir)
            if success:
                st.success("Cesium view successfully regenerated! Rerunning...")
                st.rerun()
                
        # Theme config
        new_theme = st.radio("Application UI Theme", ["Light", "Dark"],
                             index=1 if st.session_state["theme"] == "Dark" else 0, horizontal=True)
        if new_theme != st.session_state["theme"]:
            st.session_state["theme"] = new_theme
            st.rerun()

        st.markdown("---")
        st.subheader("Discovered schema")
        # KpiInfo has no free-text description; synthesize a useful one from the
        # discovered metadata (was `v.desc` → AttributeError crash).
        schema = pd.DataFrame([{
            "Column": k, "Type": v.kind, "Unit": v.unit, "Category": E.categorize2(k),
            "Unique": v.n_unique, "Missing": v.n_missing,
            "Description": f"{E.categorize2(k)} · {v.kind}" + (f" [{v.unit}]" if v.unit else ""),
        } for k, v in info.items()])
        st.dataframe(schema, use_container_width=True)

# Aerospace-Grade Simulation Timeline & Playback
if selected_tab != '🌍 3D Map':
    st.markdown("---")
    with st.container():
        tcol1, tcol2, tcol3, tcol4 = st.columns([1, 1, 8, 2])
        with tcol1:
            if st.button("⏮ Step Back", use_container_width=True): pass
        with tcol2:
            if st.button("▶ Play", use_container_width=True): pass
        with tcol3:
            st.slider("Simulation Timeline", min_value=0, max_value=1000, value=250, label_visibility="collapsed")
        with tcol4:
            st.selectbox("Speed", ["1x", "10x", "100x"], index=1, label_visibility="collapsed")
            st.caption(f"{len(df):,} rows · {df['series'].nunique()} series · "
                       f"time column = {time_col or '—'}")
            with st.expander("Preview (first 200 rows)"):
                st.dataframe(df.head(200), use_container_width=True)
