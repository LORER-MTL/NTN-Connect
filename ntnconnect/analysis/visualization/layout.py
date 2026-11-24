"""
Enterprise UI Layout & Theming Component - Mission Control Edition
Matches the KuiperSim reference image with:
  - Fixed full-width top nav bar
  - KPI strip row
  - Left icon-only sidebar rail
  - Right floating tool rail
  - Bottom timeline bar
  - Overlay panels (Events + Legend)
"""
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go


def apply_enterprise_theme(page_title: str = "SkyNTNPro", page_icon: str = "📡"):
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      /* ── Reset & Tokens ─────────────────────────────────── */
      :root {
        --top-h: 52px;
        --kpi-h: 72px;
        --bottom-h: 58px;
        --left-w: 68px;   /* matches sidebar width */
        --right-w: 56px;
        --bg: #080c18;
        --surface: rgba(10,16,36,0.92);
        --glass: rgba(18,28,54,0.80);
        --border: rgba(255,255,255,0.08);
        --primary: #3b82f6;
        --green:   #10b981;
        --amber:   #f59e0b;
        --red:     #ef4444;
        --ink:     #f1f5f9;
        --muted:   #64748b;
        --font: 'Inter', -apple-system, sans-serif;
        --font-head: 'Outfit', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
      }

      html, body, [class*="css"] {
        font-family: var(--font);
        background: var(--bg) !important;
        color: var(--ink);
      }

      /* ── Hide all default Streamlit chrome ──────────────── */
      header[data-testid="stHeader"]         { display: none !important; }
      #MainMenu                              { display: none !important; }
      footer                                 { display: none !important; }
      .stDeployButton                        { display: none !important; }

      /* ── Main app background ────────────────────────────── */
      .stApp {
        background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 20%, rgba(16,185,129,0.05) 0%, transparent 40%),
                    var(--bg) !important;
        padding-top: 0 !important;
      }

      /* ── Block container: only top/bottom shift (sidebar handles left) ──── */
      .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        padding-top: calc(var(--top-h) + var(--kpi-h)) !important;
        padding-bottom: var(--bottom-h) !important;
      }

      /* ── Sidebar → narrow icon rail ───────────────────────── */
      section[data-testid="stSidebar"] {
        width: var(--left-w) !important;
        min-width: var(--left-w) !important;
        max-width: var(--left-w) !important;
        background: rgba(8,12,24,0.99) !important;
        border-right: 1px solid var(--border) !important;
        overflow: visible !important;
      }
      /* Hide sidebar toggle/resize handle */
      section[data-testid="stSidebar"] + div > button { display: none !important; }
      [data-testid="collapsedControl"] { display: none !important; }
      /* Shrink inner padding */
      section[data-testid="stSidebar"] > div:first-child {
        padding: 8px 0 4px !important;
        overflow: visible !important;
      }
      /* Hide EVERYTHING in sidebar except the nav radio widget (first child) */
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        display: none !important;
      }
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child {
        display: block !important;
      }

      /* ── Buttons ────────────────────────────────────────── */
      .stButton > button {
        border-radius: 6px;
        font-family: var(--font-head);
        font-weight: 500;
        padding: 0.35rem 0.8rem;
        background: rgba(30,41,59,0.7);
        color: var(--ink);
        border: 1px solid var(--border);
        transition: all 0.18s ease;
        font-size: 13px;
      }
      .stButton > button:hover {
        border-color: rgba(255,255,255,0.25);
        background: rgba(51,65,85,0.9);
        transform: translateY(-1px);
      }

      /* ── Sidebar Nav Icon Rail ──────────────────────────── */
      /* Target the nav radio (7 options) specifically using :has() */
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) {
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
        align-items: center !important;
        padding: 12px 0 !important;
        width: 100% !important;
      }
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) label {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        width: 56px !important;
        height: 56px !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.18s !important;
        margin: 0 !important;
        padding: 4px 0 !important;
        box-sizing: border-box !important;
      }
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) label:hover {
        background: rgba(255,255,255,0.07) !important;
        color: #e2e8f0 !important;
      }
      /* Active (selected) nav item */
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) label:has(input:checked) {
        background: rgba(59,130,246,0.15) !important;
        color: #60a5fa !important;
        box-shadow: inset 3px 0 0 #3b82f6 !important;
      }
      /* Hide the radio circle dot */
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) input[type="radio"] {
        display: none !important;
      }
      /* Style the option text: large emoji centered */
      section[data-testid="stSidebar"] div[role="radiogroup"]:has(> label:nth-child(7):last-child) label p {
        font-size: 24px !important;
        margin: 0 !important;
        line-height: 1 !important;
        text-align: center !important;
      }

      /* ── All other radios in sidebar: compact horizontal ─── */
      .stRadio div[role="radiogroup"] { gap: 0 !important; }
      .stRadio label {
        font-size: 0.9rem !important;
        padding: 0.3rem 0 !important;
        cursor: pointer;
      }

      /* ── Plotly charts ──────────────────────────────────── */
      .stPlotlyChart {
        background: rgba(10,15,30,0.35);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
      }

      /* ── iframe (3D globe) fills all remaining space ─────── */
      iframe {
        height: calc(100vh - var(--top-h) - var(--kpi-h) - 2px) !important;
        width: 100% !important;
        border: none !important;
        border-radius: 0 !important;
        display: block;
      }

      /* ── Click-through fix for Streamlit element containers wrapping fixed chrome ── */
      div[data-testid="element-container"]:has(#mc-topbar),
      div[data-testid="element-container"]:has(#mc-kpistrip),
      div[data-testid="element-container"]:has(#mc-events) {
        position: fixed !important;
        z-index: 99999 !important;
        pointer-events: none !important;
      }
      div[data-testid="element-container"]:has(iframe) {
        z-index: 1 !important;
      }

      /* ──────────────────────────────────────────────────────
         FIXED UI CHROME (all position: fixed)
         ────────────────────────────────────────────────────── */

      /* TOP BAR */
      #mc-topbar {
        position: fixed; top: 0; left: 0; right: 0;
        height: var(--top-h);
        background: rgba(7,12,28,0.96);
        backdrop-filter: blur(20px) saturate(180%);
        border-bottom: 1px solid var(--border);
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 16px;
        z-index: 99999;
        box-shadow: 0 2px 20px rgba(0,0,0,0.6);
        pointer-events: auto !important;
      }
      #mc-topbar .tb-left {
        display: flex; align-items: center; gap: 12px; min-width: 200px;
      }
      #mc-topbar .tb-logo {
        font-family: var(--font-head);
        font-size: 18px; font-weight: 700; color: #fff;
        letter-spacing: -0.5px;
      }
      #mc-topbar .tb-mode {
        display: flex; align-items: center; gap: 5px;
        font-size: 11px; color: #94a3b8;
      }
      #mc-topbar .tb-mode .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 6px var(--green);
        animation: pulse 2s infinite;
      }
      @keyframes pulse {
        0%,100% { opacity:1; } 50% { opacity:0.5; }
      }

      /* FILTER STRIP (center of topbar) */
      #mc-topbar .tb-center {
        display: flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,0,0,0.35);
        border-radius: 8px;
        padding: 4px 12px;
        box-shadow: 0 0 12px rgba(255,60,60,0.12);
      }
      #mc-topbar .tb-center .filter-label {
        font-size: 11px; color: #64748b; margin-right: 2px;
      }
      #mc-topbar .tb-center .pill {
        font-size: 12px; font-weight: 500;
        padding: 3px 10px; border-radius: 20px; cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.15s;
      }
      #mc-topbar .tb-center .pill.active {
        background: #2563eb; color: #fff; border-color: #3b82f6;
        box-shadow: 0 0 10px rgba(59,130,246,0.4);
      }
      #mc-topbar .tb-center .pill.inactive {
        background: rgba(255,255,255,0.07); color: #94a3b8;
      }
      #mc-topbar .tb-center .pill.inactive:hover {
        background: rgba(255,255,255,0.12); color: #e2e8f0;
      }
      #mc-topbar .tb-center .sep {
        width: 1px; height: 20px; background: var(--border); margin: 0 4px;
      }
      #mc-topbar .tb-center .dropdown {
        display: flex; align-items: center; gap: 4px;
        font-size: 12px; color: #cbd5e1;
        background: rgba(255,255,255,0.06);
        padding: 3px 10px; border-radius: 5px;
        border: 1px solid rgba(255,255,255,0.1);
        cursor: pointer;
      }
      #mc-topbar .tb-center .show-all {
        display: flex; align-items: center; gap: 5px;
        font-size: 12px; color: #e2e8f0; margin-left: 6px;
      }
      #mc-topbar .tb-center .show-all input[type=checkbox] {
        accent-color: var(--primary); width: 13px; height: 13px;
      }

      #mc-topbar .tb-right {
        display: flex; align-items: center; gap: 12px;
        min-width: 260px; justify-content: flex-end;
      }
      #mc-topbar .tb-right .timestamp {
        font-family: var(--font-mono); font-size: 12px; color: #94a3b8;
      }
      #mc-topbar .tb-right .live-badge {
        color: var(--green); font-size: 12px; font-weight: 600;
        animation: pulse 2s infinite;
      }
      #mc-topbar .tb-right .deploy-btn {
        background: #1d4ed8; color: #fff;
        border: 1px solid #3b82f6; border-radius: 6px;
        padding: 5px 16px; font-size: 13px; font-weight: 600;
        cursor: pointer; font-family: var(--font);
        transition: all 0.15s;
        box-shadow: 0 0 12px rgba(59,130,246,0.3);
      }
      #mc-topbar .tb-right .deploy-btn:hover {
        background: #2563eb; box-shadow: 0 0 18px rgba(59,130,246,0.5);
      }
      #mc-topbar .tb-right .more-btn {
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--border); border-radius: 6px;
        color: #94a3b8; padding: 5px 10px; cursor: pointer; font-size: 16px;
      }

      /* KPI STRIP */
      #mc-kpistrip {
        position: fixed;
        top: var(--top-h); left: var(--left-w); right: 0;
        height: var(--kpi-h);
        background: rgba(8,13,28,0.95);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--border);
        display: flex; align-items: stretch;
        z-index: 9998;
        overflow: hidden;
        pointer-events: auto !important;
      }
      .kpi-cell {
        flex: 1; display: flex; align-items: center; gap: 10px;
        padding: 0 16px;
        border-right: 1px solid var(--border);
        position: relative;
        transition: background 0.15s;
      }
      .kpi-cell:hover { background: rgba(255,255,255,0.03); }
      .kpi-cell:last-child { border-right: none; }
      .kpi-cell .kc-icon {
        width: 32px; height: 32px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; flex-shrink: 0;
      }
      .kpi-cell .kc-text { display: flex; flex-direction: column; gap: 1px; }
      .kpi-cell .kc-label {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em;
        color: #64748b; font-weight: 600; font-family: var(--font);
      }
      .kpi-cell .kc-value {
        font-family: var(--font-head); font-size: 18px; font-weight: 700;
        color: #f1f5f9; line-height: 1; font-variant-numeric: tabular-nums;
      }
      .kpi-cell .kc-value span { font-size: 12px; color: #64748b; font-weight: 400; }
      .kpi-cell .kc-delta {
        font-size: 11px; font-weight: 600; margin-top: 2px;
      }
      .kpi-cell .kc-delta.up { color: var(--green); }
      .kpi-cell .kc-delta.down { color: var(--red); }

      /* LEFT ICON RAIL */
      #mc-leftrail {
        position: fixed; top: var(--top-h); left: 0; bottom: var(--bottom-h);
        width: var(--left-w);
        background: rgba(7,12,28,0.96);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
        display: flex; flex-direction: column; align-items: center;
        padding: 8px 0; gap: 4px;
        z-index: 9997;
      }
      .rail-icon {
        width: 42px; height: 42px; border-radius: 10px;
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; cursor: pointer; position: relative;
        font-size: 18px; color: #475569;
        transition: all 0.15s;
        text-decoration: none;
      }
      .rail-icon:hover { background: rgba(255,255,255,0.07); color: #e2e8f0; }
      .rail-icon.active {
        background: rgba(59,130,246,0.18);
        color: var(--primary);
        box-shadow: inset 0 0 0 1px rgba(59,130,246,0.3);
      }
      .rail-icon .badge {
        position: absolute; top: 4px; right: 4px;
        background: var(--red); color: #fff;
        font-size: 8px; font-weight: 700; font-family: var(--font);
        width: 14px; height: 14px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
      }
      .rail-icon .ri-label {
        font-size: 8px; color: #475569; margin-top: 2px; font-family: var(--font);
      }
      .rail-icon.active .ri-label { color: var(--primary); }
      .rail-icon:hover .ri-label { color: #cbd5e1; }
      .rail-sep { width: 28px; height: 1px; background: var(--border); margin: 4px 0; }

      /* RIGHT TOOL RAIL */
      #mc-rightrail {
        position: fixed;
        top: calc(var(--top-h) + var(--kpi-h) + 16px);
        right: 12px;
        display: flex; flex-direction: column; gap: 4px;
        z-index: 9996;
      }
      .tool-btn {
        width: 44px; height: 44px; border-radius: 10px;
        background: rgba(8,13,28,0.88);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; cursor: pointer;
        font-size: 16px; color: #64748b;
        transition: all 0.15s;
        gap: 2px;
      }
      .tool-btn span.tl { font-size: 8px; color: #475569; font-family: var(--font); }
      .tool-btn:hover {
        background: rgba(18,28,56,0.95);
        border-color: rgba(255,255,255,0.15);
        color: #e2e8f0;
      }
      .tool-btn:hover span.tl { color: #94a3b8; }

      /* EVENTS PANEL */
      #mc-events {
        position: fixed;
        bottom: calc(var(--bottom-h) + 16px);
        left: calc(var(--left-w) + 12px);
        width: 210px;
        background: rgba(7,12,28,0.90);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        z-index: 9995;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        pointer-events: auto !important;
      }
      #mc-events .ev-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 8px;
      }
      #mc-events .ev-title {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: #94a3b8;
      }
      #mc-events .ev-count {
        font-size: 10px; background: rgba(255,255,255,0.08);
        padding: 1px 6px; border-radius: 10px; color: #64748b;
      }
      .ev-item { margin-bottom: 8px; }
      .ev-item .ev-type {
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 2px;
      }
      .ev-item .ev-type.crit { color: var(--red); }
      .ev-item .ev-type.warn { color: var(--amber); }
      .ev-item .ev-msg { font-size: 12px; color: #e2e8f0; font-weight: 500; }
      .ev-item .ev-time {
        font-size: 10px; color: #475569;
        font-family: var(--font-mono); margin-top: 1px;
      }
      .ev-sep { height: 1px; background: var(--border); margin: 6px 0; }
      .ev-viewall {
        font-size: 11px; color: var(--primary); cursor: pointer; font-weight: 500;
      }
      .ev-viewall:hover { text-decoration: underline; }

      /* LEGEND PANEL */
      #mc-legend {
        position: fixed;
        bottom: calc(var(--bottom-h) + 16px);
        right: calc(var(--right-w) + 20px);
        background: rgba(7,12,28,0.88);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        z-index: 9995;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      }
      .lg-item {
        display: flex; align-items: center; gap: 7px;
        font-size: 11px; color: #cbd5e1; margin-bottom: 5px;
      }
      .lg-item:last-child { margin-bottom: 0; }
      .lg-dot {
        width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
      }
      .lg-dash {
        width: 18px; height: 2px; flex-shrink: 0; border-radius: 1px;
      }

      /* BOTTOM TIMELINE */
      #mc-timeline {
        position: fixed; bottom: 0; left: var(--left-w); right: 0;
        height: var(--bottom-h);
        background: rgba(7,12,28,0.97);
        backdrop-filter: blur(20px);
        border-top: 1px solid var(--border);
        display: flex; align-items: center; gap: 10px;
        padding: 0 16px;
        z-index: 9999;
        box-shadow: 0 -2px 20px rgba(0,0,0,0.5);
      }
      .tl-btn {
        width: 32px; height: 32px; border-radius: 6px;
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; color: #94a3b8; cursor: pointer;
        transition: all 0.15s; flex-shrink: 0;
      }
      .tl-btn:hover { background: rgba(255,255,255,0.1); color: #f1f5f9; }
      .tl-btn.play {
        width: 36px; height: 36px; border-radius: 50%;
        background: var(--primary); border-color: #60a5fa; color: #fff;
        font-size: 15px;
        box-shadow: 0 0 14px rgba(59,130,246,0.5);
      }
      .tl-btn.play:hover {
        background: #2563eb; box-shadow: 0 0 20px rgba(59,130,246,0.7);
      }
      .tl-speed {
        display: flex; align-items: center; gap: 4px;
        font-size: 12px; color: #94a3b8;
        background: rgba(255,255,255,0.06); border: 1px solid var(--border);
        padding: 3px 8px; border-radius: 5px; cursor: pointer;
      }
      .tl-live {
        display: flex; align-items: center; gap: 5px;
        font-size: 12px; color: var(--red); font-weight: 700;
      }
      .tl-live .live-dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--red);
        animation: pulse 1.2s infinite;
      }
      .tl-timestamp {
        font-family: var(--font-mono); font-size: 22px; font-weight: 700;
        color: var(--primary); letter-spacing: -0.5px; line-height: 1;
      }
      .tl-timestamp sub {
        font-size: 10px; color: #64748b; display: block; letter-spacing: 0.05em;
      }
      .tl-scrubber {
        flex: 1; display: flex; flex-direction: column; gap: 3px;
        padding: 0 4px;
      }
      .tl-track {
        position: relative; height: 4px; background: rgba(255,255,255,0.08);
        border-radius: 2px; cursor: pointer;
      }
      .tl-fill {
        position: absolute; left: 0; top: 0; height: 100%;
        width: 25%; background: linear-gradient(90deg, var(--primary), #60a5fa);
        border-radius: 2px;
      }
      .tl-handle {
        position: absolute; top: 50%; transform: translateY(-50%);
        left: 25%; width: 12px; height: 12px; border-radius: 50%;
        background: #fff; border: 2px solid var(--primary);
        box-shadow: 0 0 8px rgba(59,130,246,0.6);
        cursor: grab; margin-left: -6px;
      }
      .tl-labels {
        display: flex; justify-content: space-between;
        font-family: var(--font-mono); font-size: 10px; color: #334155;
        padding: 0 2px;
      }
      .tl-labels span.active { color: #64748b; }
      .tl-duration {
        display: flex; align-items: center; gap: 4px;
        font-size: 12px; color: #64748b;
        background: rgba(255,255,255,0.06); border: 1px solid var(--border);
        padding: 3px 8px; border-radius: 5px; cursor: pointer; flex-shrink: 0;
      }
      .tl-cal {
        width: 28px; height: 28px; border-radius: 6px;
        background: rgba(255,255,255,0.06); border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; cursor: pointer; color: #64748b;
      }

      /* Metric cards in Executive Summary */
      .kpi-metric-card {
        flex: 1; min-width: 140px;
        background: linear-gradient(145deg, rgba(18,28,54,0.7), rgba(8,13,28,0.8));
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-top: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 8px 24px -8px rgba(0,0,0,0.6);
        transition: all 0.25s ease;
        position: relative; overflow: hidden;
      }
      .kpi-metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
        background: linear-gradient(to bottom, #0EA5A4, #3b82f6);
      }
      .kpi-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 36px -8px rgba(0,0,0,0.8), 0 0 16px rgba(14,165,164,0.15);
        border-color: rgba(255,255,255,0.18);
      }
      .kpi-metric-label {
        color: #64748b; font-weight: 600; text-transform: uppercase;
        font-size: 0.7rem; letter-spacing: 0.06em; margin-bottom: 4px;
      }
      .kpi-metric-value {
        color: #f1f5f9; font-family: var(--font-head);
        font-weight: 700; font-size: 1.8rem; line-height: 1.1;
        font-variant-numeric: tabular-nums;
      }
      .metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }

      .insight {
        background: linear-gradient(to right, rgba(16,185,129,0.12), rgba(16,185,129,0.01));
        border-left: 3px solid var(--green);
        padding: 12px 16px; border-radius: 0 10px 10px 0;
        font-size: 0.9rem; color: #e2e8f0; line-height: 1.5;
        margin-bottom: 20px;
      }
      .kpi-card {
        background: rgba(10,16,36,0.5); border: 1px solid var(--border);
        border-radius: 16px; padding: 24px; margin-bottom: 24px;
      }

      h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-head) !important;
        color: var(--ink) !important;
        font-weight: 600; letter-spacing: -0.02em;
      }
      p, span, div { color: var(--ink); }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: rgba(18,28,54,0.5);
        backdrop-filter: blur(12px); padding: 5px;
        border-radius: 10px; border: 1px solid var(--border);
      }
      .stTabs [data-baseweb="tab"] {
        height: 40px; border-radius: 7px; padding: 0 16px;
        color: #64748b; font-family: var(--font-head); font-weight: 500;
        background: transparent; transition: all 0.18s;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.15) !important;
        color: var(--primary) !important;
        border: 1px solid rgba(59,130,246,0.3);
      }
    </style>
    """, unsafe_allow_html=True)

    configure_plotly_template()


def render_mission_control_chrome(kpis: dict | None = None,
                                   cnst: str = "kuiper",
                                   strat: str = "farthest_approaching",
                                   show_all: bool = True):
    """
    Inject all fixed-position UI panels as a single HTML block.
    kpis:  optional dict with keys matching KPI cell ids for dynamic values.
    cnst:  active constellation key ('kuiper' | 'oneweb' | 'starlink').
    strat: active strategy key (e.g. 'farthest_approaching').
    show_all: bool to show all satellites.
    """
    served_users  = kpis.get("served_users",  "70")     if kpis else "70"
    median_sinr   = kpis.get("median_sinr",   "10.4")   if kpis else "10.4"
    pct5_sinr     = kpis.get("pct5_sinr",     "5.0")    if kpis else "5.0"
    handovers     = kpis.get("handovers",     "104")     if kpis else "104"
    active_sats   = kpis.get("active_sats",   "516/648") if kpis else "516/648"
    system_load   = kpis.get("system_load",   "62%")     if kpis else "62%"

    # Constellation options
    _CNSTS = [
        ("kuiper", "Kuiper"),
        ("oneweb", "OneWeb"),
        ("starlink", "Starlink"),
    ]
    s_val = 'true' if show_all else 'false'
    
    current_cnst_label = next((lb for k, lb in _CNSTS if k == cnst), "Select Constellation")
    _cnst_opts = "".join(
        f'<a href="?cnst={k}&strat={strat}&show_all={s_val}">{lb}</a>'
        for k, lb in _CNSTS
    )
    _cnst_select = (
        f'<div class="mc-dropdown">'
        f'<div class="mc-dropbtn">{current_cnst_label} ▼</div>'
        f'<div class="mc-dropdown-content">{_cnst_opts}</div>'
        f'</div>'
    )

    # Algorithm options
    _STRATS = [
        ("max_sinr",            "Max-SINR (greedy)"),
        ("persistent_nearest",  "Persistent nearest"),
        ("farthest_approaching","Farthest-approaching"),
        ("max_elevation",       "Max elevation"),
    ]
    
    current_strat_label = next((lb for k, lb in _STRATS if k == strat), "Select Strategy")
    
    _dropdown_css = """
    <style>
      .mc-dropdown { position: relative; display: inline-block; cursor: pointer; z-index: 1000; }
      .mc-dropbtn {
        background: rgba(15,23,42,0.8); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.15);
        border-radius: 6px; padding: 4px 8px; font-size: 12px;
      }
      .mc-dropdown-content {
        display: none; position: absolute; background-color: #1e293b; min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 1001; border-radius: 6px; border: 1px solid var(--border);
        top: 100%; left: 0;
      }
      .mc-dropdown-content a {
        color: #e2e8f0; padding: 8px 12px; text-decoration: none; display: block; font-size: 12px;
      }
      .mc-dropdown-content a:hover { background-color: #334155; }
      .mc-dropdown:hover .mc-dropdown-content { display: block; }
      .mc-checkbox {
        display: flex; align-items: center; cursor: pointer; text-decoration: none; color: inherit;
      }
      .mc-checkbox-box {
        width: 14px; height: 14px; border: 1px solid rgba(255,255,255,0.3); border-radius: 3px;
        margin-right: 6px; display: flex; align-items: center; justify-content: center;
      }
      .mc-checkbox.checked .mc-checkbox-box::after { content: '✓'; font-size: 10px; color: #0EA5A4; }
    </style>
    """
    
    _algo_opts = "".join(
        f'<a href="?cnst={cnst}&strat={k}&show_all={s_val}">{lb}</a>'
        for k, lb in _STRATS
    )
    
    _algo_select = (
        f'<div class="mc-dropdown">'
        f'<div class="mc-dropbtn">{current_strat_label} ▼</div>'
        f'<div class="mc-dropdown-content">{_algo_opts}</div>'
        f'</div>'
    )
    
    toggle_s_val = 'false' if show_all else 'true'
    chk_class = "mc-checkbox checked" if show_all else "mc-checkbox"
    kpi_html = ""
    if kpis:
        # Map common KPI terms to icons and background colors
        _icon_map = {
            'user': ('👤', 'rgba(100,116,139,0.15)'),
            'sinr': ('📶', 'rgba(59,130,246,0.15)'),
            'snr': ('📶', 'rgba(59,130,246,0.15)'),
            'outage': ('⚠', 'rgba(239,68,68,0.15)'),
            'handover': ('🔄', 'rgba(245,158,11,0.15)'),
            'sat': ('🛰️', 'rgba(139,92,246,0.15)'),
            'load': ('⚡', 'rgba(236,72,153,0.15)'),
            'elev': ('🔭', 'rgba(16,185,129,0.15)'),
            'signal': ('📡', 'rgba(16,185,129,0.15)'),
            'noise': ('〰', 'rgba(100,116,139,0.15)'),
        }
        
        for k, v in kpis.items():
            k_lower = k.lower()
            icon = '📊'
            bg = 'rgba(255,255,255,0.1)'
            for term, icon_info in _icon_map.items():
                if term in k_lower:
                    icon, bg = icon_info
                    break
                    
            label = k.replace('_', ' ').title()
            
            # Split value and unit for styling
            parts = str(v).strip().split(' ', 1)
            if len(parts) > 1:
                val = parts[0]
                unit = f" <span>{parts[1]}</span>"
            else:
                val = parts[0]
                unit = ""
                
            kpi_html += f'''
<div class="kpi-cell">
  <div class="kc-icon" style="background:{bg};">{icon}</div>
  <div class="kc-text">
    <div class="kc-label">{label}</div>
    <div class="kc-value">{val}{unit}</div>
  </div>
</div>
'''
            
    html_content = _dropdown_css + f"""
<!-- ═══════════════ TOP BAR ═══════════════ -->
<div id="mc-topbar">
  <div class="tb-left">
    <div class="tb-logo">SkyNTN</div>
    <div class="tb-mode">
      <div class="dot"></div>
      Operations Mode
    </div>
  </div>

  <div class="tb-center">
    <span class="filter-label">Constellation</span>
    {_cnst_select}
    <div class="sep"></div>
    <span class="filter-label">Handover Algorithm</span>
    {_algo_select}
    <div class="show-all">
      <a href="?cnst={cnst}&strat={strat}&show_all={toggle_s_val}" class="{chk_class}">
        <div class="mc-checkbox-box"></div>
        <label style="cursor:pointer;">Show ALL satellites</label>
      </a>
    </div>
  </div>

  <div class="tb-right">
    <span class="timestamp">May 4, 2026 00:05:20 UTC</span>
    <span class="live-badge">● Live</span>
    <button class="deploy-btn">Deploy</button>
    <button class="more-btn">⋮</button>
  </div>
</div>

<!-- ═══════════════ KPI STRIP ═══════════════ -->
<div id="mc-kpistrip">
{kpi_html}
</div>



"""
    st.markdown(html_content, unsafe_allow_html=True)


def render_app_header(title: str, subtitle: str):
    """Renders the common application header."""
    st.markdown(f'''
        <h2 style="margin-bottom:0;">{title}</h2>
        <p style="color:#64748b;margin-top:0;font-size:0.9rem;">{subtitle}</p>
        <hr style="height:1px;border:none;margin:.4rem 0 1rem;
                   background:linear-gradient(90deg,#3b82f6,#10b981,transparent);
                   border-radius:3px;opacity:0.4;"/>
    ''', unsafe_allow_html=True)


def configure_plotly_template():
    PRO_PALETTE = ["#0EA5A4", "#3b82f6", "#8b5cf6", "#f43f5e", "#f59e0b",
                   "#10b981", "#db2777", "#6366f1", "#ea580c", "#2dd4bf"]
    pio.templates["skyntn"] = go.layout.Template(
        layout=dict(
            colorway=PRO_PALETTE,
            font=dict(family="Inter, Segoe UI, Helvetica, Arial", size=13, color="#cbd5e1"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(font=dict(size=16, color="#f1f5f9", family="Outfit"), x=0.01, xanchor="left"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                       linecolor="rgba(255,255,255,0.08)", ticks="outside", ticklen=4,
                       title=dict(font=dict(size=12, color="#64748b"))),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                       linecolor="rgba(255,255,255,0.08)", ticks="outside", ticklen=4,
                       title=dict(font=dict(size=12, color="#64748b"))),
            legend=dict(bgcolor="rgba(8,13,28,0.7)", bordercolor="rgba(255,255,255,0.08)",
                        borderwidth=1, orientation="h", yanchor="bottom", y=1.04,
                        xanchor="right", x=1, font=dict(color="#f1f5f9", size=12)),
            margin=dict(l=48, r=16, t=52, b=36),
            hoverlabel=dict(bgcolor="rgba(8,13,28,0.95)", font=dict(family="Inter", size=12, color="#f1f5f9"),
                            bordercolor="rgba(255,255,255,0.15)"),
            hovermode="x unified",
        )
    )
    pio.templates.default = "skyntn"
