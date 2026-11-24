# Copyright (c) 2026 Mohammed Al-Mekhlafi, Email: mhdalmekhlafi@gmail.com, mohammed.al-mekhlafi@polymtl.ca
"""Generate a standalone CesiumJS 3D visualization HTML file.

The page must be served over http:// (not file://): Cesium's tile web workers
refuse to load from a file:// origin, so this script starts a local HTTP server.
"""
from __future__ import annotations
import argparse, json, os, sys, webbrowser
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent          # .../ntnconnect/visualization
_V1   = _HERE.parents[1]                          # .../v1
if str(_HERE) not in sys.path: sys.path.insert(0, str(_HERE))
try:
    from ntnconnect.analysis.visualization.czml import build_czml_json, build_tick_stats, build_all_sats_czml_json
except ImportError:
    from czml import build_czml_json, build_tick_stats, build_all_sats_czml_json

_OP_LABEL = {"oneweb": "OneWeb", "starlink": "Starlink", "kuiper": "Kuiper"}
_STRAT_LABEL = {
    "max_sinr": "Max-RSRP (nearest)",
    "persistent_nearest": "Persistent nearest",
    "farthest_approaching": "Farthest-approaching",
    "max_elevation": "Max elevation",
}

CESIUM_HTML = r"""<!DOCTYPE html>
<!-- Copyright (c) 2026 Mohammed Al-Mekhlafi, Email: mhdalmekhlafi@gmail.com, mohammed.al-mekhlafi@polymtl.ca -->
<html lang="en">
<head>
<meta charset="utf-8">
<title>SkyNTN Live</title>
<script src="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
  html, body, #cesiumContainer {
    width:100%; height:100%; margin:0; padding:0; overflow:hidden;
    background:#01030a;
    font-family:"Inter","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  /* Cinematic vignette + top/bottom gradient bars for framing */
  #vignette {
    position:absolute; inset:0; z-index:5; pointer-events:none;
    box-shadow: inset 0 0 240px 40px rgba(0,0,0,0.55);
    background:
      linear-gradient(180deg, rgba(2,6,16,0.55) 0%, rgba(0,0,0,0) 16%,
                      rgba(0,0,0,0) 84%, rgba(2,6,16,0.6) 100%);
  }
  #hud {
    position:absolute; top:14px; left:14px; z-index:999; min-width:240px;
    padding:14px 16px; background:rgba(10,16,30,0.55);
    backdrop-filter:blur(9px) saturate(140%);
    -webkit-backdrop-filter:blur(9px) saturate(140%);
    border:1px solid rgba(120,170,255,0.25); border-radius:14px;
    box-shadow:0 6px 26px rgba(0,0,0,0.45); color:#eaf2ff;
  }
  #hud .title { font-size:16px; font-weight:700; letter-spacing:.3px;
    background:linear-gradient(90deg,#6fd3ff,#9be7a8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:2px; }
  #hud .sub { font-size:11px; color:#9fb4d8; margin-bottom:10px; }
  #hud .row { display:flex; justify-content:space-between; align-items:baseline; margin:5px 0; }
  #hud .k { font-size:11px; color:#9fb4d8; }
  #hud .v { font-size:20px; font-weight:700; font-variant-numeric:tabular-nums; }
  #hud .v.sm { font-size:14px; }
  #clk { font-size:12px; color:#7fd0ff; margin-top:8px; font-variant-numeric:tabular-nums; }
  /* ── Control panel (top-right): constellation + algorithm selectors ── */
  #controls {
    position:absolute; top:14px; right:14px; z-index:1000; min-width:230px;
    padding:14px 16px; background:rgba(10,16,30,0.62);
    backdrop-filter:blur(9px) saturate(140%);
    -webkit-backdrop-filter:blur(9px) saturate(140%);
    border:1px solid rgba(120,170,255,0.28); border-radius:14px;
    box-shadow:0 6px 26px rgba(0,0,0,0.45); color:#eaf2ff;
  }
  #controls .hd { font-size:12px; font-weight:700; letter-spacing:.4px;
    text-transform:uppercase; color:#8fd3ff; margin-bottom:10px; }
  #controls label { display:block; font-size:11px; color:#9fb4d8;
    margin:10px 0 4px; }
  #controls select {
    width:100%; padding:8px 10px; font-size:13px; color:#eaf2ff;
    background:rgba(20,30,52,0.95); border:1px solid rgba(120,170,255,0.35);
    border-radius:9px; outline:none; cursor:pointer;
    font-family:inherit; appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,#8fd3ff 50%),
                     linear-gradient(135deg,#8fd3ff 50%,transparent 50%);
    background-position:calc(100% - 18px) calc(50% - 2px),
                        calc(100% - 13px) calc(50% - 2px);
    background-size:5px 5px,5px 5px; background-repeat:no-repeat;
  }
  #controls select:hover { border-color:rgba(140,200,255,0.7); }
  #controls .seg { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; }
  #controls .chip {
    flex:1 1 auto; text-align:center; padding:7px 8px; font-size:12px;
    color:#cfe0ff; background:rgba(20,30,52,0.8);
    border:1px solid rgba(120,170,255,0.3); border-radius:9px; cursor:pointer;
    transition:all .12s; white-space:nowrap;
  }
  #controls .chip:hover { border-color:rgba(140,200,255,0.8); }
  #controls .chip.active {
    background:linear-gradient(90deg,#2a6df0,#36c3a0); color:#fff;
    border-color:transparent; font-weight:700;
    box-shadow:0 3px 12px rgba(40,120,240,0.4);
  }
  #legend { position:absolute; top:292px; left:calc(var(--mc-left) + 16px); z-index:999;
    padding:10px 13px; font-size:11px; color:#dce8ff; white-space:nowrap;
    background:rgba(10,16,30,0.6); backdrop-filter:blur(8px);
    border:1px solid rgba(120,170,255,0.2); border-radius:12px; }
  /* ════════ MISSION-CONTROL CHROME (matches reference design) ════════ */
  :root{ --mc-top:60px; --mc-left:64px; --mc-bottom:30px; --mc-ink:#eaf2ff;
         --mc-mut:#9fb4d8; --mc-line:rgba(120,170,255,0.16); --mc-pri:#3b82f6; }
  /* shift the globe so it sits inside the chrome frame */
  #cesiumContainer{ position:absolute; top:0; left:0; right:0; bottom:0; }
  /* ── full-width TOP BAR ── */
  #mctop{ position:absolute; top:0; left:0; right:0; height:var(--mc-top); z-index:1100;
    display:flex; align-items:center; gap:14px; padding:0 16px;
    background:rgba(7,12,26,0.92); backdrop-filter:blur(18px) saturate(150%);
    border-bottom:1px solid var(--mc-line); box-shadow:0 2px 20px rgba(0,0,0,0.5); }
  #mctop .brand{ display:flex; align-items:center; gap:9px; min-width:172px; }
  #mctop .brand .logo{ font-size:20px; }
  #mctop .brand .bt{ display:flex; flex-direction:column; line-height:1.1; }
  #mctop .brand .bt b{ font-size:15px; font-weight:800; color:#fff; letter-spacing:.2px; }
  #mctop .brand .bt small{ font-size:10px; color:var(--mc-mut); }
  #mctop .live{ display:flex; align-items:center; gap:5px; font-size:11px;
    color:#37d67a; font-weight:700; }
  #mctop .live .d{ width:7px; height:7px; border-radius:50%; background:#37d67a;
    box-shadow:0 0 7px #37d67a; animation:mcpulse 1.6s infinite; }
  @keyframes mcpulse{ 0%,100%{opacity:1} 50%{opacity:.45} }
  #mctop .simclk{ font-size:11px; color:var(--mc-mut); font-variant-numeric:tabular-nums; }
  #mctop .kpis{ display:flex; gap:8px; margin-left:6px; }
  .mck{ min-width:118px; padding:6px 12px; border-radius:10px;
    background:rgba(18,28,52,0.6); border:1px solid var(--mc-line); }
  .mck .l{ font-size:9px; letter-spacing:.07em; text-transform:uppercase;
    color:var(--mc-mut); font-weight:700; }
  .mck .vv{ font-size:18px; font-weight:800; color:#fff; line-height:1.1;
    font-variant-numeric:tabular-nums; }
  .mck .vv small{ font-size:11px; color:var(--mc-mut); font-weight:500; }
  #mctop .spacer{ flex:1; }
  #mctop .filters{ display:flex; align-items:center; gap:10px;
    border:1px solid rgba(120,170,255,0.3); border-radius:10px; padding:5px 12px;
    background:rgba(255,255,255,0.03); }
  #mctop .filters .fl{ font-size:9px; text-transform:uppercase; letter-spacing:.06em;
    color:var(--mc-mut); }
  /* constellation pills + algorithm dropdown inside the top bar */
  #mctop .filters .seg{ display:flex; gap:5px; }
  #mctop .filters .chip{ padding:5px 11px; font-size:12px; font-weight:600; cursor:pointer;
    color:#cfe0ff; background:rgba(20,30,52,0.85); border:1px solid rgba(120,170,255,0.28);
    border-radius:8px; white-space:nowrap; transition:.12s; }
  #mctop .filters .chip:hover{ border-color:rgba(140,200,255,0.7); color:#fff; }
  #mctop .filters .chip.active{ background:linear-gradient(90deg,#2563eb,#3b82f6); color:#fff;
    border-color:transparent; box-shadow:0 2px 10px rgba(40,120,240,0.4); }
  #mctop .filters select{ padding:6px 30px 6px 11px; font-size:12px; color:#eaf2ff; cursor:pointer;
    background:rgba(20,30,52,0.95); border:1px solid rgba(120,170,255,0.3); border-radius:8px;
    outline:none; font-family:inherit; appearance:none;
    background-image:linear-gradient(45deg,transparent 50%,#8fd3ff 50%),
                     linear-gradient(135deg,#8fd3ff 50%,transparent 50%);
    background-position:calc(100% - 16px) calc(50% - 2px),calc(100% - 11px) calc(50% - 2px);
    background-size:5px 5px,5px 5px; background-repeat:no-repeat; }
  #mctop .filters select:hover{ border-color:rgba(140,200,255,0.7); }
  #mctop .icbtn{ width:34px; height:34px; border-radius:9px; cursor:pointer;
    display:flex; align-items:center; justify-content:center; font-size:16px;
    color:var(--mc-mut); background:rgba(255,255,255,0.04);
    border:1px solid var(--mc-line); position:relative; }
  #mctop .icbtn:hover{ color:#fff; background:rgba(255,255,255,0.09); }
  #mctop .icbtn .bdg{ position:absolute; top:-4px; right:-4px; background:#ef4444;
    color:#fff; font-size:9px; font-weight:700; width:15px; height:15px;
    border-radius:50%; display:flex; align-items:center; justify-content:center; }
  #mctop .avatar{ width:34px; height:34px; border-radius:9px; cursor:pointer; overflow:hidden;
    background:#fff; display:flex; align-items:center; justify-content:center; }
  #mctop .avatar img{ width:100%; height:100%; object-fit:contain; padding:3px; box-sizing:border-box; }
  #mctop .avatar.brandtxt{ background:linear-gradient(135deg,#0d2f6b,#1e63b8);
    color:#fff; font-weight:800; font-size:12px; letter-spacing:0.5px; }
  /* ── About / contact overlay ── */
  #mcabout{ position:fixed; inset:0; z-index:4000; display:none;
    background:rgba(2,5,14,0.62); backdrop-filter:blur(3px); }
  #mcabout .abcard{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:340px; background:rgba(10,15,32,0.98); border:1px solid rgba(120,170,255,0.3);
    border-radius:16px; padding:22px 24px; box-shadow:0 24px 70px rgba(0,0,0,0.7);
    font-family:Inter,Segoe UI,sans-serif; color:#cfe0ff; }
  #mcabout .abx{ position:absolute; top:12px; right:14px; cursor:pointer; color:#7f93b8; font-size:14px; }
  #mcabout .abx:hover{ color:#fff; }
  #mcabout .ablogo{ display:block; width:210px; max-width:100%; background:#fff;
    border-radius:8px; padding:12px 14px; box-sizing:border-box; margin-bottom:16px; }
  #mcabout .abtitle{ font-size:18px; font-weight:700; color:#fff; }
  #mcabout .absub{ font-size:11px; color:#7f93b8; }
  #mcabout .abtheme{ font-size:12.5px; font-style:italic; color:#5aa9ff; margin:6px 0 10px; }
  #mcabout .abrow{ font-size:13px; margin:6px 0; color:#c7d6f0; }
  #mcabout .abrow b{ color:#fff; }
  #mcabout .abmail{ margin:6px 0; }
  #mcabout .abmail a{ font-size:13px; color:#5aa9ff; text-decoration:none; }
  #mcabout .abmail a:hover{ color:#8ecbff; text-decoration:underline; }
  #mcabout .abhint{ font-size:10.5px; color:#7f93b8; margin-top:12px; }
  #mcabout .abcopy{ font-size:10px; color:#5f7093; margin-top:10px;
    border-top:1px solid rgba(120,170,255,0.12); padding-top:10px; }
  /* ── left icon NAV rail ── */
  #mcnav{ position:absolute; top:var(--mc-top); left:0; bottom:var(--mc-bottom);
    width:var(--mc-left); z-index:1090; background:rgba(7,12,26,0.92);
    backdrop-filter:blur(18px); border-right:1px solid var(--mc-line);
    display:flex; flex-direction:column; align-items:center; padding:10px 0; gap:3px; }
  .mcni{ width:50px; padding:7px 0; border-radius:11px; cursor:pointer;
    display:flex; flex-direction:column; align-items:center; gap:3px;
    color:#5b6b86; font-size:18px; position:relative; transition:.15s; }
  .mcni small{ font-size:8.5px; }
  .mcni:hover{ background:rgba(255,255,255,0.06); color:#cfe0ff; }
  .mcni.on{ background:rgba(59,130,246,0.16); color:#6fb1ff;
    box-shadow:inset 2px 0 0 var(--mc-pri); }
  .mcni .bdg{ position:absolute; top:3px; right:8px; background:#ef4444; color:#fff;
    font-size:8px; font-weight:700; width:14px; height:14px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; }
  /* ── Layer Summary panel ── */
  #mclayers{ position:absolute; top:78px; left:calc(var(--mc-left) + 16px); z-index:1000;
    width:186px; padding:12px 14px; color:var(--mc-ink);
    background:rgba(10,16,30,0.6); backdrop-filter:blur(10px);
    border:1px solid var(--mc-line); border-radius:13px; }
  #mclayers .hd{ font-size:10px; letter-spacing:.08em; text-transform:uppercase;
    color:var(--mc-mut); font-weight:700; margin-bottom:9px; }
  #mclayers .lr{ display:flex; align-items:center; justify-content:space-between;
    margin:6px 0; font-size:12.5px; }
  #mclayers .lr .lf{ display:flex; align-items:center; gap:8px; color:#cfe0ff; }
  #mclayers .lr .dot{ width:9px; height:9px; border-radius:50%; box-shadow:0 0 6px currentColor; }
  #mclayers .lr .sq{ width:9px; height:9px; border-radius:2px; }
  #mclayers .lr b{ font-variant-numeric:tabular-nums; font-weight:800; }
  #mclayers .show{ margin-top:8px; font-size:11px; color:#6fb1ff; cursor:pointer; }
  /* ── right tool rail ── */
  #mctools{ position:absolute; top:96px; right:18px; z-index:1000;
    display:flex; flex-direction:column; gap:6px; }
  /* legend sits below the toolrail */
  .mctool{ width:42px; height:42px; border-radius:11px; cursor:pointer;
    display:flex; align-items:center; justify-content:center; font-size:17px;
    color:var(--mc-mut); background:rgba(10,16,30,0.7); backdrop-filter:blur(10px);
    border:1px solid var(--mc-line); transition:.15s; }
  .mctool:hover{ color:#fff; background:rgba(40,120,240,0.4); }
  .mctool.on{ color:#6fb1ff; border-color:rgba(59,130,246,0.6);
    background:rgba(59,130,246,0.18); }
  /* layers popover (anchored left of the tool rail) */
  #mclayerpop{ position:absolute; top:96px; right:72px; z-index:1100; display:none;
    width:180px; padding:11px 13px; color:var(--mc-ink);
    background:rgba(10,16,30,0.96); backdrop-filter:blur(12px);
    border:1px solid rgba(120,170,255,0.3); border-radius:12px;
    box-shadow:0 10px 30px rgba(0,0,0,0.55); }
  #mclayerpop .lp-hd{ font-size:10px; letter-spacing:.08em; color:var(--mc-mut);
    font-weight:700; margin-bottom:8px; }
  #mclayerpop .lp-row{ display:flex; align-items:center; gap:8px; font-size:12.5px;
    color:#cfe0ff; margin:7px 0; cursor:pointer; }
  #mclayerpop input{ accent-color:#3b82f6; width:14px; height:14px; }
  /* measure readout */
  #mcmeasure{ position:absolute; bottom:54px; left:50%; transform:translateX(-50%);
    z-index:1100; display:none; padding:8px 16px; font-size:13px; font-weight:600;
    color:#eaf2ff; background:rgba(20,30,52,0.95); border:1px solid rgba(120,170,255,0.4);
    border-radius:10px; }
  /* ── selected-entity inspector ── */
  #mcsel{ position:absolute; bottom:54px; right:18px; z-index:1000; width:230px;
    padding:13px 15px; color:var(--mc-ink); display:none;
    background:rgba(10,16,30,0.78); backdrop-filter:blur(12px);
    border:1px solid rgba(59,130,246,0.4); border-radius:13px;
    box-shadow:0 10px 30px rgba(0,0,0,0.55); }
  #mcsel .hd{ display:flex; align-items:center; gap:7px; font-size:11px;
    color:var(--mc-mut); margin-bottom:9px; }
  #mcsel .hd b{ color:#fff; font-size:13px; }
  #mcsel .sr{ display:flex; justify-content:space-between; margin:6px 0; font-size:12.5px; }
  #mcsel .sr .k{ color:var(--mc-mut); } #mcsel .sr .v{ font-weight:700; color:#cfe0ff; }
  #mcsel .x{ position:absolute; top:9px; right:11px; cursor:pointer; color:var(--mc-mut); }
  /* ── status bar ── */
  #mcstat{ position:absolute; left:0; right:0; bottom:0; height:var(--mc-bottom);
    z-index:1100; display:flex; align-items:center; gap:22px; padding:0 16px;
    font-size:11px; color:var(--mc-mut); background:rgba(7,12,26,0.95);
    border-top:1px solid var(--mc-line); font-variant-numeric:tabular-nums; }
  #mcstat .ok{ color:#37d67a; font-weight:700; }
  #mcstat .sp{ flex:1; }
  /* lift Cesium's native timeline + playback above the status bar, clear of the nav rail */
  .cesium-viewer-bottom{ display:none !important; }
  .cesium-viewer-timelineContainer{ left:calc(var(--mc-left) + 84px) !important;
    right:0 !important; bottom:var(--mc-bottom) !important; }
  .cesium-viewer-animationContainer{ bottom:calc(var(--mc-bottom) + 6px) !important;
    left:calc(var(--mc-left) + 8px) !important; }
  .cesium-viewer-fullscreenContainer{ display:none !important; }
  /* nav action toast */
  #mctoast{ position:absolute; left:50%; top:calc(var(--mc-top) + 14px);
    transform:translateX(-50%); z-index:1200; padding:9px 16px; font-size:12.5px;
    color:#eaf2ff; background:rgba(20,30,52,0.95); border:1px solid rgba(120,170,255,0.35);
    border-radius:10px; box-shadow:0 8px 24px rgba(0,0,0,0.5); opacity:0;
    transition:opacity .25s; pointer-events:none; }
  /* analytics overlay panel */
  #mcpanel{ position:absolute; top:calc(var(--mc-top) + 14px);
    left:calc(var(--mc-left) + 14px); right:14px; bottom:calc(var(--mc-bottom) + 14px);
    z-index:1300; display:none; flex-direction:column;
    background:rgba(8,13,28,0.97); backdrop-filter:blur(16px);
    border:1px solid rgba(120,170,255,0.28); border-radius:16px;
    box-shadow:0 20px 60px rgba(0,0,0,0.7); }
  #mcpanel .mcp-hd{ display:flex; align-items:center; justify-content:space-between;
    padding:14px 20px; border-bottom:1px solid var(--mc-line); }
  #mcpanel .mcp-hd b{ font-size:16px; color:#fff; font-weight:700; }
  #mcpanel .mcp-x{ cursor:pointer; font-size:13px; color:var(--mc-mut);
    padding:5px 12px; border:1px solid var(--mc-line); border-radius:8px; }
  #mcpanel .mcp-x:hover{ color:#fff; background:rgba(255,255,255,0.07); }
  #mcp-kpis{ display:flex; flex-wrap:wrap; gap:6px; flex:1; margin:0 16px; }
  #mcp-kpis .kbtn{ padding:5px 11px; font-size:11.5px; cursor:pointer; color:#cfe0ff;
    background:rgba(20,30,52,0.85); border:1px solid rgba(120,170,255,0.25);
    border-radius:8px; white-space:nowrap; transition:.12s; }
  #mcp-kpis .kbtn:hover{ border-color:rgba(140,200,255,0.7); color:#fff; }
  #mcp-kpis .kbtn.on{ background:linear-gradient(90deg,#2563eb,#3b82f6); color:#fff;
    border-color:transparent; font-weight:600; }
  #mcp-kpis .kgrp{ align-self:center; font-size:9.5px; font-weight:700; color:#7f93b8;
    text-transform:uppercase; letter-spacing:0.6px; padding:0 4px; }
  #mcp-kpis .kgrp:not(:first-child){ margin-left:6px; padding-left:12px;
    border-left:1px solid rgba(120,170,255,0.25); }
  #mcp-kpis .kpiSelLab{ align-self:center; font-size:10px; font-weight:700; color:#7f93b8;
    text-transform:uppercase; letter-spacing:0.6px; margin-right:8px; }
  #mcp-kpis .kpiSel{ align-self:center; background:rgba(20,30,52,0.92); color:#e6eeff;
    border:1px solid rgba(120,170,255,0.32); border-radius:9px; padding:8px 14px;
    font-size:12.5px; font-family:inherit; cursor:pointer; outline:none;
    min-width:320px; max-width:560px; }
  #mcp-kpis .kpiSel:hover{ border-color:rgba(140,200,255,0.7); }
  #mcp-kpis .kpiSel optgroup{ background:#0e1628; color:#7f93b8; font-weight:700; }
  #mcp-kpis .kpiSel option{ background:#0e1628; color:#e6eeff; font-weight:400; }
  #mcp-chart{ flex:1; padding:10px 14px 14px; min-height:0; }
  /* Reports = dashboard: a grid of every KPI at once (small multiples). */
  #mcp-dash{ flex:1; display:none; overflow:auto; padding:14px 16px 18px; }
  #mcp-dash .dgrid{ display:grid; gap:12px;
    grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); }
  #mcp-dash .dcard{ background:rgba(14,22,42,0.75);
    border:1px solid rgba(120,170,255,0.20); border-radius:12px;
    padding:8px 10px 4px; display:flex; flex-direction:column; }
  #mcp-dash .dcard .dttl{ font-size:12px; color:#cfe0ff; font-weight:600;
    margin:2px 4px 4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #mcp-dash .dcard .dch{ height:220px; }
  #mcp-dash .dnote{ color:#9fb4d8; padding:24px; }
  /* Export toolbar (Reports header, left of Close). */
  #mcp-exp{ display:none; gap:8px; align-items:center; margin-right:10px; }
  #mcp-exp .xbtn{ padding:5px 11px; font-size:11.5px; cursor:pointer; color:#cfe0ff;
    background:rgba(20,30,52,0.85); border:1px solid rgba(120,170,255,0.25);
    border-radius:8px; white-space:nowrap; transition:.12s; }
  #mcp-exp .xbtn:hover{ border-color:rgba(140,200,255,0.7); color:#fff; }
  #mcp-dash .drow{ display:flex; align-items:center; justify-content:space-between;
    gap:12px; margin-bottom:12px; flex-wrap:wrap; }
  #mcp-dash .rchips{ display:flex; gap:6px; }
  #mcp-dash .rchip{ padding:5px 13px; font-size:12px; cursor:pointer; color:#cfe0ff;
    background:rgba(20,30,52,0.85); border:1px solid rgba(120,170,255,0.25); border-radius:8px; }
  #mcp-dash .rchip.on{ background:linear-gradient(90deg,#2563eb,#3b82f6); color:#fff;
    border-color:transparent; font-weight:600; }
  #mcp-dash .ctxstrip{ display:flex; gap:8px; flex-wrap:wrap; }
  #mcp-dash .ctile{ background:rgba(20,30,52,0.7); border:1px solid rgba(120,170,255,0.15);
    border-radius:8px; padding:5px 10px; min-width:74px; }
  #mcp-dash .ctile .cl{ font-size:9.5px; color:#7f93b8; }
  #mcp-dash .ctile .cv{ font-size:14px; color:#e6eeff; font-weight:600; }
  #mcp-dash .ctile .cu{ font-size:9.5px; color:#7f93b8; font-weight:400; }
  #mcp-dash .dtab{ width:100%; border-collapse:collapse; font-size:11.5px; }
  #mcp-dash .dtab .dhr th{ color:#9fb4d8; font-weight:500; text-align:right; padding:3px 5px; }
  #mcp-dash .dtab .dhr th.lft{ text-align:left; }
  #mcp-dash .dtab .ku{ font-size:9px; color:#7f93b8; }
  #mcp-dash .dtab td{ text-align:right; padding:4px 5px; }
  #mcp-dash .dtab td.sn{ text-align:left; color:#e6eeff; }
  #mcp-dash .dhint{ font-size:9.5px; color:#7f93b8; margin-top:6px; }
  #mcp-dash .rsel{ background:rgba(20,30,52,0.9); color:#cfe0ff; font-size:10.5px;
    border:1px solid rgba(120,170,255,0.25); border-radius:6px; padding:2px 4px; }
  #mcp-dash .dins{ font-size:11px; line-height:1.55; color:#c7d6f0; }
  #mcp-dash .dins > div{ margin-bottom:7px; }
  #mcp-dash .sgrid{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  #mcp-dash .stile{ border-radius:8px; padding:8px 10px; }
  #mcp-dash .stile .sl{ font-size:9.5px; }
  #mcp-dash .stile .sv{ font-size:18px; font-weight:600; margin-top:2px; }
  #mcp-dash{ overflow:hidden; padding:12px 14px; }
  #mcp-dash .hgrid{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:10px; }
  #mcp-dash .htile{ border-radius:12px; padding:9px 12px; color:#fff; }
  #mcp-dash .htile.g1{ background:linear-gradient(135deg,#6d5cff,#a34bff); }
  #mcp-dash .htile.g2{ background:linear-gradient(135deg,#22d3ee,#0ea5e9); }
  #mcp-dash .htile.g3{ background:linear-gradient(135deg,#1fd18a,#0ea55e); }
  #mcp-dash .htile.g4{ background:linear-gradient(135deg,#f472b6,#a34bff); }
  #mcp-dash .htile .hl{ font-size:10.5px; opacity:.92; }
  #mcp-dash .htile .hv{ font-size:22px; font-weight:700; line-height:1.1; }
  #mcp-dash .htile .hv .hu{ font-size:11px; font-weight:500; }
  #mcp-dash .htile .hs{ font-size:9.5px; opacity:.85; }
  #mcp-dash .dgrid2{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }
  #mcp-dash .dgrid2.wide{ grid-template-columns:1.5fr 1fr; }
  #mcp-dash .dcard{ padding:8px 10px 6px; }
  #mcp-dash .pch{ height:158px; }
  #mcp-dash .dttl.clk{ cursor:pointer; user-select:none; }
  #mcp-dash .dttl.clk:hover{ color:#fff; }
  #mcp-over{ flex:1; display:none; overflow:auto; padding:16px 18px; }
  #mcp-over .ostrip{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }
  #mcp-over .ofact{ background:rgba(20,30,52,0.7); border:1px solid rgba(120,170,255,0.18);
    border-radius:12px; padding:10px 14px; }
  #mcp-over .ofact .ol{ font-size:10.5px; color:#7f93b8; margin-bottom:3px; }
  #mcp-over .ofact .ov{ font-size:13px; color:#e6eeff; font-weight:600; line-height:1.3; }
  #mcp-over .ogrid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }
  #mcp-over .ocard{ background:rgba(14,22,42,0.8); border:1px solid rgba(120,170,255,0.22);
    border-radius:14px; padding:14px 16px; }
  #mcp-over .och{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
  #mcp-over .ocn{ font-size:17px; font-weight:700; color:#fff; }
  #mcp-over .opill{ font-size:11px; font-weight:600; padding:3px 10px; border-radius:20px; }
  #mcp-over .opill.ok{ background:#12492f; color:#7ff0b8; }
  #mcp-over .opill.bad{ background:#4a1f1f; color:#ff9a9a; }
  #mcp-over .omet{ display:flex; align-items:baseline; justify-content:space-between;
    padding:7px 0; border-top:1px solid rgba(120,170,255,0.1); }
  #mcp-over .omet .ok2{ font-size:11px; color:#9fb4d8; }
  #mcp-over .omet .ov2{ font-size:16px; font-weight:600; color:#e6eeff; }
  #mcp-over .omet .ov2 span{ font-size:11px; font-weight:400; color:#7f93b8; }
  #mcp-over .omet .os2{ font-size:10px; color:#7f93b8; min-width:74px; text-align:right; }
  #mcp-over .ohint{ font-size:11px; color:#7f93b8; margin-top:16px; }
  #mcp-over .ohint b{ color:#cfe0ff; }
  #mcp-over .osrow2{ display:grid; grid-template-columns:2fr 1fr; gap:14px; margin-top:14px; }
  #mcp-over .osettings{ background:rgba(14,22,42,0.8); border:1px solid rgba(120,170,255,0.22);
    border-radius:14px; padding:12px 16px; }
  #mcp-over .osh{ font-size:12px; font-weight:700; color:#cfe0ff; margin-bottom:10px;
    text-transform:uppercase; letter-spacing:0.5px; }
  #mcp-over .osgrid{ display:grid; grid-template-columns:1fr 1fr; gap:5px 20px; }
  #mcp-over .osrow{ display:flex; justify-content:space-between; align-items:baseline;
    padding:5px 0; border-top:1px solid rgba(120,170,255,0.08); }
  #mcp-over .osk{ font-size:11px; color:#9fb4d8; }
  #mcp-over .osv{ font-size:12px; color:#e6eeff; font-weight:600; text-align:right; }
  /* ── Compare: live KPI statistics ── */
  #mcp-cmp{ flex:1; display:none; overflow:auto; padding:16px 18px; }
  #mcp-cmp .lhint{ font-size:11.5px; color:#9fb4d8; margin-bottom:14px; }
  #mcp-cmp .lgrid{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
  #mcp-cmp .ltile{ background:rgba(14,22,42,0.85); border:1px solid rgba(120,170,255,0.22);
    border-radius:12px; padding:12px 14px; }
  #mcp-cmp .ltile .ll{ font-size:10.5px; color:#7f93b8; margin-bottom:4px; }
  #mcp-cmp .ltile .lv{ font-size:26px; font-weight:700; color:#e6eeff; line-height:1.1;
    font-variant-numeric:tabular-nums; }
  #mcp-cmp .ltile .lv .lu{ font-size:12px; font-weight:500; color:#7f93b8; }
  #mcp-cmp .lcard{ background:rgba(14,22,42,0.75); border:1px solid rgba(120,170,255,0.2);
    border-radius:12px; padding:10px 14px; }
  #mcp-cmp .lcard .dttl{ font-size:12px; font-weight:600; color:#cfe0ff; margin-bottom:6px; }
  #mcp-cmp .rsel{ background:rgba(20,30,52,0.92); color:#e6eeff; font-size:11.5px;
    border:1px solid rgba(120,170,255,0.32); border-radius:7px; padding:4px 8px; margin-left:8px; cursor:pointer; }
  #mcp-cmp .rsel option{ background:#0e1628; color:#e6eeff; }
  #legend .item { display:flex; align-items:center; margin:3px 0; }
  #legend .dot { width:11px; height:11px; border-radius:50%; margin-right:8px; box-shadow:0 0 6px currentColor; }
  #legend .sq { width:11px; height:11px; margin-right:8px; }
  #legend .ln { width:16px; height:3px; margin-right:8px; border-radius:2px; }
  /* ── Camera pan controls ── */
  #panV { position:absolute; left:18px; bottom:130px; height:280px; z-index:1000;
    writing-mode: vertical-lr; direction: rtl; accent-color:#56B4E9; cursor:pointer; }
  #panH { position:absolute; bottom:96px; left:50%; transform:translateX(-50%);
    width:44%; z-index:1000; accent-color:#56B4E9; cursor:pointer; }
  .panlbl { position:absolute; z-index:1000; font-size:10px; letter-spacing:.4px;
    color:#9fb4d8; background:rgba(10,16,30,0.55); padding:2px 7px;
    border-radius:7px; border:1px solid rgba(120,170,255,0.18); pointer-events:none; }
  #panVlbl { left:8px; bottom:418px; }
  #panHlbl { bottom:120px; left:50%; transform:translateX(-50%); }
  #btnHome { position:absolute; left:14px; bottom:96px; z-index:1000;
    padding:6px 10px; font-size:11px; font-weight:700; color:#eaf2ff; cursor:pointer;
    background:rgba(20,30,52,0.85); border:1px solid rgba(120,170,255,0.35);
    border-radius:9px; }
  #btnHome:hover { border-color:rgba(140,200,255,0.8); }
  /* ── Zoom controls (right-centre) ── */
  #zoomctl { position:absolute; right:16px; top:50%; transform:translateY(-50%);
    z-index:1000; display:flex; flex-direction:column; gap:6px; }
  #zoomctl button { width:36px; height:36px; font-size:20px; font-weight:700;
    line-height:1; color:#eaf2ff; background:rgba(20,30,52,0.85); cursor:pointer;
    border:1px solid rgba(120,170,255,0.35); border-radius:10px; }
  #zoomctl button:hover { border-color:rgba(140,200,255,0.85);
    background:rgba(40,120,240,0.45); }
  #zoomctl .zlbl { text-align:center; font-size:9px; color:#9fb4d8;
    letter-spacing:.5px; }
</style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="vignette"></div>
<!-- ════════ TOP BAR ════════ -->
<div id="mctop">
  <div class="brand">
    <span class="logo">🛰️</span>
    <div class="bt"><b id="op_title">SkyNTN</b></div>
  </div>
  <div class="kpis">
    <div class="mck"><div class="l">Served Users</div><div class="vv" id="s_served">–</div></div>
    <div class="mck"><div class="l">Median SINR</div><div class="vv" id="s_med">– <small>dB</small></div></div>
    <div class="mck"><div class="l">5th-pct SINR</div><div class="vv" id="s_p5">– <small>dB</small></div></div>
    <div class="mck"><div class="l">Handovers</div><div class="vv" id="s_cho">0</div></div>
  </div>
  <div class="spacer"></div>
  <div class="filters">
    <span class="fl">Scenario</span><select id="seg_cnst"></select>
    <span class="fl">Algorithm</span><select id="sel_strat"></select>
    <span class="fl">User</span><select id="sel_user"><option value="">— select —</option></select>
  </div>
  <div class="icbtn" id="aboutBtn" title="About / contact the developer">？</div>
  <div class="avatar" id="brandLogo" title="Polytechnique Montréal — click for about"><img id="brandImg" src="assets/polymtl-mark.svg" alt="Polytechnique Montréal" onerror="this.remove();var p=document.getElementById('brandLogo');p.classList.add('brandtxt');p.textContent='PM';"></div>
</div>
<!-- ════════ LEFT NAV RAIL ════════ -->
<div id="mcnav">
  <div class="mcni on" data-k="overview">🌐<small>Overview</small></div>
  <div class="mcni" data-k="analytics">📊<small>Analytics</small></div>
  <div class="mcni" data-k="compare">📈<small>Live</small></div>
  <div class="mcni" data-k="map">🌍<small>Map 3D</small></div>
  <div class="mcni" data-k="reports">📄<small>Reports</small></div>
</div>
<div id="mctoast"></div>
<!-- ════════ ABOUT / CONTACT ════════ -->
<div id="mcabout">
  <div class="abcard">
    <span class="abx" id="aboutClose">✕</span>
    <img class="ablogo" src="assets/polymtl-logo.svg" alt="Polytechnique Montréal" onerror="this.remove();">
    <div class="abtitle">SkyNTN</div>
    <div class="absub">NR-NTN mission-control &amp; analysis</div>
    <div class="abtheme">New Frontiers in Space Communications</div>
    <div class="abrow">Developed by <b>Mohammed Almekhlafi</b></div>
    <div class="abrow">Department of Electrical Engineering</div>
    <div class="abrow">Poly-GRAMES Research Center</div>
    <div class="abrow">Polytechnique Montréal</div>
    <div class="abmail"><a href="https://www.google.com/maps/search/?api=1&amp;query=2500%20Chemin%20de%20Polytechnique%2C%20Montr%C3%A9al%2C%20QC%20H3T%200A3" target="_blank" rel="noopener"><span class="abpin">📍</span> 2500 Chem. de Polytechnique, Montréal, QC H3T 0A3</a></div>
    <div class="abmail"><a href="mailto:mohammed.al-mekhlafi@polymtl.ca?subject=SkyNTN">✉ mohammed.al-mekhlafi@polymtl.ca</a></div>
    <div class="abmail"><a href="mailto:mhdalmekhlafi@gmail.com?subject=SkyNTN">✉ mhdalmekhlafi@gmail.com</a></div>
    <div class="abcopy">© 2026 Mohammed Al-Mekhlafi</div>
  </div>
</div>
<!-- ════════ ANALYTICS OVERLAY PANEL (figures from outputs/analysis) ════════ -->
<div id="mcpanel">
  <div class="mcp-hd">
    <b id="mcp-title">Analytics</b>
    <div id="mcp-kpis"></div>
    <div id="mcp-exp">
      <span class="xbtn" id="expCsv">⭳ CSV</span>
      <span class="xbtn" id="expPdf">🖶 PDF</span>
    </div>
    <span class="mcp-x" id="palBtn" title="Cycle colour scheme" style="display:none">🎨 Neon</span>
    <span class="mcp-x" onclick="document.getElementById('mcpanel').style.display='none'">✕ Close</span>
  </div>
  <div id="mcp-chart"></div>
  <div id="mcp-dash"></div>
  <div id="mcp-over"></div>
  <div id="mcp-cmp"></div>
</div>
<!-- ════════ LAYER SUMMARY ════════ -->
<div id="mclayers">
  <div class="lr"><span class="lf"><span class="dot" style="background:#38bdf8;color:#38bdf8"></span>Satellites</span></div>
  <div class="lr"><span class="lf"><span class="dot" style="background:#ff466e;color:#ff466e"></span>Users</span></div>
  <div class="lr"><span class="lf"><span class="dot" style="background:#facc15;color:#facc15"></span>Gateways</span></div>
  <div class="lr"><span class="lf"><span class="sq" style="background:#5affaa"></span>Served Links</span></div>
  <div class="lr"><span class="lf"><span class="dot" style="background:#fff14b;color:#fff14b"></span>Handover Flash</span></div>
</div>
<!-- ════════ RIGHT TOOL RAIL ════════ -->
<div id="mctools">
  <div class="mctool on" id="tlSelect" title="Select / inspect">▢</div>
  <div class="mctool" id="tlDemo" title="Cinematic demo — auto-orbit tour">🎬</div>
  <div class="mctool" id="tlFollow" title="Follow selected satellite">🛰️</div>
  <div class="mctool" id="tlSpeed" title="Playback speed"><b id="tlSpeedTxt" style="font-size:11px">2×</b></div>
  <div class="mctool" id="tlSnap" title="Snapshot (whole view) → PNG">📷</div>
  <div class="mctool" id="tlRec" title="Record video of the whole view → WEBM">⏺</div>
  <div class="mctool" id="tlLayers" title="Layers">▣</div>
  <div class="mctool" id="btnFull" title="Fullscreen">⛶</div>
</div>
<div id="mclayerpop">
  <div class="lp-hd">LAYERS</div>
  <label class="lp-row"><input type="checkbox" id="lpUsers" checked> Users</label>
  <label class="lp-row"><input type="checkbox" id="lpGw" checked> Gateways</label>
  <label class="lp-row"><input type="checkbox" id="lpLinks" checked> Served links</label>
  <label class="lp-row"><input type="checkbox" id="lpAtm" checked> Atmosphere &amp; lighting</label>
  <label class="lp-row"><input type="checkbox" id="lpAll"> All satellites</label>
</div>
<div id="mcmeasure"></div>
<!-- ════════ SELECTED-ENTITY INSPECTOR ════════ -->
<div id="mcsel">
  <span class="x" onclick="document.getElementById('mcsel').style.display='none'">✕</span>
  <div class="hd">📍 SELECTED: <b id="sel_name">–</b></div>
  <div id="sel_body"></div>
</div>
<!-- ════════ STATUS BAR ════════ -->
<div id="mcstat">
  <span class="ok">● CONNECTED</span>
  <span>DATA STREAM: <b style="color:#cfe0ff">sinr_snapshot</b></span>
  <span>RUN: <b id="st_run" style="color:#cfe0ff">–</b></span>
  <span>TICKS/BATCH: <b style="color:#cfe0ff">900</b></span>
  <span class="sp"></span>
  <span id="st_upd">LAST UPDATE: –</span>
</div>
<!-- (legend removed: the Layer Summary panel already lists the same layers) -->
<script>
  Cesium.Ion.defaultAccessToken = '';
  // ── True-colour satellite imagery (Esri World Imagery — token-free, CORS,
  //    high-res Maxar/Earthstar). This is the "real Earth" base. ──
  const esri = new Cesium.UrlTemplateImageryProvider({
    url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    maximumLevel: 17, credit: 'Esri · Maxar · Earthstar Geographics'
  });
  const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: new Cesium.ImageryLayer(esri),
    baseLayerPicker:false, geocoder:false, homeButton:false,
    sceneModePicker:false, navigationHelpButton:false,
    timeline:false, animation:false, infoBox:true,
    selectionIndicator:true, fullscreenButton:true,
    // preserveDrawingBuffer lets the Snapshot tool read the canvas to a PNG.
    contextOptions:{ webgl:{ preserveDrawingBuffer:true } }
  });
  const scene = viewer.scene;
  // safe(): apply an optional Cesium API without letting a version mismatch
  // throw and abort the whole script. Logs once per failure.
  function safe(label, fn){ try { fn(); } catch(e){ console.warn('skip '+label, e); } }
  safe('baseColor', ()=>{ scene.globe.baseColor = Cesium.Color.fromCssColorString('#0b2545'); });

  // ── City lights on the night side (NASA GIBS VIIRS Black Marble). ──
  // OFF by default: the GIBS WMTS endpoint is finicky and a wrong tile-matrix
  // config floods the console with tile-load errors. Flip ENABLE_NIGHT=true
  // only after verifying the URL/tileMatrixSetID in a browser. See HANDOFF.md.
  const ENABLE_NIGHT = false;
  if (ENABLE_NIGHT) safe('nightLayer', ()=>{
    const night = viewer.imageryLayers.addImageryProvider(
      new Cesium.WebMapTileServiceImageryProvider({
        url: 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/VIIRS_Black_Marble/default/2016-01-01/500m/{TileMatrix}/{TileRow}/{TileCol}.jpg',
        layer: 'VIIRS_Black_Marble', style: 'default', format: 'image/jpeg',
        tileMatrixSetID: '500m', maximumLevel: 8,
        tilingScheme: new Cesium.GeographicTilingScheme()
      }));
    night.dayAlpha = 0.0; night.nightAlpha = 1.0;
  });

  // ── Sun-driven lighting + atmosphere + bloom (each guarded) ──
  safe('lighting', ()=>{ scene.globe.enableLighting = true; });
  safe('dynAtm',   ()=>{ scene.globe.dynamicAtmosphereLighting = true; });
  safe('atmLightG',()=>{ scene.globe.atmosphereLightIntensity = 12.0; });
  safe('groundAtm',()=>{ scene.globe.showGroundAtmosphere = true; });
  safe('skyAtm',   ()=>{ scene.skyAtmosphere.show = true;
                         scene.skyAtmosphere.saturationShift = 0.12;
                         scene.skyAtmosphere.brightnessShift = 0.05; });
  safe('atmLightS',()=>{ scene.skyAtmosphere.atmosphereLightIntensity = 22.0; });
  safe('hdr',      ()=>{ scene.highDynamicRange = true; });
  safe('fog',      ()=>{ scene.fog.enabled = true; scene.fog.density = 1.2e-4; });
  safe('skybox',   ()=>{ scene.backgroundColor = Cesium.Color.TRANSPARENT;
                         scene.skyBox.show = true; });
  safe('bloom',    ()=>{
    const b = scene.postProcessStages.bloom;
    b.enabled = true; b.uniforms.glowOnly = false; b.uniforms.contrast = 120;
    b.uniforms.brightness = -0.35; b.uniforms.delta = 1.2;
    b.uniforms.sigma = 2.2; b.uniforms.stepSize = 1.0;
  });
  safe('fxaa',     ()=>{ scene.postProcessStages.fxaa.enabled = true; });
  safe('resScale', ()=>{ viewer.resolutionScale = Math.min(window.devicePixelRatio||1, 2); });
  safe('msaa',     ()=>{ scene.msaaSamples = 4; });
  safe('sse',      ()=>{ scene.globe.maximumScreenSpaceError = 1.5; });
  scene.globe.tileCacheSize = 1000;
  viewer.cesiumWidget.creditContainer.style.display = 'none';

  // ── Cinematic idle auto-orbit: slowly rotate the camera around Earth
  //    when the user isn't interacting (pauses on mouse/touch, resumes
  //    after a short idle). Gives the "mission-control" feel. ──
  let lastInteract = Date.now();
  ['mousedown','wheel','touchstart','pointerdown'].forEach(function(ev){
    viewer.canvas.addEventListener(ev, function(){ lastInteract = Date.now(); });
  });
  // Idle auto-orbit DISABLED by default so the region of interest stays
  // framed; the pan sliders give explicit control instead. (Re-enable by
  // setting AUTO_ORBIT=true.)
  const AUTO_ORBIT = false;
  viewer.clock.onTick.addEventListener(function(){
    if (AUTO_ORBIT && Date.now() - lastInteract > 4000) {
      viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.0006);
    }
  });

  // ── View centred on the region (Quebec) showing the FULL Earth disk ──
  // High altitude + straight-down pitch → the whole globe is visible and
  // centred; wheel-zoom or the pan sliders to move in on the region.
  const ROI = { lon: -70.0, lat: 50.0, alt: 2.4e7, pitch: -90.0 };
  let viewLon = ROI.lon, viewLat = ROI.lat;
  function flyToROI(dur){
    viewLon = ROI.lon; viewLat = ROI.lat;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(ROI.lon, ROI.lat, ROI.alt),
      orientation:{ heading:0.0, pitch:Cesium.Math.toRadians(ROI.pitch), roll:0.0 },
      duration: dur || 1.5 });
    syncPanSliders();
  }
  function applyPan(){
    const h = viewer.camera.positionCartographic.height || ROI.alt;  // keep zoom
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(viewLon, viewLat, h),
      orientation:{ heading:0.0, pitch:Cesium.Math.toRadians(ROI.pitch), roll:0.0 } });
    lastInteract = Date.now();
  }
  function syncPanSliders(){
    const h=document.getElementById('panH'), v=document.getElementById('panV');
    if(h) h.value = viewLon; if(v) v.value = viewLat;
  }
  (function(){
    const h=document.getElementById('panH'), v=document.getElementById('panV'),
          home=document.getElementById('btnHome'),
          zin=document.getElementById('btnZin'), zout=document.getElementById('btnZout');
    if(h) h.addEventListener('input', ()=>{ viewLon=parseFloat(h.value); applyPan(); });
    if(v) v.addEventListener('input', ()=>{ viewLat=parseFloat(v.value); applyPan(); });
    if(home) home.addEventListener('click', ()=>flyToROI(1.4));
    function zstep(){ return Math.max(2.0e5,
        (viewer.camera.positionCartographic.height || 1.0e7) * 0.45); }
    if(zin)  zin.addEventListener('click', ()=>{ viewer.camera.zoomIn(zstep()); lastInteract=Date.now(); });
    if(zout) zout.addEventListener('click', ()=>{ viewer.camera.zoomOut(zstep()); lastInteract=Date.now(); });
  })();
  // Ensure interactive zoom (wheel / pinch) is on and sensibly bounded.
  safe('zoomctl', ()=>{
    const sscc = scene.screenSpaceCameraController;
    sscc.enableZoom = true; sscc.enableRotate = true; sscc.enableTilt = true;
    sscc.minimumZoomDistance = 2.0e5;        // 200 km
    sscc.maximumZoomDistance = 6.0e7;        // 60 000 km
  });

  // SCENES[cnst] = {label, batch, nbatch, strategies:{strat:{label,czml,stats}}}
  const SCENES = __SCENES__;

  let currentDS = null;
  let clockStart = null;
  let currentStats = [];
  let didFly = false;

  function loadScene(cnst, strat){
    const cs = SCENES[cnst]; if(!cs) return;
    const sc = cs.strategies[strat]; if(!sc) return;
    currentStats = sc.stats;
    document.getElementById('op_title').textContent = 'SkyNTN';
    if(currentDS){ viewer.dataSources.remove(currentDS, true); currentDS=null; }
    const p = Cesium.CzmlDataSource.load(sc.czml);
    viewer.dataSources.add(p);
    p.then(function(ds){
      currentDS = ds;
      viewer.clock.shouldAnimate = true;
      viewer.clock.multiplier = 2;
      if(ds.clock){
        viewer.clock.startTime  = ds.clock.startTime.clone();
        viewer.clock.stopTime   = ds.clock.stopTime.clone();
        viewer.clock.currentTime = ds.clock.startTime.clone();
        viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
        clockStart = ds.clock.startTime;
        if(viewer.timeline)
          viewer.timeline.zoomTo(ds.clock.startTime, ds.clock.stopTime);
      } else { clockStart = viewer.clock.startTime; }
      if(!didFly){
        didFly = true;
        // Frame the REGION OF INTEREST (Quebec, ~45–62°N, −80…−57°W) with an
        // oblique angle so the users/satellites over the region are centred.
        flyToROI(2.6);
      }
    });
  }

  // ── Build the selector controls from SCENES ──
  const segC = document.getElementById('seg_cnst');
  const selS = document.getElementById('sel_strat');
  const cnstKeys = Object.keys(SCENES);
  let curCnst = cnstKeys[0];

  // ── Full-constellation ("all satellites") layer toggle ──
  let allSatsDS = null;
  function setAllSats(){
    const chk = document.getElementById('lpAll');   // the Layers-panel toggle
    const on = chk && chk.checked;
    if(allSatsDS){ viewer.dataSources.remove(allSatsDS, true); allSatsDS=null; }
    const acz = SCENES[curCnst] && SCENES[curCnst].allCzml;
    if(on && acz){
      const p = Cesium.CzmlDataSource.load(acz);
      viewer.dataSources.add(p);
      p.then(function(ds){ allSatsDS = ds; });
    } else if(on && !acz && chk){ chk.checked = false; }
  }

  function refreshStrategies(cnst){
    selS.innerHTML = '';
    const strats = SCENES[cnst].strategies;
    Object.keys(strats).forEach(function(k){
      const o = document.createElement('option');
      o.value = k; o.textContent = strats[k].label;
      selS.appendChild(o);
    });
  }
  function setActiveChip(cnst){ segC.value = cnst; }   // constellation = dropdown
  cnstKeys.forEach(function(cnst){
    const o = document.createElement('option');
    o.value = cnst; o.textContent = SCENES[cnst].label;
    segC.appendChild(o);
  });
  segC.onchange = function(){
    curCnst = segC.value; refreshStrategies(curCnst);
    loadScene(curCnst, selS.value);
    setAllSats();                         // reload full-constellation for new cnst
  };
  selS.onchange = function(){ loadScene(curCnst, selS.value); };
  // (all-satellites toggle lives in the Layers panel; its onchange calls setAllSats)

  // Initial load
  setActiveChip(curCnst);
  refreshStrategies(curCnst);
  loadScene(curCnst, selS.value);

  // ── Live HUD driven by the Cesium clock ──
  const elS=document.getElementById('s_served'), elM=document.getElementById('s_med'),
        elP=document.getElementById('s_p5'), elC=document.getElementById('s_cho'),
        elK=document.getElementById('clk');
  viewer.clock.onTick.addEventListener(function(clock){
    if(!clockStart) return;
    const e=Cesium.JulianDate.secondsDifference(clock.currentTime, clockStart);
    if(elK) elK.textContent='t = '+e.toFixed(0)+' s (sim)';
    let row=null;
    for(let i=0;i<currentStats.length;i++){ if(currentStats[i].t_s<=e) row=currentStats[i]; else break; }
    if(row){
      elS.textContent=row.n;
      elM.innerHTML=row.med.toFixed(1)+' <span style="font-size:11px">dB</span>';
      elP.textContent=row.p5.toFixed(1)+' dB';
      elC.textContent=row.cho;
      elM.style.color = row.med>=-8?'#9be7a8':row.med>=-12?'#ffd166':'#ff7a7a';
      window.mcRow=row;   // current values for the Live tiles (trend reads currentStats)
    }
    if(window.mcUpdateCompare) window.mcUpdateCompare();
  });

  // ════════ MISSION-CONTROL WIRING ════════
  function mcTxt(id,v){ var e=document.getElementById(id); if(e) e.textContent=v; }
  // Fullscreen — the WHOLE app (globe + overlays), not just the Cesium canvas.
  (function(){ var b=document.getElementById('btnFull'); if(!b) return;
    b.addEventListener('click', function(){ safe('fullscreen', function(){
      var d=document, fe=d.fullscreenElement||d.webkitFullscreenElement||d.msFullscreenElement;
      if(fe){ var ex=d.exitFullscreen||d.webkitExitFullscreen||d.msExitFullscreen; if(ex) ex.call(d); }
      else { var el=d.documentElement,
             req=el.requestFullscreen||el.webkitRequestFullscreen||el.msRequestFullscreen;
             if(req) req.call(el); else mcToast('Fullscreen not supported by this browser'); }
    }); }); })();
  // Layer Summary — count active entities by type
  function mcLayers(){
    if(!currentDS) return;
    var nSat=0,nUsr=0,nGw=0,nLnk=0,nHo=0;
    currentDS.entities.values.forEach(function(ent){
      if(ent.polyline){ nLnk++; return; }
      var id=String(ent.id||'').toLowerCase();
      if(id.indexOf('hoflash')>=0||id.indexOf('flash')>=0){ nHo++; return; }  // HO flashes, not sats
      if(id.indexOf('gw')>=0||id.indexOf('gate')>=0) nGw++;
      else if(id.indexOf('user')>=0||id.indexOf('usr')>=0||id.indexOf('ue')>=0) nUsr++;
      else if(ent.point||ent.billboard||ent.model||ent.label) nSat++;
    });
    mcTxt('lc_sat',nSat); mcTxt('lc_usr',nUsr); mcTxt('lc_gw',nGw); mcTxt('lc_lnk',nLnk); mcTxt('lc_ho',nHo);
    window.mcCounts={sat:nSat,usr:nUsr,gw:nGw,lnk:nLnk,ho:nHo};   // for the live Compare panel
  }
  var _ls=loadScene;
  loadScene=function(c,s){ _ls(c,s); setTimeout(function(){ mcLayers(); mcPopulateUsers(); },1400); };
  setInterval(mcLayers,3000);
  // ── "View user" selector — list users in the active scene; fly to + inspect ──
  function mcPopulateUsers(){
    var sel=document.getElementById('sel_user'); if(!sel||!currentDS) return;
    var ids=[];
    currentDS.entities.values.forEach(function(e){ var id=String(e.id||'').toLowerCase();
      if(id.indexOf('user')>=0||id.indexOf('usr')>=0||id.indexOf('ue')>=0) ids.push(e.id); });
    ids.sort(function(a,b){ return (parseInt(String(a).replace(/\D/g,''))||0)
                                  -(parseInt(String(b).replace(/\D/g,''))||0); });
    sel.innerHTML='<option value="">— select ('+ids.length+') —</option>'+
      ids.map(function(id){ var e=currentDS.entities.getById(id);
        return '<option value="'+id+'">'+((e&&e.name)||id)+'</option>'; }).join('');
  }
  // Filter served-links to one user (empty selection = show ALL links).
  function mcFilterLinks(uidNum){
    if(!currentDS) return;
    var pref = uidNum ? ('link_'+uidNum+'_') : null;
    currentDS.entities.values.forEach(function(e){
      if(e.polyline){ e.show = (!pref) || String(e.id||'').indexOf(pref)===0; }
    });
  }
  (function(){ var sel=document.getElementById('sel_user'); if(!sel) return;
    sel.onchange=function(){
      var v=this.value;
      var uidNum = v ? String(v).replace(/\D/g,'') : null;   // numeric user id
      mcFilterLinks(uidNum);                                  // ← show only this user's link
      if(!v||!currentDS){ document.getElementById('mcsel').style.display='none'; return; }
      var ent=currentDS.entities.getById(v); if(!ent) return;
      mcLastPicked=ent;
      safe('flyUser', function(){ viewer.flyTo(ent,{duration:1.2}); });
      mcTxt('sel_name', ent.name||ent.id);
      var html='<div class="sr"><span class="k">Type</span><span class="v">User terminal</span></div>';
      var e=clockStart?Cesium.JulianDate.secondsDifference(viewer.clock.currentTime,clockStart):0;
      var st=null; for(var i=0;i<currentStats.length;i++){ if(currentStats[i].t_s<=e) st=currentStats[i]; else break; }
      if(st) html+='<div class="sr"><span class="k">Cell median SINR</span><span class="v">'+st.med.toFixed(1)+' dB</span></div>'
                  +'<div class="sr"><span class="k">Served users (cell)</span><span class="v">'+st.n+'</span></div>';
      document.getElementById('sel_body').innerHTML=html;
      document.getElementById('mcsel').style.display='block';
    };
  })();
  // Selected-entity inspector (click a satellite / user / gateway)
  var mcH=new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  mcH.setInputAction(function(click){
    var picked=viewer.scene.pick(click.position);
    var box=document.getElementById('mcsel');
    if(!(Cesium.defined(picked)&&picked.id)){ return; }
    var ent=picked.id, id=String(ent.id||'').toLowerCase();
    mcLastPicked=ent;                       // remembered for the Follow tool
    mcTxt('sel_name', ent.name||ent.id||'entity');
    function R(k,v){ return '<div class="sr"><span class="k">'+k+'</span><span class="v">'+v+'</span></div>'; }
    var e=clockStart?Cesium.JulianDate.secondsDifference(viewer.clock.currentTime,clockStart):0;
    var st=null; for(var i=0;i<currentStats.length;i++){ if(currentStats[i].t_s<=e) st=currentStats[i]; else break; }
    var kind=(id.indexOf('gw')>=0||id.indexOf('gate')>=0)?'Gateway':
             (id.indexOf('user')>=0||id.indexOf('usr')>=0||id.indexOf('ue')>=0)?'User terminal':'Satellite';
    var html=R('Type',kind);
    if(kind==='User terminal'&&st){ html+=R('Cell median SINR',st.med.toFixed(1)+' dB')+R('Cell served users',st.n); }
    if(ent.description){ safe('desc',function(){ var d=ent.description.getValue(viewer.clock.currentTime);
      if(d) html+='<div style="margin-top:9px;font-size:11px;color:#9fb4d8;line-height:1.4">'+d+'</div>'; }); }
    document.getElementById('sel_body').innerHTML=html;
    box.style.display='block';
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  // Left nav rail — open the real analytics figures (outputs/analysis/*.png,
  // served at ./analysis/ since the globe is served from outputs/).
  var ALERTS={count:0,message:'no alert data (build analysis/interactive.json)'};
  (function(){
    var items=document.querySelectorAll('#mcnav .mcni');
    function toast(msg){ var t=document.getElementById('mctoast'); if(!t) return;
      t.textContent=msg; t.style.opacity='1'; clearTimeout(t._h);
      t._h=setTimeout(function(){ t.style.opacity='0'; },2800); }
    // KPI sets per nav item (keys into analysis/interactive.json)
    var KSET={
      analytics:['fig13_sinr_ccdf','fig16_ion_ccdf','fig1_visibility_ccdf',
                 'fig5_contact_time_cdf','fig9_ihi_cdf','fig25_access_delay_ccdf',
                 'fig26_e2e_rtt_ccdf'],
      compare:['fig8_ho_rate_bar','fig19_availability_bar','fig17_visibility_outage_ccdf'],
      reports:['fig13_sinr_ccdf','fig16_ion_ccdf','fig1_visibility_ccdf',
               'fig5_contact_time_cdf','fig9_ihi_cdf','fig17_visibility_outage_ccdf',
               'fig19_availability_bar','fig8_ho_rate_bar','fig25_access_delay_ccdf',
               'fig26_e2e_rtt_ccdf']
    };
    var INTER=null;
    // Panel has two modes: single-chart explorer (Analytics/Compare) and the
    // Reports dashboard grid. setMode toggles which sub-elements are visible.
    function setMode(m){
      var chart=(m==='chart');
      document.getElementById('mcp-chart').style.display=chart?'':'none';
      document.getElementById('mcp-kpis').style.display=chart?'':'none';
      document.getElementById('mcp-dash').style.display=(m==='dash')?'block':'none';
      document.getElementById('mcp-exp').style.display=(m==='dash')?'flex':'none';
      document.getElementById('mcp-over').style.display=(m==='over')?'block':'none';
      document.getElementById('mcp-cmp').style.display=(m==='cmp')?'block':'none';
      var pb=document.getElementById('palBtn'); if(pb) pb.style.display=(m==='chart')?'':'none';
    }
    var DARK={ paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      font:{color:'#cfe0ff',family:'Inter,Segoe UI,sans-serif',size:13},
      legend:{bgcolor:'rgba(8,13,28,0.55)',bordercolor:'rgba(120,170,255,0.2)',borderwidth:1,
              orientation:'h',y:1.08,x:1,xanchor:'right'},
      margin:{l:62,r:22,t:46,b:54},
      xaxis:{gridcolor:'rgba(255,255,255,0.08)',zerolinecolor:'rgba(255,255,255,0.12)'},
      yaxis:{gridcolor:'rgba(255,255,255,0.08)',zerolinecolor:'rgba(255,255,255,0.12)'} };
    // Bright, high-contrast palettes for the dark UI (no dark lines/bars). The
    // user can cycle them from the 🎨 button. Each series also cycles a dash
    // style and a marker symbol so lines stay distinguishable beyond colour.
    var PALETTES=[
      {name:'Neon',   c:['#00E5FF','#39FF14','#FF3DAE','#FFD400','#B388FF','#FF6D00','#18FFFF','#EEFF41']},
      {name:'Aurora', c:['#4CC9F0','#F72585','#4ADE80','#FFB703','#C77DFF','#FF7B00','#48CAE4','#F15BB5']},
      {name:'Sunset', c:['#FF6D00','#FFD400','#FF3DAE','#00E5FF','#B5FF7D','#7DF9FF','#FF9E80','#E0AAFF']},
      {name:'Ice',    c:['#8AE9FF','#C4F1FF','#6EE7B7','#A7F3D0','#93C5FD','#DDD6FE','#5EEAD4','#BAE6FD']}
    ];
    var DASHES=['solid','dash','dot','dashdot','longdash','longdashdot'];
    var SYMBOLS=['circle','diamond','square','triangle-up','star','cross','x','triangle-down'];
    var ANALYTICS_PAL=0, CUR_KPI=null, ANALYTICS_CNST='all', ANALYTICS_SCALE='linear';
    function _palC(){ return PALETTES[ANALYTICS_PAL%PALETTES.length].c; }
    function _subIdx(n,m){ if(n<=m) { var a=[]; for(var i=0;i<n;i++) a.push(i); return a; }
      var idx=[],step=(n-1)/(m-1); for(var i=0;i<m;i++) idx.push(Math.round(i*step)); return idx; }
    function renderKpi(stem){
      CUR_KPI=stem;
      var rec=INTER&&INTER[stem]; var div=document.getElementById('mcp-chart');
      if(!rec){ div.innerHTML='<div style="color:#9fb4d8;padding:24px">No data for this KPI.</div>'; return; }
      var lay=JSON.parse(JSON.stringify(DARK)), traces=[], cols=_palC();
      lay.margin.t=18;                       // no chart title (the dropdown names it)
      lay.xaxis.title=rec.xlabel||''; lay.yaxis.title=rec.ylabel||'';
      if(ANALYTICS_SCALE==='logy'||ANALYTICS_SCALE==='logxy') lay.yaxis.type='log';
      if(ANALYTICS_SCALE==='logx'||ANALYTICS_SCALE==='logxy') lay.xaxis.type='log';
      if(rec.kind==='ccdf'){
        var series=rec.series;
        if(ANALYTICS_CNST&&ANALYTICS_CNST!=='all'){
          var tagged=rec.series.some(function(s){ return String(s.name).indexOf(ANALYTICS_CNST)>=0; });
          if(tagged) series=rec.series.filter(function(s){ return String(s.name).indexOf(ANALYTICS_CNST)>=0; });
        }
        series.forEach(function(s,i){
          var col=cols[i%cols.length], dash=DASHES[i%DASHES.length], sym=SYMBOLS[i%SYMBOLS.length];
          traces.push({x:s.x,y:s.y,name:s.name,mode:'lines',legendgroup:s.name,
            line:{color:col,width:2.6,dash:dash,shape:'spline',smoothing:0.35},
            hovertemplate:'%{x:.2f} · %{y:.3f}<extra>'+s.name+'</extra>'});
          var id=_subIdx((s.x||[]).length,14);                 // sparse accent markers
          traces.push({x:id.map(function(j){return s.x[j];}),y:id.map(function(j){return s.y[j];}),
            mode:'markers',legendgroup:s.name,showlegend:false,hoverinfo:'skip',
            marker:{color:col,size:8,symbol:sym,line:{color:'#0a0f20',width:1.2}}});
        });
      } else {
        var bars=rec.bars;
        if(ANALYTICS_CNST&&ANALYTICS_CNST!=='all'){
          var bt=rec.bars.some(function(b){ return String(b.label).indexOf(ANALYTICS_CNST)>=0; });
          if(bt) bars=rec.bars.filter(function(b){ return String(b.label).indexOf(ANALYTICS_CNST)>=0; });
        }
        var xs=bars.map(function(b){return b.label;}),
            ys=bars.map(function(b){return b.value;}),
            bcols=bars.map(function(b,i){return cols[i%cols.length];});
        traces.push({type:'bar',x:xs,y:ys,marker:{color:bcols,line:{color:'#0a0f20',width:1.2}},
          error_y:{type:'data',symmetric:false,
            array:bars.map(function(b){return b.hi-b.value;}),
            arrayminus:bars.map(function(b){return b.value-b.lo;}),
            color:'#e2e8f0',thickness:1.3,width:4},
          hovertemplate:'%{x}: %{y:.3f}<extra></extra>'});
        lay.showlegend=false; lay.xaxis.tickangle=-28;
      }
      Plotly.react(div,traces,lay,
        {responsive:true,displaylogo:false,
         modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});
      setTimeout(function(){ Plotly.Plots.resize(div); },60);
    }
    function cyclePalette(){ ANALYTICS_PAL=(ANALYTICS_PAL+1)%PALETTES.length;
      var pb=document.getElementById('palBtn'); if(pb) pb.textContent='🎨 '+PALETTES[ANALYTICS_PAL].name;
      if(CUR_KPI) renderKpi(CUR_KPI); }
    function buildBtns(keys){
      var bar=document.getElementById('mcp-kpis'); bar.innerHTML='';
      keys.filter(function(k){return INTER&&INTER[k];}).forEach(function(k,i){
        var b=document.createElement('div'); b.className='kbtn'+(i===0?' on':'');
        b.textContent=INTER[k].label;
        b.onclick=function(){ Array.from(bar.children).forEach(function(x){x.classList.remove('on');});
          b.classList.add('on'); renderKpi(k); };
        bar.appendChild(b);
      });
    }
    function openPanel(title,keys){
      document.getElementById('mcp-title').textContent=title;
      document.getElementById('mcpanel').style.display='flex';
      function go(){ setMode('chart'); buildBtns(keys); var f=keys.filter(function(k){return INTER&&INTER[k];})[0];
        if(f) renderKpi(f); }
      if(INTER){ go(); }
      else { fetch('analysis/interactive.json?cb='+Date.now()).then(function(r){return r.json();})
             .then(function(j){ INTER=j; go(); })
             .catch(function(){ document.getElementById('mcp-chart').innerHTML=
               '<div style="color:#9fb4d8;padding:24px">Interactive data not found. Build it once:<br>'+
               '<code style="color:#cfe0ff">python -m ntnconnect.analysis.visualization.build_interactive '+
               '--scan-root &lt;runs&gt; --eph-dir &lt;dataset&gt; --out outputs/analysis</code></div>'; }); }
    }
    // ── Reports = decision dashboard: leaderboard + Pareto + insights + SLA ──
    var REP_CNST=null, REP_YKEY='sinr_p50', REP_DONUT='ho_rate';
    function _fmt(v,d){ return (v===null||v===undefined||isNaN(v))?'–':Number(v).toFixed(d); }
    function _dec(k){ return k.key==='ho_rate'?3:(k.unit==='s'?1:2); }
    function _colBg(nm){ return nm>=0.66?'#173a26':nm>=0.33?'#3a2a12':'#4a1f1f'; }
    function _colTx(nm){ return nm>=0.66?'#8fe7b4':nm>=0.33?'#f4c778':'#f0a0a0'; }
    function openReports(){
      document.getElementById('mcp-title').textContent='Reports — decision dashboard';
      document.getElementById('mcpanel').style.display='flex';
      function go(){ setMode('dash'); renderDash(); }
      if(INTER){ go(); }
      else { fetch('analysis/interactive.json?cb='+Date.now()).then(function(r){return r.json();})
             .then(function(j){ INTER=j; go(); })
             .catch(function(){ setMode('dash'); document.getElementById('mcp-dash').innerHTML=
               '<div class="dnote">Interactive data not found. Build analysis/interactive.json first.</div>'; }); }
    }
    // ── Analytics = pick a figure from two groups: Coverage + Strategy ──
    function openAnalytics(){
      document.getElementById('mcp-title').textContent='Analytics';
      document.getElementById('mcpanel').style.display='flex';
      function go(){ setMode('chart'); buildAnalyticsBtns(); }
      if(INTER){ go(); }
      else { fetch('analysis/interactive.json?cb='+Date.now()).then(function(r){return r.json();})
             .then(function(j){ INTER=j; go(); })
             .catch(function(){ setMode('chart'); document.getElementById('mcp-chart').innerHTML=
               '<div style="color:#9fb4d8;padding:24px">Interactive data not found.</div>'; }); }
    }
    function buildAnalyticsBtns(){
      var bar=document.getElementById('mcp-kpis'); bar.innerHTML='';
      var figs=(INTER&&INTER._figures)||[];
      if(!figs.length){ buildBtns(KSET.analytics);
        var f0=KSET.analytics.filter(function(k){return INTER&&INTER[k];})[0]; if(f0) renderKpi(f0); return; }
      var sel=document.createElement('select'); sel.className='kpiSel'; sel.id='kpiSel';
      var groups=[['coverage','Coverage'],['strategy','Performance']], first=null;
      groups.forEach(function(g){
        var items=figs.filter(function(x){return x.group===g[0] && INTER[x.stem];});
        if(!items.length) return;
        var og=document.createElement('optgroup'); og.label=g[1];
        items.forEach(function(x){ var o=document.createElement('option'); o.value=x.stem;
          o.textContent=x.label; og.appendChild(o); if(!first) first=x.stem; });
        sel.appendChild(og);
      });
      sel.onchange=function(){ renderKpi(sel.value); };
      bar.appendChild(sel);
      // Constellation filter — plot all, or just one constellation's series.
      var cnsts=(INTER._matrix&&INTER._matrix.constellations)||[];
      if(cnsts.length>1){
        var csel=document.createElement('select'); csel.className='kpiSel'; csel.id='kpiCnst';
        csel.style.minWidth='150px'; csel.style.marginLeft='8px';
        var oa=document.createElement('option'); oa.value='all'; oa.textContent='All constellations'; csel.appendChild(oa);
        cnsts.forEach(function(c){ var o=document.createElement('option'); o.value=c; o.textContent=c; csel.appendChild(o); });
        if(ANALYTICS_CNST&&[].concat('all',cnsts).indexOf(ANALYTICS_CNST)<0) ANALYTICS_CNST='all';
        csel.value=ANALYTICS_CNST;
        csel.onchange=function(){ ANALYTICS_CNST=this.value; if(CUR_KPI) renderKpi(CUR_KPI); };
        bar.appendChild(csel);
      }
      // Axis scale (linear / log) — log is handy for reading CCDF tails.
      var scl=document.createElement('select'); scl.className='kpiSel'; scl.id='kpiScale';
      scl.style.minWidth='120px'; scl.style.marginLeft='8px';
      [['linear','Linear'],['logy','Log Y'],['logx','Log X'],['logxy','Log X & Y']].forEach(function(o){
        var op=document.createElement('option'); op.value=o[0]; op.textContent=o[1];
        if(o[0]===ANALYTICS_SCALE) op.selected=true; scl.appendChild(op); });
      scl.onchange=function(){ ANALYTICS_SCALE=this.value; if(CUR_KPI) renderKpi(CUR_KPI); };
      bar.appendChild(scl);
      if(first){ sel.value=first; renderKpi(first); }
    }
    // ── Overview = at-a-glance mission snapshot (scenario + per-cnst headline) ──
    function openOverview(){
      document.getElementById('mcp-title').textContent='Mission overview';
      document.getElementById('mcpanel').style.display='flex';
      function go(){ setMode('over'); renderOverview(); }
      if(INTER){ go(); }
      else { fetch('analysis/interactive.json?cb='+Date.now()).then(function(r){return r.json();})
             .then(function(j){ INTER=j; go(); })
             .catch(function(){ setMode('over'); document.getElementById('mcp-over').innerHTML=
               '<div class="dnote">Summary data not found. Build analysis/interactive.json first.</div>'; }); }
    }
    function renderOverview(){
      var M=INTER&&INTER._matrix, host=document.getElementById('mcp-over');
      if(!M||!M.constellations||!M.constellations.length){
        host.innerHTML='<div class="dnote">No summary data in interactive.json.</div>'; return; }
      var seen={}, strategies=[];
      M.constellations.forEach(function(c){ Object.keys(M.data[c].strategies).forEach(function(s){
        if(!seen[s]){ seen[s]=1; strategies.push(s); } }); });
      var facts=[
        ['Constellations', M.constellations.length+' · '+M.constellations.join(', ')],
        ['Handover methods', strategies.length+' · '+strategies.join(', ')]
      ].map(function(f){ return '<div class="ofact"><div class="ol">'+f[0]+'</div><div class="ov">'+f[1]+'</div></div>'; }).join('');
      // ── Run information: simulation parameters & assumptions (NO statistics) ──
      var S=INTER._scenario||{}, sc=S.constellations||[];
      function perC(key,unit){ var parts=sc.filter(function(c){return c[key]!=null;})
        .map(function(c){return c.name+' '+c[key];});
        return parts.length?(parts.join(' · ')+(unit?' '+unit:'')):'–'; }
      var satStr=sc.map(function(c){return c.name+' '+(c.n_sats!=null?c.n_sats:'?');}).join(' · ') || '–';
      var gwTot=sc.reduce(function(a,c){return a+(c.n_gateways||0);},0);
      var setRows=[
        ['Satellites (in TLE)', satStr],
        ['User terminals', S.n_users!=null?S.n_users:'–'],
        ['Gateways', gwTot||'–'],
        ['Elevation mask', S.elev_min_user_deg!=null?S.elev_min_user_deg+'°':'–'],
        ['Time step', S.fine_step_s!=null?(S.fine_step_s+' s fine / '+(S.coarse_step_s||'?')+' s coarse'):'–'],
        ['Duration', S.duration_h!=null?(S.duration_h+' h · '+(S.n_batches||'?')+' batches'):'–'],
        ['Handover methods', (S.strategies&&S.strategies.length)?S.strategies.join(', '):strategies.join(', ')],
        ['User grid', S.user_grid||'–'],
        ['Antenna gain', perC('antenna_gain_db','dBi')],
        ['Tx power', perC('tx_power_dbw','dBW')],
        ['Noise power', perC('noise_dbw','dBW')]
      ];
      var settings='<div class="osettings"><div class="osh">Run information — parameters &amp; assumptions</div><div class="osgrid">'+
        setRows.map(function(r){ return '<div class="osrow"><span class="osk">'+r[0]+
          '</span><span class="osv">'+r[1]+'</span></div>'; }).join('')+'</div></div>';
      // ── Display-style switcher (changes the globe background / scene look) ──
      var disp='<div class="osettings"><div class="osh">Display style</div>'+
        '<select id="dispSel" class="kpiSel" style="min-width:220px">'+
        DISPLAYS.map(function(d,i){ return '<option value="'+i+'"'+(i===CUR_DISPLAY?' selected':'')+'>'+d.name+'</option>'; }).join('')+
        '</select><div class="osk" style="margin-top:8px">Changes the 3D scene background, stars and atmosphere.</div></div>';
      host.innerHTML='<div class="ostrip">'+facts+'</div>'+
        '<div class="osrow2">'+settings+disp+'</div>'+
        '<div class="ohint">Open <b>Reports</b> for the results dashboard, or <b>Analytics</b> to drill into a single KPI distribution.</div>';
      var dv=document.getElementById('dispSel'); if(dv) dv.onchange=function(){ applyDisplay(parseInt(this.value,10)); };
    }
    // Display presets — recolour the Cesium scene (background/stars/atmosphere).
    var DISPLAYS=[
      {name:'Night sky (default)', bg:'#01030a', stars:true,  atmo:true,  light:true,  base:'#0b2545'},
      {name:'Deep space',          bg:'#000000', stars:true,  atmo:false, light:false, base:'#050a14'},
      {name:'Daylight',            bg:'#0a1a2f', stars:false, atmo:true,  light:false, base:'#0b2545'},
      {name:'Minimal dark',        bg:'#0a0f20', stars:false, atmo:false, light:false, base:'#0e1830'}
    ];
    var CUR_DISPLAY=0;
    function applyDisplay(i){ var d=DISPLAYS[i]; if(!d) return; CUR_DISPLAY=i;
      try{ viewer.scene.backgroundColor=Cesium.Color.fromCssColorString(d.bg); }catch(e){}
      try{ if(viewer.scene.skyBox) viewer.scene.skyBox.show=d.stars; }catch(e){}
      try{ if(viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show=d.atmo; }catch(e){}
      try{ viewer.scene.globe.showGroundAtmosphere=d.atmo;
           viewer.scene.globe.enableLighting=d.light;
           viewer.scene.globe.baseColor=Cesium.Color.fromCssColorString(d.base); }catch(e){}
      try{ viewer.scene.requestRender(); }catch(e){}
    }
    // ── Compare = LIVE KPI statistics that move with the simulation clock ──
    var CMP_METRICS=[
      {k:'med',lab:'Median SINR',u:'dB'},
      {k:'p5', lab:'5th-pct SINR',u:'dB'},
      {k:'n',  lab:'Served users',u:''},
      {k:'cho',lab:'Handovers (cumulative)',u:''},
      {k:'pwr',lab:'Mean transmit power',u:'dBW'},
      {k:'thr',lab:'Total throughput (load)',u:'Mbps'},
      {k:'se', lab:'Mean spectral efficiency',u:'b/s/Hz'}
    ];
    var CMP_METRIC='med';
    function openCompare(){
      document.getElementById('mcp-title').textContent='Live KPI statistics';
      document.getElementById('mcpanel').style.display='flex';
      window.mcCmpHist=[];                       // fresh trend for this session
      setMode('cmp'); renderCompare(); updateCompare();
    }
    function renderCompare(){
      var host=document.getElementById('mcp-cmp');
      function tile(id,lab,unit){ return '<div class="ltile"><div class="ll">'+lab+'</div>'+
        '<div class="lv"><span id="'+id+'">–</span> <span class="lu">'+(unit||'')+'</span></div></div>'; }
      host.innerHTML=
        '<div class="lhint">Live statistics for the current 3D scene ('+curCnst+'). Press play ▶ on the timeline and the numbers move with the clock.</div>'+
        '<div class="lgrid">'+
          tile('cmp_med','Median SINR','dB')+tile('cmp_p5','5th-pct SINR','dB')+
          tile('cmp_n','Served users','')+tile('cmp_ho','Handovers','cum')+
          tile('cmp_sat','Satellites up','')+tile('cmp_lnk','Served links','')+
          tile('cmp_flash','Handover flashes','')+tile('cmp_t','Sim time','s')+
        '</div>'+
        '<div class="lcard"><div class="dttl">Live trend &nbsp;<select id="cmpMetric" class="rsel">'+
          CMP_METRICS.map(function(m){ return '<option value="'+m.k+'"'+(m.k===CMP_METRIC?' selected':'')+'>'+m.lab+'</option>'; }).join('')+
          '</select></div><div id="cmpSpark" style="width:100%;height:180px"></div></div>';
      var ms=document.getElementById('cmpMetric'); if(ms) ms.onchange=function(){ CMP_METRIC=this.value; drawSpark(true); };
    }
    function updateCompare(){
      var host=document.getElementById('mcp-cmp'); if(!host||host.style.display==='none') return;
      var r=window.mcRow, c=window.mcCounts||{};
      function set(id,v){ var e=document.getElementById(id); if(e) e.textContent=v; }
      if(r){ set('cmp_med',r.med.toFixed(1)); set('cmp_p5',r.p5.toFixed(1));
        set('cmp_n',r.n); set('cmp_ho',r.cho); set('cmp_t',r.t_s!=null?Math.round(r.t_s):'–');
        var m=document.getElementById('cmp_med');
        if(m) m.style.color=r.med>=-8?'#7ff0b8':r.med>=-12?'#f4c778':'#ff9a9a'; }
      set('cmp_sat',c.sat!=null?c.sat:'–'); set('cmp_lnk',c.lnk!=null?c.lnk:'–');
      set('cmp_flash',c.ho!=null?c.ho:'–');
      drawSpark();
    }
    var _sparkT=0;
    function drawSpark(force){
      var div=document.getElementById('cmpSpark'); if(!div) return;
      var now=Date.now(); if(!force && (now-_sparkT)<400) return; _sparkT=now;   // throttle live redraws
      var meta=CMP_METRICS.filter(function(m){return m.k===CMP_METRIC;})[0]||{lab:'',u:''};
      // Trend the full per-tick stats from t=0 up to the current sim time, so the
      // curve always starts at 0 and GROWS to the right (not a scrolling window
      // that only starts when the panel was opened).
      var e = clockStart ? Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, clockStart) : 1e12;
      var stats = (typeof currentStats!=='undefined' && currentStats) ? currentStats : [];
      var pts=[];
      for(var i=0;i<stats.length;i++){ var p=stats[i]; if(p.t_s>e+0.01) break;
        var v=p[CMP_METRIC]; if(v!=null&&!isNaN(v)) pts.push([p.t_s/60.0, v]); }
      if(pts.length<2){ div.innerHTML='<div class="dnote">Press play ▶ — '+meta.lab+' will trend here.</div>'; return; }
      var stride=Math.max(1, Math.floor(pts.length/800)), xs=[], ys=[];   // thin dense stats for plotting
      for(var j=0;j<pts.length;j+=stride){ xs.push(pts[j][0]); ys.push(pts[j][1]); }
      var last=pts[pts.length-1]; if(xs[xs.length-1]!==last[0]){ xs.push(last[0]); ys.push(last[1]); }
      if(div.querySelector('.dnote')) div.innerHTML='';   // drop the placeholder before plotting
      var lay=JSON.parse(JSON.stringify(DARK)); lay.margin={l:60,r:18,t:12,b:36}; lay.showlegend=false;
      lay.xaxis.title='time [min]'; lay.yaxis.title=meta.lab+(meta.u?' ['+meta.u+']':'');
      lay.xaxis.autorange=false; lay.xaxis.range=[0, Math.max(xs[xs.length-1], 0.1)];
      Plotly.react(div,[{x:xs,y:ys,mode:'lines',
        line:{color:'#22d3ee',width:2.6,shape:'spline',smoothing:0.3},
        fill:'tozeroy',fillcolor:'rgba(34,211,238,0.12)',
        hovertemplate:'%{x:.1f} min · %{y:.2f} '+(meta.u||'')+'<extra></extra>'}],
        lay,{responsive:true,displaylogo:false,displayModeBar:false});
    }
    window.mcUpdateCompare=updateCompare;
    var STRATPAL=['#3b82f6','#22d3ee','#a34bff','#f472b6','#1fd18a','#f4c778','#7ff0b8','#e2726e'];
    function stratColor(nm){ var M=INTER._matrix;
      if(!M._colors){ var seen={}, list=[];
        M.constellations.forEach(function(c){ Object.keys(M.data[c].strategies).forEach(function(s){
          if(!seen[s]){ seen[s]=1; list.push(s); } }); });
        list.sort(); M._colors={}; list.forEach(function(s,i){ M._colors[s]=STRATPAL[i%STRATPAL.length]; }); }
      return M._colors[nm]||'#8aa0c8'; }
    function renderDash(){
      var M=INTER&&INTER._matrix, host=document.getElementById('mcp-dash');
      if(!M||!M.constellations||!M.constellations.length){
        host.innerHTML='<div class="dnote">No KPI matrix in interactive.json — rebuild with the current build_interactive.</div>'; return; }
      if(!REP_CNST||M.constellations.indexOf(REP_CNST)<0) REP_CNST=M.constellations[0];
      var D=M.data[REP_CNST], strat=D.strategies, names=Object.keys(strat), sk=M.strat_kpis, ck=M.ctx_kpis;
      // ── hero tiles (best-in-class headline KPIs for this constellation) ──
      function bestBy(key,dir){ var bn=null,bv=null; names.forEach(function(n){ var v=strat[n][key];
        if(v==null||isNaN(v)) return; if(bv==null||(dir==='hi'?v>bv:v<bv)){ bv=v; bn=n; } }); return {n:bn,v:bv}; }
      var bS=bestBy('sinr_p50','hi'), bH=bestBy('ho_rate','lo');
      // Mean interval between handovers = 1 / handover-rate, derived from the
      // trusted ho_rate bar (consistent by construction). This is NOT ihi_p50 —
      // that is the median gap among users who actually hand over, a different
      // (and, when handovers are rare, noisier) statistic kept in the leaderboard.
      var mInt=(bH.v!=null&&bH.v>0)?(1.0/bH.v):null;
      var av=D.context.availability, avOk=(av==null||av>=99.0);
      var hero=[
        {g:'g1',lab:'Best median SINR',val:_fmt(bS.v,1),unit:'dB',sub:bS.n||''},
        {g:'g2',lab:'Availability',val:_fmt(av,1),unit:'%',sub:avOk?'SLA met':'below SLA'},
        {g:'g3',lab:'Lowest handover rate',val:_fmt(bH.v,3),unit:'/min',sub:bH.n||''},
        {g:'g4',lab:'Mean HO interval',val:_fmt(mInt,1),unit:'min',sub:(bH.n||'')+' · 1/rate'}
      ].map(function(t){ return '<div class="htile '+t.g+'"><div class="hl">'+t.lab+'</div>'+
        '<div class="hv">'+t.val+' <span class="hu">'+t.unit+'</span></div><div class="hs">'+t.sub+'</div></div>'; }).join('');
      // ── leaderboard heatmap (collapsible detail) ──
      var stats={}; sk.forEach(function(k){
        var vs=names.map(function(n){return strat[n][k.key];}).filter(function(v){return v!=null&&!isNaN(v);});
        stats[k.key]=vs.length?{mn:Math.min.apply(null,vs),mx:Math.max.apply(null,vs),better:k.better}:null; });
      function norm(key,v){ var s=stats[key]; if(!s||v==null||isNaN(v)) return null;
        if(s.mx===s.mn) return 1; var n=(v-s.mn)/(s.mx-s.mn); return s.better==='lo'?1-n:n; }
      var th='<th class="lft">strategy</th>'+sk.map(function(k){
        return '<th>'+k.label.replace(/ \(.*\)/,'')+'<br><span class="ku">'+k.unit+(k.better==='hi'?' ↑':' ↓')+'</span></th>'; }).join('');
      var rows=names.map(function(n){
        var tds=sk.map(function(k){ var v=strat[n][k.key], nm=norm(k.key,v);
          var st=(nm==null?'color:#7f93b8':'background:'+_colBg(nm)+';color:'+_colTx(nm)+(nm===1?';border:1.5px solid #37d98a;border-radius:4px':''));
          return '<td style="'+st+'">'+_fmt(v,_dec(k))+'</td>'; }).join('');
        return '<tr><td class="sn">'+n+'</td>'+tds+'</tr>'; }).join('');
      var leader='<table class="dtab"><tr class="dhr">'+th+'</tr>'+rows+'</table>';
      // ── Pareto trade-off + insights + SLA ──
      var ysel='<select id="repY" class="rsel">'+sk.filter(function(k){return k.key!=='ho_rate';}).map(function(k){
        return '<option value="'+k.key+'"'+(k.key===REP_YKEY?' selected':'')+'>'+k.label.replace(/ \(.*\)/,'')+'</option>'; }).join('')+'</select>';
      var dsel='<select id="repDonutSel" class="rsel">'+sk.map(function(k){
        return '<option value="'+k.key+'"'+(k.key===REP_DONUT?' selected':'')+'>'+k.label.replace(/ \(.*\)/,'')+'</option>'; }).join('')+'</select>';
      var pareto=_paretoSvg(names,strat,sk);
      var ctx=ck.map(function(k){ var v=D.context[k.key];
        return '<div class="ctile"><div class="cl">'+k.label.replace(/ \(.*\)/,'')+'</div><div class="cv">'+_fmt(v,1)+' <span class="cu">'+k.unit+'</span></div></div>'; }).join('');
      var chips=M.constellations.map(function(c){
        return '<span class="rchip'+(c===REP_CNST?' on':'')+'" data-c="'+c+'">'+c+'</span>'; }).join('');
      host.innerHTML=
        '<div class="drow"><div class="rchips">'+chips+'</div></div>'+
        '<div class="hgrid">'+hero+'</div>'+
        '<div class="dgrid2">'+
          '<div class="dcard"><div class="dttl">By strategy &nbsp;'+dsel+'</div><div id="repDonut" class="pch"></div></div>'+
          '<div class="dcard"><div class="dttl">Trade-off vs handover rate &nbsp;'+ysel+'</div>'+
            '<div class="pch">'+pareto+'</div>'+
            '<div class="dhint">frontier = non-dominated (min handovers, max metric)</div></div>'+
        '</div>'+
        '<div class="dcard"><div class="dttl">Strategy leaderboard — '+REP_CNST+'</div>'+
          '<div style="overflow-x:auto">'+leader+'</div></div>';
      Array.prototype.forEach.call(host.querySelectorAll('.rchip'),function(c){
        c.onclick=function(){ REP_CNST=c.getAttribute('data-c'); renderDash(); }; });
      var ys=document.getElementById('repY'); if(ys) ys.onchange=function(){ REP_YKEY=ys.value; renderDash(); };
      var ds=document.getElementById('repDonutSel'); if(ds) ds.onchange=function(){ REP_DONUT=ds.value; _plotDonut(names,strat); };
      _plotDonut(names,strat);
    }
    function _plotTrend(){
      var T=INTER._matrix.trend&&INTER._matrix.trend[REP_CNST], div=document.getElementById('repTrend');
      if(!div) return;
      if(!T||!T.series){ div.innerHTML='<div class="dnote">no trend data.</div>'; return; }
      var traces=Object.keys(T.series).map(function(nm){
        return {x:T.x,y:T.series[nm],name:nm,mode:'lines',connectgaps:true,
                line:{width:2,color:stratColor(nm)},hovertemplate:'%{x:.1f} min · %{y:.1f} dB<extra>'+nm+'</extra>'}; });
      var lay=JSON.parse(JSON.stringify(DARK)); lay.margin={l:46,r:14,t:8,b:34};
      lay.xaxis.title='time [min]'; lay.yaxis.title='median SINR [dB]';
      lay.legend={orientation:'h',y:1.16,x:1,xanchor:'right',font:{size:9.5},bgcolor:'rgba(0,0,0,0)',bordercolor:'rgba(0,0,0,0)'};
      Plotly.react(div,traces,lay,{responsive:true,displaylogo:false,displayModeBar:false});
      setTimeout(function(){ Plotly.Plots.resize(div); },50);
    }
    function _plotDonut(names,strat){
      var div=document.getElementById('repDonut'); if(!div) return;
      var key=REP_DONUT||'ho_rate';
      var meta=(INTER._matrix.strat_kpis.filter(function(k){return k.key===key;})[0])||{label:key,unit:''};
      var short=meta.label.replace(/ \(.*\)/,'');
      var keep=names.filter(function(n){ var v=strat[n][key]; return v!=null&&!isNaN(v)&&v>0; });
      if(!keep.length){ div.innerHTML='<div class="dnote">no positive values for this metric.</div>'; return; }
      var vals=keep.map(function(n){return strat[n][key];}), cols=keep.map(function(n){return stratColor(n);});
      var lay=JSON.parse(JSON.stringify(DARK)); lay.margin={l:8,r:8,t:8,b:8}; lay.showlegend=true;
      lay.legend={orientation:'v',font:{size:9.5},x:1,y:0.5,bgcolor:'rgba(0,0,0,0)'};
      lay.annotations=[{text:short+(meta.unit?'<br><span style="font-size:8px">['+meta.unit+']</span>':''),x:0.5,y:0.5,
        showarrow:false,font:{size:12,color:'#cfe0ff'}}];
      Plotly.react(div,[{type:'pie',hole:0.62,values:vals,labels:keep,sort:false,direction:'clockwise',
        marker:{colors:cols,line:{color:'#0a0f20',width:1.5}},textinfo:'percent',
        textfont:{size:9.5,color:'#fff'},
        hovertemplate:'%{label}: %{value:.3f} '+(meta.unit||'')+' (%{percent} of total)<extra></extra>'}],
        lay,{responsive:true,displaylogo:false,displayModeBar:false});
      setTimeout(function(){ Plotly.Plots.resize(div); },50);
    }
    function _paretoSvg(names,strat,sk){
      var yk=REP_YKEY, better=(sk.filter(function(k){return k.key===yk;})[0]||{}).better||'hi';
      var pts=names.map(function(n){return {n:n,x:strat[n].ho_rate,y:strat[n][yk]};})
        .filter(function(p){return p.x!=null&&p.y!=null&&!isNaN(p.x)&&!isNaN(p.y);});
      if(pts.length<2) return '<div class="dnote">not enough data for a trade-off.</div>';
      var xs=pts.map(function(p){return p.x;}), ys=pts.map(function(p){return p.y;});
      var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
      var px=(x1-x0)*0.15||0.05, py=(y1-y0)*0.15||1; x0-=px;x1+=px;y0-=py;y1+=py;
      var W=300,H=185,L=44,R=14,T=12,B=30;
      function sx(x){return L+(x-x0)/(x1-x0)*(W-L-R);}
      function sy(y){return T+(1-(y-y0)/(y1-y0))*(H-T-B);}
      // y better=hi means maximise y; if lo, maximise -y. minimise x (handovers).
      function yv(p){return better==='lo'?-p.y:p.y;}
      function dom(a,b){return b.x<=a.x&&yv(b)>=yv(a)&&(b.x<a.x||yv(b)>yv(a));}
      var front=pts.filter(function(p){return !pts.some(function(q){return dom(p,q);});})
        .sort(function(a,b){return a.x-b.x;});
      var poly=front.map(function(p){return sx(p.x).toFixed(1)+','+sy(p.y).toFixed(1);}).join(' ');
      // Build markers, then de-collide labels vertically so clustered strategies
      // (near-identical trade-off points) don't print on top of each other.
      var L=pts.map(function(p){ return {cx:sx(p.x),cy:sy(p.y),ly:sy(p.y)+3,n:p.n,onf:front.indexOf(p)>=0}; });
      L.sort(function(a,b){return a.ly-b.ly;});
      for(var li=1;li<L.length;li++){ if(L[li].ly-L[li-1].ly<10) L[li].ly=L[li-1].ly+10; }
      var dots=L.map(function(l){
        return '<circle cx="'+l.cx.toFixed(1)+'" cy="'+l.cy.toFixed(1)+'" r="4.5" fill="'+(l.onf?'#7ff0b8':'#f4c778')+'"'+(l.onf?' stroke="#fff" stroke-width="1"':'')+'/>'; }).join('')+
        L.map(function(l){
        return '<text x="'+(l.cx+7).toFixed(1)+'" y="'+l.ly.toFixed(1)+'" fill="#cfe0ff" font-size="8">'+l.n+'</text>'; }).join('');
      var yl=(sk.filter(function(k){return k.key===yk;})[0]||{}).label||yk;
      return '<svg viewBox="0 0 300 185" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%">'+
        '<line x1="'+L+'" y1="'+T+'" x2="'+L+'" y2="'+(H-B)+'" stroke="#3a4a6a"/>'+
        '<line x1="'+L+'" y1="'+(H-B)+'" x2="'+(W-R)+'" y2="'+(H-B)+'" stroke="#3a4a6a"/>'+
        '<text x="10" y="'+((H-B+T)/2)+'" fill="#9fb4d8" font-size="8.5" transform="rotate(-90 12,'+((H-B+T)/2)+')">'+yl+'</text>'+
        '<text x="'+((L+W-R)/2)+'" y="'+(H-6)+'" fill="#9fb4d8" font-size="8.5" text-anchor="middle">handover rate [/min]  (lower better →)</text>'+
        (poly?'<polyline points="'+poly+'" fill="none" stroke="#37d98a" stroke-width="1.4" stroke-dasharray="4 3"/>':'')+
        dots+'</svg>';
    }
    function _tile(kind,label,val){
      var bg=kind==='bad'?'#4a1f1f':kind==='warn'?'#3a2a12':'#12492f';
      var tc=kind==='bad'?'#ff9a9a':kind==='warn'?'#f4c778':'#7ff0b8';
      return '<div class="stile" style="background:'+bg+'"><div class="sl" style="color:'+tc+'">'+label+'</div>'+
        '<div class="sv" style="color:'+tc+'">'+val+'</div></div>';
    }
    function _dashCsv(){
      var M=INTER&&INTER._matrix; if(!M) return '';
      var sk=M.strat_kpis, ck=M.ctx_kpis;
      var head=['constellation','strategy'].concat(sk.map(function(k){return k.label+' ['+k.unit+']';}))
        .concat(ck.map(function(k){return k.label+' ['+k.unit+']';}));
      var lines=[head.join(',')];
      M.constellations.forEach(function(c){ var D=M.data[c];
        Object.keys(D.strategies).forEach(function(n){ var s=D.strategies[n];
          var row=[c,n].concat(sk.map(function(k){var v=s[k.key];return v==null?'':v;}))
            .concat(ck.map(function(k){var v=D.context[k.key];return v==null?'':v;}));
          lines.push(row.join(',')); }); });
      return lines.join('\n');
    }
    (function(){
      var pbtn=document.getElementById('palBtn'); if(pbtn) pbtn.onclick=cyclePalette;
      var c=document.getElementById('expCsv'), p=document.getElementById('expPdf');
      if(c) c.onclick=function(){ var blob=new Blob([_dashCsv()],{type:'text/csv'});
        var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
        a.download='reports_kpi_matrix.csv'; document.body.appendChild(a); a.click();
        setTimeout(function(){ URL.revokeObjectURL(a.href); a.remove(); },500); };
      if(p) p.onclick=function(){ var w=window.open('','_blank');
        if(!w){ window.print(); return; }
        w.document.write('<html><head><title>SkyNTN Reports</title></head><body style="background:#0a0f20;margin:16px">'+
          document.getElementById('mcp-dash').innerHTML+'</body></html>');
        w.document.close(); setTimeout(function(){ w.print(); },300); };
    })();
    items.forEach(function(it){ it.addEventListener('click', function(){
      items.forEach(function(x){ x.classList.remove('on'); }); it.classList.add('on');
      var k=it.getAttribute('data-k'), sel=document.getElementById('mcsel');
      if(k==='map'){ document.getElementById('mcpanel').style.display='none';
        flyToROI(1.6); if(sel) sel.style.display='none'; }
      else if(k==='overview'){ if(sel) sel.style.display='none'; openOverview(); }
      else if(k==='analytics'){ openAnalytics(); }
      else if(k==='compare'){ openCompare(); }
      else if(k==='reports'){ openReports(); }
    }); });
  })();
  // About / contact popup (the "?" button and the Polytechnique Montréal logo).
  (function(){
    var ab=document.getElementById('mcabout');
    function show(){ if(ab) ab.style.display='block'; }
    function hide(){ if(ab) ab.style.display='none'; }
    ['aboutBtn','brandLogo'].forEach(function(id){ var e=document.getElementById(id);
      if(e) e.addEventListener('click', show); });
    var x=document.getElementById('aboutClose'); if(x) x.addEventListener('click', hide);
    if(ab) ab.addEventListener('click', function(ev){ if(ev.target===ab) hide(); });
    document.addEventListener('keydown', function(ev){ if(ev.key==='Escape') hide(); });
  })();
  // ════════ ADVANCED TOOL RAIL ════════
  var mcLastPicked=null;
  function mcToast(msg){ var t=document.getElementById('mctoast'); if(!t) return;
    t.textContent=msg; t.style.opacity='1'; clearTimeout(t._h);
    t._h=setTimeout(function(){ t.style.opacity='0'; },2800); }
  function tlOn(id,on){ var e=document.getElementById(id); if(e) e.classList.toggle('on',!!on); }
  // Demo — cinematic auto-orbit
  var demoOn=false;
  viewer.clock.onTick.addEventListener(function(){
    if(demoOn) safe('demo', function(){ viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.0008); }); });
  (function(){ var b=document.getElementById('tlDemo'); if(b) b.onclick=function(){
    demoOn=!demoOn; tlOn('tlDemo',demoOn); mcToast(demoOn?'Demo: cinematic auto-orbit ON':'Demo OFF'); }; })();
  // Follow — track the selected entity
  (function(){ var b=document.getElementById('tlFollow'); if(b) b.onclick=function(){
    if(viewer.trackedEntity){ viewer.trackedEntity=undefined; tlOn('tlFollow',false); mcToast('Follow OFF'); }
    else if(mcLastPicked){ viewer.trackedEntity=mcLastPicked; tlOn('tlFollow',true);
      mcToast('Following '+(mcLastPicked.name||mcLastPicked.id)); }
    else mcToast('Click a satellite first, then Follow'); }; })();
  // Speed — cycle the playback multiplier
  (function(){ var b=document.getElementById('tlSpeed'), t=document.getElementById('tlSpeedTxt');
    var speeds=[1,2,5,10,30,60], i=1;
    if(b) b.onclick=function(){ i=(i+1)%speeds.length; viewer.clock.multiplier=speeds[i];
      if(t) t.textContent=speeds[i]+'×'; mcToast('Playback '+speeds[i]+'× real-time'); }; })();
  // Snapshot — the WHOLE view (globe + overlays) via html2canvas; falls back to
  // the globe canvas alone if html2canvas is unavailable or fails.
  function _dlPng(dataUrl,label){ var a=document.createElement('a'); a.href=dataUrl;
    a.download='skyntn_'+curCnst+'_'+Date.now()+'.png'; a.click(); mcToast(label); }
  (function(){ var b=document.getElementById('tlSnap'); if(b) b.onclick=function(){
    safe('snap', function(){ viewer.render();
      var gcv=viewer.scene.canvas;
      var W=window.innerWidth, H=window.innerHeight, dpr=Math.min(window.devicePixelRatio||1,2);
      // Composite: the WebGL globe (bottom) + the DOM overlays (top). html2canvas
      // can't reliably rasterise a WebGL canvas, so we IGNORE the globe subtree and
      // let it render only the UI panels on a transparent layer, then draw both.
      function compose(overlay){
        var out=document.createElement('canvas'); out.width=Math.round(W*dpr); out.height=Math.round(H*dpr);
        var ctx=out.getContext('2d');
        try{ ctx.drawImage(gcv,0,0,out.width,out.height); }catch(e){}
        if(overlay){ try{ ctx.drawImage(overlay,0,0,out.width,out.height); }catch(e){} }
        _dlPng(out.toDataURL('image/png'), overlay?'Snapshot saved (whole view)':'Snapshot saved (globe only)');
      }
      if(window.html2canvas){
        mcToast('Capturing whole view…');
        html2canvas(document.body,{backgroundColor:null,logging:false,useCORS:true,scale:dpr,
          width:W,height:H,windowWidth:W,windowHeight:H,
          ignoreElements:function(el){ return el.id==='cesiumContainer'
            || (el.classList && (el.classList.contains('cesium-widget')||el.classList.contains('cesium-viewer'))); }})
          .then(compose).catch(function(){ compose(null); });
      } else { compose(null); }
    }); }; })();
  // Record — SMOOTH video. Primary path captures the WebGL globe canvas directly
  // at 60 fps (GPU-paced, independent of the desktop compositor and window focus,
  // so no dropped/duplicated frames), at a high bitrate. Globe only. If the canvas
  // can't be captured, fall back to the whole-view screen-capture API.
  (function(){ var b=document.getElementById('tlRec'); if(!b) return; var rec=null, chunks=[];
    function record(stream, whole){
      chunks=[]; var mt=(window.MediaRecorder&&MediaRecorder.isTypeSupported('video/webm;codecs=vp9'))?
        'video/webm;codecs=vp9':'video/webm';
      try{ rec=new MediaRecorder(stream,{mimeType:mt, videoBitsPerSecond:16000000}); }
      catch(e){ rec=new MediaRecorder(stream); }
      rec.ondataavailable=function(ev){ if(ev.data&&ev.data.size) chunks.push(ev.data); };
      rec.onstop=function(){ var blob=new Blob(chunks,{type:'video/webm'});
        var a=document.createElement('a'); a.href=URL.createObjectURL(blob);
        a.download='skyntn_'+curCnst+'_'+Date.now()+'.webm'; a.click();
        setTimeout(function(){ URL.revokeObjectURL(a.href); },1000);
        stream.getTracks().forEach(function(t){ t.stop(); }); tlOn('tlRec',false);
        mcToast('Video saved (WEBM'+(whole?', whole view':', globe @60 fps')+')'); };
      var vt=stream.getVideoTracks()[0]; if(vt) vt.onended=function(){ if(rec&&rec.state==='recording') rec.stop(); };
      rec.start(); tlOn('tlRec',true);
      mcToast('Recording '+(whole?'whole view':'globe @60 fps')+'… click ⏺ again to stop');
    }
    b.onclick=function(){
      if(rec && rec.state==='recording'){ rec.stop(); return; }
      var gcv=viewer.scene.canvas, s=null;
      if(gcv && gcv.captureStream){ try{ s=gcv.captureStream(60); }catch(e){ s=null; } }
      if(s){ record(s,false); return; }
      if(navigator.mediaDevices&&navigator.mediaDevices.getDisplayMedia){
        navigator.mediaDevices.getDisplayMedia({video:{frameRate:60},audio:false})
          .then(function(st){ record(st,true); }).catch(function(){ mcToast('Recording cancelled'); });
      } else { mcToast('Recording not supported in this browser'); }
    }; })();
  // Measure — two clicks → great-circle ground distance
  var measOn=false, measPts=[], measMarker=null;
  (function(){ var b=document.getElementById('tlMeasure'); if(b) b.onclick=function(){
    measOn=!measOn; measPts=[]; tlOn('tlMeasure',measOn);
    var m=document.getElementById('mcmeasure'); m.style.display=measOn?'block':'none';
    m.textContent=measOn?'Measure: click two points on the globe':'';
    if(measMarker){ viewer.entities.remove(measMarker); measMarker=null; } }; })();
  (function(){ var h=new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    h.setInputAction(function(click){
      if(!measOn) return;
      var c=null; safe('pickpos', function(){ c=viewer.scene.pickPosition(click.position); });
      if(!Cesium.defined(c)) c=viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
      if(!Cesium.defined(c)) return;
      measPts.push(c);
      if(measPts.length===2){
        var g=new Cesium.EllipsoidGeodesic(Cesium.Cartographic.fromCartesian(measPts[0]),
                                            Cesium.Cartographic.fromCartesian(measPts[1]));
        document.getElementById('mcmeasure').textContent='Ground distance: '+(g.surfaceDistance/1000).toFixed(1)+' km';
        if(measMarker) viewer.entities.remove(measMarker);
        measMarker=viewer.entities.add({ polyline:{ positions:measPts.slice(), width:3,
          material:new Cesium.PolylineDashMaterialProperty({color:Cesium.Color.fromCssColorString('#6fd3ff')}) }});
        measPts=[];
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK); })();
  // Layers popover — toggle entity types + atmosphere + all-sats
  (function(){ var b=document.getElementById('tlLayers'), pop=document.getElementById('mclayerpop');
    if(b) b.onclick=function(){ var sh=pop.style.display!=='block'; pop.style.display=sh?'block':'none'; tlOn('tlLayers',sh); };
    function isU(e){ var id=String(e.id||'').toLowerCase(); return id.indexOf('user')>=0||id.indexOf('usr')>=0||id.indexOf('ue')>=0; }
    function isG(e){ var id=String(e.id||'').toLowerCase(); return id.indexOf('gw')>=0||id.indexOf('gate')>=0; }
    function setType(pred,on){ if(!currentDS) return; currentDS.entities.values.forEach(function(e){ if(pred(e)) e.show=on; }); }
    var U=document.getElementById('lpUsers'),G=document.getElementById('lpGw'),L=document.getElementById('lpLinks'),
        A=document.getElementById('lpAtm'),AL=document.getElementById('lpAll');
    if(U) U.onchange=function(){ setType(isU,U.checked); };
    if(G) G.onchange=function(){ setType(isG,G.checked); };
    if(L) L.onchange=function(){ setType(function(e){return !!e.polyline;}, L.checked); };
    if(A) A.onchange=function(){ safe('atm',function(){ scene.skyAtmosphere.show=A.checked;
      scene.globe.showGroundAtmosphere=A.checked; scene.globe.enableLighting=A.checked; }); };
    if(AL) AL.onchange=function(){ setAllSats(); };
  })();
  // Status bar last-update clock
  viewer.clock.onTick.addEventListener(function(clock){
    mcTxt('st_upd','LAST UPDATE: '+new Date().toLocaleTimeString());
  });
</script>
</body>
</html>"""


def _load_stream(rd: Path, stream: str) -> pd.DataFrame:
    """Load a run output stream via the canonical run reader."""
    from ntnconnect.common.utils.run_io import read_run_table
    return read_run_table(rd, stream)


def _discover_runs(scan_roots: list[Path]) -> dict:
    """Map (constellation, strategy) -> run_dir for every run with SINR data (first match wins)."""
    found: dict[tuple[str, str], Path] = {}
    for root in scan_roots:
        if not root.exists(): continue
        # A "cell" is a <cnst>__<strat> dir; run data is in run_* or the cell itself
        for cell in sorted(root.glob("**/*__*")):
            if not cell.is_dir(): continue
            name = cell.name
            if "__" not in name: continue
            cnst_, strat_ = name.split("__", 1)
            if cnst_ not in _OP_LABEL: continue
            if (cnst_, strat_) in found: continue
            # run_* directories only (not files); else the cell holds the run data
            from ntnconnect.common.utils.run_io import has_run_table
            for rd in (sorted(d for d in cell.glob("run_*") if d.is_dir()) or [cell]):
                if rd.is_dir() and has_run_table(rd, "sinr_snapshot"):
                    found[(cnst_, strat_)] = rd
                    break
    return found


def _resolve_eph(eph_dir: Path, cnst: str) -> Path:
    """Resolve the ephemeris directory holding this constellation's satellites."""
    p = Path(eph_dir)
    def _has(d: Path) -> bool:
        s = d / "satellites"
        return bool(list(s.glob(f"{cnst}*.parquet"))) if s.is_dir() else False
    if _has(p) or _has(p / "coverage"):
        return p
    posix = p.as_posix()
    for other in ("oneweb", "starlink", "kuiper"):
        if other == cnst:
            continue
        if f"/{other}/" in posix or posix.endswith(f"/{other}"):
            cand = Path(posix.replace(f"/{other}/", f"/{cnst}/")
                        if f"/{other}/" in posix
                        else posix[: -len(other)] + cnst)
            if _has(cand) or _has(cand / "coverage"):
                return cand
    return p


def _build_scene(eph_dir: Path, cnst: str, batch: int, run_dir: Path,
                 step_s: float, orbits: bool) -> dict:
    eph_dir = _resolve_eph(eph_dir, cnst)
    sin = _load_stream(run_dir, "sinr_snapshot")
    ho  = _load_stream(run_dir, "handover_events")
    czml = build_czml_json(
        eph_dir, cnst, batch,
        downsample_every_n=20, include_users=True, include_gateways=True,
        include_orbits=orbits,
        sinr_snapshot=sin if not sin.empty else None,
        ho_events=ho if not ho.empty else None,
        phase2_step_s=step_s, flash_duration_s=10.0,
        all_batches=True)
    stats = build_tick_stats(
        eph_dir, cnst, batch,
        sin if not sin.empty else None,
        ho if not ho.empty else None, phase2_step_s=step_s,
        all_batches=True)
    return {"czml": json.loads(czml), "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                     help="Force a single run dir (otherwise auto-discover all)")
    ap.add_argument("--eph-dir", type=Path,
                     default=_V1 / "outputs" / "ephemeris")
    ap.add_argument("--scan-root", action="append", type=Path, default=None,
                     help="Root(s) to search for <cnst>__<strat>/run_* dirs "
                          "(repeatable). Order = preference.")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--step-s", type=float, default=10.0)
    ap.add_argument("--orbits", action="store_true", help="show orbit trails")
    ap.add_argument("--out", type=Path,
                     default=_V1 / "outputs" / "cesium_view.html")
    ap.add_argument("--port", type=int, default=0, help="http port (0=auto)")
    ap.add_argument("--open", action="store_true", help="serve + open in browser")
    args = ap.parse_args()

    manifest = json.loads((args.eph_dir / "manifest.json").read_text())
    # Derive n_batches when a new dataset omits it
    if "n_batches" in manifest:
        n_batches = int(manifest["n_batches"])
    elif "batch_size_s" in manifest and float(manifest["batch_size_s"]) > 0:
        _dur = float(manifest.get("hours", 0) * 3600.0) or float(manifest["batch_size_s"])
        n_batches = max(1, int(round(_dur / float(manifest["batch_size_s"]))))
    else:
        n_batches = 1

    # Decide which runs to embed
    if args.run_dir:
        name = args.run_dir.parent.name
        cnst_, strat_ = (name.split("__", 1) if "__" in name
                         else ("oneweb", "max_sinr"))
        runs = {(cnst_, strat_): args.run_dir}
    else:
        v1_out = _V1 / "outputs"
        roots = args.scan_root or [v1_out / "runs", v1_out]
        runs = _discover_runs([Path(r) for r in roots])
    if not runs:
        print("No runs with SINR data found. Pass --run-dir or --scan-root.")
        sys.exit(1)

    # Auto-detect the true SINR/link tick step (sub-slot) from resource_profile
    _rs = None
    for _rd in runs.values():
        _rp = Path(_rd) / "resource_profile.json"
        if _rp.is_file():
            try:
                _v = float(json.loads(_rp.read_text()).get("step_size_s") or 0.0)
                if _v > 0:
                    _rs = _v; break
            except Exception:
                pass
    if _rs and abs(_rs - float(args.step_s)) > 1e-9:
        print(f"  auto-detected SINR/link step {_rs}s (was --step-s {args.step_s})")
        args.step_s = _rs

    # Build the nested SCENES structure
    scenes: dict[str, dict] = {}
    for (cnst, strat), rd in sorted(runs.items()):
        print(f"  embedding {cnst}/{strat}  <- {rd}")
        sc = _build_scene(args.eph_dir, cnst, args.batch, rd,
                          args.step_s, args.orbits)
        scenes.setdefault(cnst, {
            "label": _OP_LABEL.get(cnst, cnst),
            "batch": args.batch, "nbatch": n_batches - 1,
            "strategies": {}})
        scenes[cnst]["strategies"][strat] = {
            "label": _STRAT_LABEL.get(strat, strat),
            "czml": sc["czml"], "stats": sc["stats"]}
        # Full-constellation layer, built once per constellation
        if "allCzml" not in scenes[cnst]:
            try:
                # Prefer the coverage sub-layer (full constellation) when present
                _eph_c = _resolve_eph(args.eph_dir, cnst)
                if (_eph_c / "coverage" / "satellites").is_dir():
                    _eph_c = _eph_c / "coverage"
                acz = build_all_sats_czml_json(_eph_c, cnst, args.batch,
                                               downsample_every_n=6,
                                               all_batches=True)
                acz = json.loads(acz)
                scenes[cnst]["allCzml"] = acz if len(acz) > 1 else None
                print(f"    + all-sats layer: {len(acz)-1 if acz else 0} sats")
            except Exception as e:
                print(f"    ! all-sats layer skipped: {e}")
                scenes[cnst]["allCzml"] = None

    n_combos = sum(len(c["strategies"]) for c in scenes.values())
    html = (CESIUM_HTML
            .replace("__VER__", "1.123")
            .replace("__SCENES__", json.dumps(scenes)))
    args.out.write_text(html, encoding="utf-8")
    print(f"\nwrote standalone Cesium view -> {args.out}  "
          f"({len(html)/1e6:.2f} MB, {len(scenes)} constellations, "
          f"{n_combos} algorithm combos)")

    # Serve over http (file:// breaks Cesium's tile workers)
    import http.server, socketserver, threading, functools
    out_dir = args.out.resolve().parent
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(out_dir))
    httpd = socketserver.TCPServer(("127.0.0.1", args.port), handler)
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}/{args.out.name}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"serving over http (required for the 3D globe): {url}")
    if args.open:
        webbrowser.open(url)
        print("opened in your browser. Press Ctrl+C here to stop the server.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nserver stopped.")
    else:
        print("re-run with --open to launch it, or open the URL above. "
              "Do NOT open the .html as file:// — the globe needs http.")


if __name__ == "__main__":
    main()
