"""SkyNTNPro - live (slot-by-slot) 3D dashboard.

Streamlit app showing:
  * a 3D globe of Earth with satellites at their ECEF positions
  * users (X markers, on-ground)
  * per-constellation Canadian gateways (square markers)
  * served-link arrows from each user to its currently-served satellite
  * a "handover flash" highlight on satellites that handed over within the
    last N ticks
  * a side panel with live KPIs read from the chosen run directory

Reads slot-by-slot from:
  data/ephemeris_12h/<cnst>/visible_sats_NN.parquet        (sat ECEF + LLA)
  data/ephemeris_12h/users.parquet                  (Quebec user grid)
  data/ephemeris_12h/<cnst>/gateways.parquet        (5 Canadian sites)
  <run_dir>/{visibility_per_tick,sinr_snapshot,handover_events}_batchNN.parquet
                                                    (when a run dir is supplied)

Run:
   pip install streamlit          # one-time
   streamlit run v1/scenarios/handover_compare/live_dashboard.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

_HERE = Path(__file__).resolve().parent          # .../ntnconnect/visualization
if str(_HERE) not in sys.path: sys.path.insert(0, str(_HERE))
# Works both as a package module (relative) and as a streamlit script (sibling).
try:
    from ntnconnect.analysis.visualization.czml import build_czml_json, build_tick_stats
except ImportError:
    from czml import build_czml_json, build_tick_stats

# Paths and constants
_REPO_ROOT     = _HERE.parents[2]                         # worktree root
_V1            = _HERE.parents[1]                          # .../v1
EPHEMERIS_DIR  = _V1 / "outputs" / "ephemeris"
_R_EQ          = 6378137.0
_OP_COLOR      = {"oneweb": "#3399ff",
                  "starlink": "#ff9933",
                  "kuiper": "#33cc66"}
_OP_LABEL      = {"oneweb": "OneWeb",
                  "starlink": "Starlink",
                  "kuiper": "Kuiper"}

# Streamlit configuration
st.set_page_config(page_title="SkyNTNPro Live", layout="wide",
                    page_icon="🛰️")
st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 1rem; }
  div[data-testid="stMetric"] { background: #1a1f2e; padding: 0.6rem;
                                 border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️  SkyNTNPro — Live 3D Dashboard")
st.caption("3D globe + slot-by-slot replay from the 12-hour ephemeris dataset")


# CesiumJS HTML template (stars + atmosphere + glassmorphism HUD +
# legend + clock-driven live statistics). Placeholders are filled in
# per render: __VER__, __CZML_DATA__, __STATS_DATA__, __OP_TITLE__,
# __BATCH__, __NBATCH__.
_CESIUM_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SkyNTNPro Live</title>
<script src="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Cesium.js"></script>
<link href="https://cesium.com/downloads/cesiumjs/releases/__VER__/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>
  html, body, #cesiumContainer {
    width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
    background: radial-gradient(ellipse at center, #060a18 0%, #01030a 100%);
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  /* ── Glassmorphism HUD (top-left) ── */
  #hud {
    position: absolute; top: 14px; left: 14px; z-index: 999;
    min-width: 230px; padding: 14px 16px;
    background: rgba(10, 16, 30, 0.55);
    backdrop-filter: blur(9px) saturate(140%);
    -webkit-backdrop-filter: blur(9px) saturate(140%);
    border: 1px solid rgba(120, 170, 255, 0.25);
    border-radius: 14px;
    box-shadow: 0 6px 26px rgba(0,0,0,0.45);
    color: #eaf2ff;
  }
  #hud .title {
    font-size: 15px; font-weight: 700; letter-spacing: .3px;
    background: linear-gradient(90deg,#6fd3ff,#9be7a8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
  }
  #hud .sub { font-size: 11px; color: #9fb4d8; margin-bottom: 10px; }
  #hud .row { display:flex; justify-content:space-between; align-items:baseline;
              margin: 5px 0; }
  #hud .k { font-size: 11px; color:#9fb4d8; }
  #hud .v { font-size: 19px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #hud .v.sm { font-size: 14px; }
  #clk { font-size: 12px; color:#7fd0ff; margin-top:8px;
         font-variant-numeric: tabular-nums; }
  /* ── Legend (bottom-left) ── */
  #legend {
    position:absolute; bottom: 36px; left: 14px; z-index: 999;
    padding: 9px 12px; font-size: 11px; color:#dce8ff;
    background: rgba(10,16,30,0.5); backdrop-filter: blur(7px);
    border:1px solid rgba(120,170,255,0.2); border-radius: 10px;
  }
  #legend .item { display:flex; align-items:center; margin: 3px 0; }
  #legend .dot { width:11px; height:11px; border-radius:50%; margin-right:8px;
                 box-shadow:0 0 6px currentColor; }
  #legend .sq  { width:11px; height:11px; margin-right:8px; }
  #legend .ln  { width:16px; height:3px; margin-right:8px; border-radius:2px; }
</style>
</head>
<body>
<div id="cesiumContainer"></div>

<div id="hud">
  <div class="title">SkyNTNPro · __OP_TITLE__</div>
  <div class="sub">batch __BATCH__ / __NBATCH__ · live link & handover view</div>
  <div class="row"><span class="k">Served users</span><span class="v" id="s_served">–</span></div>
  <div class="row"><span class="k">Median SINR</span><span class="v" id="s_med">– <span style="font-size:11px">dB</span></span></div>
  <div class="row"><span class="k">5th-pct SINR</span><span class="v sm" id="s_p5">– dB</span></div>
  <div class="row"><span class="k">Handovers (cum.)</span><span class="v" id="s_cho">0</span></div>
  <div id="clk">t = 0 s</div>
</div>

<div id="legend">
  <div class="item"><span class="dot" style="background:#ff9a3c;color:#ff9a3c"></span>satellites</div>
  <div class="item"><span class="dot" style="background:#ff466e;color:#ff466e"></span>users</div>
  <div class="item"><span class="sq"  style="background:#00e676"></span>gateways</div>
  <div class="item"><span class="ln"  style="background:#5affaa"></span>served link</div>
  <div class="item"><span class="dot" style="background:#fff14b;color:#fff14b"></span>handover flash</div>
</div>

<script>
  // Avoid any Cesium Ion token requirement (we use bundled imagery).
  Cesium.Ion.defaultAccessToken = '';
  // NOTE: in Cesium >=1.104 use `baseLayer: false` (NOT imageryProvider).
  // Use Cesium's BUNDLED Natural Earth II world imagery (ships with the CDN
  // build under Assets/Textures/NaturalEarthII — no Ion token, no external
  // tile server). Canonical token-free offline globe, renders reliably.
  const viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayer: Cesium.ImageryLayer.fromProviderAsync(
      Cesium.TileMapServiceImageryProvider.fromUrl(
        Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII'))),
    baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false,
    timeline: true, animation: true, infoBox: true,
    selectionIndicator: true, fullscreenButton: true
  });
  viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#1d4e89');
  // Beautify the scene: stars, atmosphere glow, hide credits.
  const scene = viewer.scene;
  // Lighting OFF → full Earth texture always lit (no black night side).
  scene.globe.enableLighting = false;
  scene.globe.showGroundAtmosphere = true;
  scene.skyAtmosphere.show = true;
  scene.skyAtmosphere.brightnessShift = 0.35;
  scene.skyAtmosphere.saturationShift = 0.15;
  scene.backgroundColor = Cesium.Color.fromCssColorString('#01030a');
  viewer.cesiumWidget.creditContainer.style.display = 'none';

  // Load the CZML.
  const czml = __CZML_DATA__;
  const stats = __STATS_DATA__;   // [{t_s, n, med, p5, cho}, ...]

  const dsPromise = Cesium.CzmlDataSource.load(czml);
  viewer.dataSources.add(dsPromise);
  let clockStart = null;
  dsPromise.then(function(ds){
    viewer.clock.shouldAnimate = true;
    viewer.clock.multiplier = 30;
    clockStart = ds.clock ? ds.clock.startTime : viewer.clock.startTime;
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(-85.0, 56.0, 14_000_000),
      orientation: { heading: 0.0,
                     pitch: Cesium.Math.toRadians(-72.0), roll: 0.0 },
      duration: 1.6
    });
  });

  // Live HUD: update statistics from the per-tick `stats` table as the
  // Cesium clock advances. We pick the latest stats row with t_s <= now.
  const elServed = document.getElementById('s_served');
  const elMed    = document.getElementById('s_med');
  const elP5     = document.getElementById('s_p5');
  const elCho    = document.getElementById('s_cho');
  const elClk    = document.getElementById('clk');
  viewer.clock.onTick.addEventListener(function(clock){
    if (!clockStart) return;
    const elapsed = Cesium.JulianDate.secondsDifference(
        clock.currentTime, clockStart);
    elClk.textContent = 't = ' + elapsed.toFixed(0) + ' s (sim)';
    // binary-ish linear scan for the latest row <= elapsed
    let row = null;
    for (let i = 0; i < stats.length; i++) {
      if (stats[i].t_s <= elapsed) row = stats[i]; else break;
    }
    if (row) {
      elServed.textContent = row.n;
      elMed.innerHTML = row.med.toFixed(1) +
          ' <span style="font-size:11px">dB</span>';
      elP5.textContent = row.p5.toFixed(1) + ' dB';
      elCho.textContent = row.cho;
      // colour the median by quality
      const c = row.med >= -8 ? '#9be7a8' : row.med >= -12 ? '#ffd166' : '#ff7a7a';
      elMed.style.color = c;
    }
  });
</script>
</body>
</html>"""


# Cached loaders
@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((EPHEMERIS_DIR / "manifest.json").read_text())


@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    return pd.read_parquet(EPHEMERIS_DIR / "users.parquet")


@st.cache_data(show_spinner=False)
def load_gateways(cnst: str) -> pd.DataFrame:
    return pd.read_parquet(EPHEMERIS_DIR / cnst / "gateways.parquet")


@st.cache_data(show_spinner=False)
def load_eph_batch(cnst: str, batch_idx: int) -> pd.DataFrame:
    p = EPHEMERIS_DIR / cnst / f"visible_sats_{batch_idx:02d}.parquet"
    if not p.exists(): return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_run_stream(run_dir: str, stream: str) -> pd.DataFrame:
    rd = Path(run_dir)
    if not rd.exists(): return pd.DataFrame()
    shards = sorted(rd.glob(f"{stream}*.parquet"))
    if not shards: return pd.DataFrame()
    try:
        return pd.concat([pd.read_parquet(s) for s in shards],
                          ignore_index=True)
    except Exception:
        return pd.DataFrame()


# Geometry helpers
def lla_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h   = np.asarray(alt_m, dtype=float)
    R = _R_EQ + h
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def make_earth_sphere() -> go.Surface:
    """A solid blue-ocean Earth sphere with continent shading via z-modulated
    colour. Lat-lon parametrisation. Lightweight, no texture image."""
    phi   = np.linspace(0, np.pi, 60)
    theta = np.linspace(0, 2 * np.pi, 90)
    phi, theta = np.meshgrid(phi, theta)
    r = _R_EQ * 0.999
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#0a2342"], [0.45, "#163e6c"],
                     [0.55, "#1c4f87"], [1, "#2b6fb4"]],
        surfacecolor=z, showscale=False,
        opacity=0.95, hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.75, specular=0.18),
        lightposition=dict(x=2e7, y=0, z=2e7),
        name="Earth")


def _scatter3d_sats(df: pd.DataFrame, op_color: str,
                     label: str, ho_set: set | None = None) -> go.Scatter3d:
    """Per-operator satellites; flash recently-handed-over sats in yellow."""
    if df.empty:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers",
                              name=label, showlegend=False)
    x = df["x"].to_numpy(); y = df["y"].to_numpy(); z = df["z"].to_numpy()
    colors = []
    sizes  = []
    for sid in df["node_id"].astype(int):
        if ho_set and int(sid) in ho_set:
            colors.append("#ffeb3b")   # flash yellow
            sizes.append(8)
        else:
            colors.append(op_color)
            sizes.append(3.2)
    hover = [f"{label} sat {n}<br>elev: lat {la:.1f} lon {lo:.1f}"
              for n, la, lo in zip(df["node_id"], df["lat"], df["lon"])]
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=sizes, color=colors,
                     line=dict(width=0)),
        name=label, hoverinfo="text", text=hover)


def _scatter3d_users(users: pd.DataFrame) -> go.Scatter3d:
    if users.empty:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers",
                              name="Users", showlegend=False)
    ux, uy, uz = lla_to_ecef(users["lat"].to_numpy(),
                              users["lon"].to_numpy())
    hover = [f"user {int(u)}<br>{la:.2f}°N, {lo:.2f}°W"
              for u, la, lo in zip(users["user_id"], users["lat"],
                                     users["lon"])]
    return go.Scatter3d(
        x=ux, y=uy, z=uz, mode="markers",
        marker=dict(size=4, color="#ff3366", symbol="x", line=dict(width=0)),
        name=f"Users ({len(users)})",
        hoverinfo="text", text=hover)


def _scatter3d_gateways(gws: pd.DataFrame) -> go.Scatter3d:
    if gws.empty:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers",
                              name="Gateways", showlegend=False)
    gx, gy, gz = lla_to_ecef(gws["lat"].to_numpy(),
                              gws["lon"].to_numpy(),
                              gws["alt_m"].to_numpy())
    hover = [f"<b>{n}</b><br>{la:.2f}°N, {lo:.2f}°W"
              for n, la, lo in zip(gws["name"], gws["lat"], gws["lon"])]
    return go.Scatter3d(
        x=gx, y=gy, z=gz, mode="markers+text",
        marker=dict(size=8, color="#00e676", symbol="square",
                     line=dict(width=1, color="white")),
        text=gws["name"], textposition="top center",
        textfont=dict(size=9, color="white"),
        name=f"Gateways ({len(gws)})",
        hoverinfo="text", hovertext=hover)


def _served_link_lines(sats: pd.DataFrame, users: pd.DataFrame,
                        assigns: dict) -> go.Scatter3d:
    """One polyline per served (user, sat) pair this tick."""
    if not assigns or sats.empty or users.empty:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines",
                              name="Served links", showlegend=False)
    user_xyz = {int(u): lla_to_ecef(la, lo)
                 for u, la, lo in zip(users["user_id"], users["lat"],
                                        users["lon"])}
    sat_xyz = {int(s): (float(x), float(y), float(z))
                for s, x, y, z in zip(sats["node_id"], sats["x"],
                                        sats["y"], sats["z"])}
    xs, ys, zs = [], [], []
    for uid, sid in assigns.items():
        if int(uid) not in user_xyz or int(sid) not in sat_xyz: continue
        ux_, uy_, uz_ = user_xyz[int(uid)]
        sx_, sy_, sz_ = sat_xyz[int(sid)]
        # Each as a 2-vertex polyline with None separators
        xs.extend([float(ux_), sx_, None])
        ys.extend([float(uy_), sy_, None])
        zs.extend([float(uz_), sz_, None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="#80ff80", width=1.5),
        opacity=0.65, hoverinfo="skip",
        name=f"Served links ({len(assigns)})")


# Sidebar (controls)
try:
    manifest = load_manifest()
    fine_step_s = float(manifest["fine_step_s"])
    batch_min   = float(manifest["batch_min"])
    n_batches   = int(manifest["n_batches"])
    ticks_per_batch = int(round(batch_min * 60.0 / fine_step_s))
except Exception as e:
    st.error(f"Cannot load ephemeris manifest at {EPHEMERIS_DIR}. "
              f"Did you run `build_ephemeris.py`?  ({e})")
    st.stop()

@st.cache_data(show_spinner=False)
def discover_runs() -> dict:
    """Auto-discover all Phase-2 run directories on disk so the user picks
    constellation + handover strategy from dropdowns (no manual paths).
    Returns {(constellation, strategy): run_dir_path}."""
    roots = [
        _V1 / "outputs" / "runs",
        _V1 / "outputs",
    ]
    found: dict = {}
    for root in roots:
        if not root.exists(): continue
        # Two layouts: <root>/skyntn_*/<cnst>__<strat>/run_NNN
        #  and <root>/<cnst>__<strat>/run_NNN
        for cell in list(root.glob("*/*__*/run_*")) + list(root.glob("*__*/run_*")):
            if not cell.is_dir(): continue
            name = cell.parent.name           # "<cnst>__<strat>"
            if "__" not in name: continue
            cnst_, strat_ = name.split("__", 1)
            if cnst_ not in ("oneweb", "starlink", "kuiper"): continue
            # Only keep runs that actually have SINR data (so overlays work)
            if not (list(cell.glob("sinr_snapshot*.parquet"))): continue
            found[(cnst_, strat_)] = cell
    return found


_RUNS = discover_runs()

with st.sidebar:
    use_cesium = True   # CesiumJS is the renderer
    st.header("Scenario")

    # Constellation + strategy come from discovered runs when available
    avail_cnst = sorted({c for (c, s) in _RUNS}) or ["oneweb", "starlink", "kuiper"]
    cnst = st.selectbox("Constellation", avail_cnst,
                         format_func=lambda c: _OP_LABEL.get(c, c))
    avail_strat = sorted({s for (c, s) in _RUNS if c == cnst})
    if avail_strat:
        strategy = st.selectbox("Handover algorithm", avail_strat,
                                 help="Discovered automatically from your "
                                      "Phase-2 run outputs.")
        run_dir_input = str(_RUNS[(cnst, strategy)])
        st.success(f"✓ data: {cnst}/{strategy}")
    else:
        strategy = None
        run_dir_input = ""
        st.warning("No runs with SINR data found on disk. The globe will "
                    "still show satellites/users/gateways, but links, "
                    "handover flashes and stats need a Phase-2 run.\n\n"
                    "Run e.g.:\n`run_with_ephemeris.py --constellation "
                    f"{cnst} --strategies max_sinr`")

    st.markdown("---")
    st.subheader("Time slot")
    batch_idx = st.slider("Batch (15-min slot)", 0, n_batches - 1, 0)
    tick_in_batch = st.slider("Fine tick within batch",
                                0, ticks_per_batch - 1, 0)
    st.markdown("---")
    st.subheader("Playback")
    auto = st.checkbox("▶ Auto-play", value=False)
    step_size = st.slider("Tick advance per refresh", 1, 60, 10)
    refresh_hz = st.slider("Refresh rate [Hz]", 1, 10, 2)
    st.markdown("---")
    with st.expander("Advanced"):
        manual = st.text_input("Manual run dir (override)", value="")
        if manual.strip():
            run_dir_input = manual.strip()
        flash_window = st.slider("HO flash window [ticks]", 0, 30, 5)
        show_orbits = st.checkbox("Show orbit trails", value=False)

# Data for the current slot
eph_batch = load_eph_batch(cnst, batch_idx)
users     = load_users()
gateways  = load_gateways(cnst)
if eph_batch.empty:
    st.warning(f"No ephemeris batch found for {cnst} batch {batch_idx}.")
    st.stop()
sats_at_t = eph_batch[eph_batch["t_idx"] == tick_in_batch]

# Try to read the run streams if a run dir was supplied
sin = load_run_stream(run_dir_input, "sinr_snapshot") if run_dir_input else pd.DataFrame()
ho  = load_run_stream(run_dir_input, "handover_events") if run_dir_input else pd.DataFrame()
vis = load_run_stream(run_dir_input, "visibility_per_tick") if run_dir_input else pd.DataFrame()

# Map fine-tick index to engine-tick index in the run streams.
# Engine ticks are at the Phase-2 step_size_s; ephemeris is at fine_step_s.
# Without external knowledge we assume Phase-2 step == fine_step; the user
# can pass any run dir but stats will be best aligned when the two match.
global_t = batch_idx * ticks_per_batch + tick_in_batch

assigns: dict[int, int] = {}
ho_set: set[int] = set()
if not sin.empty and "t_idx" in sin.columns:
    # Pick rows where the (run) t_idx matches the current global tick.
    served_now = sin[sin["t_idx"] == global_t]
    if not served_now.empty:
        assigns = {int(u): int(s) for u, s in
                   zip(served_now["user_id"], served_now["sat_id"])}
if not ho.empty and "t_idx" in ho.columns and flash_window > 0:
    recent = ho[(ho["t_idx"] >= global_t - flash_window) &
                (ho["t_idx"] <= global_t)]
    if "is_handover" in recent.columns:
        recent = recent[recent["is_handover"].astype(bool)]
    ho_set = set(int(s) for s in recent["to_sat_id"] if int(s) >= 0)


# 3D globe (Plotly only - Cesium is built later from CZML)
fig = None
if not use_cesium:
    fig = go.Figure()
    fig.add_trace(make_earth_sphere())
    fig.add_trace(_scatter3d_sats(sats_at_t, _OP_COLOR[cnst],
                                    _OP_LABEL[cnst], ho_set))
    fig.add_trace(_scatter3d_users(users))
    fig.add_trace(_scatter3d_gateways(gateways))
    fig.add_trace(_served_link_lines(sats_at_t, users, assigns))
    # Initial camera: tilt down on Canada
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            bgcolor="#02050d",
            camera=dict(
                eye=dict(x=-0.55, y=-0.85, z=0.85),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#02050d", plot_bgcolor="#02050d",
        margin=dict(l=0, r=0, t=10, b=0),
        height=720,
        legend=dict(orientation="h", x=0, y=1.06,
                     bgcolor="rgba(0,0,0,0.4)",
                     font=dict(color="white", size=10)),
    )

# Layout: globe (left) + KPI panel (right)
col_map, col_stats = st.columns([2.4, 1])

with col_map:
    st.subheader(
        f"{_OP_LABEL[cnst]}  ·  batch {batch_idx:02d}/{n_batches-1}  ·  "
        f"fine tick {tick_in_batch:03d}/{ticks_per_batch-1}  ·  "
        f"sim t = {global_t * fine_step_s:.0f} s")
    if use_cesium:
        # CesiumJS branch
        # Try to infer Phase-2 step from repro_meta if a run dir was given.
        phase2_step_s = float(fine_step_s)
        if run_dir_input:
            rmp = Path(run_dir_input).parent.parent / "repro_meta.json"
            if rmp.exists():
                try:
                    phase2_step_s = float(
                        json.loads(rmp.read_text()).get("step_size_s",
                                                          fine_step_s))
                except Exception:
                    pass
        czml_json = build_czml_json(
            EPHEMERIS_DIR, cnst, batch_idx,
            downsample_every_n=10,
            include_users=True,
            include_gateways=True,
            include_orbits=bool(show_orbits),
            sinr_snapshot=sin if not sin.empty else None,
            ho_events=ho if not ho.empty else None,
            phase2_step_s=phase2_step_s,
            flash_duration_s=10.0)
        stats_json = json.dumps(build_tick_stats(
            EPHEMERIS_DIR, cnst, batch_idx,
            sin if not sin.empty else None,
            ho if not ho.empty else None,
            phase2_step_s=phase2_step_s))
        cesium_version = "1.123"
        op_title = _OP_LABEL[cnst]
        html = _CESIUM_HTML
        html = (html.replace("__VER__", cesium_version)
                     .replace("__CZML_DATA__", czml_json)
                     .replace("__STATS_DATA__", stats_json)
                     .replace("__OP_TITLE__", op_title)
                     .replace("__BATCH__", str(batch_idx))
                     .replace("__NBATCH__", str(n_batches - 1)))
        components.html(html, height=760, scrolling=False)
    else:
        st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.subheader("Slot KPIs")
    c1, c2 = st.columns(2)
    c1.metric("Sats in slot", f"{len(sats_at_t):,}")
    c2.metric("Users", f"{len(users)}")
    c3, c4 = st.columns(2)
    c3.metric("Served now",
               f"{len(assigns) if assigns else 0}")
    c4.metric("HOs in last N",
               f"{len(ho_set)}")

    # SINR + availability snapshot
    if not sin.empty:
        sin_now = sin[sin["t_idx"] == global_t]
        if not sin_now.empty:
            st.markdown("**Served-link SINR (this slot)**")
            st.metric("median [dB]",
                       f"{float(sin_now['sinr_db'].median()):.1f}")
            st.metric("p5 [dB]",
                       f"{float(np.percentile(sin_now['sinr_db'], 5)):.1f}")

    # Cumulative SINR time series
    if not sin.empty and len(sin) > 1:
        st.markdown("**SINR median over time**")
        df_med = sin.groupby("t_idx")["sinr_db"].median().reset_index()
        st.line_chart(df_med.set_index("t_idx"))

    # HO cumulative
    if not ho.empty and "is_handover" in ho.columns:
        ho_real = ho[ho["is_handover"].astype(bool)]
        if not ho_real.empty:
            st.markdown("**HOs cumulative**")
            cum = (ho_real.groupby("t_idx").size().cumsum()
                    .rename("cum_HOs").reset_index())
            st.line_chart(cum.set_index("t_idx"))

    # Static manifest reference
    with st.expander("ℹ Ephemeris info"):
        st.json({"start_utc":      manifest["start_utc"],
                  "duration_h":     manifest["duration_h"],
                  "fine_step_s":    manifest["fine_step_s"],
                  "batch_min":      manifest["batch_min"],
                  "n_batches":      manifest["n_batches"],
                  "elev_min_user":  manifest.get("elev_min_user_deg"),
                  "elev_min_gateway": manifest.get("elev_min_gateway_deg"),
                  "n_users":        manifest["n_users"],
                  "constellation":  cnst,
                  "n_sats_in_tle":  manifest["constellations"][cnst]["n_sats_in_tle"],
                  "n_gateways":     manifest["constellations"][cnst]["n_gateways"]})


# Auto-play loop - advance the slider and st.rerun()
if auto:
    time.sleep(max(0.05, 1.0 / refresh_hz))
    new_tick = tick_in_batch + step_size
    new_batch = batch_idx
    if new_tick >= ticks_per_batch:
        new_tick = new_tick - ticks_per_batch
        new_batch = (batch_idx + 1) % n_batches
    # Streamlit ≥ 1.27: use st.query_params; older: experimental_rerun.
    st.query_params["batch"] = str(new_batch)
    st.query_params["tick"]  = str(new_tick)
    st.rerun()
