"""Builds the full set of paper-ready figures from handover_compare runs.

Reads per-cell parquet logs produced by the visibility_log + handover_events_log
hooks, plus the manifest produced by run_compare.py, and emits:

  Visibility / coverage
    fig1_visibility_ccdf.png         (A1) CCDF of n_visible_sats
    fig2_coverage_bar.png            (A2) P[user covered]
    fig3_mean_nvis_vs_lat.png        (A3) mean n_vis vs latitude
    fig4_visibility_gap_cdf.png      (A5) CDF of consecutive-outage duration

  Contact times
    fig5_contact_time_cdf.png        (B1) single-link contact CDF
    fig6_contact_vs_peakelev.png     (B2) contact vs peak elevation
    fig7_intercontact_gap_cdf.png    (B3) inter-contact gap CDF

  Handover behaviour
    fig8_ho_rate_bar.png             (C1) HO rate per user-minute
    fig9_ihi_cdf.png                 (C2) inter-handover interval CDF
    fig10_pingpong_bar.png           (C3) ping-pong ratio (A->B->A in window)
    fig11_service_gap_cdf.png        (C4) service-interruption CDF
    fig12_ho_geo_map.png             (C7) geographic HO density map

  Link quality
    fig13_sinr_ccdf.png              (D1) CCDF of served SINR
    fig14_sinr_vs_elev.png           (D3) SINR vs serving-sat elevation
    fig15_sinr_drop_around_ho.png    (D5) aligned SINR trace around HO
    fig16_ion_ccdf.png               (D6) CCDF of I/N
    fig19_rate_ccdf.png              (D7) CCDF of served throughput (Mbps)

  Summary
    strategy_summary.csv             (E)  one row per (constellation, strategy)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE single-column figure size (inches)
_IEEE_FIGSIZE = (3.5, 2.6)

_V1 = Path(__file__).resolve().parents[2]           # .../v1

# Per-scenario ephemeris override; None -> _find_eph_dir() auto-locates.
_EPH_DIR_OVERRIDE = None


def set_eph_dir(path) -> None:
    """Point the RTT-routing code at a scenario's ephemeris dataset."""
    global _EPH_DIR_OVERRIDE
    _EPH_DIR_OVERRIDE = Path(path) if path is not None else None


def short_label(name: str) -> str:                  # default label map; overridable
    return {"oneweb": "OneWeb", "starlink": "Starlink",
            "kuiper": "Kuiper"}.get(name.lower(), name)

# Constellation colour by POSITION in a fixed ordering, stable across figures.
import hashlib as _hashlib
_CONSTELLATION_ORDER = ["oneweb", "starlink", "kuiper"]
_PALETTE = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def constellation_color(name):
    key = str(name).lower()
    if key in _CONSTELLATION_ORDER:
        return _PALETTE[_CONSTELLATION_ORDER.index(key) % len(_PALETTE)]
    h = int(_hashlib.md5(key.encode()).hexdigest(), 16)
    return _PALETTE[(len(_CONSTELLATION_ORDER) + h) % len(_PALETTE)]


class _ColorMap(dict):
    """dict whose .get() returns the positional colour for any key."""
    def get(self, key, default=None):
        return constellation_color(key)


_COLORS = _ColorMap()
_LSTYLE = {"max_sinr": "-", "persistent_nearest": "--", "farthest_approaching": ":",
           "max_rx_power": "--", "nearest_sat": ":", "load_aware": (0, (3, 1, 1, 1))}
_PINGPONG_WINDOW_TICKS = 10
_HO_ALIGN_WINDOW       = 5


# WGS-84 LLA -> ECEF
_R_EQ = 6378137.0
_F    = 1.0 / 298.257223563
_E2   = 2 * _F - _F * _F


def _lla_to_ecef_arr(lat_deg, lon_deg, alt_m):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h   = np.asarray(alt_m, dtype=float)
    N = _R_EQ / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - _E2) + h) * np.sin(lat)
    return x, y, z


def _find_eph_dir() -> Path | None:
    """Locate the ephemeris dataset (override, else auto-locate)."""
    if _EPH_DIR_OVERRIDE is not None:
        p = _EPH_DIR_OVERRIDE
        if p.exists() and (p / "manifest.json").exists():
            return p
    candidates = [
        _V1 / "outputs" / "ephemeris",
        Path.cwd() / "v1" / "outputs" / "ephemeris",
    ]
    for p in candidates:
        if p.exists() and (p / "manifest.json").exists():
            return p
    return None


def _compute_e2e_rtt_for_cell(rd: Path, sin: pd.DataFrame,
                               lvi: pd.DataFrame, cnst: str,
                               step_size_s: float,
                               max_isl_km: float = 5000.0,
                               feeder_top_k: int = 3,
                               routing_mode: str = "isl",
                               max_feeder_km: float = 3000.0
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Per served row, end-to-end RTT (ms) and ISL hop count; NaN-padded to len(sin).

    routing_mode "isl": multi-source Dijkstra over the per-tick ISL graph plus
    per-gateway top-K feeder edges. "bent_pipe": single feeder hop to the
    nearest gateway within max_feeder_km.
    """
    if sin is None or sin.empty:
        return np.array([]), np.array([])
    eph_dir = _find_eph_dir()
    if eph_dir is None:
        return (np.full(len(sin), np.nan), np.full(len(sin), np.nan))
    try:
        manifest = json.loads((eph_dir / "manifest.json").read_text())
    except Exception:
        return (np.full(len(sin), np.nan), np.full(len(sin), np.nan))
    # Support old (fine_step_s/batch_min) and new (step_s/batch_size_s/hours) manifests.
    fine_step = float(manifest.get("fine_step_s",
                                   manifest.get("step_s", step_size_s or 1.0)))
    batch_min = float(manifest.get("batch_min",
                                   (float(manifest.get("batch_size_s", 0.0)) / 60.0)
                                   or (float(manifest.get("hours", 0.0)) * 60.0)
                                   or 15.0))
    batch_ticks = int(round(batch_min * 60.0 / fine_step))
    # Re-solve ISL shortest paths every ~30 s (topology changes slowly).
    _ROUTE_STRIDE_TICKS = max(1, int(round(30.0 / fine_step)))

    # Gateways: batched dataset layout (ground_stations/<cnst>__batch_NN.parquet,
    # unioned) with a legacy fallback (<cnst>/gateways.parquet). Static -> dedup.
    from ntnconnect.common.utils.partition_writer import read_partitioned
    gws = read_partitioned(eph_dir, "ground_stations")
    if gws is None or gws.empty:
        gw_p = eph_dir / cnst / "gateways.parquet"
        gws = pd.read_parquet(gw_p) if gw_p.exists() else None
    if gws is None or gws.empty:
        return (np.full(len(sin), np.nan), np.full(len(sin), np.nan))
    if "node_id" in gws.columns:
        gws = gws.drop_duplicates("node_id")
    gx, gy, gz = _lla_to_ecef_arr(
        gws["lat"].to_numpy(), gws["lon"].to_numpy(), gws["alt_m"].to_numpy())
    gw_ecef = np.column_stack([gx, gy, gz])      # (n_gw, 3)
    n_gw = gw_ecef.shape[0]
    # Satellites: batched layout (satellites/<cnst>__batch_NN, global t_idx),
    # pre-split by GLOBAL t_idx; legacy per-batch visible_sats_NN handled in-loop.
    _allsat = read_partitioned(eph_dir, "satellites")
    _sat_by_t = ({int(t): g for t, g in _allsat.groupby("t_idx", sort=False)}
                 if _allsat is not None and not _allsat.empty else None)

    # Access-leg slant range from link_visibility.
    if lvi is not None and "dist_m" in lvi.columns:
        sin = sin.merge(lvi[["t_idx", "user_id", "sat_id", "dist_m"]],
                         on=["t_idx", "user_id", "sat_id"], how="left",
                         suffixes=("", "_lvi"))
    else:
        sin = sin.copy(); sin["dist_m"] = np.nan

    # Engine tick -> ephemeris fine tick.
    ratio = float(step_size_s) / fine_step

    tick_cache: dict[tuple[int, int], tuple] = {}
    batch_cache: dict[int, pd.DataFrame] = {}
    c_m_s = 299_792_458.0
    max_isl_m = max_isl_km * 1000.0
    max_feeder_m = max_feeder_km * 1000.0

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except Exception:
        return (np.full(len(sin), np.nan), np.full(len(sin), np.nan))

    rtts  = np.full(len(sin), np.nan, dtype=float)
    hops  = np.full(len(sin), np.nan, dtype=float)
    sat_id_col  = sin["sat_id"].astype(int).to_numpy()
    t_idx_col   = sin["t_idx"].astype(int).to_numpy()
    access_col  = sin["dist_m"].to_numpy(dtype=float)

    for i in range(len(sin)):
        engine_t = int(t_idx_col[i])
        total_eph_t = int(round(engine_t * ratio))
        if _sat_by_t is not None:                        # batched layout: global t_idx
            eph_t = (total_eph_t // _ROUTE_STRIDE_TICKS) * _ROUTE_STRIDE_TICKS
            df_by_t = _sat_by_t
            cache_key = eph_t
        else:                                            # legacy visible_sats_NN
            batch_idx = total_eph_t // batch_ticks
            eph_t     = total_eph_t -  batch_idx * batch_ticks
            if batch_idx not in batch_cache:
                p = eph_dir / cnst / f"visible_sats_{batch_idx:02d}.parquet"
                if not p.exists():
                    continue
                try:
                    _df = pd.read_parquet(p)
                    batch_cache[batch_idx] = {int(t): g for t, g
                                              in _df.groupby("t_idx", sort=False)}
                except Exception:
                    continue
            df_by_t = batch_cache[batch_idx]
            eph_t = (eph_t // _ROUTE_STRIDE_TICKS) * _ROUTE_STRIDE_TICKS
            cache_key = (batch_idx, eph_t)
        if cache_key not in tick_cache:
            sub = df_by_t.get(eph_t)
            if sub is None or sub.empty:
                tick_cache[cache_key] = None
            else:
                sat_ids  = sub["node_id"].to_numpy(dtype=int)
                sat_pos  = np.column_stack([
                    sub["x"].to_numpy(), sub["y"].to_numpy(),
                    sub["z"].to_numpy()])
                n = len(sat_ids)
                gw2sat = np.linalg.norm(
                    gw_ecef[:, None, :] - sat_pos[None, :, :], axis=2)
                if routing_mode == "bent_pipe":
                    nearest = gw2sat.min(axis=0)                       # (n,)
                    dgw = np.where(nearest <= max_feeder_m, nearest, np.inf)
                    hops_arr = np.where(np.isfinite(dgw), 1.0, np.nan)
                    sat_id_to_idx = {int(sid): k for k, sid in enumerate(sat_ids)}
                    tick_cache[cache_key] = (sat_id_to_idx, dgw, hops_arr)
                else:
                    dmat = np.linalg.norm(
                        sat_pos[:, None, :] - sat_pos[None, :, :], axis=2)
                    mask = (dmat > 0) & (dmat <= max_isl_m)
                    rows_isl, cols_isl = np.where(mask)
                    w_isl = dmat[rows_isl, cols_isl]
                    # Feeder edges to top-K nearest sats per gateway
                    rows_f, cols_f, w_f = [], [], []
                    for gi in range(n_gw):
                        top = np.argsort(gw2sat[gi])[:feeder_top_k]
                        for sj in top:
                            d = float(gw2sat[gi, sj])
                            if d > max_isl_m * 2:    # gateway out of reach
                                continue
                            rows_f.extend([n + gi, sj])
                            cols_f.extend([sj, n + gi])
                            w_f.extend([d, d])
                    rows_all = np.concatenate([rows_isl, np.array(rows_f, dtype=int)])
                    cols_all = np.concatenate([cols_isl, np.array(cols_f, dtype=int)])
                    w_all    = np.concatenate([w_isl, np.array(w_f, dtype=float)])
                    total_n = n + n_gw
                    if rows_all.size == 0:
                        tick_cache[cache_key] = None
                    else:
                        g = csr_matrix((w_all, (rows_all, cols_all)),
                                        shape=(total_n, total_n))
                        # min_only + return_predecessors -> (dist, predecessors, sources)
                        dist_to_gw, predecessors, _sources = dijkstra(
                            g, indices=list(range(n, n + n_gw)),
                            return_predecessors=True, min_only=True)
                        hops_arr = np.full(n, np.nan, dtype=float)
                        for s in range(n):
                            if not np.isfinite(dist_to_gw[s]):
                                continue
                            cur, h = s, 0
                            while cur >= 0 and cur < n and h < 64:
                                nxt = int(predecessors[cur])
                                if nxt < 0: break
                                h += 1
                                cur = nxt
                            hops_arr[s] = float(h)
                        sat_id_to_idx = {int(sid): k for k, sid in enumerate(sat_ids)}
                        tick_cache[cache_key] = (sat_id_to_idx,
                                                  dist_to_gw[:n], hops_arr)
        tc = tick_cache[cache_key]
        if tc is None: continue
        sat_id_to_idx, dist_to_gw_arr, hops_arr = tc
        sat_idx = sat_id_to_idx.get(int(sat_id_col[i]))
        if sat_idx is None: continue
        d_isl_plus_feeder = float(dist_to_gw_arr[sat_idx])
        if not np.isfinite(d_isl_plus_feeder): continue
        d_access = float(access_col[i]) if np.isfinite(access_col[i]) else 0.0
        rtts[i] = 2.0 * (d_access + d_isl_plus_feeder) / c_m_s * 1000.0
        hops[i] = float(hops_arr[sat_idx])

    return rtts, hops


# IO + style helpers
def _load(rd: Path, name: str) -> pd.DataFrame | None:
    """Load a stream from a single Parquet and/or per-batch shards, de-duplicated."""
    from ntnconnect.common.utils.run_io import run_shards
    stem = Path(name).stem
    files = run_shards(rd, stem)
    frames = []
    for s in files:
        try: frames.append(pd.read_parquet(s))
        except Exception: pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    n_sources = len(files)
    # Guard against overlapping batch ranges: check disjoint tick-ranges and
    # drop exact-duplicate rows, warning if either fires.
    if n_sources > 1:
        if len(files) > 1 and "t_idx" in df.columns:
            ranges = []
            for s in files:
                try:
                    ti = pd.read_parquet(s, columns=["t_idx"])["t_idx"]
                    ranges.append((int(ti.min()), int(ti.max())))
                except Exception:
                    pass
            ranges.sort()
            overlap = any(ranges[i][1] >= ranges[i + 1][0]
                          for i in range(len(ranges) - 1))
            if overlap:
                print(f"WARN: _load({name}): OVERLAPPING batch tick-ranges "
                      f"{ranges} — batch boundaries conflict; de-duplicating.")
        n0 = len(df)
        df = df.drop_duplicates(ignore_index=True)
        if len(df) < n0:
            print(f"WARN: _load({name}): dropped {n0 - len(df)} duplicate rows "
                  f"across {n_sources} batch sources (overlapping batches).")
    return df


def _resolve_step_s(out: Path, cells: "pd.DataFrame") -> float:
    """Simulation step (s): manifest step_s, then repro_meta.json, then 1.0."""
    try:
        if "step_s" in cells.columns and cells["step_s"].notna().any():
            return float(cells["step_s"].dropna().iloc[0])
    except Exception:
        pass
    rm = out / "repro_meta.json"
    if rm.exists():
        try:
            return float(json.loads(rm.read_text()).get("step_size_s", 1.0))
        except Exception:
            pass
    return 1.0


_STRAT_ABBREV = {
    "max_sinr": "MRSRP",             # greedy strongest received power
    "gpp_a3": "3GPP-A3",             # RSRP + A3 offset + TTT (TS 38.331)
    "persistent_nearest": "MEA",     # highest elevation, hold until set
    "farthest_approaching": "MST",   # max service time (longest dwell)
    "max_rx_power": "MaxRxP",        # strongest mean received power
    "load_aware": "LoadAware",       # load-penalised association
    "utility_based": "Utility",      # utility-maximising association
}

# Per-strategy marker (distinguishes strategies where line styles overlap).
_MARKER = {
    "max_sinr": "s", "gpp_a3": "v", "persistent_nearest": "^",
    "farthest_approaching": "D", "max_rx_power": "o",
    "load_aware": "d", "utility_based": "p",
}


def _strat_abbrev(ho: str) -> str:
    return _STRAT_ABBREV.get(str(ho).lower(), str(ho))


def _style(cnst: str, ho: str) -> dict:
    return dict(color=_COLORS.get(cnst, "k"),
                linestyle=_LSTYLE.get(ho, "-"),
                marker=_MARKER.get(str(ho).lower(), None),
                markevery=0.12, markersize=5, markerfacecolor="none",
                label=f"{short_label(cnst)}-{_strat_abbrev(ho)}", lw=1.5)


# Canonical per-strategy order + distinct colour, unified across figures.
_STRAT_ORDER = ["max_sinr", "farthest_approaching", "max_rx_power",
                "load_aware", "utility_based", "gpp_a3", "persistent_nearest"]
_STRAT_COLOR = {
    "max_sinr": "#2ca02c", "farthest_approaching": "#8c564b",   # MRSRP green, MST brown
    "max_rx_power": "#1f77b4", "load_aware": "#17becf",
    "utility_based": "#ff7f0e",
    "gpp_a3": "#7f7f7f", "persistent_nearest": "#aec7e8"}
_STRAT_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd",
                  "#ff7f0e", "#8c564b", "#17becf", "#bcbd22"]
_EMPH_STRATS = set()
_LW_EMPH, _LW_BASE = 1.4, 0.9


def strategy_color(name):
    key = str(name).lower()
    if key in _STRAT_COLOR:
        return _STRAT_COLOR[key]
    h = int(_hashlib.md5(key.encode()).hexdigest(), 16)
    return _STRAT_PALETTE[h % len(_STRAT_PALETTE)]


def _strat_sort(strats):
    """Order strategy keys canonically (unknowns last, a-z)."""
    def rank(s):
        s = str(s).lower()
        return (_STRAT_ORDER.index(s),) if s in _STRAT_ORDER else (len(_STRAT_ORDER), s)
    return sorted(strats, key=rank)


def _split_by_cnst(d):
    """{(cnst, strat): v} -> {cnst: {strat: v}}."""
    out = {}
    for k, v in d.items():
        c, s = k if isinstance(k, tuple) else (k, "")
        out.setdefault(c, {})[s] = v
    return out


def _frame(ax):
    """Full black axes box (all four spines)."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)


def _bordered_save(fig, path, dpi):
    """Save with a black border around the whole figure."""
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(1.3)
    fig.savefig(path, dpi=dpi, edgecolor="black", facecolor="white")


def _style_strat(ho):
    """Line style keyed by strategy."""
    key = str(ho).lower()
    return dict(color=strategy_color(key), linestyle="-",
                marker=_MARKER.get(key, None), markevery=0.15, markersize=3.2,
                markerfacecolor="white", markeredgewidth=0.8,
                label=_strat_abbrev(ho),
                lw=_LW_EMPH if key in _EMPH_STRATS else _LW_BASE,
                zorder=6 if key in _EMPH_STRATS else 3)


def _fig_bar_by_strategy(per_user_dict, out, fname_base, ylabel,
                         ylim=None, seed=42, fmt="{:.2f}"):
    """One bar figure per constellation; bars are strategies (mean +/- 95% CI)."""
    if not per_user_dict:
        return
    for cnst, series in _split_by_cnst(per_user_dict).items():
        strats = _strat_sort(series)
        labels = [_strat_abbrev(s) for s in strats]
        colours = [strategy_color(s) for s in strats]
        means, lo_err, hi_err, samples = [], [], [], []
        for i, s in enumerate(strats):
            a = np.array(list(series[s].values()), dtype=float)
            samples.append(a)
            m, lo, hi = _bootstrap_mean_ci(a, seed=seed + i)
            means.append(m)
            lo_err.append(max(0.0, m - lo) if m == m else 0.0)
            hi_err.append(max(0.0, hi - m) if m == m else 0.0)
        fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
        xs = np.arange(len(labels))
        ax.bar(xs, means, width=0.68, color=colours,
               yerr=[lo_err, hi_err], capsize=3, ecolor="0.3",
               edgecolor="black", linewidth=0.6, zorder=2,
               error_kw=dict(lw=1.0))
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(short_label(cnst))
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        _frame(ax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for j, m in enumerate(means):
            if m == m:
                top = m + (hi_err[j] if hi_err[j] == hi_err[j] else 0.0)
                ax.text(j, top, fmt.format(m), ha="center", va="bottom", fontsize=6.5)
        fig.tight_layout()
        fname = f"{fname_base}_{cnst}.png"
        _bordered_save(fig, out / fname, 140); plt.close(fig)
        bars = [dict(label=l, value=m, ci_lo=m - le, ci_hi=m + he, color=c, samples=sa)
                for l, m, le, he, c, sa
                in zip(labels, means, lo_err, hi_err, colours, samples)]
        _save_scalar_data(out, fname, title=str(cnst), ylabel=ylabel, bars=bars)


def _save_curve_data(out: Path, fname: str, *,
                      title: str = "", xlabel: str = "", ylabel: str = "",
                      curves: list | None = None) -> None:
    """Save curve plot data as JSON next to the PNG for re-rendering."""
    if curves is None: curves = []
    safe = []
    for c in curves:
        d = dict(c)
        for k in ("x", "y"):
            if k in d: d[k] = _to_list(d[k])
        safe.append(d)
    p = out / (Path(fname).stem + "_data.json")
    p.write_text(json.dumps(
        dict(kind="curves", title=title, xlabel=xlabel, ylabel=ylabel,
             curves=safe),
        indent=2))


def _save_scalar_data(out: Path, fname: str, *,
                       title: str = "", ylabel: str = "", xlabel: str = "",
                       bars: list | None = None) -> None:
    """Saved-data sibling for bar charts. bars: {label, value, ci_lo, ci_hi, color, samples?}."""
    if bars is None: bars = []
    safe = []
    for b in bars:
        d = dict(b)
        if "samples" in d and d["samples"] is not None:
            d["samples"] = (d["samples"].tolist()
                              if hasattr(d["samples"], "tolist")
                              else list(d["samples"]))
        safe.append(d)
    p = out / (Path(fname).stem + "_data.json")
    p.write_text(json.dumps(
        dict(kind="bars", title=title, xlabel=xlabel, ylabel=ylabel,
             bars=safe), indent=2))


def _to_list(v):
    if v is None: return v
    if hasattr(v, "tolist"): return v.tolist()          # numpy array / Series
    # numpy scalars are not JSON-safe
    return [x.item() if hasattr(x, "item") else x for x in v]


# Sidecar-only caps to keep the JSON small; never affect the PNG.
_SCATTER_MAX = 4000        # points per scatter panel
_CURVE_MAX   = 300         # points per (downsampled) curve


def _subsample_cloud(x, y, n=_SCATTER_MAX, seed=0):
    # Deterministic subsample of a scatter cloud.
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size <= n:
        return x, y
    idx = np.sort(np.random.default_rng(seed).choice(x.size, n, replace=False))
    return x[idx], y[idx]


def _stride_curve(x, y, n=_CURVE_MAX):
    # Thin a monotonic curve to <= n points, keeping endpoints.
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size <= n:
        return x, y
    idx = np.unique(np.linspace(0, x.size - 1, n).astype(int))
    return x[idx], y[idx]


def _save_scatter_data(out: Path, fname: str, *, title: str = "",
                        xlabel: str = "", ylabel: str = "",
                        panels: list | None = None) -> None:
    """Saved-data sibling for scatter figures."""
    panels = panels or []
    safe = []
    for i, p in enumerate(panels):
        xs, ys = _subsample_cloud(p.get("x", []), p.get("y", []), seed=i)
        safe.append(dict(label=p.get("label", ""), color=p.get("color"),
                         x=_to_list(xs), y=_to_list(ys)))
    p = out / (Path(fname).stem + "_data.json")
    p.write_text(json.dumps(
        dict(kind="scatter", title=title, xlabel=xlabel, ylabel=ylabel,
             panels=safe), indent=2))


def _save_overlay_data(out: Path, fname: str, *, title: str = "",
                        xlabel: str = "", panels: list | None = None) -> None:
    """Saved-data sibling for per-user CCDF overlays."""
    panels = panels or []
    safe = []
    for p in panels:
        ov = p.get("overall") or {}
        ox, oy = _stride_curve(ov.get("x", []), ov.get("y", []))
        users = []
        for u in p.get("users", []):
            ux, uy = _stride_curve(u.get("x", []), u.get("y", []))
            users.append(dict(x=_to_list(ux), y=_to_list(uy)))
        safe.append(dict(label=p.get("label", ""),
                         overall=dict(x=_to_list(ox), y=_to_list(oy)),
                         users=users))
    p = out / (Path(fname).stem + "_data.json")
    p.write_text(json.dumps(
        dict(kind="overlay", title=title, xlabel=xlabel,
             ylabel="P[>= x]", panels=safe), indent=2))


def _save_hexbin_data(out: Path, fname: str, *, title: str = "",
                       xlabel: str = "", ylabel: str = "",
                       panels: list | None = None) -> None:
    """Saved-data sibling for geographic density (hexbin) figures."""
    panels = panels or []
    safe = []
    for i, p in enumerate(panels):
        xs, ys = _subsample_cloud(p.get("x", []), p.get("y", []),
                                  n=20000, seed=i)
        safe.append(dict(label=p.get("label", ""),
                         x=_to_list(xs), y=_to_list(ys)))
    p = out / (Path(fname).stem + "_data.json")
    p.write_text(json.dumps(
        dict(kind="hexbin", title=title, xlabel=xlabel, ylabel=ylabel,
             panels=safe), indent=2))


def _cdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / max(len(x), 1)
    return x, y


def _ccdf(values):
    x, y = _cdf(values)
    return x, 1.0 - y


def _pool_by_constellation(metric: dict) -> dict:
    """Collapse {(cnst, strat): array} to {cnst: pooled_array} for geometry metrics."""
    out: dict = {}
    for key, vals in metric.items():
        cnst = key[0] if isinstance(key, tuple) else key
        out.setdefault(cnst, []).append(np.asarray(vals, dtype=float).ravel())
    return {c: (np.concatenate(a) if a else np.array([]))
            for c, a in out.items()}


# Aggregation across (constellation, strategy) cells
def _clip_ticks(df, max_ticks):
    """Keep only rows with t_idx < max_ticks (fast preview)."""
    if df is None or max_ticks is None or "t_idx" not in getattr(df, "columns", []):
        return df
    return df[df["t_idx"] < int(max_ticks)]


def _build_aggregates(cells: pd.DataFrame, step_size_s: float = 1.0,
                      compute_rtt: bool = True, max_ticks=None) -> dict:
    A = dict(visibility={}, cover_prob={}, lat_curves={}, vis_gap={},
             vis_gap_s={}, service_gap_s_all={},
             outage_summary={},
             contact={}, contact_peak_pairs={}, inter_contact_gap={},
             ho_rate={}, ihi={}, pingpong={}, service_gap={},
             ho_geo={}, sinr={}, sinr_elev_pairs={}, ho_sinr_align={},
             ion={}, sinr_summary={}, contact_summary={},
             rate={},   # served-link throughput [Mbps]
             # Per-user breakdowns: key -> {user_id -> array}
             sinr_per_user={}, ion_per_user={}, ihi_per_user={},
             rate_per_user={},
             vis_gap_s_per_user={}, coverage_per_user={},
             # Per-user scalars for bar charts: key -> {user_id -> scalar}
             ho_rate_per_user={}, pingpong_per_user={},
             system_power={},   # key -> {t_idx -> total tx power [W]}
             availability_per_user={},
             access_delay_ms={},   # access-link one-way delay [ms], geometry-only
             e2e_rtt_ms={},        # end-to-end RTT [ms]
             isl_hops={})          # ISL hops along the routed path
    A["step_s"] = float(step_size_s)
    for _, row in cells.iterrows():
        rd = Path(row["run_dir"]) if row["run_dir"] else None
        if rd is None or not rd.exists():
            continue
        key = (row["constellation"], row["strategy"])
        vis = _clip_ticks(_load(rd, "visibility_per_tick.parquet"), max_ticks)
        ctx = _load(rd, "contact_events.parquet")
        sin = _clip_ticks(_load(rd, "sinr_snapshot.parquet"), max_ticks)
        ho  = _clip_ticks(_load(rd, "handover_events.parquet"), max_ticks)
        lvi = _clip_ticks(_load(rd, "link_visibility.parquet"), max_ticks)

        # Access-link one-way delay = dist_m / c; slant range from link_visibility.
        access_dist_arr = None
        if sin is not None and not sin.empty and lvi is not None \
           and "dist_m" in lvi.columns:
            try:
                lvi_keyed = lvi[["t_idx", "user_id", "sat_id", "dist_m"]]
                sin_with_d = sin[["t_idx", "user_id", "sat_id"]].merge(
                    lvi_keyed, on=["t_idx", "user_id", "sat_id"], how="left")
                d = sin_with_d["dist_m"].astype(float).to_numpy()
                fin = np.isfinite(d)
                if fin.any():
                    A["access_delay_ms"][key] = (d[fin] / 299_792_458.0) * 1000.0
                    access_dist_arr = d        # length matches sin row count
            except Exception:
                pass

        # End-to-end RTT + ISL hops; only when the RTT figure is requested (costly).
        if compute_rtt:
            try:
                rtt_arr, hops_arr = _compute_e2e_rtt_for_cell(
                    rd, sin, lvi, str(row["constellation"]), step_size_s)
                if rtt_arr is not None and rtt_arr.size:
                    A["e2e_rtt_ms"][key] = rtt_arr[np.isfinite(rtt_arr)]
                if hops_arr is not None and hops_arr.size:
                    A["isl_hops"][key] = hops_arr[np.isfinite(hops_arr)]
            except Exception as e:
                print(f"WARN: e2e RTT failed for {key}: {e}")

        # A1, A2, A3, A5
        if vis is not None and not vis.empty:
            A["visibility"][key] = vis["n_visible_sats"].astype(int)
            A["cover_prob"][key] = float((vis["n_visible_sats"] >= 1).mean())
            if "user_lat" in vis.columns:
                lat_bins = np.arange(40, 86, 2.0)
                g = (vis.assign(lat_bin=pd.cut(vis["user_lat"], lat_bins))
                       .groupby("lat_bin", observed=True)["n_visible_sats"].mean())
                A["lat_curves"][key] = g.dropna()
            # Per-user visibility-outage runs (ticks and seconds)
            gaps_ticks = []
            per_user_total_outage_s = []
            per_user_worst_outage_s = []
            n_ticks_total = (vis["t_idx"].max() - vis["t_idx"].min() + 1)
            sim_seconds   = float(n_ticks_total) * float(step_size_s)
            per_user_gaps_s: dict[int, list] = {}
            per_user_coverage: dict[int, float] = {}
            for uid, sub in vis.sort_values("t_idx").groupby("user_id"):
                uid = int(uid)
                run = 0
                user_runs = []
                nvis_arr = sub["n_visible_sats"].to_numpy()
                for n in nvis_arr:
                    if n <= 0: run += 1
                    elif run > 0: user_runs.append(run); gaps_ticks.append(run); run = 0
                if run > 0: user_runs.append(run); gaps_ticks.append(run)
                tot_s = sum(user_runs) * step_size_s
                worst_s = (max(user_runs) if user_runs else 0) * step_size_s
                per_user_total_outage_s.append(tot_s)
                per_user_worst_outage_s.append(worst_s)
                per_user_gaps_s[uid] = [r * step_size_s for r in user_runs]
                per_user_coverage[uid] = float((nvis_arr > 0).mean())
            A["vis_gap_s_per_user"][key] = per_user_gaps_s
            A["coverage_per_user"][key]  = per_user_coverage
            A["availability_per_user"][key] = {
                u: 100.0 * c for u, c in per_user_coverage.items()
            }
            if gaps_ticks:
                A["vis_gap"][key]   = np.array(gaps_ticks, dtype=float)
                A["vis_gap_s"][key] = np.array(gaps_ticks, dtype=float) * float(step_size_s)
            mean_tot   = float(np.mean(per_user_total_outage_s)) if per_user_total_outage_s else 0.0
            worst_glob = float(np.max(per_user_worst_outage_s))  if per_user_worst_outage_s else 0.0
            vis_med_s  = float(np.median(A["vis_gap_s"][key])) if key in A["vis_gap_s"] else 0.0
            A["outage_summary"][key] = {
                "sim_seconds":         sim_seconds,
                "mean_total_outage_s": mean_tot,
                "worst_outage_s":      worst_glob,
                "vis_gap_med_s":       vis_med_s,
                "availability_pct":    100.0 * (1.0 - mean_tot / max(sim_seconds, 1e-9)),
            }

        # B1, B2, B3 - contact-duration uses only fully-observed passes, dropping
        # left/right-censored and zero-length ones so the short tail is real.
        if ctx is not None and not ctx.empty:
            cc = ctx
            t_last = int(pd.to_numeric(ctx["t_leave_idx"], errors="coerce").max())
            if "open_at_end" in cc.columns:
                # NaN (closed-contact rows) compares False -> kept; only True drops
                cc = cc[cc["open_at_end"] != True]   # noqa: E712
            if "t_enter_idx" in cc.columns:
                cc = cc[cc["t_enter_idx"].astype(int) > 0]      # left-censored
            if "t_leave_idx" in cc.columns:
                cc = cc[cc["t_leave_idx"].astype(int) < t_last] # right-censored
            cc = cc[pd.to_numeric(cc["contact_duration_s"], errors="coerce") > 0]
        else:
            cc = None
        if cc is not None and not cc.empty:
            A["contact"][key] = cc["contact_duration_s"].astype(float)
            A["contact_summary"][key] = (float(cc["contact_duration_s"].mean()),
                                         float(cc["contact_duration_s"].median()))
            if vis is not None and {"user_id", "t_idx", "best_elev_deg"}.issubset(vis.columns):
                peaks, durs = [], []
                for r in cc.itertuples(index=False):
                    sub = vis[(vis["user_id"] == r.user_id) &
                              (vis["t_idx"] >= r.t_enter_idx) &
                              (vis["t_idx"] <= r.t_leave_idx)]
                    if not sub.empty:
                        peaks.append(float(sub["best_elev_deg"].max()))
                        durs.append(float(r.contact_duration_s))
                if peaks:
                    A["contact_peak_pairs"][key] = (np.array(durs), np.array(peaks))

        # Per-user coverage-outage gaps [s]: interior gaps between merged
        # per-user visibility intervals; edge gaps are censored and excluded.
        if ctx is not None and not ctx.empty \
           and {"user_id", "t_enter_idx", "t_leave_idx"}.issubset(ctx.columns):
            ig = []
            for uid, sub in ctx.groupby("user_id"):
                iv = sorted(zip(sub["t_enter_idx"].astype(int).tolist(),
                                sub["t_leave_idx"].astype(int).tolist()))
                merged = []
                for a, b in iv:
                    # overlapping or <=1 tick apart -> continuous
                    if merged and a <= merged[-1][1] + 1:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                    else:
                        merged.append((a, b))
                for (a0, b0), (a1, b1) in zip(merged[:-1], merged[1:]):
                    gap_ticks = a1 - b0 - 1
                    if gap_ticks > 0:
                        ig.append(gap_ticks * step_size_s)
            if ig:
                A["inter_contact_gap"][key] = np.array(ig, dtype=float)

        # D1, D3, D6
        if sin is not None and not sin.empty:
            A["sinr"][key] = sin["sinr_db"].astype(float)
            A["sinr_summary"][key] = (float(sin["sinr_db"].median()),
                                      float(np.percentile(sin["sinr_db"], 5)),
                                      float(sin["link_rate_bps"].median())
                                       if "link_rate_bps" in sin.columns else float("nan"))
            if "serving_sat_elev_deg" in sin.columns:
                A["sinr_elev_pairs"][key] = (
                    sin["serving_sat_elev_deg"].astype(float).to_numpy(),
                    sin["sinr_db"].astype(float).to_numpy())
            if "i_over_n_db" in sin.columns:
                A["ion"][key] = sin["i_over_n_db"].astype(float)
            # Served-link throughput [Mbps]: throughput_mbps else link_rate_bps.
            _rate_col = ("throughput_mbps" if "throughput_mbps" in sin.columns
                         else None)
            if _rate_col is not None:
                rate_mbps = sin[_rate_col].astype(float)
            elif "link_rate_bps" in sin.columns:
                rate_mbps = sin["link_rate_bps"].astype(float) / 1e6
            else:
                rate_mbps = None
            if rate_mbps is not None:
                A["rate"][key] = rate_mbps.to_numpy()
                A["rate_per_user"][key] = {
                    int(uid): g.to_numpy()
                    for uid, g in rate_mbps.groupby(sin["user_id"].to_numpy())}
            # System transmit power: total P_tx summed over served users per tick [W].
            if "tx_power_dbw" in sin.columns:
                srv = (sin[sin["link_rate_bps"] > 0]
                       if "link_rate_bps" in sin.columns else sin)
                if not srv.empty:
                    _ptx = 10.0 ** (srv["tx_power_dbw"].astype(float).to_numpy() / 10.0)
                    _tot = pd.Series(_ptx).groupby(srv["t_idx"].to_numpy()).sum()
                    # W -> dBW
                    A["system_power"][key] = {int(t): float(10.0 * np.log10(max(v, 1e-30)))
                                              for t, v in _tot.items()}
            A["sinr_per_user"][key] = {int(uid): g["sinr_db"].astype(float).to_numpy()
                                         for uid, g in sin.groupby("user_id")}
            if "i_over_n_db" in sin.columns:
                A["ion_per_user"][key] = {int(uid): g["i_over_n_db"].astype(float).to_numpy()
                                            for uid, g in sin.groupby("user_id")}

        # C1..C4, C7, D5
        if ho is not None and not ho.empty:
            # Drop init/leave: is_handover flag, else (from>=0 & to>=0 & differ).
            if "is_handover" in ho.columns:
                ho_real = ho[ho["is_handover"].astype(bool)]
            else:
                ho_real = ho[(ho["from_sat_id"] >= 0) & (ho["to_sat_id"] >= 0)
                              & (ho["from_sat_id"] != ho["to_sat_id"])]
            n_users = int(ho["user_id"].nunique())
            tot_minutes = 1.0
            if vis is not None and not vis.empty:
                # ticks -> minutes (t_idx is in step_s units, not seconds)
                tot_minutes = ((vis["t_idx"].max() - vis["t_idx"].min() + 1)
                               * float(step_size_s) / 60.0)
                tot_minutes = max(tot_minutes, 1e-6)
            n_changes = int(len(ho_real))
            A["ho_rate"][key] = n_changes / max(tot_minutes * n_users, 1e-6)
            # Per-user HO rate [HO/(user·min)]
            per_user_rate = {}
            try:
                per_user_changes = ho_real.groupby("user_id").size().to_dict()
            except Exception:
                per_user_changes = {}
            # Every visibility-set user gets a rate (zero if no HOs)
            if vis is not None and not vis.empty:
                vis_users = set(int(u) for u in vis["user_id"].unique())
            else:
                vis_users = set(int(u) for u in per_user_changes.keys())
            for u in vis_users:
                per_user_rate[int(u)] = float(per_user_changes.get(int(u), 0)) / max(tot_minutes, 1e-6)
            A["ho_rate_per_user"][key] = per_user_rate

            # Inter-handover interval [s]: gaps between serving-sat changes read
            # from sinr_snapshot (counts soft switches AND forced reconnections).
            ihi_all: list = []
            ihi_pu: dict[int, list] = {}
            if sin is not None and not sin.empty \
               and {"user_id", "sat_id", "t_idx"}.issubset(sin.columns):
                for uid, ssub in sin.sort_values("t_idx").groupby("user_id"):
                    tt  = ssub["t_idx"].to_numpy()
                    sat = ssub["sat_id"].to_numpy()
                    chg = tt[1:][sat[1:] != sat[:-1]]
                    if chg.size >= 2:
                        d = (np.diff(chg).astype(float) * step_size_s)
                        ihi_all.extend(d.tolist())
                        ihi_pu[int(uid)] = d.tolist()
            if ihi_all: A["ihi"][key] = np.array(ihi_all, dtype=float)
            A["ihi_per_user"][key] = ihi_pu

            pp, n_ho = 0, 0
            pp_per_user = {}
            for uid, sub in ho_real.sort_values("t_idx").groupby("user_id"):
                u_pp, u_n = 0, 0
                evs = sub[["t_idx", "from_sat_id", "to_sat_id"]].to_numpy()
                for i in range(len(evs) - 1):
                    t1, f1, t1_to = evs[i]
                    t2, f2, t2_to = evs[i + 1]
                    if (t2 - t1) <= _PINGPONG_WINDOW_TICKS and t1_to == f2 and t2_to == f1:
                        pp += 1; u_pp += 1
                pp_per_user[int(uid)] = (u_pp / len(evs)) if len(evs) > 0 else 0.0
                n_ho += len(evs)
            A["pingpong"][key] = (pp / n_ho) if n_ho > 0 else 0.0
            A["pingpong_per_user"][key] = pp_per_user

            if "service_gap_s" in ho.columns:
                sg = ho["service_gap_s"].dropna().astype(float)
                if not sg.empty:
                    A["service_gap"][key]      = sg.to_numpy()
                    A["service_gap_s_all"][key] = sg.to_numpy()
                    if key in A["outage_summary"]:
                        A["outage_summary"][key]["service_gap_med_s"] = float(sg.median())

            if {"user_lat", "user_lon"}.issubset(ho.columns):
                A["ho_geo"][key] = ho[["user_lat", "user_lon"]].dropna()

            if sin is not None and not sin.empty and "t_idx" in sin.columns:
                # Dedup on (user_id, t_idx) so the lookup returns a scalar per tick
                sin_idx = (sin.drop_duplicates(subset=["user_id", "t_idx"])
                              .set_index(["user_id", "t_idx"])["sinr_db"])
                segs = []
                for r in ho.itertuples(index=False):
                    if r.to_sat_id < 0 or r.from_sat_id < 0: continue
                    u, t0 = int(r.user_id), int(r.t_idx)
                    seg = [sin_idx.get((u, t0 + dt), np.nan)
                           for dt in range(-_HO_ALIGN_WINDOW, _HO_ALIGN_WINDOW + 1)]
                    segs.append(seg)
                if segs:
                    A["ho_sinr_align"][key] = np.asarray(segs, dtype=float)
    return A


# Figure builders (each emits one png; absent data -> no-op)
def _bootstrap_mean_ci(samples: np.ndarray,
                        n_boot: int = 1000,
                        ci: float = 95.0,
                        seed: int = 42) -> tuple[float, float, float]:
    """(mean, lo, hi) of a 95% bootstrap CI on the sample mean; empty -> NaNs."""
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0: return (float("nan"),) * 3
    if a.size == 1: return (float(a[0]), float(a[0]), float(a[0]))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = a[idx].mean(axis=1)
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return (float(a.mean()), lo, hi)


def _fig_bar_with_ci(per_user_dict: dict, out: Path, fname: str,
                      title: str, ylabel: str,
                      ylim: tuple | None = None,
                      show_dots: bool = True,
                      seed: int = 42) -> None:
    """Bar chart with mean +/- 95% bootstrap CI + jittered per-user dots."""
    if not per_user_dict: return
    labels, means, lo_err, hi_err, colours = [], [], [], [], []
    per_cell_samples: list[np.ndarray] = []
    for k, d in per_user_dict.items():
        labels.append(f"{short_label(k[0])}-{_strat_abbrev(k[1])}" if k[1]
                      else short_label(k[0]))
        colours.append(_COLORS.get(k[0], "k"))
        a = np.array(list(d.values()), dtype=float)
        per_cell_samples.append(a)
        m, lo, hi = _bootstrap_mean_ci(a, seed=seed + len(labels))
        means.append(m)
        lo_err.append(max(0.0, m - lo) if m == m else 0.0)
        hi_err.append(max(0.0, hi - m) if m == m else 0.0)
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    xs = np.arange(len(labels))
    ax.bar(xs, means, color=colours,
            yerr=[lo_err, hi_err], capsize=4, ecolor="black",
            edgecolor="black", linewidth=0.6, zorder=2,
            error_kw=dict(lw=1.2))
    if show_dots:
        rng = np.random.default_rng(int(seed))
        for x, a, col in zip(xs, per_cell_samples, colours):
            if a.size == 0: continue
            jitter = rng.uniform(-0.18, 0.18, size=a.size)
            ax.scatter(x + jitter, a, s=10, color="gray", alpha=0.35,
                        edgecolor="none", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None: ax.set_ylim(*ylim)
    for j, m in enumerate(means):
        if m == m:
            ax.text(j, m, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out / fname, dpi=140)
    plt.close(fig)
    bars = [dict(label=lbl, value=m,
                  ci_lo=m - lo_e if m == m else float("nan"),
                  ci_hi=m + hi_e if m == m else float("nan"),
                  color=col,
                  samples=samp)
             for lbl, m, lo_e, hi_e, col, samp
             in zip(labels, means, lo_err, hi_err, colours, per_cell_samples)]
    _save_scalar_data(out, fname,
        title=title, ylabel=ylabel, bars=bars)


# Per-user CCDF helpers (overlay + small-multiples)
def _ccdf_per_user_overlay(per_user: dict, out: Path, fname: str,
                            title: str, xlabel: str,
                            x_left: float | None = None,
                            x_right: float | None = None,
                            min_samples: int = 2) -> None:
    """Grey per-user lines + a bold black overall curve, one panel per cell."""
    if not per_user: return
    keys = list(per_user.keys())
    cols = min(len(keys), 3)
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5.2 * cols, 3.6 * rows),
                              squeeze=False, sharex=True, sharey=True)
    panels = []
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        ax.set_title(f"{short_label(k[0])} / {k[1]}", fontsize=8)
        ax.grid(alpha=0.3); ax.set_xlabel(xlabel)
        ax.set_ylabel(r"P[$\geq$ x]")
        # per-user grey lines
        all_samples = []
        user_curves = []
        for uid, arr in per_user[k].items():
            arr = np.asarray(arr, dtype=float)
            if arr.size < min_samples: continue
            x, y = _ccdf(arr)
            ax.plot(x, y, color="gray", alpha=0.35, lw=0.6, zorder=2)
            all_samples.append(arr)
            user_curves.append(dict(x=x, y=y))
        # overall bold curve
        overall = {}
        if all_samples:
            cat = np.concatenate(all_samples)
            x, y = _ccdf(cat)
            ax.plot(x, y, color="black", lw=2.0, zorder=3,
                     label="over-all")
            overall = dict(x=x, y=y)
        if x_left is not None: ax.set_xlim(left=x_left)
        if x_right is not None: ax.set_xlim(right=x_right)
        ax.legend(loc="best", fontsize=7)
        panels.append(dict(label=f"{short_label(k[0])}-{_strat_abbrev(k[1])}",
                           overall=overall, users=user_curves))
    for j in range(len(keys), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout(); fig.savefig(out / fname, dpi=140,
                                     bbox_inches="tight")
    plt.close(fig)
    _save_overlay_data(out, fname, title=title, xlabel=xlabel, panels=panels)


def _ccdf_per_user_grid(per_user: dict, out: Path, fname: str,
                         title: str, xlabel: str,
                         min_samples: int = 2) -> None:
    """Small-multiples: one CCDF panel per user, all (cnst, strat)
    curves overlaid in each panel. Auto-sized grid."""
    if not per_user: return
    # Collect global user_id list
    all_uids = sorted({uid for d in per_user.values() for uid in d.keys()})
    if not all_uids: return
    cols = min(int(np.ceil(np.sqrt(len(all_uids)))), 8)
    rows = int(np.ceil(len(all_uids) / cols))
    fig, axes = plt.subplots(rows, cols,
                              figsize=(2.4 * cols, 1.7 * rows),
                              squeeze=False, sharex=True, sharey=True)
    for i, uid in enumerate(all_uids):
        ax = axes[i // cols][i % cols]
        plotted = False
        for k, d in per_user.items():
            arr = np.asarray(d.get(uid, []), dtype=float)
            if arr.size < min_samples: continue
            x, y = _ccdf(arr)
            ax.plot(x, y, **{**_style(*k), "lw": 0.7})
            plotted = True
        ax.set_title(f"u{uid}", fontsize=6)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=5)
        if not plotted:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                     ha="center", va="center", fontsize=6, color="gray")
    for j in range(len(all_uids), rows * cols):
        axes[j // cols][j % cols].axis("off")
    axes[0, 0].set_ylabel(r"P[$\geq$ x]", fontsize=7)
    fig.suptitle(title + f"  ({len(all_uids)} users)", y=1.005, fontsize=10)
    fig.tight_layout(); fig.savefig(out / fname, dpi=140,
                                     bbox_inches="tight")
    plt.close(fig)


def _fig_visibility_ccdf(A, out):
    if not A["visibility"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CCDF of $n_{vis}$ (≥ k visible sats)")
    ax.set_xlabel("Number of visible satellites, $k$")
    ax.set_ylabel(r"P$[n_\mathrm{vis} \geq k]$"); ax.grid(alpha=0.3)
    curves = []
    # Visibility is geometry -> strategy-independent: one curve per constellation.
    for cnst, s in _pool_by_constellation(A["visibility"]).items():
        ks = np.arange(0, max(int(s.max()) + 1, 20))
        ccdf = np.array([(s >= ki).mean() for ki in ks])
        color = _COLORS.get(cnst.lower())
        ax.plot(ks, ccdf, label=short_label(cnst), color=color, lw=1.8)
        curves.append(dict(label=short_label(cnst), color=color,
                            linestyle="-", x=ks, y=ccdf))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 20); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(out / "fig1_visibility_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig1_visibility_ccdf.png",
        title="CCDF of n_vis (≥ k visible sats)",
        xlabel="Number of visible satellites k", ylabel="P[n_vis >= k]", curves=curves)


def _fig_coverage_bar(A, out):
    if not A["cover_prob"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("P[user covered] (≥1 visible sat)")
    labels, vals, cols = [], [], []
    for k, p in A["cover_prob"].items():
        labels.append(f"{short_label(k[0])}-{_strat_abbrev(k[1])}" if k[1]
                      else short_label(k[0]))
        vals.append(p); cols.append(_COLORS.get(k[0], "k"))
    ax.bar(labels, vals, color=cols)
    ax.set_ylabel("Coverage probability")
    ax.set_ylim(0, 1.02); ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    for j, v in enumerate(vals):
        ax.text(j, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig2_coverage_bar.png", dpi=140)
    plt.close(fig)
    _save_scalar_data(out, "fig2_coverage_bar.png",
        title="P[user covered] (>=1 visible sat)", ylabel="Coverage probability",
        bars=[dict(label=lbl, value=v, color=c)
              for lbl, v, c in zip(labels, vals, cols)])


def _fig_mean_nvis_vs_lat(A, out):
    if not A["lat_curves"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Mean $n_{vis}$ vs. user latitude")
    ax.set_xlabel("Latitude [°N]"); ax.set_ylabel(r"Mean $n_\mathrm{vis}$")
    ax.grid(alpha=0.3)
    # n_vis is geometry -> strategy-independent: average to one curve per constellation.
    by_c: dict[str, list] = {}
    for (cnst, _strat), g in A["lat_curves"].items():
        by_c.setdefault(cnst, []).append(g)
    curves = []
    markers = {"oneweb": "o", "starlink": "s", "kuiper": "^"}
    for cnst, gs in by_c.items():
        g = pd.concat(gs, axis=1).mean(axis=1)
        xs = [(iv.left + iv.right) / 2 for iv in g.index]
        color = _COLORS.get(cnst.lower(), "k")
        ax.plot(xs, g.values, color=color, lw=1.8,
                marker=markers.get(cnst.lower(), "o"), markersize=5,
                markerfacecolor="none", label=short_label(cnst))
        curves.append(dict(label=short_label(cnst), color=color,
                            linestyle="-", x=xs, y=list(g.values)))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3_mean_nvis_vs_lat.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig3_mean_nvis_vs_lat.png",
        title="Mean n_vis vs. user latitude",
        xlabel="Latitude [deg N]", ylabel="Mean n_vis", curves=curves)


def _fig_visibility_gap_cdf(A, out):
    """fig4 - CDF of visibility-outage duration (kept for backwards compat)."""
    if not A["vis_gap"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CDF of visibility-gap duration"); ax.grid(alpha=0.3)
    ax.set_xlabel("Visibility-gap duration [s]"); ax.set_ylabel("CDF")
    curves = []
    for k, s in A["vis_gap_s"].items():
        x, y = _cdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig4_visibility_gap_cdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig4_visibility_gap_cdf.png",
        title="CDF of visibility-gap duration",
        xlabel="Visibility-gap duration [s]", ylabel="CDF", curves=curves)


def _fig_visibility_gap_ccdf_seconds(A, out):
    """fig17 - CCDF of visibility-outage duration [s]."""
    if not A["vis_gap_s"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CCDF of visibility-outage duration (no sat above elev_min)")
    ax.set_xlabel("Visibility-outage duration [s]"); ax.set_ylabel(r"P[outage $\geq$ t]")
    ax.grid(alpha=0.3)
    curves = []
    for k, s in A["vis_gap_s"].items():
        if s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig17_visibility_outage_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig17_visibility_outage_ccdf.png",
        title="Visibility-outage duration CCDF",
        xlabel="Visibility-outage duration [s]", ylabel="P[outage ≥ t]", curves=curves)


def _fig_service_outage_ccdf_seconds(A, out):
    """fig18 - CCDF of served-link outage (HO-induced gap) in SECONDS."""
    if not A["service_gap_s_all"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CCDF of service-interruption duration (HO-induced)")
    ax.set_xlabel("Service-interruption duration [s]"); ax.set_ylabel(r"P[gap $\geq$ t]")
    ax.grid(alpha=0.3)
    curves = []
    for k, s in A["service_gap_s_all"].items():
        if s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig18_service_outage_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig18_service_outage_ccdf.png",
        title="Service-interruption duration CCDF",
        xlabel="Service-interruption duration [s]", ylabel="P[gap ≥ t]", curves=curves)


# Per-user CCDF figures (overlay + grid for SINR, I/N, IHI, vis-gap)
def _fig_sinr_per_user_overlay(A, out):
    _ccdf_per_user_overlay(A.get("sinr_per_user", {}), out,
        "fig20_sinr_ccdf_per_user_overlay.png",
        "SINR CCDF — per-user (grey) and over-all (black)",
        "SINR [dB]")


def _fig_sinr_per_user_grid(A, out):
    _ccdf_per_user_grid(A.get("sinr_per_user", {}), out,
        "fig21_sinr_ccdf_per_user_grid.png",
        "SINR CCDF — per-user panel", "SINR [dB]")


def _fig_ion_per_user_overlay(A, out):
    _ccdf_per_user_overlay(A.get("ion_per_user", {}), out,
        "fig22_ion_ccdf_per_user_overlay.png",
        "I/N CCDF — per-user (grey) and over-all (black)",
        "I/N [dB]")


def _fig_ihi_per_user_overlay(A, out):
    _ccdf_per_user_overlay(A.get("ihi_per_user", {}), out,
        "fig23_ihi_ccdf_per_user_overlay.png",
        "IHI CCDF — per-user (grey) and over-all (black)",
        "Inter-handover interval [s]")


def _fig_visgap_per_user_overlay(A, out):
    _ccdf_per_user_overlay(A.get("vis_gap_s_per_user", {}), out,
        "fig24_visgap_ccdf_per_user_overlay.png",
        "Visibility-outage CCDF — per-user (grey) and over-all (black)",
        "Visibility-outage duration [s]")


def _fig_access_delay_ccdf(A, out):
    """fig25 - Over-all CCDF of access-link one-way delay (ms)."""
    if not A.get("access_delay_ms"): return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Access-link one-way delay CCDF (user ↔ served sat)")
    ax.set_xlabel("delay [ms]"); ax.set_ylabel(r"P[delay $\geq$ t]")
    ax.grid(alpha=0.3)
    curves = []
    for k, s in A["access_delay_ms"].items():
        if s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig25_access_delay_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig25_access_delay_ccdf.png",
        title="Access-link one-way delay CCDF", xlabel="delay [ms]",
        ylabel="P[delay ≥ t]", curves=curves)


def _fig_e2e_rtt_ccdf(A, out):
    """fig26 - Over-all CCDF of end-to-end RTT (ms)."""
    if not A.get("e2e_rtt_ms"): return
    from matplotlib.ticker import MaxNLocator
    for cnst, series in _split_by_cnst(A["e2e_rtt_ms"]).items():
        if all(np.asarray(s).size == 0 for s in series.values()):
            continue
        fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
        ax.set_title(f"End-to-end RTT CCDF ({short_label(cnst)})")
        ax.set_xlabel("RTT [ms]"); ax.set_ylabel("CCDF")
        curves = []
        for strat in _strat_sort(series):
            s = np.asarray(series[strat])
            if s.size == 0: continue
            x, y = _ccdf(s); st = _style_strat(strat); ax.plot(x, y, **st)   # CCDF
            curves.append(dict(label=st["label"], color=st["color"],
                                linestyle=str(st["linestyle"]), x=x, y=y))
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer x ticks
        _frame(ax)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fname = f"fig26_e2e_rtt_ccdf_{cnst}.png"
        _bordered_save(fig, out / fname, 140); plt.close(fig)
        _save_curve_data(out, fname,
            title=f"End-to-end RTT CCDF ({cnst})", xlabel="RTT [ms]",
            ylabel="CCDF", curves=curves)


def _fig_availability_bar(A, out):
    """fig19 - Per-cell availability % (mean +/- 95% CI); strategies pooled per constellation."""
    src = A.get("availability_per_user", {})
    agg: dict = {}
    for (cnst, strat), d in src.items():
        bucket = agg.setdefault((cnst, ""), {})
        for uid, v in d.items():
            bucket[f"{strat}:{uid}"] = v       # unique keys pool all strat x user
    _fig_bar_with_ci(agg, out,
        "fig19_availability_bar.png",
        "Per-cell availability (mean ± 95% bootstrap CI)",
        "Availability [%]",
        ylim=(0, 101))


def _fig_contact_cdf(A, out):
    if not A["contact"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Contact-time CDF (single-link runs)")
    ax.set_xlabel("Contact duration [s]"); ax.set_ylabel("CDF"); ax.grid(alpha=0.3)
    curves = []
    # Contact time is geometry -> strategy-independent: one curve per constellation.
    for cnst, s in _pool_by_constellation(A["contact"]).items():
        x, y = _cdf(s); color = _COLORS.get(cnst.lower())
        ax.plot(x, y, label=short_label(cnst), color=color, lw=1.8)
        curves.append(dict(label=short_label(cnst), color=color,
                            linestyle="-", x=x, y=y))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig5_contact_time_cdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig5_contact_time_cdf.png",
        title="Contact-time CDF", xlabel="Contact duration [s]",
        ylabel="CDF", curves=curves)


def _fig_contact_vs_peakelev(A, out):
    if not A["contact_peak_pairs"]: return
    n = len(A["contact_peak_pairs"])
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 4),
                              sharey=True, squeeze=False)
    panels = []
    for ax, (k, (d, p)) in zip(axes[0], A["contact_peak_pairs"].items()):
        col = _COLORS.get(k[0], "k")
        ax.scatter(p, d, s=5, alpha=0.4, color=col)
        ax.set_title(f"{short_label(k[0])} / {k[1]}", fontsize=8)
        ax.set_xlabel("Peak elevation [°]"); ax.grid(alpha=0.3)
        panels.append(dict(label=f"{short_label(k[0])}-{_strat_abbrev(k[1])}",
                           color=col, x=p, y=d))
    axes[0, 0].set_ylabel("contact duration [s]")
    fig.suptitle("Contact duration vs. peak elevation", y=1.02)
    fig.tight_layout(); fig.savefig(out / "fig6_contact_vs_peakelev.png",
                                     dpi=140, bbox_inches="tight")
    plt.close(fig)
    _save_scatter_data(out, "fig6_contact_vs_peakelev.png",
        title="Contact duration vs. peak elevation",
        xlabel="Peak elevation [°]", ylabel="Contact duration [s]", panels=panels)


def _fig_intercontact_gap_cdf(A, out):
    # Prefer vis_gap_s (all zero-visibility runs); fall back to interior gaps.
    src = A["vis_gap_s"] if A.get("vis_gap_s") else A.get("inter_contact_gap", {})
    if not src: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CDF of per-user coverage-outage duration"); ax.grid(alpha=0.3)
    ax.set_xlabel("Inter-contact gap [s]"); ax.set_ylabel("CDF")
    # Coverage is geometry -> strategy-independent: one curve per constellation.
    curves = []
    for cnst, s in _pool_by_constellation(src).items():
        if np.asarray(s).size == 0:
            continue
        x, y = _cdf(s); color = _COLORS.get(cnst.lower())
        ax.plot(x, y, label=short_label(cnst), color=color, lw=1.8)
        curves.append(dict(label=short_label(cnst), color=color,
                            linestyle="-", x=x, y=y))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig7_intercontact_gap_cdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig7_intercontact_gap_cdf.png",
        title="CDF of per-user coverage-outage duration",
        xlabel="Inter-contact gap [s]", ylabel="CDF", curves=curves)


def _fig_ho_rate_bar(A, out):
    """Per-constellation HO-rate bars, one bar per strategy (colour = strategy)."""
    _fig_bar_by_strategy(A.get("ho_rate_per_user", {}), out,
        "fig8_ho_rate_bar", "Handover rate [HO/(user·min)]")


def _fig_ihi_cdf(A, out):
    """One IHI CDF per constellation; one line per strategy."""
    if not A["ihi"]: return
    from matplotlib.ticker import MaxNLocator
    for cnst, series in _split_by_cnst(A["ihi"]).items():
        fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
        ax.set_title(f"Inter-handover interval CDF ({short_label(cnst)})")
        ax.set_xlabel("Inter-handover interval, IHI [min]"); ax.set_ylabel("CDF")
        curves = []
        for strat in _strat_sort(series):
            x, y = _cdf(np.asarray(series[strat], float) / 60.0)   # s -> min
            st = _style_strat(strat); ax.plot(x, y, **st)
            curves.append(dict(label=st["label"], color=st["color"],
                                linestyle=str(st["linestyle"]), x=x, y=y))
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer x ticks
        _frame(ax)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fname = f"fig9_ihi_cdf_{cnst}.png"
        _bordered_save(fig, out / fname, 140); plt.close(fig)
        _save_curve_data(out, fname,
            title=f"Inter-handover interval CDF ({cnst})", xlabel="Inter-handover interval, IHI [min]",
            ylabel="CDF", curves=curves)


def _fig_pingpong_bar(A, out):
    """Bar of mean per-user ping-pong ratio with 95% bootstrap CI + dots."""
    _pp_win_s = _PINGPONG_WINDOW_TICKS * A.get("step_s", 1.0)   # ping-pong window in seconds
    _fig_bar_with_ci(A.get("pingpong_per_user", {}), out,
        "fig10_pingpong_bar.png",
        f"Ping-pong ratio (A→B→A within {_pp_win_s:g} s) — "
        "mean ± 95% bootstrap CI",
        "Ping-pong ratio [per HO]",
        ylim=(0, 1.02))


def _fig_service_gap_cdf(A, out):
    if not A["service_gap"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Service-interruption duration CDF"); ax.grid(alpha=0.3)
    ax.set_xlabel("Service-interruption duration [s]"); ax.set_ylabel("CDF")
    curves = []
    for k, s in A["service_gap"].items():
        x, y = _cdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig11_service_gap_cdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig11_service_gap_cdf.png",
        title="Service-interruption duration CDF",
        xlabel="Service-interruption duration [s]", ylabel="CDF", curves=curves)


def _fig_ho_geo_map(A, out):
    if not A["ho_geo"]: return
    keys = list(A["ho_geo"].keys())
    n = len(keys); cols = min(n, 3); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                              squeeze=False)
    panels = []
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        d = A["ho_geo"][k]
        ax.hexbin(d["user_lon"], d["user_lat"], gridsize=30,
                   cmap="viridis", mincnt=1)
        ax.set_title(f"{short_label(k[0])} / {k[1]}", fontsize=8)
        ax.set_xlabel("Longitude [°]"); ax.set_ylabel("Latitude [°]"); ax.grid(alpha=0.3)
        panels.append(dict(label=f"{short_label(k[0])}-{_strat_abbrev(k[1])}",
                           x=list(d["user_lon"]), y=list(d["user_lat"])))
    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Handover density over Canada", y=1.02)
    fig.tight_layout(); fig.savefig(out / "fig12_ho_geo_map.png", dpi=140,
                                     bbox_inches="tight")
    plt.close(fig)
    _save_hexbin_data(out, "fig12_ho_geo_map.png",
        title="Handover density over Canada",
        xlabel="Longitude [°]", ylabel="Latitude [°]", panels=panels)


def _fig_sinr_ccdf(A, out):
    if not A["sinr"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("SINR CCDF (served links)")
    ax.set_xlabel("SINR [dB]"); ax.set_ylabel(r"P$[\mathrm{SINR} \geq \gamma]$"); ax.grid(alpha=0.3)
    curves = []
    for k, s in A["sinr"].items():
        x, y = _ccdf(s.values); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig13_sinr_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig13_sinr_ccdf.png",
        title="SINR CCDF (served links)", xlabel="SINR [dB]",
        ylabel="P[SINR ≥ γ]", curves=curves)


def _fig_rate_ccdf(A, out):
    """CCDF of served-link throughput [Mbps], one curve per (constellation, strategy)."""
    if not A.get("rate"): return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Throughput CCDF (served links)")
    ax.set_xlabel("Throughput [Mbit/s]"); ax.set_ylabel(r"P[rate $\geq$ R]")
    ax.grid(alpha=0.3)
    curves = []
    for k, s in A["rate"].items():
        s = np.asarray(s, float); s = s[np.isfinite(s)]
        if s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig19_rate_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig19_rate_ccdf.png",
        title="Throughput CCDF (served links)", xlabel="Throughput [Mbit/s]",
        ylabel="P[rate >= R]", curves=curves)


def _representative_user(per_user):
    """Pick the user with the most valid samples (avoids cross-user over-counting)."""
    best, bestn = None, -1
    for _uid, v in per_user.items():
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if v.size > bestn:
            bestn, best = v.size, v
    return best


def _fig_sinr_ccdf_user(A, out):
    """SINR CCDF for a single representative user, one curve per (constellation, strategy)."""
    src = A.get("sinr_per_user", {})
    if not src: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("SINR CCDF (representative user)")
    ax.set_xlabel("SINR [dB]"); ax.set_ylabel(r"P[SINR $\geq$ x]"); ax.grid(alpha=0.3)
    curves = []
    for k, per_user in src.items():
        s = _representative_user(per_user)
        if s is None or s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig13u_sinr_ccdf_user.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig13u_sinr_ccdf_user.png",
        title="SINR CCDF (representative user)", xlabel="SINR [dB]",
        ylabel="P[SINR >= x]", curves=curves)


def _fig_rate_ccdf_user(A, out):
    """Throughput CCDF for a single representative user (no cross-user pooling)."""
    src = A.get("rate_per_user", {})
    if not src: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Throughput CCDF (representative user)")
    ax.set_xlabel("Throughput [Mbit/s]"); ax.set_ylabel(r"P[rate $\geq$ R]")
    ax.grid(alpha=0.3)
    curves = []
    for k, per_user in src.items():
        s = _representative_user(per_user)
        if s is None or s.size == 0: continue
        x, y = _ccdf(s); st = _style(*k); ax.plot(x, y, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]), x=x, y=y))
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig19u_rate_ccdf_user.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig19u_rate_ccdf_user.png",
        title="Throughput CCDF (representative user)", xlabel="Throughput [Mbit/s]",
        ylabel="P[rate >= R]", curves=curves)


def _fig_power_bar(A, out):
    """Bar of system total transmit power, time-mean +/- 95% CI, per (constellation, strategy)."""
    _fig_bar_by_strategy(A.get("system_power", {}), out,
        "fig30_power_bar", "Mean transmit power [dBW]")


def _fig_avg_rate_bar(A, out):
    """Bar of mean per-user average throughput with 95% bootstrap CI."""
    ru = A.get("rate_per_user", {})
    if not ru: return
    avg = {}
    for k, per_user in ru.items():
        d = {}
        for uid, s in per_user.items():
            s = np.asarray(s, float); s = s[np.isfinite(s)]
            if s.size:
                d[int(uid)] = float(s.mean())
        if d:
            avg[k] = d
    _fig_bar_with_ci(avg, out,
        "fig28_avg_rate_bar.png",
        "Average per-user throughput (mean ± 95% bootstrap CI)",
        "Mean user throughput [Mbit/s]", show_dots=False)


def _fig_avg_rate_bar_norm(A, out):
    """Per-user throughput normalized by each constellation's best strategy (best = 1.0)."""
    ru = A.get("rate_per_user", {})
    if not ru: return
    # Per-cell mean over users, then the per-constellation max (the normalizer).
    cell_mean, by_cnst = {}, {}
    for k, per_user in ru.items():
        vals = [np.asarray(s, float)[np.isfinite(np.asarray(s, float))].mean()
                for s in per_user.values()
                if np.asarray(s, float)[np.isfinite(np.asarray(s, float))].size]
        if vals:
            cell_mean[k] = float(np.mean(vals))
            by_cnst.setdefault(k[0], []).append(cell_mean[k])
    cmax = {c: max(v) for c, v in by_cnst.items() if v}
    norm = {}
    for k, per_user in ru.items():
        mx = cmax.get(k[0], 0.0)
        if not mx:
            continue
        d = {}
        for uid, s in per_user.items():
            s = np.asarray(s, float); s = s[np.isfinite(s)]
            if s.size:
                d[int(uid)] = float(s.mean()) / mx
        if d:
            norm[k] = d
    _fig_bar_with_ci(norm, out,
        "fig29_avg_rate_bar_norm.png",
        "Normalized average throughput (per constellation, best strategy = 1.0)",
        "Normalized throughput", ylim=(0, 1.08), show_dots=False)


def _fig_rate_per_user_overlay(A, out):
    _ccdf_per_user_overlay(A.get("rate_per_user", {}), out,
        "fig27_rate_ccdf_per_user_overlay.png",
        "Throughput CCDF — per-user (grey) and over-all (black)",
        "Throughput [Mbit/s]")


def _fig_sinr_vs_elev(A, out):
    if not A["sinr_elev_pairs"]: return
    n = len(A["sinr_elev_pairs"]); cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                              squeeze=False, sharex=True, sharey=True)
    panels = []
    for i, (k, (el, sin)) in enumerate(A["sinr_elev_pairs"].items()):
        ax = axes[i // cols][i % cols]
        col = _COLORS.get(k[0], "k")
        ax.scatter(el, sin, s=4, alpha=0.3, color=col)
        ax.set_title(f"{short_label(k[0])} / {k[1]}", fontsize=8)
        ax.grid(alpha=0.3); ax.set_xlabel("Serving-satellite elevation [°]")
        panels.append(dict(label=f"{short_label(k[0])}-{_strat_abbrev(k[1])}",
                           color=col, x=el, y=sin))
    axes[0, 0].set_ylabel("SINR [dB]")
    fig.suptitle("SINR vs. serving-sat elevation", y=1.02)
    fig.tight_layout(); fig.savefig(out / "fig14_sinr_vs_elev.png",
                                     dpi=140, bbox_inches="tight")
    plt.close(fig)
    _save_scatter_data(out, "fig14_sinr_vs_elev.png",
        title="SINR vs. serving-sat elevation",
        xlabel="Serving-satellite elevation [°]", ylabel="SINR [dB]", panels=panels)


def _fig_sinr_drop_around_ho(A, out):
    if not A["ho_sinr_align"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("Mean SINR trajectory around handover")
    ax.set_xlabel("Time since handover [s]"); ax.set_ylabel("SINR [dB]")
    ax.grid(alpha=0.3)
    xs = np.arange(-_HO_ALIGN_WINDOW, _HO_ALIGN_WINDOW + 1) * A.get("step_s", 1.0)
    curves = []
    for k, M in A["ho_sinr_align"].items():
        mean = np.nanmean(M, axis=0)
        st = _style(*k); ax.plot(xs, mean, **st)
        curves.append(dict(label=st["label"], color=st["color"],
                            linestyle=str(st["linestyle"]),
                            x=xs, y=mean))       # arrays -> saver does.tolist()
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig15_sinr_drop_around_ho.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig15_sinr_drop_around_ho.png",
        title="Mean SINR trajectory around handover",
        xlabel="Time since handover [s]", ylabel="SINR [dB]", curves=curves)


def _fig_ion_ccdf(A, out):
    if not A["ion"]: return
    fig, ax = plt.subplots(figsize=_IEEE_FIGSIZE)
    ax.set_title("CCDF of I/N (interference-vs-noise regime)")
    ax.set_xlabel("I/N [dB]"); ax.set_ylabel("P[I/N ≥ γ]"); ax.grid(alpha=0.3)
    # Interference-free links (I/N = -inf) clamp to a -40 dB noise-limited floor.
    _ION_FLOOR_DB = -40.0
    # I/N is geometry-driven -> one curve per constellation, strategies pooled.
    by_cnst: dict[str, list] = {}
    for (cnst, strat), s in A["ion"].items():
        by_cnst.setdefault(cnst, []).append(s.dropna())
    curves = []
    for cnst, parts in by_cnst.items():
        vals = np.clip(pd.concat(parts).values, _ION_FLOOR_DB, None)
        if vals.size == 0: continue
        x, y = _ccdf(vals)
        col = _COLORS.get(cnst, "k")
        ax.plot(x, y, color=col, linestyle="-", lw=2.0, label=short_label(cnst))
        curves.append(dict(label=short_label(cnst), color=col,
                            linestyle="-", x=x, y=y))
    ax.set_xlim(left=_ION_FLOOR_DB)
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig16_ion_ccdf.png", dpi=140)
    plt.close(fig)
    _save_curve_data(out, "fig16_ion_ccdf.png",
        title="I/N CCDF", xlabel="I/N [dB]", ylabel="P[I/N ≥ γ]",
        curves=curves)


# Summary table (E)
def _build_summary(A) -> pd.DataFrame:
    rows = []
    for k in A["visibility"]:
        sinr_med, sinr_p5, rate_med = A["sinr_summary"].get(
            k, (float("nan"),) * 3)
        cnt_mean, cnt_med = A["contact_summary"].get(
            k, (float("nan"), float("nan")))
        rows.append({
            "constellation":      k[0],
            "strategy":           k[1],
            "mean_n_visible":     float(A["visibility"][k].mean()),
            "p_covered":          float(A["cover_prob"].get(k, float("nan"))),
            "mean_contact_s":     cnt_mean,
            "median_contact_s":   cnt_med,
            "ho_rate_per_min":    float(A["ho_rate"].get(k, float("nan"))),
            "pingpong_ratio":     float(A["pingpong"].get(k, float("nan"))),
            "service_gap_med_s":  (float(np.median(A["service_gap"][k]))
                                    if k in A["service_gap"] else float("nan")),
            "median_sinr_db":     sinr_med,
            "p5_sinr_db":         sinr_p5,
            "median_rate_mbps":   (rate_med / 1e6) if rate_med == rate_med
                                   else float("nan"),
            "median_i_over_n_db": (float(A["ion"][k].median())
                                    if k in A["ion"] else float("nan")),
            # Outage / availability
            "sim_seconds":         float(A["outage_summary"][k]["sim_seconds"])
                                    if k in A["outage_summary"] else float("nan"),
            "mean_total_outage_s": float(A["outage_summary"][k]["mean_total_outage_s"])
                                    if k in A["outage_summary"] else float("nan"),
            "worst_outage_s":      float(A["outage_summary"][k]["worst_outage_s"])
                                    if k in A["outage_summary"] else float("nan"),
            "vis_gap_med_s":       float(A["outage_summary"][k]["vis_gap_med_s"])
                                    if k in A["outage_summary"] else float("nan"),
            "availability_pct":    float(A["outage_summary"][k]["availability_pct"])
                                    if k in A["outage_summary"] else float("nan"),
            # Per-user spread companions
            "n_users_zero_coverage": int(sum(
                1 for v in A.get("coverage_per_user", {}).get(k, {}).values()
                if v == 0.0)),
            "p5_sinr_db_all":     float(np.percentile(
                np.concatenate(list(A.get("sinr_per_user", {}).get(k, {}).values())
                                or [np.array([np.nan])]), 5))
                if k in A.get("sinr_per_user", {}) else float("nan"),
            "min_sinr_user_p50":  float(min(
                (np.median(np.asarray(a)) for a in
                 A.get("sinr_per_user", {}).get(k, {}).values()
                 if np.asarray(a).size > 0), default=float("nan"))),
            "max_sinr_user_p50":  float(max(
                (np.median(np.asarray(a)) for a in
                 A.get("sinr_per_user", {}).get(k, {}).values()
                 if np.asarray(a).size > 0), default=float("nan"))),
            "n_users_zero_ho":    int(sum(
                1 for v in A.get("ihi_per_user", {}).get(k, {}).values()
                if len(v) == 0)),
            # Routing / delay companions
            "median_access_delay_ms": float(np.median(A.get("access_delay_ms", {}).get(k, np.array([np.nan])))),
            "median_e2e_rtt_ms":      float(np.median(A.get("e2e_rtt_ms",     {}).get(k, np.array([np.nan])))),
            "p95_e2e_rtt_ms":         float(np.percentile(
                A.get("e2e_rtt_ms", {}).get(k, np.array([np.nan])), 95)
                if A.get("e2e_rtt_ms", {}).get(k, np.array([])).size else float("nan")),
            "mean_isl_hops":          float(np.mean(A.get("isl_hops", {}).get(k, np.array([np.nan])))),
            "p95_isl_hops":           float(np.percentile(
                A.get("isl_hops", {}).get(k, np.array([np.nan])), 95)
                if A.get("isl_hops", {}).get(k, np.array([])).size else float("nan")),
        })
    if not rows:
        return pd.DataFrame(columns=["constellation", "strategy"])
    return pd.DataFrame(rows).sort_values(["constellation", "strategy"])


# Main
# Selectable figure registry: short name -> figure function.
_FIG_REGISTRY = {
    "visibility_ccdf":       _fig_visibility_ccdf,
    "coverage_bar":          _fig_coverage_bar,
    "mean_nvis_vs_lat":      _fig_mean_nvis_vs_lat,
    "visibility_gap_cdf":    _fig_visibility_gap_cdf,
    "outage_ccdf":           _fig_visibility_gap_ccdf_seconds,
    "service_outage_ccdf":   _fig_service_outage_ccdf_seconds,
    "access_delay_ccdf":     _fig_access_delay_ccdf,
    "e2e_rtt_ccdf":          _fig_e2e_rtt_ccdf,
    "availability_bar":      _fig_availability_bar,
    "contact_cdf":           _fig_contact_cdf,
    "contact_vs_peakelev":   _fig_contact_vs_peakelev,
    "intercontact_gap_cdf":  _fig_intercontact_gap_cdf,
    "ho_rate_bar":           _fig_ho_rate_bar,
    "power_bar":             _fig_power_bar,
    "ihi_cdf":               _fig_ihi_cdf,
    "pingpong_bar":          _fig_pingpong_bar,
    "service_gap_cdf":       _fig_service_gap_cdf,
    "ho_geo_map":            _fig_ho_geo_map,
    "sinr_ccdf":             _fig_sinr_ccdf,
    "rate_ccdf":             _fig_rate_ccdf,
    "sinr_ccdf_user":        _fig_sinr_ccdf_user,   # single representative user
    "rate_ccdf_user":        _fig_rate_ccdf_user,
    "avg_rate_bar":          _fig_avg_rate_bar,
    "avg_rate_bar_norm":     _fig_avg_rate_bar_norm,
    "rate_per_user_overlay": _fig_rate_per_user_overlay,
    "sinr_vs_elev":          _fig_sinr_vs_elev,
    "sinr_drop_around_ho":   _fig_sinr_drop_around_ho,
    "ion_ccdf":              _fig_ion_ccdf,
    "sinr_per_user_overlay": _fig_sinr_per_user_overlay,
    "sinr_per_user_grid":    _fig_sinr_per_user_grid,
    "ion_per_user_overlay":  _fig_ion_per_user_overlay,
    "ihi_per_user_overlay":  _fig_ihi_per_user_overlay,
    "visgap_per_user_overlay": _fig_visgap_per_user_overlay,
}
# The curated default set (used when no selection is given).
_DEFAULT_FIGS = ("visibility_ccdf", "contact_cdf", "ho_rate_bar", "ihi_cdf",
                 "sinr_ccdf", "rate_ccdf", "avg_rate_bar", "avg_rate_bar_norm",
                 "ion_ccdf", "outage_ccdf", "availability_bar",
                 "access_delay_ccdf", "e2e_rtt_ccdf")

# Per-figure category tags (kpi, scope, stage) for family-based selection.
_FIG_CATEGORY = {
    "visibility_ccdf":       ("coverage",    "user",    "geometry"),
    "coverage_bar":          ("coverage",    "network", "geometry"),
    "mean_nvis_vs_lat":      ("coverage",    "network", "geometry"),
    "visibility_gap_cdf":    ("coverage",    "user",    "geometry"),
    "outage_ccdf":           ("reliability", "network", "geometry"),
    "service_outage_ccdf":   ("reliability", "network", "sim"),
    "access_delay_ccdf":     ("latency",     "link",    "geometry"),
    "e2e_rtt_ccdf":          ("latency",     "network", "sim"),
    "availability_bar":      ("coverage",    "network", "sim"),
    "contact_cdf":           ("mobility",    "link",    "geometry"),
    "contact_vs_peakelev":   ("mobility",    "link",    "geometry"),
    "intercontact_gap_cdf":  ("mobility",    "link",    "geometry"),
    "ho_rate_bar":           ("mobility",    "network", "sim"),
    "power_bar":             ("energy",      "network", "sim"),
    "ihi_cdf":               ("mobility",    "network", "sim"),
    "pingpong_bar":          ("mobility",    "user",    "sim"),
    "service_gap_cdf":       ("reliability", "network", "sim"),
    "ho_geo_map":            ("mobility",    "network", "sim"),
    "sinr_ccdf":             ("capacity",    "link",    "sim"),
    "rate_ccdf":             ("capacity",    "link",    "sim"),
    "sinr_ccdf_user":        ("capacity",    "user",    "sim"),
    "rate_ccdf_user":        ("capacity",    "user",    "sim"),
    "avg_rate_bar":          ("capacity",    "user",    "sim"),
    "avg_rate_bar_norm":     ("capacity",    "network", "sim"),
    "rate_per_user_overlay": ("capacity",    "user",    "sim"),
    "sinr_vs_elev":          ("capacity",    "link",    "sim"),
    "sinr_drop_around_ho":   ("mobility",    "link",    "sim"),
    "ion_ccdf":              ("reliability", "link",    "sim"),
    "sinr_per_user_overlay": ("capacity",    "user",    "sim"),
    "sinr_per_user_grid":    ("capacity",    "user",    "sim"),
    "ion_per_user_overlay":  ("reliability", "user",    "sim"),
    "ihi_per_user_overlay":  ("mobility",    "user",    "sim"),
    "visgap_per_user_overlay": ("coverage",  "user",    "sim"),
    # extra-KPI figures: produced from a run snapshot by extra_kpis
    "energy_power_ccdf":        ("energy",     "network", "sim"),
    "energy_efficiency_bar":    ("energy",     "network", "sim"),
    "connection_admission_bar": ("connection", "network", "sim"),
    "connection_blocking_ccdf": ("connection", "network", "sim"),
    "fairness_jain_bar":        ("fairness",   "network", "sim"),
    "doppler_ccdf":             ("doppler",    "link",    "sim"),
}
# Snapshot-based extra-KPI figures, dispatched per (constellation, strategy) run.
_EXTRA_KPI_FIGS = ("energy_power_ccdf", "energy_efficiency_bar",
                   "connection_admission_bar", "connection_blocking_ccdf",
                   "fairness_jain_bar", "doppler_ccdf")
_FIG_AXES = {
    "kpi":   sorted({c[0] for c in _FIG_CATEGORY.values()}),
    "scope": sorted({c[1] for c in _FIG_CATEGORY.values()}),
    "stage": sorted({c[2] for c in _FIG_CATEGORY.values()}),
}
_AXIS_IDX = {"kpi": 0, "scope": 1, "stage": 2}


def _expand_selectors(sel):
    """Expand 'axis:value' selectors to figure names; explicit names pass through."""
    out = []
    for s in sel:
        if isinstance(s, str) and ":" in s and s.split(":", 1)[0] in _AXIS_IDX:
            axis, val = s.split(":", 1)
            if val not in _FIG_AXES[axis]:
                raise ValueError(f"unknown {axis} '{val}' (valid: {_FIG_AXES[axis]})")
            i = _AXIS_IDX[axis]
            out.extend(n for n, c in _FIG_CATEGORY.items() if c[i] == val)
        else:
            out.append(s)
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq


def figure_categories():
    """(_FIG_CATEGORY, _FIG_AXES) - the selectable taxonomy."""
    return dict(_FIG_CATEGORY), {k: list(v) for k, v in _FIG_AXES.items()}


# AUTO mode: aggregate key(s) each built-in figure needs to be renderable.
_FIG_DATA_KEY = {
    "visibility_ccdf":       ("visibility",),
    "coverage_bar":          ("cover_prob",),
    "mean_nvis_vs_lat":      ("lat_curves",),
    "visibility_gap_cdf":    ("vis_gap_s",),
    "outage_ccdf":           ("vis_gap_s",),
    "service_outage_ccdf":   ("service_gap_s_all",),
    "access_delay_ccdf":     ("access_delay_ms",),
    "e2e_rtt_ccdf":          ("e2e_rtt_ms",),
    "availability_bar":      ("availability_per_user",),
    "contact_cdf":           ("contact",),
    "contact_vs_peakelev":   ("contact_peak_pairs",),
    "intercontact_gap_cdf":  ("vis_gap_s", "inter_contact_gap"),
    "ho_rate_bar":           ("ho_rate_per_user",),
    "power_bar":             ("system_power",),
    "ihi_cdf":               ("ihi",),
    "pingpong_bar":          ("pingpong_per_user",),
    "service_gap_cdf":       ("service_gap",),
    "ho_geo_map":            ("ho_geo",),
    "sinr_ccdf":             ("sinr",),
    "rate_ccdf":             ("rate",),
    "avg_rate_bar":          ("rate_per_user",),
    "avg_rate_bar_norm":     ("rate_per_user",),
    "rate_per_user_overlay": ("rate_per_user",),
    "sinr_vs_elev":          ("sinr_elev_pairs",),
    "sinr_drop_around_ho":   ("ho_sinr_align",),
    "ion_ccdf":              ("ion",),
    "sinr_per_user_overlay": ("sinr_per_user",),
    "sinr_per_user_grid":    ("sinr_per_user",),
    "ion_per_user_overlay":  ("ion_per_user",),
    "ihi_per_user_overlay":  ("ihi_per_user",),
    "visgap_per_user_overlay": ("vis_gap_s_per_user",),
}
# AUTO mode: snapshot columns each extra-KPI figure needs.
_EXTRA_KPI_COLS = {
    "energy_power_ccdf":        {"min_power_w", "slot_admitted"},
    "energy_efficiency_bar":    {"min_power_w", "slot_admitted", "sinr_db_paper"},
    "connection_admission_bar": {"slot_admitted"},
    "connection_blocking_ccdf": {"slot_admitted"},
    "fairness_jain_bar":        {"slot_admitted", "sinr_db_paper", "user_id"},
    "doppler_ccdf":             {"sat_id", "t_idx", "user_id",
                                 "ground_dist_m", "serving_sat_elev_deg"},
}


def _snapshot_columns(cells) -> set:
    """Union of sinr_snapshot column names across cells (metadata only)."""
    cols: set = set()
    try:
        import pyarrow.parquet as pq
    except Exception:
        return cols
    for _, row in cells.iterrows():
        rd = Path(row["run_dir"]) if row.get("run_dir") else None
        from ntnconnect.common.utils.run_io import run_shards
        _snaps = run_shards(rd, "sinr_snapshot") if rd else []
        snap = _snaps[0] if _snaps else None
        if snap and snap.exists():
            try:
                cols |= set(pq.ParquetFile(snap).schema_arrow.names)
            except Exception:
                pass
    return cols


def _prune_empty(sel, A, cells):
    """AUTO mode: keep only figures whose data exists; returns (kept, dropped)."""
    kept, dropped, snap_cols = [], [], None
    for name in sel:
        if name in _FIG_REGISTRY:
            keys = _FIG_DATA_KEY.get(name, ())
            ok = (any(A.get(k) for k in keys) if keys else True)
        elif name in _EXTRA_KPI_FIGS:
            if snap_cols is None:
                snap_cols = _snapshot_columns(cells)
            need = _EXTRA_KPI_COLS.get(name, set())
            ok = bool(need) and need.issubset(snap_cols)
        else:
            ok = True
        (kept if ok else dropped).append(name)
    return kept, dropped


# Unicode -> mathtext map for LaTeX/Computer-Modern rendering.
_UNI_MATH = {
    "≥": r"\geq", "≤": r"\leq", "Δ": r"\Delta", "γ": r"\gamma", "μ": r"\mu",
    "µ": r"\mu", "·": r"\cdot", "→": r"\rightarrow", "×": r"\times",
    "−": "-", "∞": r"\infty", "≈": r"\approx", "°": r"^{\circ}",
    "↔": r"\leftrightarrow", "↑": r"\uparrow", "↓": r"\downarrow",
    "±": r"\pm", "≠": r"\neq", "α": r"\alpha", "β": r"\beta",
    "θ": r"\theta", "λ": r"\lambda", "σ": r"\sigma", "τ": r"\tau",
}


def _sanitize_text(s, use_tex: bool):
    """Convert unicode symbols to math and (for usetex) escape LaTeX specials."""
    if not isinstance(s, str) or not s:
        return s
    # Computer Modern has no em/en-dash glyph
    s = s.replace("—", "-").replace("–", "-")
    for u, r in _UNI_MATH.items():
        s = s.replace(u, f"${r}$")
    if use_tex:
        parts = s.split("$")
        for i in range(0, len(parts), 2):      # even indices = non-math text
            p = parts[i]
            for ch, esc in (("%", r"\%"), ("_", r"\_"), ("&", r"\&"),
                            ("#", r"\#")):
                p = p.replace(ch, esc)
            parts[i] = p
        s = "$".join(parts)
    return s


def _apply_label_overrides(fig, stem, label_overrides):
    """Apply per-figure {xlabel, ylabel, title, legend} overrides before saving."""
    if not label_overrides:
        return
    spec = None
    for key, s in label_overrides.items():
        if key == stem or key in stem:        # "sinr_ccdf" matches "fig13_sinr_ccdf"
            spec = s
            break
    if not spec:
        return
    for ax in fig.axes:
        if "xlabel" in spec: ax.set_xlabel(spec["xlabel"])
        if "ylabel" in spec: ax.set_ylabel(spec["ylabel"])
        if "title" in spec:
            ax.title.set_text(spec["title"]); ax.title.set_visible(True)
        leg = ax.get_legend()
        if leg is not None and "legend" in spec:
            m = spec["legend"]
            for t in leg.get_texts():
                cur = t.get_text()
                if callable(m):
                    t.set_text(m(cur))
                elif isinstance(m, dict) and cur in m:
                    t.set_text(m[cur])


def _apply_output_style(show_titles: bool, dpi: int, fmt: str,
                        paper_style: bool = True, use_tex: bool = False,
                        base_fontsize: int = 9, label_overrides: dict = None) -> None:
    """Apply title/dpi/format/typography/label-override shims globally via matplotlib."""
    import matplotlib, matplotlib.axes, matplotlib.figure, matplotlib.text
    import shutil
    if use_tex and shutil.which("latex") is None:   # LaTeX optional: mathtext fallback
        print("[style] no LaTeX engine found -> using matplotlib mathtext (Computer Modern)")
        use_tex, paper_style = False, True

    if paper_style or use_tex:
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.serif": ["cmr10", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.titlesize":  base_fontsize + 1,
            "axes.labelsize":  base_fontsize,
            "legend.fontsize": base_fontsize - 1,
            "xtick.labelsize": base_fontsize - 1,
            "ytick.labelsize": base_fontsize - 1,
            "axes.formatter.use_mathtext": True,
        })
    if use_tex:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })

    # Force fonts globally, overriding per-figure fontsize=7 calls.
    matplotlib.rcParams.update({
        "axes.labelsize":  base_fontsize,
        "axes.titlesize":  base_fontsize + 1,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "figure.figsize":  _IEEE_FIGSIZE,
        "text.color":      "black",
        "axes.labelcolor": "black",
        "xtick.color":     "black",
        "ytick.color":     "black",
        "axes.edgecolor":  "black",
        "lines.linewidth":      1.6,
        "lines.markersize":     4.0,
        "lines.markeredgewidth": 0.6,
        "axes.axisbelow":       True,
        "grid.alpha":           0.35,
        "grid.linewidth":       0.6,
        "grid.linestyle":       ":",
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "0.7",
        "legend.borderpad":     0.4,
        "legend.handlelength":  1.8,
        "legend.labelspacing":  0.3,
    })
    _legend_fs = base_fontsize - 1
    _orig_legend = matplotlib.axes.Axes.legend
    def _legend(self, *a, **k):
        k["fontsize"] = _legend_fs
        return _orig_legend(self, *a, **k)
    matplotlib.axes.Axes.legend = _legend
    _orig_xlabel = matplotlib.axes.Axes.set_xlabel
    _orig_ylabel = matplotlib.axes.Axes.set_ylabel
    def _xlabel(self, xlabel, *a, **k):
        k.setdefault("fontsize", base_fontsize)
        return _orig_xlabel(self, xlabel, *a, **k)
    def _ylabel(self, ylabel, *a, **k):
        k.setdefault("fontsize", base_fontsize)
        return _orig_ylabel(self, ylabel, *a, **k)
    matplotlib.axes.Axes.set_xlabel = _xlabel
    matplotlib.axes.Axes.set_ylabel = _ylabel

    # Global text sanitizer
    if paper_style or use_tex:
        _orig_set_text = matplotlib.text.Text.set_text
        def _set_text(self, s):
            return _orig_set_text(self, _sanitize_text(s, use_tex))
        matplotlib.text.Text.set_text = _set_text

    if not show_titles:
        matplotlib.axes.Axes.set_title = lambda self, *a, **k: None
        matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

    _orig = matplotlib.figure.Figure.savefig
    def _savefig(self, fname, *a, **k):
        k["dpi"] = dpi
        fn = str(fname)
        _apply_label_overrides(self, Path(fn).stem, label_overrides)
        if fmt != "png" and fn.lower().endswith(".png"):
            fn = fn[:-4] + "." + fmt
        return _orig(self, fn, *a, **k)
    matplotlib.figure.Figure.savefig = _savefig


def list_figure_names():
    """Return (default_set, all_names) for the selectable figures."""
    return tuple(_DEFAULT_FIGS), tuple(_FIG_REGISTRY) + tuple(_EXTRA_KPI_FIGS)


def resolve_figures(figures=None, all_figures=False, exclude=None):
    """Resolve a figure selection from a CLI/config value (None/'all'/comma/list)."""
    _all = list(_FIG_REGISTRY) + list(_EXTRA_KPI_FIGS)
    _selectable = set(_all)
    if all_figures or (isinstance(figures, str) and figures.strip().lower() == "all"):
        sel = list(_all)
    elif isinstance(figures, str):
        sel = [s.strip() for s in figures.split(",") if s.strip()]
    elif figures:
        sel = list(figures)
        if tuple(sel) == ("all",):
            sel = list(_all)
    else:
        sel = list(_DEFAULT_FIGS)
    # Category selectors expand to figure names; explicit names pass through.
    if any(isinstance(s, str) and ":" in s for s in sel):
        sel = _expand_selectors(sel)
    unknown = [s for s in sel if s not in _selectable]
    if unknown:
        print(f"WARN: unknown figure name(s) ignored: {unknown} (see list_figure_names)")
    sel = [s for s in sel if s in _selectable]
    if exclude:
        drop = ({s.strip() for s in exclude.split(",") if s.strip()}
                if isinstance(exclude, str) else set(exclude))
        if drop:
            print(f"analysis: excluding {len(drop & set(sel))} figure(s): "
                  f"{sorted(drop & set(sel))}")
        sel = [s for s in sel if s not in drop]
    return sel


def _run_extra_kpi_figures(cells, out, names, step_s, bandwidth_hz, carrier_hz, dpi):
    """Render snapshot-based extra-KPI figures per cell into per-(cnst, strategy) subfolders."""
    from ntnconnect.analysis.extra_kpis import make_extra_kpi_figures
    kw = {}
    if bandwidth_hz is not None:
        kw["bandwidth_hz"] = bandwidth_hz
    if carrier_hz is not None:
        kw["carrier_hz"] = carrier_hz
    made = 0
    for _, row in cells.iterrows():
        rd = Path(row["run_dir"]) if row.get("run_dir") else None
        from ntnconnect.common.utils.run_io import run_shards
        _snaps = run_shards(rd, "sinr_snapshot") if rd else []
        if not _snaps:
            continue
        snap = _snaps if len(_snaps) > 1 else _snaps[0]
        cell_out = out / f"{row['constellation']}__{row['strategy']}"
        try:
            made += len(make_extra_kpi_figures(snap, cell_out, step_s=step_s,
                                               dpi=dpi, figures=list(names), **kw))
        except Exception as e:
            print(f"WARN: extra-KPI figures for "
                  f"{row['constellation']}/{row['strategy']} failed: {e}")
    print(f"analysis: extra-KPI -> {made} figure(s) across "
          f"{len(cells)} cell(s) in per-run subfolders")


def run_analysis(out_dir, *, runs_dir=None, figures=None, all_figures=False,
                 exclude=None, exclude_strategies=None,
                 show_titles=True, dpi=140, fig_format="png",
                 paper_style=True, use_tex=False, base_fontsize=9,
                 write_tables=True, eph_dir=None,
                 label_fn=None, color_map=None, label_overrides=None,
                 bandwidth_hz=None, carrier_hz=None, auto=False, max_ticks=None,
                 build_globe_json=True, globe_max_batches=None):
    """Scenario-independent analysis runner: reads manifest.csv + per-run Parquet,
    computes metrics, emits selected figures + tables, returns the outputs-index dict.

    runs_dir defaults to out_dir; set it when runs and outputs are in separate folders.
    """
    out = Path(out_dir)
    src = Path(runs_dir) if runs_dir is not None else out
    out.mkdir(parents=True, exist_ok=True)
    if eph_dir is not None:
        set_eph_dir(eph_dir)
    if label_fn is not None:
        globals()["short_label"] = label_fn
    if color_map is not None:
        _COLORS.update(color_map)

    _apply_output_style(show_titles, dpi, fig_format, paper_style=paper_style,
                        use_tex=use_tex, base_fontsize=base_fontsize,
                        label_overrides=label_overrides)
    sel = resolve_figures(figures, all_figures, exclude=exclude)

    man_p = src / "manifest.csv"
    if not man_p.exists():
        raise FileNotFoundError(f"manifest not found: {man_p}")
    man = pd.read_csv(man_p)
    cells = man[man["status"] == "ok"].copy()
    if exclude_strategies:
        drop = {str(s).lower() for s in exclude_strategies}
        cells = cells[~cells["strategy"].astype(str).str.lower().isin(drop)].copy()
        print(f"analysis: excluding strategies {sorted(drop)} "
              f"-> {len(cells)} cells remain")
    if cells.empty:
        raise ValueError("no successful cells in manifest")

    step_size_s = _resolve_step_s(src, cells)

    A = _build_aggregates(cells, step_size_s=step_size_s,
                          compute_rtt=("e2e_rtt_ccdf" in sel), max_ticks=max_ticks)
    if auto:
        sel, dropped = _prune_empty(sel, A, cells)
        if dropped:
            print(f"analysis[auto]: dropped {len(dropped)} figure(s) with no data "
                  f"in this experiment: {sorted(dropped)}")
    builtin = [n for n in sel if n in _FIG_REGISTRY]
    extra   = [n for n in sel if n in _EXTRA_KPI_FIGS]
    print(f"analysis: {len(sel)} figures selected "
          f"({len(builtin)} aggregate + {len(extra)} extra-KPI; "
          f"titles={'on' if show_titles else 'off'}, {fig_format}@{dpi}dpi)")
    for name in builtin:
        fn = _FIG_REGISTRY[name]
        try:
            fn(A, out)
        except Exception as e:
            print(f"WARN: figure '{name}' ({fn.__name__}) failed: {e}")
    if extra:
        _run_extra_kpi_figures(cells, out, extra, step_size_s,
                               bandwidth_hz, carrier_hz, dpi)

    if write_tables:
        summary = _build_summary(A)
        summary.to_csv(out / "strategy_summary.csv", index=False)
        print(summary.to_string(index=False))

    idx = {
        "figures": sorted(p.name for p in out.glob(f"fig*.{fig_format}")),
        "extra_kpi_figures": sorted(str(p.relative_to(out)).replace("\\", "/")
                                    for n in extra
                                    for p in out.glob(f"*__*/{n}.png")),
        "tables":  sorted(p.name for p in out.glob("*.csv")),
        "selected_figures": sel,
        "style": {"show_titles": show_titles, "dpi": dpi, "format": fig_format,
                  "paper_style": paper_style, "use_tex": use_tex,
                  "base_fontsize": base_fontsize},
        "meta":    "repro_meta.json" if (out / "repro_meta.json").exists() else None,
        "per_cell_runs": sorted(p.name for p in out.glob("*__*") if p.is_dir()),
    }
    (out / "outputs_index.json").write_text(json.dumps(idx, indent=2))
    print(f"\nWrote {len(idx['figures'])} figures + {len(idx['tables'])} tables to {out}")

    # Emit the globe rail data here; guarded so PA never fails on a globe hiccup.
    if build_globe_json:
        try:
            from ntnconnect.analysis.visualization.build_interactive import build_interactive
            _eph = eph_dir if eph_dir is not None else _find_eph_dir()
            build_interactive([src], _eph, out, max_batches=globe_max_batches)
            idx["interactive"] = "interactive.json"
            (out / "outputs_index.json").write_text(json.dumps(idx, indent=2))
            print(f"PA: wrote {out / 'interactive.json'} (globe rail data)")
        except Exception as e:
            print(f"PA: interactive.json build skipped: {e}")
    return idx


# Elevation-threshold sweep (re-derived from link_visibility; no engine re-run)
def _derive_geometry_at_threshold(lv: pd.DataFrame, theta: float, step_s: float):
    """At mask theta, re-derive n_visible (incl. zeros), complete contact
    durations [s], and per-user coverage-outage gaps [s] from link_visibility."""
    vis = lv[lv["elev_deg"] >= float(theta)]
    ticks = pd.to_numeric(lv["t_idx"], errors="coerce")
    t0, t1 = int(ticks.min()), int(ticks.max())
    n_users = int(lv["user_id"].nunique())
    total_cells = n_users * (t1 - t0 + 1)
    # n_visible per (tick,user), zero-padded for cells with no visible sat
    cnt = vis.groupby(["t_idx", "user_id"]).size().to_numpy().astype(float)
    nvis = np.concatenate([cnt, np.zeros(max(total_cells - len(cnt), 0))])
    # complete contact durations: consecutive-tick runs per (user,sat)
    durs = []
    for _, g in vis.groupby(["user_id", "sat_id"], sort=False):
        ts = np.unique(g["t_idx"].astype(int).to_numpy())
        for run in np.split(ts, np.where(np.diff(ts) > 1)[0] + 1):
            r0, r1 = int(run[0]), int(run[-1])
            if r0 > t0 and r1 < t1:                     # exclude boundary-censored
                durs.append((r1 - r0) * step_s)
    # per-user coverage-outage gaps
    gaps = []
    for _, g in vis.groupby("user_id", sort=False):
        cov = np.unique(g["t_idx"].astype(int).to_numpy())
        if len(cov) >= 2:
            d = np.diff(cov)
            for hole in d[d > 1]:
                gaps.append((int(hole) - 1) * step_s)
    return nvis, np.array(durs, dtype=float), np.array(gaps, dtype=float)


def elevation_sweep(out_dir, *, thresholds=(20, 25, 30, 35), eph_dir=None,
                    strategy=None, show_titles=True, dpi=140, fig_format="png",
                    paper_style=True, use_tex=False, base_fontsize=9,
                    label_fn=None, color_map=None):
    """Sweep the elevation mask over `thresholds` and emit faceted geometry
    figures (n_visible CCDF, contact-time CDF, coverage-outage CDF)."""
    out = Path(out_dir)
    if eph_dir is not None:
        set_eph_dir(eph_dir)
    if label_fn is not None:
        globals()["short_label"] = label_fn
    if color_map is not None:
        _COLORS.update(color_map)
    _apply_output_style(show_titles, dpi, fig_format, paper_style=paper_style,
                        use_tex=use_tex, base_fontsize=base_fontsize)

    man = pd.read_csv(src / "manifest.csv")
    cells = man[man["status"] == "ok"].copy()
    step_s = _resolve_step_s(out, cells)

    # one run per constellation (geometry is strategy-independent)
    cnsts, used = [], {}
    for _, r in cells.iterrows():
        c = r["constellation"]
        if c in used:
            continue
        if strategy and r["strategy"] != strategy:
            continue
        used[c] = Path(r["run_dir"]); cnsts.append(c)
    if not used:  # fall back: first run per constellation regardless of strategy
        for _, r in cells.iterrows():
            c = r["constellation"]
            used.setdefault(c, Path(r["run_dir"]))
        cnsts = list(used)

    # derive metrics[cnst][theta] = (nvis, durs, gaps)
    data = {}
    for c in cnsts:
        lv = _load(used[c], "link_visibility.parquet")
        if lv is None or "elev_deg" not in lv.columns:
            continue
        data[c] = {th: _derive_geometry_at_threshold(lv, th, step_s) for th in thresholds}

    if not data:
        print("elevation_sweep: no link_visibility data found")
        return
    cnsts = list(data)
    th_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(thresholds)))

    def _facet(metric_idx, transform, title, xlabel, ylabel, fname, xlim=None):
        fig, axes = plt.subplots(1, len(cnsts), figsize=(4.2 * len(cnsts), 3.6),
                                  sharey=True, squeeze=False)
        for ax, c in zip(axes[0], cnsts):
            for th, col in zip(thresholds, th_colors):
                arr = data[c][th][metric_idx]
                xy = transform(arr)
                if xy is None:
                    continue
                ax.plot(xy[0], xy[1], color=col, lw=1.6, label=f"{th}°")
            ax.set_title(short_label(c)); ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
            if xlim: ax.set_xlim(*xlim)
        axes[0, 0].set_ylabel(ylabel)
        # Legend from a panel that has every threshold curve (handles may be
        # missing on a constellation whose data empties at a high mask).
        h, l = max((ax.get_legend_handles_labels() for ax in axes[0]),
                   key=lambda hl: len(hl[0]))
        if h:
            axes[0, -1].legend(h, l, title="elev. mask", fontsize=7, loc="best")
        fig.suptitle(title)
        fig.tight_layout(); fig.savefig(out / fname, dpi=dpi); plt.close(fig)

    def _ccdf_k(arr):
        if arr.size == 0: return None
        ks = np.arange(0, int(arr.max()) + 1)
        return ks, np.array([(arr >= k).mean() for k in ks])
    def _cdf_t(arr):
        if arr.size == 0: return None
        return _cdf(arr)

    _facet(0, _ccdf_k, "n_visible vs elevation mask",
           "k (visible sats)", "P[$n_{vis}$ ≥ k]", "fig_elev_visibility_ccdf.png",
           xlim=(0, 25))
    _facet(1, _cdf_t, "Contact-time vs elevation mask",
           "contact duration [s]", "CDF", "fig_elev_contact_cdf.png")
    _facet(2, _cdf_t, "Coverage-outage gap vs elevation mask",
           "no-coverage gap [s]", "CDF", "fig_elev_coverage_gap_cdf.png")

    # quick numeric summary so the Kuiper degradation is visible in text
    print(f"{'cnst':9} {'mask':>5} {'min_nvis':>9} {'med_contact_s':>14} {'n_gaps':>7} {'max_gap_s':>9}")
    for c in cnsts:
        for th in thresholds:
            nv, du, ga = data[c][th]
            print(f"{c:9} {th:5}° {int(nv.min()):9d} "
                  f"{(np.median(du) if du.size else 0):14.0f} {ga.size:7d} "
                  f"{(ga.max() if ga.size else 0):9.0f}")
    print(f"\nWrote 3 elevation-sweep figures to {out}")

