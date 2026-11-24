"""Convert a Phase-1 ephemeris batch into a CesiumJS CZML document."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# WGS-84
_R_EQ = 6378137.0
_F    = 1.0 / 298.257223563
_E2   = 2 * _F - _F * _F

import base64

def makeDiamondSVG(op_color):
    r, g, b, _ = op_color

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="64"
         height="64"
         viewBox="0 0 64 64">
      <polygon
        points="32,4 60,32 32,60 4,32"
        fill="rgb({r},{g},{b})"
        stroke="white"
        stroke-width="3"/>
    </svg>
    """

    return (
        "data:image/svg+xml;base64," +
        base64.b64encode(svg.encode()).decode()
    )


def _lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h   = np.asarray(alt_m, dtype=float)
    N = _R_EQ / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - _E2) + h) * np.sin(lat)
    return x, y, z


_OP_COLOR_RGBA = {
    "oneweb":   [70, 160, 255, 255],
    "starlink": [255, 160, 60, 255],
    "kuiper":   [70, 220, 130, 255],
}


def _iso(dt: datetime) -> str:
    """ISO-8601 with Z suffix, no microseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_manifest(m: dict) -> dict:
    """Adapt a new-layout manifest to the fields the CZML builders expect."""
    if "fine_step_s" not in m:
        m["fine_step_s"] = float(m.get("step_s", 1.0))
    if "batch_min" not in m:
        if "batch_size_s" in m:
            m["batch_min"] = float(m["batch_size_s"]) / 60.0
        elif "hours" in m:
            m["batch_min"] = float(m["hours"]) * 60.0
        else:
            m["batch_min"] = 15.0
    if "n_batches" not in m:
        try:
            _hours = float(m.get("hours", m.get("duration_h", 0.0)))
            _bs = float(m.get("batch_size_s", 0.0))
            m["n_batches"] = max(1, round(_hours * 3600.0 / _bs)) if _bs > 0 else 1
        except (TypeError, ValueError):
            m["n_batches"] = 1
    if "duration_h" not in m:
        if "hours" in m:
            m["duration_h"] = float(m["hours"])
        elif "batch_size_s" in m:
            m["duration_h"] = float(m["batch_size_s"]) / 3600.0
    return m


def _sat_shards(eph_dir: Path, constellation: str) -> list:
    """Ordered satellite shard paths (batched layout, or single-file legacy)."""
    d = Path(eph_dir) / "satellites"
    shards = sorted(d.glob(f"{constellation}__batch_*.parquet"))
    if shards:
        return shards
    single = d / f"{constellation}.parquet"
    return [single] if single.exists() else []


def _read_sats_streamed(eph_dir: Path, constellation: str, every_n: int,
                        max_sim_tick=None) -> "pd.DataFrame":
    """Read satellite ephemeris one shard at a time, down-sampling as read."""
    parts = []
    for f in _sat_shards(eph_dir, constellation):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if max_sim_tick is not None:
            df = df[df["t_idx"].astype(float) <= max_sim_tick]
        if every_n > 1:
            df = df[df["t_idx"].astype(np.int64) % int(every_n) == 0]
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sat_file_new(eph_dir: Path, constellation: str):
    """True-ish if the new satellite layout exists."""
    return bool(_sat_shards(eph_dir, constellation))


def _gw_df(eph_dir: Path, constellation: str) -> "pd.DataFrame":
    """Gateways/ground stations across batched, single-file and legacy layouts."""
    gs = Path(eph_dir) / "ground_stations"
    shards = sorted(gs.glob(f"{constellation}__batch_*.parquet"))
    if shards:
        return pd.read_parquet(shards[0])          # static -> first shard suffices
    for p in (gs / f"{constellation}.parquet",
              Path(eph_dir) / constellation / "gateways.parquet"):
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()


def _users_df(eph_dir: Path) -> "pd.DataFrame":
    """Users across the batched and single-file layouts."""
    ud = Path(eph_dir) / "users"
    shards = sorted(ud.glob("users__batch_*.parquet"))
    if shards:
        return pd.read_parquet(shards[0])
    single = Path(eph_dir) / "users.parquet"
    return pd.read_parquet(single) if single.exists() else pd.DataFrame()


def build_czml(eph_dir: Path,
               constellation: str,
               batch_idx: int,
               downsample_every_n: int = 10,
               include_users: bool = True,
               include_gateways: bool = True,
               include_orbits: bool = True,
               sinr_snapshot: "pd.DataFrame | None" = None,
               ho_events: "pd.DataFrame | None" = None,
               phase2_step_s: float = 1.0,
               flash_duration_s: float = 10.0,
               all_batches: bool = False) -> list[dict]:
    """CZML packets for a (constellation, batch); all_batches stitches one timeline."""
    eph_dir = Path(eph_dir)
    manifest = _norm_manifest(json.loads((eph_dir / "manifest.json").read_text()))
    fine_step_s = float(manifest["fine_step_s"])
    batch_min   = float(manifest["batch_min"])
    start_dt = datetime.fromisoformat(
        manifest["start_utc"].replace("Z", "+00:00"))
    ticks_per_batch = int(round(batch_min * 60.0 / fine_step_s))
    # Clip ephemeris to the sim's last tick so motion and links share one clock
    max_sim_tick = None
    if sinr_snapshot is not None and not sinr_snapshot.empty \
       and "t_idx" in sinr_snapshot.columns:
        max_sim_tick = float(sinr_snapshot["t_idx"].astype(float).max())
    _shards = _sat_shards(eph_dir, constellation)
    sat_downsample = downsample_every_n
    if _shards:
        # New batched layout: stream and down-sample per shard
        eph = _read_sats_streamed(eph_dir, constellation, downsample_every_n,
                                  max_sim_tick)
        if eph is None or eph.empty:
            return []
        sat_downsample = 1
        batch_start = start_dt
        total_s = (float(eph["t_idx"].max()) + 1.0) * fine_step_s
        batch_end = start_dt + timedelta(seconds=total_s)
    elif all_batches:
        # Old layout: stitch per-batch visible_sats_NN into one timeline
        n_b = int(manifest.get("n_batches", 1))
        if max_sim_tick is not None:
            n_b = min(n_b, int(max_sim_tick // ticks_per_batch) + 1)
        frames = []
        for b in range(n_b):
            bp = eph_dir / constellation / f"visible_sats_{b:02d}.parquet"
            if not bp.exists():
                continue
            dfb = pd.read_parquet(bp)
            if dfb.empty:
                continue
            dfb = dfb.copy()
            dfb["t_idx"] = dfb["t_idx"].astype(float) + b * ticks_per_batch
            frames.append(dfb)
        if not frames:
            return []
        eph = pd.concat(frames, ignore_index=True)
        if max_sim_tick is not None:
            eph = eph[eph["t_idx"] <= max_sim_tick]
        batch_start = start_dt
        total_s = (float(eph["t_idx"].max()) + 1.0) * fine_step_s
        batch_end = start_dt + timedelta(seconds=total_s)
    else:
        batch_start = start_dt + timedelta(minutes=batch_idx * batch_min)
        batch_end   = batch_start + timedelta(minutes=batch_min)
        eph_p = eph_dir / constellation / f"visible_sats_{batch_idx:02d}.parquet"
        if not eph_p.exists():
            return []
        eph = pd.read_parquet(eph_p)
        if eph.empty:
            return []
    users_df = _users_df(eph_dir)
    gws_df   = _gw_df(eph_dir, constellation)

    op_color = _OP_COLOR_RGBA.get(constellation,
                                     [200, 200, 200, 255])

    packets: list[dict] = []

    # Document packet: clock + interval
    packets.append({
        "id":      "document",
        "name":    f"SkyNTN {constellation} batch {batch_idx:02d}",
        "version": "1.0",
        "clock": {
            "interval":    f"{_iso(batch_start)}/{_iso(batch_end)}",
            "currentTime": _iso(batch_start),
            "multiplier":  20,         # 20× real time by default
            "range":       "LOOP_STOP",
            "step":        "SYSTEM_CLOCK_MULTIPLIER",
        },
    })

    avail = f"{_iso(batch_start)}/{_iso(batch_end)}"

    # Satellites
    epoch_str = _iso(batch_start)
    for sid, sub in eph.groupby("node_id", sort=False):
        sub = sub.sort_values("t_idx")
        sub = sub.iloc[::max(1, int(sat_downsample))]
        if len(sub) < 2: continue
        ts  = (sub["t_idx"].to_numpy(dtype=float) * fine_step_s).tolist()
        xs  = sub["x"].to_numpy(dtype=float).tolist()
        ys  = sub["y"].to_numpy(dtype=float).tolist()
        zs  = sub["z"].to_numpy(dtype=float).tolist()
        cart = []
        for t, xi, yi, zi in zip(ts, xs, ys, zs):
            cart.extend([t, xi, yi, zi])
        sat_packet = {
            "id":           f"sat_{int(sid)}",
            "name":         f"{constellation.upper()}-{int(sid)}",
            "availability": avail,

            "billboard": {
                "image": makeDiamondSVG(op_color),
                "scale": .08,
                "verticalOrigin": "CENTER",
                "disableDepthTestDistance": 1e9,
                "scaleByDistance": {
                    "nearFarScalar": [1.5e6, 1.8, 4.0e7, 0.5]
                },
            },

            "position": {
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree":    5,
                "referenceFrame":         "FIXED",
                "epoch":                  epoch_str,
                "cartesian":              cart,
            },
        }            
           
            

        if include_orbits:
            sat_packet["path"] = {
                "leadTime":  150.0,
                "trailTime": 350.0,
                "material": {
                    "polylineGlow": {
                        "color": {"rgba": op_color[:3] + [160]},
                        "glowPower": 0.12,
                        "taperPower": 0.5,
                    }
                },
                "width": 2.0,
                "resolution": 20.0,
            }
        packets.append(sat_packet)

    # Users (static ground points)
    if include_users and not users_df.empty:
        ux, uy, uz = _lla_to_ecef(users_df["lat"].to_numpy(),
                                     users_df["lon"].to_numpy(),
                                     np.zeros(len(users_df)))
        for u, x, y, z in zip(users_df["user_id"], ux, uy, uz):
            packets.append({
                "id":   f"user_{int(u)}",
                "name": f"User {int(u)}",
                "availability": avail,
                "position": {"cartesian": [float(x), float(y), float(z)]},
                "point":    {"pixelSize": 3,
                              "color": {"rgba": [255, 70, 110, 240]},
                              "outlineColor": {"rgba": [255, 220, 230, 220]},
                              "outlineWidth": 1.5,
                              "scaleByDistance": {"nearFarScalar":
                                  [1.5e6, 1.6, 4.0e7, 0.6]}},
            })

    # Gateways (static, labeled)
    if include_gateways and not gws_df.empty:
        gx, gy, gz = _lla_to_ecef(gws_df["lat"].to_numpy(),
                                     gws_df["lon"].to_numpy(),
                                     gws_df["alt_m"].to_numpy())
        for n, x, y, z in zip(gws_df["name"], gx, gy, gz):
            packets.append({
                "id":   f"gw_{n}",
                "name": str(n),
                "availability": avail,
                "position": {"cartesian": [float(x), float(y), float(z)]},
                "point":    {"pixelSize": 10,
                              "color": {"rgba": [0, 230, 120, 240]},
                              "outlineColor": {"rgba": [255, 255, 255, 230]},
                              "outlineWidth": 1.5},
                "label":    {"text": str(n),
                              "font": "12px Helvetica",
                              "pixelOffset": {"cartesian2": [0, -16]},
                              "fillColor": {"rgba": [255, 255, 255, 240]},
                              "scale": 0.8,
                              "showBackground": True,
                              "backgroundColor": {"rgba": [0, 0, 0, 140]}},
            })

    # Served-link polylines: one per consecutive-tick run of a served (user, sat) pair
    if sinr_snapshot is not None and not sinr_snapshot.empty:
        try:
            need = {"t_idx", "user_id", "sat_id"}
            if need.issubset(sinr_snapshot.columns):
                if all_batches:
                    t_start_s = 0.0
                    t_end_s   = (batch_end - batch_start).total_seconds()
                else:
                    t_start_s = batch_idx * batch_min * 60.0
                    t_end_s   = t_start_s + batch_min * 60.0
                ts_sec = sinr_snapshot["t_idx"].to_numpy(
                            dtype=float) * float(phase2_step_s)
                in_batch = (ts_sec >= t_start_s) & (ts_sec < t_end_s)
                sin_b = sinr_snapshot.loc[in_batch].copy()
                if not sin_b.empty:
                    sin_b = sin_b.sort_values(["user_id", "sat_id", "t_idx"])
                    grp = sin_b.groupby(["user_id", "sat_id"])
                    pair_idx = 0
                    for (uid, sid), g in grp:
                        ticks = g["t_idx"].to_numpy()
                        if ticks.size == 0: continue
                        # Gaps > 1 tick start a new run
                        gaps = np.where(np.diff(ticks) > 1)[0]
                        run_starts = np.concatenate([[ticks[0]],
                                                       ticks[gaps + 1]])
                        run_ends   = np.concatenate([ticks[gaps],
                                                       [ticks[-1]]])
                        for ts_tick, te_tick in zip(run_starts, run_ends):
                            t_s = float(ts_tick) * float(phase2_step_s)
                            t_e = (float(te_tick) + 1) * float(phase2_step_s)
                            dt_s = start_dt + timedelta(seconds=t_s)
                            dt_e = start_dt + timedelta(seconds=t_e)
                            packets.append({
                                "id":            f"link_{int(uid)}_{int(sid)}_{pair_idx}",
                                "availability":  f"{_iso(dt_s)}/{_iso(dt_e)}",
                                "polyline": {
                                    "positions": {
                                        "references": [
                                            f"user_{int(uid)}#position",
                                            f"sat_{int(sid)}#position"]
                                    },
                                    "material": {"polylineGlow": {
                                        "color": {"rgba": [90, 255, 170, 230]},
                                        "glowPower": 0.22,
                                        "taperPower": 0.45,
                                    }},
                                    "width":   5.0,
                                    "arcType": "NONE",
                                },
                            })
                            pair_idx += 1
        except Exception as e:
            _log.warning("served-link CZML build skipped: %s", e)

    # Handover flashes: a short-lived halo Point on the destination sat
    if ho_events is not None and not ho_events.empty:
        try:
            need = {"t_idx", "to_sat_id"}
            if need.issubset(ho_events.columns):
                hev = ho_events
                if "is_handover" in hev.columns:
                    hev = hev[hev["is_handover"].astype(bool)]
                hev = hev[hev["to_sat_id"].astype(int) >= 0]
                if all_batches:
                    t_start_s = 0.0
                    t_end_s   = (batch_end - batch_start).total_seconds()
                else:
                    t_start_s = batch_idx * batch_min * 60.0
                    t_end_s   = t_start_s + batch_min * 60.0
                ts_sec = hev["t_idx"].to_numpy(dtype=float) \
                            * float(phase2_step_s)
                in_batch = (ts_sec >= t_start_s) & (ts_sec < t_end_s)
                hev = hev.loc[in_batch]
                fd = float(flash_duration_s)
                for i, r in enumerate(hev.itertuples(index=False)):
                    t_s  = float(r.t_idx) * float(phase2_step_s)
                    dt_s = start_dt + timedelta(seconds=t_s)
                    dt_e = dt_s + timedelta(seconds=fd)
                    packets.append({
                        "id":           f"hoflash_{int(r.t_idx)}_{i}",
                        "availability": f"{_iso(dt_s)}/{_iso(dt_e)}",
                        "position":     {"reference":
                                          f"sat_{int(r.to_sat_id)}#position"},
                        "point": {
                            "pixelSize": {
                                "epoch": _iso(dt_s),
                                "number": [0.0, 12.0, fd*0.25, 42.0,
                                            fd*0.6, 22.0, fd, 6.0],
                            },
                            "color": {
                                "epoch": _iso(dt_s),
                                "rgba": [0.0, 255, 245, 90, 255,
                                          fd, 255, 170, 0, 0],
                            },
                            "outlineColor": {"rgba": [255, 120, 0, 230]},
                            "outlineWidth": 2.5,
                        },
                    })
        except Exception as e:
            _log.warning("handover-flash CZML build skipped: %s", e)

    return packets


def build_tick_stats(eph_dir: Path, constellation: str, batch_idx: int,
                      sinr_snapshot, ho_events, phase2_step_s: float = 1.0,
                      all_batches: bool = False):
    """Per-tick statistics for the live Cesium HUD, keyed by seconds since batch start."""
    import json as _json
    manifest = _norm_manifest(_json.loads((Path(eph_dir) / "manifest.json").read_text()))
    batch_min = float(manifest["batch_min"])
    if all_batches:
        t0 = 0.0
        t1 = float(manifest.get("n_batches", 1)) * batch_min * 60.0
    else:
        t0 = batch_idx * batch_min * 60.0
        t1 = t0 + batch_min * 60.0
    rows = []
    if sinr_snapshot is not None and not sinr_snapshot.empty \
       and "t_idx" in sinr_snapshot.columns:
        s = sinr_snapshot.copy()
        s["t_s"] = s["t_idx"].astype(float) * float(phase2_step_s)
        s = s[(s["t_s"] >= t0) & (s["t_s"] < t1)]
        cum = 0
        ho_by_t = {}
        if ho_events is not None and not ho_events.empty \
           and "t_idx" in ho_events.columns:
            h = ho_events
            if "is_handover" in h.columns:
                h = h[h["is_handover"].astype(bool)]
            h = h.copy(); h["t_s"] = h["t_idx"].astype(float) * float(phase2_step_s)
            h = h[(h["t_s"] >= t0) & (h["t_s"] < t1)]
            ho_by_t = h.groupby("t_s").size().to_dict()
        def _agg(gg, col, how):
            if col not in gg.columns:
                return None
            v = gg[col].astype(float)
            v = v[np.isfinite(v)]
            if not len(v):
                return None
            return round(float(v.sum() if how == "sum" else v.mean()), 2)
        for t_s, g in s.groupby("t_s"):
            cum += int(ho_by_t.get(t_s, 0))
            sd = g["sinr_db"].astype(float)
            rows.append({
                "t_s":   float(t_s - t0),   # seconds since batch start
                "n":     int(len(g)),
                "med":   round(float(sd.median()), 2),
                "p5":    round(float(np.percentile(sd, 5)), 2),
                "cho":   int(cum),
                "pwr":   _agg(g, "tx_power_dbw", "mean"),          # mean Tx power [dBW]
                "thr":   _agg(g, "throughput_mbps", "sum"),        # total throughput (load) [Mbps]
                "se":    _agg(g, "spectral_efficiency_bps_hz", "mean"),  # mean SE [b/s/Hz]
            })
    return rows


def build_czml_json(eph_dir: Path, constellation: str,
                     batch_idx: int, **kwargs) -> str:
    return json.dumps(build_czml(eph_dir, constellation, batch_idx,
                                   **kwargs), separators=(",", ":"))


# Full-constellation overview layer (points-only CZML from all_sats_NN.parquet)
def build_all_sats_czml(eph_dir: Path, constellation: str, batch_idx: int,
                        downsample_every_n: int = 1,
                        all_batches: bool = False) -> list[dict]:
    eph_dir = Path(eph_dir)
    manifest = _norm_manifest(json.loads((eph_dir / "manifest.json").read_text()))
    batch_min = float(manifest["batch_min"])
    step_s = float(manifest.get("all_sats_step_s") or manifest.get("step_s") or 60.0)
    start_dt = datetime.fromisoformat(manifest["start_utc"].replace("Z", "+00:00"))

    _shards = _sat_shards(eph_dir, constellation)
    swarm_downsample = downsample_every_n
    if _shards:
        # New batched layout: stream and down-sample per shard
        eph = _read_sats_streamed(eph_dir, constellation,
                                  max(1, int(downsample_every_n)))
        swarm_downsample = 1
        win_start = start_dt
        total_s = (float(eph["t_idx"].max()) + 1.0) * step_s if not eph.empty else 0.0
        win_end = start_dt + timedelta(seconds=total_s)
        name_sfx = "all"
    elif all_batches:
        # Whole-window swarm: monolithic all_sats.parquet, else stitch per-batch files
        win_start = start_dt
        win_end   = start_dt + timedelta(hours=float(manifest.get("duration_h", 0.25)))
        name_sfx  = "all"
        mono = eph_dir / constellation / "all_sats.parquet"
        if mono.exists():
            eph = pd.read_parquet(mono)
        else:
            n_batches = int(manifest.get("n_batches", 1))
            ticks_per_batch = int(round(batch_min * 60.0 / step_s))
            parts = []
            for b in range(n_batches):
                bp = eph_dir / constellation / f"all_sats_{b:02d}.parquet"
                if bp.exists():
                    d = pd.read_parquet(bp)
                    if not d.empty:
                        d = d.copy()
                        d["t_idx"] = d["t_idx"].astype(float) + b * ticks_per_batch
                        parts.append(d)
            eph = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        p = eph_dir / constellation / f"all_sats_{batch_idx:02d}.parquet"
        win_start = start_dt + timedelta(minutes=batch_idx * batch_min)
        win_end   = win_start + timedelta(minutes=batch_min)
        name_sfx  = f"{batch_idx:02d}"
        eph = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    if eph is None or eph.empty:
        return []
    # Cap the overview swarm to an evenly-strided sample (a browser can't animate 10k+)
    _MAX_OVERVIEW_SATS = 600
    _ids = list(dict.fromkeys(eph["node_id"].tolist()))
    if len(_ids) > _MAX_OVERVIEW_SATS:
        _stride = max(1, len(_ids) // _MAX_OVERVIEW_SATS)
        _keep = set(_ids[::_stride][:_MAX_OVERVIEW_SATS])
        eph = eph[eph["node_id"].isin(_keep)]
    col = _OP_COLOR_RGBA.get(constellation, [200, 200, 200, 255])
    dim = [col[0], col[1], col[2], 200]
    avail = f"{_iso(win_start)}/{_iso(win_end)}"
    epoch_str = _iso(win_start)
    packets = [{
        "id": "document", "name": f"{constellation} all-sats {name_sfx}",
        "version": "1.0",
        "clock": {"interval": avail, "currentTime": _iso(win_start),
                  "multiplier": 20, "range": "LOOP_STOP",
                  "step": "SYSTEM_CLOCK_MULTIPLIER"},
    }]
    for sid, sub in eph.groupby("node_id", sort=False):
        sub = sub.sort_values("t_idx").iloc[::max(1, int(swarm_downsample))]
        if len(sub) < 2:
            continue
        ts = (sub["t_idx"].to_numpy(dtype=float) * step_s)
        xs = sub["x"].to_numpy(dtype=float); ys = sub["y"].to_numpy(dtype=float)
        zs = sub["z"].to_numpy(dtype=float)
        cart = []
        for t, xi, yi, zi in zip(ts, xs, ys, zs):
            cart.extend([float(t), float(xi), float(yi), float(zi)])
        packets.append({
            "id": f"allsat_{int(sid)}",
            "availability": avail,
            "billboard": {
            "image": makeDiamondSVG(dim),
            "scale": 0.08,
            "verticalOrigin": "CENTER",
            # 0 = depth test on, so far-side sats are occluded by the globe
            "disableDepthTestDistance": 0.0,
            "scaleByDistance": {
                "nearFarScalar": [1.5e6, 1.6, 6.0e7, 0.5]
            }
        },
            "position": {"interpolationAlgorithm": "LAGRANGE",
                         "interpolationDegree": 5, "referenceFrame": "FIXED",
                         "epoch": epoch_str, "cartesian": cart},
        })
    return packets


def build_all_sats_czml_json(eph_dir: Path, constellation: str,
                             batch_idx: int, **kwargs) -> str:
    return json.dumps(build_all_sats_czml(eph_dir, constellation, batch_idx,
                                          **kwargs), separators=(",", ":"))
