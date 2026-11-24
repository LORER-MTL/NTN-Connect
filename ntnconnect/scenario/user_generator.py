# Phase-1 user generator.
#
# Produces a per-tick `users.parquet` with one row per (user_id, timestamp).
# Sampling methods supported:
#  - random : uniform within env.lat_bounds / lon_bounds
#  - matern : iid random + min_dist_km hard-core rejection
#  - fibonacci : Fibonacci spiral inside the bounding box
# Mobility hooks live in stages/mobility.py (mobility) - this
# generator only emits the INITIAL positions, replicated per timestep so
# downstream readers see a populated users_t even when mobility is static.

from __future__ import annotations
from datetime import datetime
from typing import Any, List
import logging
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _sample_positions(cfg, rng: np.random.Generator) -> pd.DataFrame:
    n = int(cfg.users.num_seed_users)
    if n <= 0: return pd.DataFrame()

    lat_lo, lat_hi = cfg.env.lat_bounds
    lon_lo, lon_hi = cfg.env.lon_bounds
    method = (cfg.users.grid_method or "random").lower()

    if method == "hotspot":
        # N users placed UNIFORMLY-IN-AREA within hotspot_radius_km of
        # (hotspot_lat, hotspot_lon) - a tight circular cluster. Ported BYTE-FOR-BYTE
        # from the WISEE make_hotspot_users: the draws MUST be interleaved scalar
        # (r_i, th_i) per user - a batched rng.random(n) consumes the stream in a
        # different order and yields different positions for the same seed.
        lat0 = float(getattr(cfg.users, "hotspot_lat", 0.5 * (lat_lo + lat_hi)))
        lon0 = float(getattr(cfg.users, "hotspot_lon", 0.5 * (lon_lo + lon_hi)))
        r_km = float(getattr(cfg.users, "hotspot_radius_km", 100.0))
        lats = np.empty(n); lons = np.empty(n)
        for i in range(n):
            r = r_km * np.sqrt(rng.random())              # uniform in disk
            th = 2.0 * np.pi * rng.random()
            lats[i] = lat0 + (r * np.cos(th)) / 111.0
            lons[i] = lon0 + (r * np.sin(th)) / (111.0 * max(np.cos(np.radians(lat0)), 0.05))
    elif method == "fibonacci":
        # Phyllotaxis spiral in 2D, scaled to bbox.
        i = np.arange(n) + 0.5
        phi = (1 + 5**0.5) / 2
        x = (i / n)                        # 0..1
        y = (i * phi) % 1.0
        lats = lat_lo + x * (lat_hi - lat_lo)
        lons = lon_lo + y * (lon_hi - lon_lo)
    elif method == "matern":
        # iid + hard-core min_dist_km rejection
        min_d_deg = cfg.users.min_dist_km / 111.0   # rough deg/km
        out_lat: list = []; out_lon: list = []
        tries = 0
        while len(out_lat) < n and tries < n * 200:
            la = rng.uniform(lat_lo, lat_hi)
            lo = rng.uniform(lon_lo, lon_hi)
            if all(((la - la2)**2 + (lo - lo2)**2) ** 0.5 >= min_d_deg
                   for la2, lo2 in zip(out_lat, out_lon)):
                out_lat.append(la); out_lon.append(lo)
            tries += 1
        if len(out_lat) < n:
            _log.warning("Matern sampling could not place all %d users (got %d)",
                         n, len(out_lat))
        lats = np.array(out_lat); lons = np.array(out_lon)
    else:
        # random uniform
        lats = rng.uniform(lat_lo, lat_hi, n)
        lons = rng.uniform(lon_lo, lon_hi, n)

    user_ids = np.arange(1, len(lats) + 1, dtype=int)
    return pd.DataFrame({
        "user_id": user_ids,
        "lat":     lats.astype(float),
        "lon":     lons.astype(float),
        "alt_m":   np.zeros(len(lats), dtype=float),
    })


def generate_users(cfg, timestamps: List[Any]) -> pd.DataFrame:
    """Phase-1 driver entry point.

    Returns a DataFrame with one row per (user, timestamp). Initial positions
    are sampled once with a deterministic RNG seeded from experiment.random_seed.

    Adds a `cluster_id` column at t=0 (Earth-cell index from geometry/grid.py).
    The `clustering` stage recomputes it every subsequent tick as users
    move under the mobility model.
    """
    rng = np.random.default_rng(int(cfg.experiment.random_seed))
    pos = _sample_positions(cfg, rng)
    if pos.empty:
        return pd.DataFrame()

    # Initial cluster_id from (lat, lon) using the simulator's grid module.
    # We do the import here to avoid a hard dependency at module load time.
    try:
        from ntnconnect.scenario.grid import lat_lon_to_cluster_id
        init_cid = lat_lon_to_cluster_id(pos["lat"].to_numpy(),
                                         pos["lon"].to_numpy(), cfg)
    except Exception:
        init_cid = np.full(len(pos), -1, dtype=np.int32)
    pos["cluster_id"]      = init_cid.astype(np.int32)
    pos["prev_cluster_id"] = np.full(len(pos), -1, dtype=np.int32)
    pos["cluster_changed"] = np.zeros(len(pos), dtype=bool)

    # Trajectory across ticks. phase==1 advances the mobility model forward so
    # users.parquet holds the real per-tick track (Phase-2 mobility is skipped);
    # otherwise the seed positions are replicated and Phase-2 mobility moves
    # them live.
    pref = (int(cfg.users.preferences.preferred_operator_id)
            if cfg.users.preferences.preferred_operator_id is not None else -1)
    mob = cfg.users.mobility
    precompute = (int(getattr(mob, "phase", 2)) == 1 and mob.model not in ("static", "dataset"))
    lat = pos["lat"].to_numpy(dtype=float); lon = pos["lon"].to_numpy(dtype=float)
    uids = pos["user_id"].to_numpy(dtype=int)
    dt = float(cfg.env.step_size_s)
    lat_b = list(cfg.env.lat_bounds); lon_b = list(cfg.env.lon_bounds)
    waypoint: dict = {}; velocity: dict = {}
    if precompute:
        from ntnconnect.scenario.mobility_model import step_users, user_params
        params = user_params(mob)
    frames = []
    for t_idx, ts in enumerate(timestamps):
        df = pos.copy()
        df["lat"] = lat; df["lon"] = lon
        df["timestamp"] = ts
        df["t_idx"] = t_idx
        df["mobility_type"] = mob.model
        df["preferred_operator_id"] = pref
        frames.append(df)
        if precompute:
            lat, lon = step_users(lat, lon, uids, mob.model, params, dt,
                                  lat_b, lon_b, waypoint, velocity, rng)
    return pd.concat(frames, ignore_index=True)
