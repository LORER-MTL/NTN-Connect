# Earth-cell grid helper.
#
# important:
#  * The grid is fixed for the entire Monte-Carlo iteration.
#  * No randomness inside this module -- the cell layout depends
#  only on cfg.env.lat_bounds, cfg.env.lon_bounds, and
#  cfg.users.cluster_size_km (+ cfg.users.cluster_grid_method).
#  * Random init of USER positions happens in user_generator.py,
#  not here.
#
# Pure vectorised functions: arrays in -> arrays out.
#
# Methods (cfg.users.cluster_grid_method):
#  regular_lat_lon -- implemented (default)
#  hex -- placeholder for future flat-hex tiling
#  h3 -- placeholder for future H3 icosahedral tiling
from __future__ import annotations
import numpy as np

EARTH_DEG_PER_KM = 1.0 / 111.0          # approximate


def _bins(cfg):
    """Return (lat_step_deg, lon_step_deg, n_lat, n_lon)."""
    cell_km = float(cfg.users.cluster_size_km)
    lat_lo, lat_hi = cfg.env.lat_bounds
    lon_lo, lon_hi = cfg.env.lon_bounds
    lat_step = cell_km * EARTH_DEG_PER_KM
    # use the mid-lat as the reference for lon scaling
    lat_mid  = 0.5 * (lat_lo + lat_hi)
    lon_step = cell_km * EARTH_DEG_PER_KM / max(np.cos(np.radians(lat_mid)), 0.05)
    n_lat = max(1, int(np.ceil((lat_hi - lat_lo) / lat_step)))
    n_lon = max(1, int(np.ceil((lon_hi - lon_lo) / lon_step)))
    return lat_step, lon_step, n_lat, n_lon


def lat_lon_to_cluster_id(lat, lon, cfg) -> np.ndarray:
    """Vectorised lat/lon -> cluster_id (int). Pure / deterministic.
    Dispatches on cfg.users.cluster_grid_method."""
    method = str(getattr(cfg.users, "cluster_grid_method", "regular_lat_lon")).lower()
    if method == "regular_lat_lon":
        return _lat_lon_regular(lat, lon, cfg)
    if method in ("hex", "h3"):
        raise NotImplementedError(
            f"cluster_grid_method='{method}' is reserved for a future "
            f"implementation. Use 'regular_lat_lon' for now.")
    raise ValueError(f"Unknown cluster_grid_method: {method!r}")


def _lat_lon_regular(lat, lon, cfg) -> np.ndarray:
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    lat_lo, _ = cfg.env.lat_bounds
    lon_lo, _ = cfg.env.lon_bounds
    lat_step, lon_step, n_lat, n_lon = _bins(cfg)
    i = np.clip(((lat - lat_lo) / lat_step).astype(int), 0, n_lat - 1)
    j = np.clip(((lon - lon_lo) / lon_step).astype(int), 0, n_lon - 1)
    return (i * n_lon + j).astype(np.int32)


def cluster_centroid(cid: int, cfg) -> tuple[float, float]:
    """Return (lat, lon) at the centre of the cluster cell."""
    lat_lo, _ = cfg.env.lat_bounds
    lon_lo, _ = cfg.env.lon_bounds
    lat_step, lon_step, _, n_lon = _bins(cfg)
    i = int(cid) // n_lon; j = int(cid) % n_lon
    return (lat_lo + (i + 0.5) * lat_step,
            lon_lo + (j + 0.5) * lon_step)


def n_clusters(cfg) -> int:
    _, _, n_lat, n_lon = _bins(cfg)
    return int(n_lat * n_lon)
