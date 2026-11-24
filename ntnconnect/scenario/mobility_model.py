# Shared mobility stepper used by both the Phase-2 mobility stage (live)
# and the Phase-1 trajectory precompute. Keeping one implementation ensures
# precomputed and live paths produce identical motion.
from __future__ import annotations
import numpy as np

_DEG_PER_M = 1.0 / 111_000.0


def step_users(lat, lon, uids, mtype, p, dt, lat_b, lon_b,
               waypoint: dict, velocity: dict, rng):
    """Advance user lat/lon one step. Updates the waypoint/velocity caches in
    place. `rng` is any object exposing uniform()/normal() (np.random module
    for the live path, a seeded Generator for precompute)."""
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    n = len(uids)
    if mtype == "random_waypoint":
        wps = np.array([waypoint.get(int(u), (np.nan, np.nan)) for u in uids],
                       dtype=float)
        need_wp = (np.isnan(wps[:, 0])
                   | ((np.abs(lat - wps[:, 0]) < 0.01)
                      & (np.abs(lon - wps[:, 1]) < 0.01)))
        if need_wp.any():
            k = int(need_wp.sum())
            new_lats = rng.uniform(*lat_b, size=k)
            new_lons = rng.uniform(*lon_b, size=k)
            wps[need_wp, 0] = new_lats
            wps[need_wp, 1] = new_lons
            for u, wl, wn in zip(np.asarray(uids)[need_wp], new_lats, new_lons):
                waypoint[int(u)] = (float(wl), float(wn))
        v       = rng.uniform(p["vmin"], p["vmax"], size=n)
        d_per_s = v * _DEG_PER_M
        dlat    = np.sign(wps[:, 0] - lat) * d_per_s * dt
        cos_lat = np.maximum(np.cos(np.radians(lat)), 0.01)
        dlon    = np.sign(wps[:, 1] - lon) * d_per_s * dt / cos_lat
    elif mtype == "gauss_markov":
        vprev = np.array([velocity.get(int(u), (0.0, 0.0)) for u in uids],
                         dtype=float)
        noise = rng.normal(0.0, p["sigma"], size=(n, 2))
        vnew  = p["alpha"] * vprev + (1 - p["alpha"]) * noise
        for u, (a, b) in zip(uids, vnew):
            velocity[int(u)] = (float(a), float(b))
        dlat = vnew[:, 0] * dt * _DEG_PER_M
        cos_lat = np.maximum(111_000.0 * np.cos(np.radians(lat)), 1.0)
        dlon = vnew[:, 1] * dt / cos_lat
    else:
        return lat, lon
    new_lat = np.clip(lat + dlat, lat_b[0], lat_b[1])
    new_lon = np.clip(lon + dlon, lon_b[0], lon_b[1])
    return new_lat, new_lon


def user_params(mob) -> dict:
    return {"vmax": float(mob.speed_max_mps), "vmin": float(mob.speed_min_mps),
            "alpha": float(mob.alpha), "sigma": float(mob.sigma)}
