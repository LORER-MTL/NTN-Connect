"""Shared coordinate-frame helpers - the single source of truth for WGS-84
geodetic <-> ECEF conversions.

Previously `lla_to_ecef` was copy-pasted into 5 modules (external_interference,
visualization.czml, visualization.dashboard, scenarios.WISEE.analyze,
scenarios.WISEE.build_ephemeris). Import it from here instead.
"""
from __future__ import annotations
import numpy as np

# WGS-84 ellipsoid constants
R_EQ = 6378137.0
F    = 1.0 / 298.257223563
E2   = 2.0 * F - F * F


def lla_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    """Geodetic (lat, lon in degrees, altitude in m) -> ECEF (x, y, z) in m.

    Vectorised; accepts scalars or arrays. Earth-fixed (rotates with Earth).
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h   = np.asarray(alt_m, dtype=float)
    N = R_EQ / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - E2) + h) * np.sin(lat)
    return x, y, z
