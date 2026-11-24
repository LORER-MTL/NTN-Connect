# SkyNTNPro v1 - Static/fixed node generators
#
# Ground stations and base stations don't move; HAPS are quasi-stationary;
# UAVs follow waypoints. All four use simple time-replication of static
# lat/lon/alt entries declared in the operator block.

from __future__ import annotations
from typing import Any, List
import numpy as np
import pandas as pd

from ntnconnect.scenario.base_generator import BaseNodeGenerator

# Per-tick drift matching stages/mobility.py the mobility stage.
AERIAL_DRIFT_SIGMA_DEG = 0.001


class _StaticPositionGenerator(BaseNodeGenerator):
    """Common implementation for any node type with fixed positions."""

    def generate(self, timestamps: List[Any]) -> pd.DataFrame:
        nodes = self._positions()
        if not nodes:
            return pd.DataFrame()

        id_lo = self._id_range()[0]
        cap = self._capability_cols()

        # phase==1: precompute the per-tick drift trajectory here (Phase-2
        # mobility skips these rows). Otherwise positions are static and
        # Phase-2 drifts them live. Base nodes never drift.
        precompute = (self._mobility_phase() == 1 and self._drift_sigma() > 0.0)
        lat = np.array([float(n.lat) for n in nodes])
        lon = np.array([float(n.lon) for n in nodes])
        rng = np.random.default_rng(
            int(self.cfg.experiment.random_seed) + int(self.op.id))

        rows = []
        for t_idx, ts in enumerate(timestamps):
            for i, n in enumerate(nodes):
                rows.append({
                    "timestamp":   ts,
                    "t_idx":       t_idx,
                    "node_id":     int(id_lo + i),
                    "node_type":   self.node_type,
                    "operator_id": int(self.op.id),
                    "name":        n.name,
                    "lat":         float(lat[i]),
                    "lon":         float(lon[i]),
                    "alt_m":       float(n.alt_m),
                    **cap,
                })
            if precompute:
                lat = lat + rng.normal(0.0, self._drift_sigma(), len(nodes))
                lon = lon + rng.normal(0.0, self._drift_sigma(), len(nodes))
        return pd.DataFrame(rows)

    def _positions(self): raise NotImplementedError
    def _id_range(self):   raise NotImplementedError
    def _mobility_phase(self) -> int: return 2
    def _drift_sigma(self) -> float:  return 0.0


class HAPSGenerator(_StaticPositionGenerator):
    node_type = "haps"
    def _positions(self): return self.op.haps.positions
    def _id_range(self):  return self.op.haps.id_range
    def _mobility_phase(self): return int(self.op.haps.mobility.phase)
    def _drift_sigma(self):
        return AERIAL_DRIFT_SIGMA_DEG if self.op.haps.mobility.model not in ("hover", "dataset", "static") else 0.0


class UAVGenerator(_StaticPositionGenerator):
    node_type = "uav"
    def _positions(self): return self.op.uavs.positions
    def _id_range(self):  return self.op.uavs.id_range
    def _mobility_phase(self): return int(self.op.uavs.mobility.phase)
    def _drift_sigma(self):
        return AERIAL_DRIFT_SIGMA_DEG if self.op.uavs.mobility.model not in ("hover", "dataset", "static") else 0.0


class GroundStationGenerator(_StaticPositionGenerator):
    node_type = "ground_station"
    def _positions(self): return self.op.ground_stations.positions
    def _id_range(self):  return self.op.ground_stations.id_range


class BaseStationGenerator(_StaticPositionGenerator):
    node_type = "base_station"
    def _positions(self): return self.op.base_stations.positions
    def _id_range(self):  return self.op.base_stations.id_range
