# SkyNTNPro v1 - Base node generator
#
# Each node type (satellite, haps, uav, ground_station, base_station)
# subclasses BaseNodeGenerator and emits rows for one partition of
# nodes.parquet/<type>.parquet/operator_id=<k>/.
#
# Phase-1 generators are time-stepped but POLICY-INDEPENDENT.
# Anything that depends on a strategy choice belongs in Phase 2.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import pandas as pd


@dataclass
class NodeRow:
    timestamp:     Any
    t_idx:         int
    node_id:       int
    node_type:     str
    operator_id:   int
    lat:           float
    lon:           float
    alt_m:         float
    vx_mps:        float = 0.0
    vy_mps:        float = 0.0
    vz_mps:        float = 0.0
    # static capability columns (populated for compute-capable types)
    cpu_cores:        int = 0
    cpu_clock_ghz:    float = 0.0
    flops_peak:       float = 0.0
    ram_gb:           float = 0.0
    storage_gb:       float = 0.0
    gpu_present:      bool = False
    power_idle_w:     float = 0.0
    power_per_flop_j: float = 0.0
    compute_role:     str = "none"


class BaseNodeGenerator(ABC):
    """Phase-1 generator for one node type within one operator."""

    node_type: str = "base"

    def __init__(self, cfg, operator):
        self.cfg = cfg
        self.op  = operator
        self.cap = operator.compute.get(self.node_type, None)

    @abstractmethod
    def generate(self, timestamps: List[Any]) -> pd.DataFrame:
        """Return a DataFrame with one row per (node, timestamp).
        Must include the columns listed in NodeRow + at least
        node_id, node_type, operator_id, timestamp, t_idx, lat, lon, alt_m.
        """

    def _capability_cols(self) -> dict:
        if self.cap is None:
            return {"cpu_cores": 0, "cpu_clock_ghz": 0.0, "flops_peak": 0.0,
                    "ram_gb": 0.0, "storage_gb": 0.0, "gpu_present": False,
                    "power_idle_w": 0.0, "power_per_flop_j": 0.0,
                    "compute_role": "none"}
        return {"cpu_cores": self.cap.cpu_cores,
                "cpu_clock_ghz": self.cap.cpu_clock_ghz,
                "flops_peak": self.cap.flops_peak,
                "ram_gb": self.cap.ram_gb,
                "storage_gb": self.cap.storage_gb,
                "gpu_present": self.cap.gpu_present,
                "power_idle_w": self.cap.power_idle_w,
                "power_per_flop_j": self.cap.power_per_flop_j,
                "compute_role": self.cap.compute_role}
