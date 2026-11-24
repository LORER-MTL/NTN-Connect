# Gateway runtime object: feeder/backhaul/compute/power state. Static (no mobility).
# Lives in engine.gateways; step() per tick, release() resets per-link state.
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, Set
import logging

_log = logging.getLogger(__name__)


# State groups
@dataclass
class FeederState:
    """Sat <-> gateway feeder link bookkeeping."""
    capacity_bps:        float = 10.0e9          # total bidirectional
    buffer_bytes:        int   = 0
    served_sats:         Set[int] = field(default_factory=set)
    elev_min_deg:        float = 10.0
    util_bps_recent:     float = 0.0


@dataclass
class BackhaulState:
    """Gateway to terrestrial-network bookkeeping + routing table."""
    capacity_bps:        float = 100.0e9
    latency_s:           float = 0.01
    drop_rate:           float = 0.0
    routing_table:       Dict[int, int] = field(default_factory=dict)
    util_bps_recent:     float = 0.0


@dataclass
class GatewayComputeState:
    """Gateway-side MEC compute pool + cache for offload studies."""
    cycles_per_s:        float = float("inf")
    memory_mb:           float = float("inf")
    cache_size_mb:       float = 0.0
    cache_hits:          int   = 0
    cache_misses:        int   = 0
    pending_tasks:       Deque = field(default_factory=deque)
    completed_tasks:     int   = 0
    cum_compute_cycles:  float = 0.0


@dataclass
class GatewayPowerState:
    """Power bookkeeping - gateways are not battery-limited."""
    source:                 str   = "grid"        # "grid" | "solar" | "hybrid"
    consumption_w_current:  float = 0.0
    cumulative_energy_j:    float = 0.0


# The Gateway class
class Gateway:
    """A simulated ground gateway / teleport site.

    Identity:
        gw_id, name, operator_id
        lat, lon, alt_m  (static)

    State groups:
        feeder, backhaul, compute, power

    Engine lifecycle mirrors User / Satellite:
        __init__(...)
        step(t_idx, dt_s, state)
        release_sat(sat_id)            # called when a sat no longer feeds via me
        attach_sat(sat_id)             # called when a sat begins feeding via me
        update_routing(table)          # routing-table updater (called by
                                       # an `agent_route` stage on cadence)
    """

    __slots__ = ("gw_id", "name", "operator_id",
                 "lat", "lon", "alt_m", "source",
                 "feeder", "backhaul", "compute", "power",
                 "_engine")

    def __init__(self, gw_id: int, name: str, operator_id: int,
                 lat: float, lon: float, alt_m: float = 0.0,
                 source: str = "filing",
                 cfg: Any = None,
                 engine: Any = None):
        self.gw_id       = int(gw_id)
        self.name        = str(name)
        self.operator_id = int(operator_id)
        self.lat         = float(lat)
        self.lon         = float(lon)
        self.alt_m       = float(alt_m)
        self.source      = str(source)
        self._engine     = engine
        self.feeder   = self._build_feeder(cfg)
        self.backhaul = self._build_backhaul(cfg)
        self.compute  = self._build_compute(cfg)
        self.power    = self._build_power(cfg)

    @staticmethod
    def _build_feeder(cfg) -> FeederState:
        f = FeederState()
        if cfg is None: return f
        gw = getattr(cfg, "gateways", None) or getattr(cfg, "ground_stations", None)
        fc = getattr(gw, "feeder", None) if gw is not None else None
        if fc is None: return f
        f.capacity_bps = float(getattr(fc, "capacity_bps", f.capacity_bps))
        f.elev_min_deg = float(getattr(fc, "elev_min_deg", f.elev_min_deg))
        return f

    @staticmethod
    def _build_backhaul(cfg) -> BackhaulState:
        b = BackhaulState()
        if cfg is None: return b
        gw = getattr(cfg, "gateways", None) or getattr(cfg, "ground_stations", None)
        bc = getattr(gw, "backhaul", None) if gw is not None else None
        if bc is None: return b
        b.capacity_bps = float(getattr(bc, "capacity_bps", b.capacity_bps))
        b.latency_s    = float(getattr(bc, "latency_s",    b.latency_s))
        b.drop_rate    = float(getattr(bc, "drop_rate",    b.drop_rate))
        return b

    @staticmethod
    def _build_compute(cfg) -> GatewayComputeState:
        c = GatewayComputeState()
        if cfg is None: return c
        gw = getattr(cfg, "gateways", None) or getattr(cfg, "ground_stations", None)
        cc = getattr(gw, "compute", None) if gw is not None else None
        if cc is None: return c
        c.cycles_per_s  = float(getattr(cc, "cycles_per_s",  c.cycles_per_s))
        c.memory_mb     = float(getattr(cc, "memory_mb",     c.memory_mb))
        c.cache_size_mb = float(getattr(cc, "cache_size_mb", c.cache_size_mb))
        return c

    @staticmethod
    def _build_power(cfg) -> GatewayPowerState:
        p = GatewayPowerState()
        if cfg is None: return p
        gw = getattr(cfg, "gateways", None) or getattr(cfg, "ground_stations", None)
        pc = getattr(gw, "power", None) if gw is not None else None
        if pc is None: return p
        p.source = str(getattr(pc, "source", p.source))
        return p

    # Per-tick agent step
    def step(self, t_idx: int, dt_s: float, state: Any = None) -> None:
        # Drain MEC compute queue if any
        if self.compute.cycles_per_s < float("inf") and self.compute.pending_tasks:
            self._drain_compute(t_idx, dt_s)
        # Power bookkeeping accumulates consumption
        if self.power.consumption_w_current > 0:
            self.power.cumulative_energy_j += \
                self.power.consumption_w_current * dt_s

    def _drain_compute(self, t_idx: int, dt_s: float) -> None:
        budget = self.compute.cycles_per_s * dt_s
        while budget > 0 and self.compute.pending_tasks:
            task = self.compute.pending_tasks[0]
            need = float(task.get("size_cycles", 0.0))
            if need <= budget:
                budget -= need
                self.compute.cum_compute_cycles += need
                self.compute.completed_tasks   += 1
                self.compute.pending_tasks.popleft()
            else:
                task["size_cycles"] = need - budget
                self.compute.cum_compute_cycles += budget
                budget = 0.0

    # Hooks used by ISL / routing / link stages
    def attach_sat(self, sat_id: int) -> None:
        self.feeder.served_sats.add(int(sat_id))

    def release_sat(self, sat_id: int) -> None:
        self.feeder.served_sats.discard(int(sat_id))

    def update_routing(self, table: Dict[int, int]) -> None:
        """Replace routing table wholesale. table[dst_node_id] = next_hop_sat_id."""
        self.backhaul.routing_table = dict(table)

    def as_row(self) -> dict:
        return {
            "gw_id":           self.gw_id,
            "name":            self.name,
            "operator_id":     self.operator_id,
            "lat":             self.lat,
            "lon":             self.lon,
            "alt_m":           self.alt_m,
            "served_sats":     len(self.feeder.served_sats),
            "feeder_util_bps": self.feeder.util_bps_recent,
            "backhaul_util_bps": self.backhaul.util_bps_recent,
            "tasks_pending":   len(self.compute.pending_tasks),
            "tasks_completed": self.compute.completed_tasks,
            "power_w":         self.power.consumption_w_current,
            "energy_j":        self.power.cumulative_energy_j,
        }

    def __repr__(self) -> str:
        return (f"Gateway({self.gw_id}, {self.name}, op={self.operator_id}, "
                 f"lat={self.lat:.2f}, lon={self.lon:.2f}, "
                 f"served={len(self.feeder.served_sats)} sats)")
