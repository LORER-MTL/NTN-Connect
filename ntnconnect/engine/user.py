# User runtime object: mobility/service/power/compute state. In engine.users;
# step() per tick (no-op default), release() on handover.
#
# The DataFrame view of users in state.data["users_t"] is preserved for
# vectorised channel / SINR code; the User class is the *agent* surface
# used by behaviour-rich logic (mobility, battery, compute, QoS).
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Optional
import logging
import math

_log = logging.getLogger(__name__)


# State groups (small dataclasses, one per concern)
@dataclass
class MobilityState:
    """Per-user mobility state. model='static' means no updates per tick.
    All position updates happen in-place; the class trusts whatever wrote
    them last (the agent_step stage, an external scenario hook, etc.)."""
    model:        str   = "static"      # "static" | "waypoint" | "pedestrian" | "vehicle"
    speed_mps:    float = 0.0
    bearing_deg:  float = 0.0           # ENU bearing for parametric models
    waypoints:    list  = field(default_factory=list)
    waypoint_idx: int   = 0


@dataclass
class ServiceState:
    """Per-user connection bookkeeping. Cross-tick state used by handover
    KPI accumulators, the HO trigger inference, and the per-user contact
    log. Resets on simulation start."""
    qos_class:           str   = "embb"  # "embb" | "urllc" | "mmtc"
    requested_rate_bps:  float = 1.0e6
    served_sat_id:       int   = -1
    last_ho_tick:        int   = -1
    contact_open_tick:   int   = -1
    sinr_db_recent:      float = float("nan")
    rate_bps_recent:     float = 0.0


@dataclass
class PowerState:
    """Per-user battery model. capacity=inf => no constraint (default).
    `consume(dt_s, tx_power_w)` deducts (idle + tx_duty * tx_power_w)·dt.
    When `remaining_j` hits 0 the user enters power_save (tx_power_w=0)."""
    capacity_j:         float = float("inf")
    remaining_j:        float = float("inf")
    idle_power_w:       float = 0.0
    tx_duty:            float = 0.0       # current Tx duty cycle [0, 1]
    cumulative_energy_j:float = 0.0
    power_save:         bool  = False


@dataclass
class ComputeState:
    """Per-user compute pool. cycles_per_s=inf => no constraint (default).
    Tasks are dicts {arrival_t, size_cycles, deadline_t}; pending_tasks
    is a FIFO queue and `step()` drains it at cycles_per_s × dt_s."""
    cycles_per_s:        float = float("inf")
    memory_mb:           float = float("inf")
    pending_tasks:       Deque = field(default_factory=deque)
    completed_tasks:     int   = 0
    dropped_tasks:       int   = 0
    cum_compute_cycles:  float = 0.0


# The User class
class User:
    """A simulated user terminal - handheld direct-to-device by default.

    Identity:
        user_id      : int
        lat, lon, alt_m : current geodetic position

    State groups (instantiated lazily; defaults are zero-cost no-ops):
        mobility, service, power, compute

    Engine lifecycle:
        __init__(user_id, lat, lon, alt_m=0.0, cfg=None)
            built once at engine.setup() from users.parquet
        step(t_idx, dt_s, state)
            called once per tick; default does nothing
        release()
            called by the handover stage when this user moves off its
            current serving sat; resets per-link bookkeeping
        consume(dt_s, tx_power_w, rate_bps)
            updates the power and compute states given a tick's outcome
    """

    __slots__ = ("user_id", "lat", "lon", "alt_m",
                 "mobility", "service", "power", "compute",
                 "_engine")

    def __init__(self, user_id: int,
                 lat: float, lon: float, alt_m: float = 0.0,
                 cfg: Any = None,
                 engine: Any = None):
        self.user_id = int(user_id)
        self.lat    = float(lat)
        self.lon    = float(lon)
        self.alt_m  = float(alt_m)
        self._engine = engine
        # State groups with defaults taken from cfg if provided
        self.mobility = self._build_mobility(cfg)
        self.service  = self._build_service(cfg)
        self.power    = self._build_power(cfg)
        self.compute  = self._build_compute(cfg)

    @staticmethod
    def _build_mobility(cfg) -> MobilityState:
        m = MobilityState()
        if cfg is None: return m
        mc = getattr(getattr(cfg, "users", cfg), "mobility", None)
        if mc is None: return m
        m.model       = str(getattr(mc, "model", m.model))
        m.speed_mps   = float(getattr(mc, "speed_mps",   m.speed_mps))
        m.bearing_deg = float(getattr(mc, "bearing_deg", m.bearing_deg))
        return m

    @staticmethod
    def _build_service(cfg) -> ServiceState:
        s = ServiceState()
        if cfg is None: return s
        sc = getattr(getattr(cfg, "users", cfg), "service", None)
        if sc is None: return s
        s.qos_class          = str(getattr(sc, "qos_class", s.qos_class))
        s.requested_rate_bps = float(getattr(sc, "requested_rate_bps",
                                                s.requested_rate_bps))
        return s

    @staticmethod
    def _build_power(cfg) -> PowerState:
        p = PowerState()
        if cfg is None: return p
        pc = getattr(getattr(cfg, "users", cfg), "power", None)
        if pc is None: return p
        cap = float(getattr(pc, "battery_capacity_j", p.capacity_j))
        p.capacity_j   = cap
        p.remaining_j  = cap
        p.idle_power_w = float(getattr(pc, "idle_power_w", p.idle_power_w))
        return p

    @staticmethod
    def _build_compute(cfg) -> ComputeState:
        c = ComputeState()
        if cfg is None: return c
        cc = getattr(getattr(cfg, "users", cfg), "compute", None)
        if cc is None: return c
        c.cycles_per_s = float(getattr(cc, "cycles_per_s", c.cycles_per_s))
        c.memory_mb    = float(getattr(cc, "memory_mb",    c.memory_mb))
        return c

    # Per-tick agent step. The base class advances mobility and drains
    # the compute queue if non-default models are configured. Specific
    # mobility / compute models are registered separately and dispatched
    # by name to keep this file dependency-free.
    def step(self, t_idx: int, dt_s: float, state: Any = None) -> None:
        # Mobility - only the parametric models incur work
        if self.mobility.model != "static" and self.mobility.speed_mps > 0:
            self._advance_position(dt_s)
        # Compute drain
        if self.compute.cycles_per_s < float("inf") and self.compute.pending_tasks:
            self._drain_compute(t_idx, dt_s)

    def _advance_position(self, dt_s: float) -> None:
        # Simple ENU bearing-based advance; replaced by a richer model
        # via the mobility-models registry in a follow-up. Negligible
        # cost vs the channel pipeline at our scale.
        dist_m = self.mobility.speed_mps * dt_s
        if dist_m < 1e-6: return
        dlat = (dist_m * math.cos(math.radians(self.mobility.bearing_deg))) / 111_111.0
        cos_phi = max(math.cos(math.radians(self.lat)), 1e-3)
        dlon = (dist_m * math.sin(math.radians(self.mobility.bearing_deg))) \
                 / (111_111.0 * cos_phi)
        self.lat += dlat
        self.lon += dlon

    def _drain_compute(self, t_idx: int, dt_s: float) -> None:
        budget_cycles = self.compute.cycles_per_s * dt_s
        while budget_cycles > 0 and self.compute.pending_tasks:
            task = self.compute.pending_tasks[0]
            need = float(task.get("size_cycles", 0.0))
            if need <= budget_cycles:
                budget_cycles -= need
                self.compute.cum_compute_cycles += need
                self.compute.completed_tasks   += 1
                self.compute.pending_tasks.popleft()
            else:
                task["size_cycles"] = need - budget_cycles
                self.compute.cum_compute_cycles += budget_cycles
                budget_cycles = 0.0

    # Hooks used by the handover and radio stages
    def assign(self, sat_id: int, t_idx: int) -> None:
        """Called by the handover stage when this user is assigned to a sat."""
        if self.service.served_sat_id == int(sat_id): return
        prev = self.service.served_sat_id
        self.service.served_sat_id = int(sat_id)
        self.service.last_ho_tick  = int(t_idx)
        if prev < 0 and sat_id >= 0:
            self.service.contact_open_tick = int(t_idx)

    def release(self, t_idx: int = -1) -> None:
        """Drop the user's per-link bookkeeping (HO cold-start helper)."""
        self.service.served_sat_id = -1
        if t_idx >= 0:
            self.service.last_ho_tick = int(t_idx)

    def consume(self, dt_s: float, tx_power_w: float, rate_bps: float) -> None:
        """Updates power state after a successful tick. Idempotent for
        scenarios with infinite battery (default)."""
        if self.power.capacity_j == float("inf"): return
        p = self.power
        p.tx_duty = 1.0 if tx_power_w > 0 else 0.0
        energy = (p.idle_power_w + float(tx_power_w)) * float(dt_s)
        p.cumulative_energy_j += energy
        p.remaining_j = max(0.0, p.remaining_j - energy)
        if p.remaining_j <= 0.0 and not p.power_save:
            p.power_save = True
            _log.info("user %d entered power_save (battery 0)", self.user_id)

    # Inspection helpers
    def as_row(self) -> dict:
        """Return a flat dict suitable for parquet logging."""
        return {
            "user_id":             self.user_id,
            "lat":                 self.lat,
            "lon":                 self.lon,
            "alt_m":               self.alt_m,
            "qos_class":           self.service.qos_class,
            "served_sat_id":       self.service.served_sat_id,
            "battery_remaining_j": self.power.remaining_j,
            "energy_consumed_j":   self.power.cumulative_energy_j,
            "tasks_pending":       len(self.compute.pending_tasks),
            "tasks_completed":     self.compute.completed_tasks,
            "power_save":          self.power.power_save,
        }

    def __repr__(self) -> str:
        return (f"User({self.user_id}, lat={self.lat:.2f}, lon={self.lon:.2f}, "
                 f"served={self.service.served_sat_id}, "
                 f"batt={self.power.remaining_j:.1f}J)")
