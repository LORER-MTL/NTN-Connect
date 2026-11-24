# Compute resource + task lifecycle + cache audit hooks.
# Best-effort population: every column the current stage stack produces is
# filled; columns that depend on stages not yet wired (energy model, full
# task lifecycle) are written as None / 0 with no schema violation.

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook
from ntnconnect.common.schemas import COMPUTE_RESOURCE, TASK_AUDIT, CACHE_AUDIT

_log = logging.getLogger(__name__)


class ComputeResourceHook(SimulationHook):
    """Per-node per-tick compute / storage / energy snapshot."""

    def __init__(self, engine):
        self.engine = engine
        self.c = COMPUTE_RESOURCE.collector(engine)
        # cumulative energy per node
        self._cum_energy: dict = {}

    def on_step_end(self, state):
        if not COMPUTE_RESOURCE.is_enabled(self.engine.cfg): return
        ns = state.data.get("node_state_now")
        if not isinstance(ns, pd.DataFrame) or ns.empty: return

        cfg = self.engine.cfg
        step_s = float(cfg.env.step_size_s)
        # Per-(op, type) capability lookup
        op_caps = {}
        for op in cfg.operators:
            for ntype, cap in op.compute.items():
                op_caps[(op.id, ntype)] = cap

        for _, r in ns.iterrows():
            nid = int(r.get("node_id", r.get("node_idx", -1)))
            op_id = int(r.get("operator_id", -1)) if r.get("operator_id") is not None else -1
            ntype = str(r.get("node_type", ""))
            cap = op_caps.get((op_id, ntype))

            cpu_peak = float(cap.flops_peak) if cap else 0.0
            n_run = int(r.get("n_tasks_running", 0))
            util = min(1.0, n_run * 0.3) if cap else 0.0
            cpu_used = util * cpu_peak

            p_idle = float(cap.power_idle_w) if cap else 0.0
            p_compute = util * cpu_peak * float(cap.power_per_flop_j) if cap else 0.0
            p_total = p_idle + p_compute
            e_tick = p_total * step_s
            self._cum_energy[nid] = self._cum_energy.get(nid, 0.0) + e_tick

            self.c.add({
                "t_idx":               state.t_idx,
                "node_id":             nid,
                "cpu_used_flops":      cpu_used,
                "cpu_capacity_flops":  cpu_peak,
                "cpu_util_pct":        100.0 * util,
                "ram_used_gb":         None,
                "ram_capacity_gb":     float(cap.ram_gb)     if cap else None,
                "storage_used_gb":     None,
                "storage_capacity_gb": float(cap.storage_gb) if cap else None,
                "power_compute_w":     p_compute,
                "power_idle_w":        p_idle,
                "power_total_w":       p_total,
                "energy_joules_tick":  e_tick,
                "energy_joules_cum":   self._cum_energy[nid],
                "n_tasks_admitted":    n_run,
                "n_tasks_rejected":    0,
                "n_tasks_completed":   0,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class TaskAuditHook(SimulationHook):
    """Per-task lifecycle - one row per task at the moment we know its fate."""

    def __init__(self, engine):
        self.engine = engine
        self.c = TASK_AUDIT.collector(engine)
        self._seen_terminal: set = set()       # task_ids already written

    def on_step_end(self, state):
        if not TASK_AUDIT.is_enabled(self.engine.cfg): return
        for t in state.data.get("task_events", []):
            tid = int(t.get("task_id", -1))
            if tid in self._seen_terminal: continue
            status = t.get("status", "queued")
            if status not in ("done","dropped","failed","timeout"):
                continue  # not terminal yet
            self._seen_terminal.add(tid)
            arrival = int(t.get("arrival_t_idx", state.t_idx))
            finish  = state.t_idx
            step_ms = float(self.engine.cfg.env.step_size_s) * 1000.0
            e2e_ms  = max(0, finish - arrival) * step_ms
            self.c.add({
                "task_id":           tid,
                "user_id":           int(t.get("user_id", -1)),
                "arrival_t_idx":     arrival,
                "queue_t_idx":       arrival,
                "start_t_idx":       arrival,
                "finish_t_idx":      finish,
                "assigned_node_id":  int(t.get("assigned_node_idx",
                                                t.get("assigned_node_id", -1))),
                "queue_latency_ms":  0.0,
                "exec_time_ms":      e2e_ms,
                "e2e_latency_ms":    e2e_ms,
                "cpu_consumed_flops":float(t.get("cpu_required_flops", 0.0)),
                "energy_joules":     0.0,
                "data_in_mb":        float(t.get("data_in_mb", 0.0)),
                "data_out_mb":       float(t.get("data_out_mb", 0.0)),
                "deadline_ms":       float(t.get("deadline_ms", 0.0)),
                "deadline_met":      int(e2e_ms <= float(t.get("deadline_ms", 1e18))),
                "final_status":      status,
                "failure_reason":    None,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class CacheAuditHook(SimulationHook):
    """Per-node cache audit. Stub-friendly: writes zeros until cache stages
    populate cache_state with hit/miss counters."""

    def __init__(self, engine):
        self.engine = engine
        self.c = CACHE_AUDIT.collector(engine)

    def on_step_end(self, state):
        if not CACHE_AUDIT.is_enabled(self.engine.cfg): return
        policy = (self.engine.cfg.cache.policy or "lru").lower()
        cache = state.data.get("cache_state", {}) or {}
        ns = state.data.get("node_state_now")
        if not isinstance(ns, pd.DataFrame) or ns.empty: return
        for _, r in ns.iterrows():
            nid = int(r.get("node_id", r.get("node_idx", -1)))
            cell = cache.get(nid, {}) if isinstance(cache, dict) else {}
            used_mb = sum(cell.values()) if isinstance(cell, dict) else 0.0
            self.c.add({
                "t_idx":             state.t_idx,
                "node_id":           nid,
                "cache_capacity_mb": float(self.engine.cfg.cache.capacity_fraction_of_storage * 1000.0),
                "cache_used_mb":     float(used_mb),
                "n_items":           len(cell) if isinstance(cell, dict) else 0,
                "n_hits":            0,
                "n_misses":          0,
                "n_evictions":       0,
                "hit_rate":          0.0,
                "policy":            policy,
            })

    def on_simulation_end(self, state):
        self.c.flush()


@register_hook("compute_resource")
def _b1(engine): return ComputeResourceHook(engine)

@register_hook("task_audit")
def _b2(engine): return TaskAuditHook(engine)

@register_hook("cache_audit")
def _b3(engine): return CacheAuditHook(engine)
