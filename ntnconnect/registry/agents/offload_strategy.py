# SkyNTNPro v1 - Offload / scheduler / cache strategies
# Same interface pattern as handover. Pluggable per config.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class Task:
    task_id: int
    user_id: int
    arrival_t_idx: int
    cpu_required_flops: float
    data_in_mb: float
    data_out_mb: float
    deadline_ms: float
    priority: str = "normal"


class BaseOffloadStrategy(ABC):
    @abstractmethod
    def assign(self, ctx) -> Dict[int, int]:
        """Pick a compute node per task. Return {task_id: node_id}; -1 if there
        is no reachable compute candidate (dropped). ctx is an OffloadContext:
        ctx.tasks (task dicts) + ctx.candidates[user_id] (OffloadCandidate list,
        each carrying achieved rate / capacity / load / latency)."""


def _completion_ms(task: dict, c, step_s: float) -> float:
    """Predicted end-to-end completion time [ms] for running `task` on candidate
    `c`: uplink transmit + queue wait + compute + downlink return + 2x prop.
    Reads the ACHIEVED rate carried on the candidate."""
    din_b  = float(task.get("data_in_mb", 0.0))  * 8.0e6
    dout_b = float(task.get("data_out_mb", 0.0)) * 8.0e6
    flops  = float(task.get("cpu_required_flops", 0.0))
    rate   = max(c.rate_bps, 1.0) if c.tier != "local" else float("inf")
    cap    = max(c.capacity_flops, 1.0)
    tx_s   = (din_b + dout_b) / rate if rate != float("inf") else 0.0
    comp_s = flops / cap
    queue_s = c.queue_len * comp_s              # rough backlog wait (~own size)
    return 2.0 * c.latency_ms + (tx_s + comp_s + queue_s) * 1000.0


class DeadlineAwareOffload(BaseOffloadStrategy):
    """Rate + deadline aware (the realistic MEC orchestrator): pick the candidate
    with the smallest predicted completion time, preferring those that meet the
    task deadline. Uses the achieved uplink rate + node capacity/load/latency."""
    def assign(self, ctx):
        out = {}
        for t in ctx.tasks:
            cand = ctx.candidates.get(int(t["user_id"]), [])
            if not cand:
                out[int(t["task_id"])] = -1; continue
            dl = float(t.get("deadline_ms", 1e12))
            scored = [(_completion_ms(t, c, ctx.step_s), c) for c in cand]
            feasible = [(m, c) for (m, c) in scored if m <= dl]
            pick = min(feasible or scored, key=lambda mc: mc[0])[1]
            out[int(t["task_id"])] = int(pick.node_id)
        return out


class NearestComputeOffload(BaseOffloadStrategy):
    """Prefer the edge (serving access node); else the first compute candidate."""
    def assign(self, ctx):
        out = {}
        for t in ctx.tasks:
            cand = ctx.candidates.get(int(t["user_id"]), [])
            if not cand:
                out[int(t["task_id"])] = -1; continue
            edge = next((c for c in cand if c.tier == "edge"), cand[0])
            out[int(t["task_id"])] = int(edge.node_id)
        return out


class LoadBalancedOffload(BaseOffloadStrategy):
    """Pick the reachable candidate with the lowest current backlog."""
    def assign(self, ctx):
        out = {}
        load = {}
        for t in ctx.tasks:
            cand = ctx.candidates.get(int(t["user_id"]), [])
            if not cand:
                out[int(t["task_id"])] = -1; continue
            best = min(cand, key=lambda c: load.get(c.node_id, c.queue_len))
            out[int(t["task_id"])] = int(best.node_id)
            load[best.node_id] = load.get(best.node_id, best.queue_len) + 1
        return out


OFFLOAD_STRATEGIES = {
    "deadline_aware":           DeadlineAwareOffload,
    "load_balanced":            LoadBalancedOffload,
    "nearest_compute_capable":  NearestComputeOffload,
}


def register_offload_strategy(name):
    def deco(cls):
        OFFLOAD_STRATEGIES[str(name).lower()] = cls
        return cls
    return deco


def list_offload_strategies():
    return dict(OFFLOAD_STRATEGIES)


def get_offload_strategy(cfg) -> BaseOffloadStrategy:
    pol = (cfg.offload.target_policy or "nearest_compute_capable").lower()
    return OFFLOAD_STRATEGIES.get(pol, NearestComputeOffload)()


# Scheduler (per node)
class BaseSchedulerStrategy(ABC):
    @abstractmethod
    def order(self, queued_tasks: List[Task], cfg) -> List[Task]: ...


class FIFOScheduler(BaseSchedulerStrategy):
    def order(self, qs, cfg): return sorted(qs, key=lambda t: t.arrival_t_idx)


class EDFScheduler(BaseSchedulerStrategy):
    def order(self, qs, cfg):
        return sorted(qs, key=lambda t: (t.arrival_t_idx + t.deadline_ms))


class SRTFScheduler(BaseSchedulerStrategy):
    def order(self, qs, cfg):
        return sorted(qs, key=lambda t: t.cpu_required_flops)


SCHEDULER_STRATEGIES = {
    "fifo": FIFOScheduler, "edf": EDFScheduler, "srtf": SRTFScheduler,
}


def register_scheduler_strategy(name):
    def deco(cls):
        SCHEDULER_STRATEGIES[str(name).lower()] = cls
        return cls
    return deco


def list_scheduler_strategies():
    return dict(SCHEDULER_STRATEGIES)


def get_scheduler_strategy(cfg) -> BaseSchedulerStrategy:
    pol = (cfg.scheduler.policy or "edf").lower()
    return SCHEDULER_STRATEGIES.get(pol, EDFScheduler)()


# Cache
class BaseCacheStrategy(ABC):
    @abstractmethod
    def update(self, node_cache_state: dict, requests: list, cfg) -> dict: ...


class LRUCache(BaseCacheStrategy):
    def update(self, cache, requests, cfg):
        # cache: {node_id: OrderedDict of content_id -> size_mb}
        for (node_id, content_id, size_mb) in requests:
            od = cache.setdefault(node_id, {})
            od.pop(content_id, None)
            od[content_id] = size_mb
            # note: capacity enforcement is the caller's responsibility
        return cache


def get_cache_strategy(cfg) -> BaseCacheStrategy:
    return LRUCache()
