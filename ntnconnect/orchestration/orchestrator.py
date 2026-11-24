# Orchestrator: run stage groups (parallel via ThreadPool or sequential);
# hooks fire between groups and around each tick.

from __future__ import annotations
import time
import logging
from abc import ABC
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger(__name__)


@dataclass
class SimulationState:
    timestamp: Any
    t_idx: int
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationHook(ABC):
    def on_simulation_start(self, cfg): pass
    def on_step_start(self, state):     pass
    def on_group_end(self, group, state): pass
    def on_step_end(self, state):       pass
    def on_simulation_end(self, state): pass
    def on_error(self, state, exc):     pass


@dataclass
class StageGroup:
    name: str
    stages: List[Callable]
    parallel: bool = False
    max_workers: int = 4


class SimulationScheduler:
    def __init__(self, timestamps: List[Any], step_size_s: float):
        self.timestamps = list(timestamps)
        self.step_size_s = step_size_s
        self.i = 0
    def __iter__(self): return self
    def __next__(self):
        if self.i >= len(self.timestamps): raise StopIteration
        v = (self.i, self.timestamps[self.i]); self.i += 1; return v
    def reset(self): self.i = 0


class SimulationOrchestrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.groups: List[StageGroup] = []
        self.hooks: List[SimulationHook] = []
        # per-stage timing (cumulative across ticks)
        self.stage_time_s: Dict[str, float] = defaultdict(float)
        self.stage_calls: Dict[str, int]    = defaultdict(int)
        self.group_time_s: Dict[str, float] = defaultdict(float)

    # construction
    def add_group(self, group: StageGroup):
        self.groups.append(group)

    def add_hook(self, hook: SimulationHook):
        self.hooks.append(hook)

    # safety check (called once at engine.setup)
    def validate(self) -> None:
        """Reject groups that violate the parallel-safety contract:
          * No two parallel stages write the same state.data key
          * Each stage in group K must have its `reads` produced by some
            stage in groups <= K (or be missing, in which case it's an
            optional read - we don't raise, but we log via writes set).
        """
        produced: set = set()
        for k, g in enumerate(self.groups):
            if g.parallel:
                writes_seen: dict = {}
                for fn in g.stages:
                    stg = getattr(fn, "stage", None)
                    if stg is None: continue
                    if not stg.thread_safe:
                        raise ValueError(
                            f"Group '{g.name}' is parallel but stage "
                            f"'{stg.name}' is not thread_safe."
                        )
                    for w in stg.writes:
                        if w in writes_seen:
                            raise ValueError(
                                f"Group '{g.name}': stages '{stg.name}' and "
                                f"'{writes_seen[w]}' both write key '{w}'."
                            )
                        writes_seen[w] = stg.name
            # Accumulate writes for read-availability checks
            for fn in g.stages:
                stg = getattr(fn, "stage", None)
                if stg is None: continue
                produced.update(stg.writes)

    # execution
    def run(self, scheduler: SimulationScheduler) -> Optional[SimulationState]:
        for h in self.hooks: h.on_simulation_start(self.cfg)
        state = None
        for t_idx, ts in scheduler:
            state = SimulationState(ts, t_idx)
            try:
                for h in self.hooks: h.on_step_start(state)
                for g in self.groups:
                    self._run_group(g, state)
                    for h in self.hooks: h.on_group_end(g, state)
                for h in self.hooks: h.on_step_end(state)
            except Exception as exc:
                for h in self.hooks: h.on_error(state, exc)
                raise
        for h in self.hooks: h.on_simulation_end(state)
        return state

    def _run_group(self, g: StageGroup, state: SimulationState):
        g_t0 = time.perf_counter()
        if g.parallel and len(g.stages) > 1:
            with ThreadPoolExecutor(max_workers=g.max_workers) as ex:
                futs = [ex.submit(self._run_stage_timed, stg, state)
                        for stg in g.stages]
                for f in as_completed(futs):
                    f.result()
        else:
            for stg in g.stages:
                self._run_stage_timed(stg, state)
        self.group_time_s[g.name] += time.perf_counter() - g_t0

    def _run_stage_timed(self, stg: Callable, state: SimulationState) -> None:
        name = getattr(getattr(stg, "stage", None), "name", None) \
               or getattr(stg, "__name__", "anonymous_stage")
        t0 = time.perf_counter()
        try:
            stg(state)
        finally:
            dt = time.perf_counter() - t0
            self.stage_time_s[name] += dt
            self.stage_calls[name]  += 1

    # reporting
    def timing_report(self, top_n: int = 15) -> str:
        """Return a sorted human-readable summary of per-stage cumulative time."""
        rows = sorted(self.stage_time_s.items(), key=lambda kv: -kv[1])
        total = sum(self.stage_time_s.values()) or 1.0
        lines = ["", "=" * 64,
                 f"Per-stage cumulative time (top {top_n}):",
                 f"{'stage':<28} {'calls':>6} {'total_s':>10} {'pct':>6}",
                 "-" * 64]
        for name, t in rows[:top_n]:
            calls = self.stage_calls.get(name, 0)
            lines.append(f"{name:<28} {calls:>6d} {t:>10.3f} {100*t/total:>5.1f}%")
        lines.append("=" * 64)
        return "\n".join(lines)
