# Stage: compute cfg.kpis.computed -> engine.kpi_results (persists across ticks).

from __future__ import annotations
import logging
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_log = logging.getLogger(__name__)
# Import all KPI modules so they self-register
from ntnconnect.analysis.metrics import (
    link_kpis, user_kpis, operator_kpis,
    network_kpis, compute_kpis, system_kpis,
    routing_kpis, energy_kpis, rrm_kpis,
    coverage_kpis,
)
from ntnconnect.analysis.metrics.base import list_kpis


@register_stage("metrics")
class MetricsStage(Stage):
    reads  = ["links", "isl_active", "assignments", "ho_classified"]
    writes = []
    thread_safe = True

    def __init__(self, cfg):
        super().__init__(cfg)
        kp = getattr(cfg, "kpis", None)
        selected = getattr(kp, "computed", None) if kp else None
        all_kpis = list_kpis()
        if selected:
            self._selected = {n: all_kpis[n] for n in selected if n in all_kpis}
            unknown = [n for n in selected if n not in all_kpis]
            if unknown:
                _log.warning("MetricsStage: unknown KPI names in cfg.kpis.computed: %s",
                             unknown)
        else:
            self._selected = all_kpis
        self._reported_errors: set = set()
        # Clear module-level KPI caches from any previous run in this process
        try:
            network_kpis.reset_module_state()
        except AttributeError:
            pass

    def run(self, state):
        links = state.data.get("links")
        if self.engine is None:
            _log.warning("MetricsStage has no engine reference; KPIs will be lost.")
            return
        results = self.engine.kpi_results
        for name, kpi in self._selected.items():
            try:
                out = kpi.compute(state, links, None)
                if out is None: continue
                if not isinstance(out, pd.DataFrame):
                    out = (pd.DataFrame([out]) if isinstance(out, dict)
                           else pd.DataFrame([{name: out,
                                               "timestamp": state.timestamp,
                                               "t_idx": state.t_idx}]))
                results.setdefault(name, []).append(out)
            except Exception as e:
                # KPI failure must not break the simulation, but it must be
                # visible. Log once per KPI name with the exception class
                # and message so silent data loss is impossible.
                if name not in self._reported_errors:
                    self._reported_errors.add(name)
                    _log.warning("KPI '%s' failed at t_idx=%s: %s: %s",
                                 name, state.t_idx, type(e).__name__, e)
