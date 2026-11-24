# Plotting hook
#
# On simulation end:
#  1. Materialize each computed KPI's per-tick frames into one DataFrame
#  2. Persist KPI tables (format selected by cfg.outputs)
#  3. Render each chart declared in cfg.plots.charts

from __future__ import annotations
import logging
from pathlib import Path
from typing import List
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.common.utils.writers import write_table
from ntnconnect.analysis.hooks.registry import register_hook
# Import plot modules so they self-register
from ntnconnect.analysis.plots import base as _plot_base
from ntnconnect.analysis.plots import line, bar, cdf, histogram, heatmap, scatter, network_graph

_log = logging.getLogger(__name__)


class PlottingHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine

    def on_simulation_end(self, state):
        # Read accumulator from the engine (survives across ticks),
        # falling back to state.metadata for backwards compat.
        kpi_results = getattr(self.engine, "kpi_results", {}) or \
                      (state.metadata.get("kpi_results", {}) if state else {})
        if not kpi_results:
            _log.warning("PlottingHook: no KPI results captured "
                         "(metrics stage off or not in pipeline?).")
            return
        cfg = self.engine.cfg

        # Persist KPI tables in the configured format
        out_dir = self.engine.sim_dir
        g_fmt = cfg.outputs.format
        per_tbl = getattr(cfg.outputs, "formats", {}) or {}
        kpi_dir = out_dir / "kpis"
        for name, frames in kpi_results.items():
            try:
                if not frames: continue
                full = pd.concat(frames, ignore_index=True)
                write_table(full, kpi_dir, f"kpi_{name}", g_fmt, per_tbl)
            except Exception as e:
                _log.warning("KPI '%s' persistence failed: %s", name, e)

        # Render plots
        plots_cfg = getattr(cfg, "plots", None)
        if plots_cfg is None or not getattr(plots_cfg, "charts", None):
            _log.info("PlottingHook: no charts declared in cfg.plots.charts.")
            return

        formats = list(getattr(plots_cfg, "formats", ["png"]))
        plot_dir = out_dir / getattr(plots_cfg, "output_dir", "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        for chart in plots_cfg.charts:
            try:
                kpi_name = chart.kpi
                plot_type = chart.type
                if kpi_name not in kpi_results: continue
                df = pd.concat(kpi_results[kpi_name], ignore_index=True)
                PlotCls = _plot_base.get_plot_class(plot_type)
                plot = PlotCls()
                stem = chart.name or f"{kpi_name}_{plot_type}"
                out_paths = [plot_dir / f"{stem}.{ext}" for ext in formats]
                plot.render(df, _chart_cfg(chart), out_paths)
            except Exception as e:
                _log.warning("Chart '%s' failed: %s", getattr(chart, "name", "?"), e)


def _chart_cfg(chart) -> dict:
    """Map ChartSpec dataclass into the dict each plot expects."""
    d = {}
    if chart.title:  d["title"] = chart.title
    if chart.x:      d["x"]     = chart.x
    if chart.y:      d["y"]     = chart.y
    if chart.group:  d["group"] = chart.group
    if chart.group_by: d["group_by"] = chart.group_by
    if chart.agg:    d["agg"]   = chart.agg
    if chart.bins:   d["bins"]  = chart.bins
    if chart.value:  d["value"] = chart.value
    if chart.layout: d["layout"] = chart.layout
    if chart.color:  d["color"] = chart.color
    return d


@register_hook("plotting")
def _build(engine): return PlottingHook(engine)
