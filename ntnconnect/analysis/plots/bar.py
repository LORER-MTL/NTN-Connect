from __future__ import annotations
import numpy as np
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("bar")
class BarPlot(BasePlot):
    """Bar chart. cfg keys:
       group_by (column to group bars by, default 'operator_id'),
       y (KPI column), agg (mean|sum|max|count, default 'mean'),
       title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        group_by = cfg.get("group_by", "operator_id")
        if group_by not in df.columns:
            cands = [c for c in df.columns
                     if c in ("operator_id", "user_id", "node_id", "sat_id", "t_idx")]
            if not cands: return
            group_by = cands[0]
        y = cfg.get("y")
        if y is None:
            for c in df.columns:
                if c not in (group_by, "timestamp", "t_idx"):
                    y = c; break
        agg = cfg.get("agg", "mean")
        s = getattr(df.groupby(group_by)[y], agg)()
        fig, ax = self._fig()
        ax.bar([str(k) for k in s.index], s.values)
        ax.set_xlabel(group_by); ax.set_ylabel(f"{agg}({y})")
        ax.set_title(cfg.get("title", f"{agg}({y}) by {group_by}"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        self._save(fig, out_paths)
