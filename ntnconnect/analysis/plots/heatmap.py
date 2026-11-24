from __future__ import annotations
import numpy as np
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("heatmap")
class HeatmapPlot(BasePlot):
    """2D heatmap. cfg keys: x (col, default 't_idx'), y (col, default 'user_id'),
    value (col), agg (mean|sum|max), title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        x = cfg.get("x", "t_idx")
        y = cfg.get("y", "user_id")
        v = cfg.get("value")
        if v is None:
            for c in df.columns:
                if c not in (x, y, "timestamp"):
                    v = c; break
        agg = cfg.get("agg", "mean")
        piv = df.pivot_table(index=y, columns=x, values=v, aggfunc=agg)
        fig, ax = self._fig(figsize=(10, 5))
        im = ax.imshow(piv.values, aspect="auto", origin="lower")
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(cfg.get("title", f"{v} heatmap"))
        fig.colorbar(im, ax=ax)
        self._save(fig, out_paths)
