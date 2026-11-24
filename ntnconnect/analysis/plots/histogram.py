from __future__ import annotations
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("hist")
class HistogramPlot(BasePlot):
    """Histogram. cfg keys: y (KPI column), bins (int, default 30), title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        y = cfg.get("y")
        if y is None:
            for c in df.columns:
                if c not in ("timestamp","t_idx","user_id","node_id","operator_id"):
                    y = c; break
        bins = int(cfg.get("bins", 30))
        fig, ax = self._fig()
        ax.hist(df[y].dropna().values, bins=bins)
        ax.set_xlabel(y); ax.set_ylabel("count")
        ax.set_title(cfg.get("title", f"Histogram of {y}"))
        ax.grid(True, alpha=0.3)
        self._save(fig, out_paths)
