from __future__ import annotations
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("scatter")
class ScatterPlot(BasePlot):
    """Scatter plot. cfg keys: x, y, color (column), title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        x, y = cfg.get("x"), cfg.get("y")
        color = cfg.get("color")
        fig, ax = self._fig()
        if color and color in df.columns:
            cats = df[color].unique()
            for c in cats:
                g = df[df[color] == c]
                ax.scatter(g[x], g[y], label=f"{color}={c}", s=12, alpha=0.7)
            ax.legend(fontsize=8)
        else:
            ax.scatter(df[x], df[y], s=12, alpha=0.7)
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(cfg.get("title", f"{y} vs {x}"))
        ax.grid(True, alpha=0.3)
        self._save(fig, out_paths)
