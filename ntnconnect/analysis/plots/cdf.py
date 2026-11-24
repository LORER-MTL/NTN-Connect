from __future__ import annotations
import numpy as np
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("cdf")
class CDFPlot(BasePlot):
    """Empirical CDF. cfg keys: y (KPI column), group (split CDFs per group), title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        y = cfg.get("y") or _first_value_col(df)
        group = cfg.get("group")
        fig, ax = self._fig()
        if group and group in df.columns:
            for k, g in df.groupby(group):
                v = np.sort(g[y].dropna().values)
                if len(v) == 0: continue
                ax.plot(v, np.arange(1, len(v)+1)/len(v), label=f"{group}={k}")
            ax.legend(fontsize=8)
        else:
            v = np.sort(df[y].dropna().values)
            if len(v) > 0:
                ax.plot(v, np.arange(1, len(v)+1)/len(v))
        ax.set_xlabel(y); ax.set_ylabel("CDF")
        ax.set_title(cfg.get("title", f"CDF of {y}"))
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        self._save(fig, out_paths)


def _first_value_col(df):
    for c in df.columns:
        if c not in ("timestamp","t_idx","user_id","node_id","operator_id","sat_id"):
            return c
    return df.columns[-1]
