from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("line")
class LinePlot(BasePlot):
    """Time-series line plot. cfg keys: x (default 't_idx'), y (KPI name),
    group (column name to draw one line per category), title."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        x = cfg.get("x", "t_idx")
        y = cfg.get("y") or _first_value_col(df)
        group = cfg.get("group")
        fig, ax = self._fig()
        if group and group in df.columns:
            for k, g in df.groupby(group):
                ax.plot(g[x], g[y], label=f"{group}={k}")
            ax.legend(fontsize=8)
        else:
            ax.plot(df[x], df[y])
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(cfg.get("title", y))
        ax.grid(True, alpha=0.3)
        self._save(fig, out_paths)


def _first_value_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c not in ("timestamp", "t_idx", "user_id", "node_id",
                     "operator_id", "sat_a", "sat_b"):
            return c
    return df.columns[-1]
