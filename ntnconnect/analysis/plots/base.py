# SkyNTNPro v1 - Plot registry
#
# Each plot type is a subclass of BasePlot, registered by name.
# The user picks plots in cfg.plots.charts. Output formats are
# decided by cfg.plots.formats (png, pdf, svg).

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Type
import pandas as pd

_REGISTRY: Dict[str, Type["BasePlot"]] = {}


def register_plot(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def list_plots() -> Dict[str, Type["BasePlot"]]:
    return dict(_REGISTRY)


def get_plot_class(name: str) -> Type["BasePlot"]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown plot '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


class BasePlot(ABC):
    """Base class for a plot type.

    Render method receives a DataFrame, a config dict (from the chart spec),
    and a list of output paths to write.
    """
    name: str = "base"

    @abstractmethod
    def render(self, df: pd.DataFrame, cfg: dict, out_paths: List[Path]) -> None: ...

    @staticmethod
    def _fig(figsize=(8, 5), dpi=120):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig, ax

    @staticmethod
    def _save(fig, out_paths: List[Path]):
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
