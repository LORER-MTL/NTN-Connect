# SkyNTNPro v1 - KPI registry
#
# Every KPI is one row in the registry:
#  name - stable identifier referenced from YAML
#  granularity - per_link | per_user | per_operator | per_tick | per_run
#  units - "dB", "bps/Hz", "ms", "%",...
#  compute(state, links_df, history) -> pd.DataFrame or scalar
#  default_plot - line | bar | cdf | hist | heatmap | scatter
#  description - one-line documentation
#
# The user toggles which KPIs to compute via cfg.kpis.computed.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

GRANULARITIES = ("per_link", "per_user", "per_operator", "per_tick", "per_run")


@dataclass
class KPI:
    name: str
    granularity: str
    units: str
    compute: Callable
    default_plot: str = "line"
    description: str = ""


_REGISTRY: Dict[str, KPI] = {}


def register_kpi(name: str, *, granularity: str, units: str,
                 default_plot: str = "line", description: str = ""):
    """Decorator: register a function as a KPI under ``name``."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")
    def deco(fn):
        _REGISTRY[name] = KPI(
            name=name, granularity=granularity, units=units,
            compute=fn, default_plot=default_plot, description=description,
        )
        return fn
    return deco


def list_kpis() -> Dict[str, KPI]:
    return dict(_REGISTRY)


def get_kpi(name: str) -> KPI:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown KPI '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def kpis_by_granularity(g: str) -> List[KPI]:
    return [k for k in _REGISTRY.values() if k.granularity == g]
