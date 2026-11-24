# Radio algorithm registry: @register_radio_algorithm; picked via
# cfg.handover.rrm_power_algorithm; scn_radio_link calls algorithm.compute().
#
# Why a registry and not @register_stage? Radio algorithms differ
# in their internal math but share the same input/output schema on
# the `links` frame. Treating them as plug-ins under one stage
# means the rest of the pipeline (handover, classify, metrics, hooks)
# is identical regardless of which algorithm runs.
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import pandas as pd


class BaseRadioAlgorithm(ABC):
    """Base class for any RRM/beamforming algorithm used by scn_radio_link.

    Subclasses must implement `compute(links, assigns)` which mutates
    `links` (or returns a new frame) with the radio columns:

        tx_power_w, signal_w, noise_w, interference_w,
        snr_db, sinr_db, spectral_efficiency_bps_hz, link_rate_bps,
        radio_status                       (string status per link)

    Optional columns the algorithm may also add:

        bf_gain_db, bf_array_db, bf_precoder_db,
        interference_intra_w, interference_ext_w,
        interference_model
    """
    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def compute(self, links: pd.DataFrame, assigns: dict,
                state: Any = None) -> pd.DataFrame:
        """`state` is the SimulationState - used by DT-aware algorithms to
        read state.data["channel_predictions"]. Pre-DT algorithms may ignore it."""
        ...


_REGISTRY: Dict[str, Type[BaseRadioAlgorithm]] = {}


def register_radio_algorithm(name: str):
    """Decorator. Adds the class to the global radio-algorithm registry."""
    def _deco(cls: Type[BaseRadioAlgorithm]) -> Type[BaseRadioAlgorithm]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _deco


def get_radio_algorithm(cfg: Any) -> Optional[BaseRadioAlgorithm]:
    """Resolve the algorithm named in cfg.handover.rrm_power_algorithm.

    Returns None when the name is not in the registry. The caller is
    expected to fall back to its closed-form path in that case.
    """
    name = (getattr(cfg.handover, "rrm_power_algorithm", "") or "").lower()
    cls = _REGISTRY.get(name)
    return cls(cfg) if cls is not None else None


def list_radio_algorithms() -> Dict[str, Type[BaseRadioAlgorithm]]:
    """Snapshot of every registered radio algorithm name -> class."""
    return dict(_REGISTRY)
