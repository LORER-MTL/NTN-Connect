# Channel-predictor registry (DT). Predict next-tick h + Sigma from history.
# Selected by cfg.handover.channel_predictor; register via @register_channel_predictor.
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np


class BaseChannelPredictor(ABC):
    """Subclass + @register_channel_predictor("name") to add a predictor.

    Each predictor receives a HistoryView for one (sat, user) link and
    returns:
        h_pred  : np.ndarray  shape (N_t,)  complex   -- predicted mean
        Sigma   : np.ndarray  shape (N_t, N_t) complex Hermitian PSD
        status  : str  ("ok" | "cold" | "fallback")
    """
    name: str = "base"

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def predict(self, history_view, lag_steps: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, str]:
        ...

    def on_tick_start(self) -> None:
        """Override for predictors that need a per-tick reset (e.g. MLP)."""
        ...

    def on_tick_end(self, ground_truth: dict) -> None:
        """Override for predictors that learn from the realised channel."""
        ...


_REGISTRY: Dict[str, Type[BaseChannelPredictor]] = {}


def register_channel_predictor(name: str):
    """Decorator: add a predictor class to the registry."""
    def _deco(cls: Type[BaseChannelPredictor]) -> Type[BaseChannelPredictor]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _deco


def get_channel_predictor(cfg: Any) -> Optional[BaseChannelPredictor]:
    name = (getattr(cfg.handover, "channel_predictor", "") or "").lower()
    cls  = _REGISTRY.get(name)
    return cls(cfg) if cls is not None else None


def list_channel_predictors() -> Dict[str, Type[BaseChannelPredictor]]:
    return dict(_REGISTRY)
