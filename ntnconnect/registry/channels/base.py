# Base types: model returns ChannelData; components mutate it in place.
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class ChannelState:
    """Per-link persistent state across timesteps.

    Holds Markov-chain state for Lutz/Perez-Fontan, last-known shadowing
    values for spatial correlation, etc. Keyed by model name to keep
    composable.
    """
    rng: Any = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelData:
    """Container for one channel realisation per link.

    Set by the model; components add/mutate fields in place.
    """
    loss_db: np.ndarray = None           # large-scale path loss, dB
    h: np.ndarray = None                  # complex channel coefficient(s) per link
    doppler_hz: np.ndarray = None        # carrier Doppler shift
    delay_s: Optional[np.ndarray] = None # propagation delay (set by PropagationDelayComponent)
    shadow_db: Optional[np.ndarray] = None
    xpr_db: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def n_links(self) -> int:
        if self.h is None: return 0
        return int(self.h.shape[0]) if self.h.ndim >= 1 else 1

    def as_tuple(self):
        """Backwards-compat 3-tuple: (loss_db, h, doppler_hz)."""
        return self.loss_db, self.h, self.doppler_hz


# Model contract
class AbstractChannelModel(ABC):
    """Channel models implement compute_channel() -> ChannelData."""

    produces_per_antenna_h: bool = False          # h shape (n_links, n_tx) vs (n_links,)
    consumes_external_k_factor: bool = True        # reads state["k_factor_db_pending"]
    has_intrinsic_spatial_structure: bool = False  # per-tap/cluster angles
    # "fast_fading" (default): only small-scale h; large-scale via components.
    # "complete_lms": owns shadowing intrinsically -> must list it in handles_internally.
    channel_class: str = "fast_fading"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def compute_channel(self,
                        dist_m: np.ndarray,
                        elev_deg: np.ndarray,
                        dt_sec: float,
                        state: Optional[ChannelState] = None,
                        pos_m: Optional[np.ndarray] = None,
                        azimuth_deg: Optional[np.ndarray] = None) -> ChannelData: ...


# Component contract
class AbstractChannelComponent(ABC):
    """Post-processing components run after the base model.

    Each component reads cfg flags to decide whether it applies, then
    mutates ChannelData in place.
    """

    name: str = "component"

    @abstractmethod
    def apply(self, data: ChannelData, cfg: Any, dt_sec: float,
              state: Optional[ChannelState] = None) -> None: ...


__all__ = [
    "AbstractChannelModel", "AbstractChannelComponent",
    "ChannelData", "ChannelState",
]
