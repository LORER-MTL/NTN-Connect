# Per-(sat, user) channel-history store for the DT predictors (read-only views).
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# A view that predictors read from
class HistoryView:
    __slots__ = ("hs_meas", "hs_nominal", "n_tx", "key")

    def __init__(self, hs_meas: Deque[np.ndarray],
                 hs_nominal: Deque[np.ndarray],
                 n_tx: int, key: Tuple[int, int]):
        self.hs_meas    = hs_meas       # ring buffer of measured h (complex N_t,)
        self.hs_nominal = hs_nominal    # ring buffer of geometry-nominal h
        self.n_tx       = n_tx
        self.key        = key           # (sat_id, user_id)

    def __len__(self) -> int:
        return len(self.hs_meas)

    def as_array(self) -> np.ndarray:
        """Return measured history as np.ndarray (T, N_t)."""
        return np.array(self.hs_meas) if len(self.hs_meas) else \
               np.empty((0, self.n_tx), dtype=complex)

    def nominal_array(self) -> np.ndarray:
        return np.array(self.hs_nominal) if len(self.hs_nominal) else \
               np.empty((0, self.n_tx), dtype=complex)


# The store (engine-level service)
class ChannelHistoryStore:
    """Per-(sat_id, user_id) ring buffer of measured and nominal channels."""

    def __init__(self, window_size: int = 20, n_tx: int = 64):
        self.window_size = int(window_size)
        self.n_tx        = int(n_tx)
        self._meas: Dict[Tuple[int, int], Deque[np.ndarray]] = {}
        self._nom:  Dict[Tuple[int, int], Deque[np.ndarray]] = {}

    # ingest
    def push(self, sat_id: int, user_id: int,
             h_meas: np.ndarray, h_nominal: Optional[np.ndarray] = None) -> None:
        key = (int(sat_id), int(user_id))
        if key not in self._meas:
            self._meas[key] = deque(maxlen=self.window_size)
            self._nom[key]  = deque(maxlen=self.window_size)
        self._meas[key].append(np.asarray(h_meas, dtype=complex).copy())
        if h_nominal is not None:
            self._nom[key].append(np.asarray(h_nominal, dtype=complex).copy())
        else:
            # Keep the buffers aligned for predictors that index both.
            self._nom[key].append(self._meas[key][-1].copy())

    # read
    def view(self, sat_id: int, user_id: int) -> Optional[HistoryView]:
        key = (int(sat_id), int(user_id))
        if key not in self._meas:
            return None
        return HistoryView(self._meas[key], self._nom[key], self.n_tx, key)

    def keys(self) -> List[Tuple[int, int]]:
        return list(self._meas.keys())

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._meas

    def __len__(self) -> int:
        return len(self._meas)
