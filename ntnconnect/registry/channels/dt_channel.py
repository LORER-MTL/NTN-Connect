# DTChannel: wraps a channel predictor into a belief CN(h_pred, Sigma) per (sat,user).
# update() ingests h_meas; mean()/cov()/sample() serve the posterior to radio algos.
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

from ntnconnect.registry.channels._history import ChannelHistoryStore

_log = logging.getLogger(__name__)


class DTChannel:
    """Digital-Twin channel model.

    Conceptually: posterior belief CN(h_pred, Sigma) per (sat, user) link,
    refreshed every tick from the predictor. Exposes a sampler so any
    stochastic optimisation algorithm can draw synthetic channels without
    knowing which predictor produced the belief.
    """

    def __init__(self,
                 predictor,
                 n_tx: int,
                 window_size: int = 20,
                 history_store: Optional[ChannelHistoryStore] = None,
                 rng: Optional[np.random.Generator] = None):
        self._predictor   = predictor
        self.n_tx         = int(n_tx)
        self.window_size  = int(window_size)
        self.history      = history_store or ChannelHistoryStore(
            window_size=window_size, n_tx=self.n_tx)
        self._rng         = rng or np.random.default_rng()
        # Cached posterior per (sat, user) - refreshed by predict().
        # Each entry: {"h_pred", "Sigma", "status", "L"}.
        self._posterior: dict = {}

    # ingest / belief refresh
    def update(self, sat_id: int, user_id: int, h_meas: np.ndarray,
               h_nominal: Optional[np.ndarray] = None) -> None:
        """Push a freshly-observed real channel into the predictor history."""
        self.history.push(int(sat_id), int(user_id), h_meas,
                          h_nominal=h_nominal if h_nominal is not None else h_meas)

    def predict(self, sat_id: int, user_id: int, lag_steps: int = 1) -> dict:
        """Run the predictor on the strict-past history and refresh the
        cached posterior for (sat, user). Returns the posterior dict.
        Safe even when the link is fresh - falls back to a cold-start
        identity-scaled Sigma."""
        key  = (int(sat_id), int(user_id))
        view = self.history.view(int(sat_id), int(user_id))
        if view is None:
            # Link not seen yet - return a placeholder.
            self._posterior[key] = {
                "h_pred": np.zeros(self.n_tx, dtype=complex),
                "Sigma":  0.1 * np.eye(self.n_tx, dtype=complex),
                "status": "no_history",
                "L":      None,
            }
            return self._posterior[key]

        h_pred, Sigma, status = self._predictor.predict(view, lag_steps=lag_steps)
        # Defensive: replace NaN/Inf with a cold-start surrogate.
        if (not np.all(np.isfinite(h_pred))) or (not np.all(np.isfinite(Sigma))):
            h_pred = np.zeros(self.n_tx, dtype=complex)
            Sigma  = 0.1 * np.eye(self.n_tx, dtype=complex)
            status = "nan_recover"
        self._posterior[key] = {
            "h_pred": h_pred,
            "Sigma":  Sigma,
            "status": status,
            "L":      None,                          # lazily computed
        }
        return self._posterior[key]

    # posterior accessors
    def has(self, sat_id: int, user_id: int) -> bool:
        return (int(sat_id), int(user_id)) in self._posterior

    def mean(self, sat_id: int, user_id: int) -> Optional[np.ndarray]:
        e = self._posterior.get((int(sat_id), int(user_id)))
        return None if e is None else e["h_pred"]

    def cov(self, sat_id: int, user_id: int) -> Optional[np.ndarray]:
        e = self._posterior.get((int(sat_id), int(user_id)))
        return None if e is None else e["Sigma"]

    def status(self, sat_id: int, user_id: int) -> Optional[str]:
        e = self._posterior.get((int(sat_id), int(user_id)))
        return None if e is None else e["status"]

    def predictions_dict(self) -> dict:
        """Snapshot in the legacy {(sat,user): {h_pred, Sigma, status}} shape
        for back-compat with the existing radio algorithms."""
        return {k: {"h_pred": v["h_pred"],
                    "Sigma":  v["Sigma"],
                    "status": v["status"]}
                for k, v in self._posterior.items()}

    # sampler - the new capability
    def _cholesky(self, sat_id: int, user_id: int) -> Optional[np.ndarray]:
        """Lazy trace-relative Cholesky cache."""
        key = (int(sat_id), int(user_id))
        e   = self._posterior.get(key)
        if e is None: return None
        if e["L"] is not None: return e["L"]
        # Defer to the shared safe_cholesky helper.
        from ntnconnect.registry.agents.radio._utils import safe_cholesky
        L = safe_cholesky(e["Sigma"])
        e["L"] = L
        return L

    def sample(self, sat_id: int, user_id: int, n: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw N synthetic channel realisations from CN(h_pred, Sigma).

        Returns array of shape (n, n_tx) complex. Uses
            H_n = h_pred + L @ (z_re + j*z_im)/sqrt(2)
        with L L^H = Sigma + trace-relative jitter so the covariance is
        respected exactly. When (sat, user) hasn't been refreshed via
        predict() yet, returns N copies of the zero vector with a
        warning.
        """
        e = self._posterior.get((int(sat_id), int(user_id)))
        if e is None:
            _log.warning("DTChannel.sample: no posterior for (%s,%s); "
                         "returning zeros", sat_id, user_id)
            return np.zeros((int(n), self.n_tx), dtype=complex)
        L      = self._cholesky(sat_id, user_id)
        h_pred = e["h_pred"]
        rng    = rng or self._rng
        # Complex circular Gaussian: z = (a + j b)/sqrt(2), a,b ~ N(0,I)
        a = rng.standard_normal((int(n), self.n_tx))
        b = rng.standard_normal((int(n), self.n_tx))
        z = (a + 1j * b) / np.sqrt(2.0)
        # H_n[n,:] = h_pred + (L @ z[n,:]). Vectorised over rows this is z @ L.T
        # (not z @ L.conj().T): with L L^H = Sigma, z @ L.T yields Cov = Sigma,
        # whereas z @ L.conj().T would yield conj(Sigma).
        innov = z @ L.T
        return h_pred[None, :] + innov

    # lifecycle hooks (pass-through to the predictor)
    def on_tick_start(self) -> None:
        self._predictor.on_tick_start()

    def on_tick_end(self, ground_truth: dict) -> None:
        self._predictor.on_tick_end(ground_truth)

    # introspection
    def __len__(self) -> int:
        return len(self._posterior)

    def keys(self):
        return self._posterior.keys()
