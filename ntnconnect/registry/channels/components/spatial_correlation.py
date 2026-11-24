# Kronecker spatial correlation: H_corr = R_rx^1/2 H R_tx^1/2. Off by default.
# Limitation: colors the whole h incl. LoS; exact only for pure-diffuse. Warns on
# Rician h. The correct fix is to apply R_tx^1/2 to the diffuse term in-model.
from __future__ import annotations
import logging
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent

_log = logging.getLogger(__name__)


def _exp_corr_matrix(n: int, rho: float) -> np.ndarray:
    if n <= 1 or abs(rho) < 1e-12:
        return np.eye(n)
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    return rho ** np.abs(i - j)


def _sqrtm_hermitian(M: np.ndarray) -> np.ndarray:
    # Cholesky is faster but requires PD; the exponential correlation matrix
    # is PD for |rho|<1.
    w, V = np.linalg.eigh(M)
    return V @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ V.conj().T


class SpatialCorrelationComponent(AbstractChannelComponent):
    name  = "spatial_correlation"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if data.h is None: return
        model = (cfg.spatial_corr_model or "iid").lower()
        if model == "iid": return
        rho_tx = float(getattr(cfg, "tx_corr_rho", 0.0))
        rho_rx = float(getattr(cfg, "rx_corr_rho", 0.0))
        if abs(rho_tx) < 1e-12 and abs(rho_rx) < 1e-12: return

        if data.h.ndim == 1:
            # Scalar per link - no spatial dimensions to correlate.
            return

        if data.h.ndim == 2:
            # per-antenna h shape (n_links, n_tx). Apply Tx-side correlation
            # only (no Rx array dimension in this view). This colors the whole h,
            # including the LoS steering vector, which is physically wrong for
            # Rician channels; correct only for pure-diffuse/low-K.
            _log.warning(
                "SpatialCorrelationComponent colors the combined per-antenna h, "
                "including any LoS steering vector. Exact only for pure-diffuse/"
                "low-K channels; for Rician links this distorts the LoS "
                "direction/gain. Prefer in-model diffuse-only correlation.")
            n_links, n_tx = data.h.shape
            R_tx   = _exp_corr_matrix(n_tx, rho_tx)
            Rtx_sq = _sqrtm_hermitian(R_tx)
            data.h = data.h @ Rtx_sq
            data.extras["spatial_corr_applied"] = "tx_kronecker"
            return

        if data.h.ndim == 3:
            n_links, n_rx, n_tx = data.h.shape
            R_tx   = _exp_corr_matrix(n_tx, rho_tx)
            R_rx   = _exp_corr_matrix(n_rx, rho_rx)
            Rtx_sq = _sqrtm_hermitian(R_tx)
            Rrx_sq = _sqrtm_hermitian(R_rx)
            out = np.empty_like(data.h)
            for i in range(n_links):
                out[i] = Rrx_sq @ data.h[i] @ Rtx_sq
            data.h = out
            data.extras["spatial_corr_applied"] = "txrx_kronecker"
