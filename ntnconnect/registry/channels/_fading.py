# Shared diffuse-fading helpers. E[||g||^2]=1. Modes: iid | ar1 | sos (Clarke).
# Models set state["model_did_temporal_fading"] so pipeline Jakes doesn't re-filter.
from __future__ import annotations
import numpy as np
from scipy.special import j0

_C = 299_792_458.0


def doppler_spread_hz(cfg, carrier_hz: float) -> float:
    """Fading Doppler spread f_D = v * f_c / c from the terminal speed."""
    v = float(getattr(cfg, "ue_speed_mps", 0.0))
    return abs(v) * float(carrier_hz) / _C


def white_diffuse(n: int, n_tx: int, rng) -> np.ndarray:
    """White diffuse draw, E[||g||^2] = 1 (per-element variance 1/n_tx)."""
    buf = (rng.standard_normal((n, 2 * n_tx)) if hasattr(rng, "standard_normal")
           else rng.randn(n, 2 * n_tx))
    return (buf[:, :n_tx] + 1j * buf[:, n_tx:]) / np.sqrt(2.0 * n_tx)


def diffuse(mode: str, n: int, n_tx: int, f_d: float, dt: float,
            state, key: str, rng, sos_M: int = 16) -> np.ndarray:
    """Time-correlated diffuse term g of shape (n, n_tx), E[||g||^2] = 1.
    State is keyed by (key, n, n_tx) so a change in the visible-link set resets
    the correlation (same convention as the K-factor / Markov models)."""
    mode = str(mode).lower()
    if mode == "iid" or state is None:
        return white_diffuse(n, n_tx, rng)
    if mode == "ar1":
        kk = f"{key}_ar1_n{n}_{n_tx}"
        w = white_diffuse(n, n_tx, rng)
        prev = state.data.get(kk)
        if prev is None or getattr(prev, "shape", None) != w.shape:
            g = w
        else:
            rho = j0(2.0 * np.pi * f_d * dt) if (f_d > 0.0 and dt > 0.0) else 1.0
            rho = float(np.clip(rho, -0.9999, 0.9999))
            g = rho * prev + np.sqrt(1.0 - rho * rho) * w
        state.data[kk] = g
        return g
    # "sos" - Clarke/Jakes sum-of-sinusoids
    M = max(4, int(sos_M))
    kph = f"{key}_sos_phi_n{n}_{n_tx}_M{M}"
    kt = f"{key}_sos_t_n{n}_{n_tx}_M{M}"
    phi = state.data.get(kph)
    if phi is None or getattr(phi, "shape", None) != (n, n_tx, M):
        phi = (rng.uniform(-np.pi, np.pi, size=(n, n_tx, M))
               if hasattr(rng, "uniform")
               else np.random.uniform(-np.pi, np.pi, size=(n, n_tx, M)))
        t = 0.0
    else:
        t = float(state.data.get(kt, 0.0)) + float(dt)
    alpha = 2.0 * np.pi * (np.arange(M) + 0.5) / M
    osc = np.exp(1j * ((2.0 * np.pi * f_d * t) * np.cos(alpha)[None, None, :] + phi))
    g = osc.sum(axis=2) / np.sqrt(M) / np.sqrt(n_tx)      # E[||g||^2] = 1
    state.data[kph] = phi
    state.data[kt] = t
    return g


def slow_lognormal_amp(mu_db: float, sigma_db: float, n: int, rho: float,
                       state, key: str, rng=None) -> np.ndarray:
    """Slowly-varying LOGNORMAL shadow amplitude (linear), from an AR-1 process
    in the log domain: 20*log10(amp) ~ correlated N(mu_db, sigma_db^2).
    Draws from `rng` (default: state.rng, else np.random) for reproducibility."""
    if rng is None:
        rng = getattr(state, "rng", None) or np.random
    rho = float(np.clip(rho, 0.0, 0.99999))
    kk = f"{key}_slog_n{n}"
    prev = state.data.get(kk) if state is not None else None
    z = rng.standard_normal(n)
    if prev is None or np.asarray(prev).shape != (n,):
        g = z
    else:
        g = rho * prev + np.sqrt(1.0 - rho * rho) * z
    if state is not None:
        state.data[kk] = g
    return 10.0 ** ((mu_db + sigma_db * g) / 20.0)
