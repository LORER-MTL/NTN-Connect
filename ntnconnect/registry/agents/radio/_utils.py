# Shared helpers used by radio algorithms (registered via
# @register_radio_algorithm). Pure numpy / no cvxpy here so the
# module imports cheaply.
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

# Physical constants
KB         = 1.380649e-23
EARTH_R_M  = 6_371_000.0


# Shared safe Cholesky (trace-relative jitter)
def safe_cholesky(Sigma: np.ndarray,
                  rel_jitter: float = 1e-6,
                  abs_jitter: float = 1e-12) -> np.ndarray:
    """Hermitian PSD Cholesky with TRACE-RELATIVE diagonal loading.

    Loading is rel_jitter * max(trace(Sigma)/N, abs_jitter). This avoids
    the failure mode where a fixed jitter (e.g. 1e-9) swamps a Sigma
    whose trace is much smaller than the jitter - typical at low SNR.

    Returns a Cholesky factor L such that L @ L.conj().T = Sigma + jit·I,
    falling back to eigendecomposition if numpy.linalg.cholesky fails.
    """
    Sigma = 0.5 * (Sigma + Sigma.conj().T)
    n = Sigma.shape[0]
    floor = max(float(np.real(np.trace(Sigma))) / max(n, 1), abs_jitter)
    jit   = rel_jitter * floor
    try:
        return np.linalg.cholesky(Sigma + jit * np.eye(n, dtype=Sigma.dtype))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.maximum(vals.real, jit)
        return vecs @ np.diag(np.sqrt(vals))


# Solver fallback chain
def solve_with_fallback(problem, primary: str = "CLARABEL",
                        fallback: str = "SCS", verbose: bool = False):
    """Solve a CVXPY problem with primary solver, falling back to SCS.

    Returns (status, used_solver). `status` is one of
    'optimal', 'optimal_inaccurate' (both treated as success by callers),
    or whatever the failing solver reported. Mirrors the
    `try: CLARABEL except: SCS` pattern from the Colab reference.
    """
    try:
        import cvxpy as cp                            # type: ignore
    except Exception:
        return "no_cvxpy", None
    try:
        primary_solver = getattr(cp, primary, cp.CLARABEL)
        problem.solve(solver=primary_solver, verbose=verbose)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return problem.status, primary
    except Exception:
        pass
    # Fall back to SCS - handles ill-conditioned cases CLARABEL rejects.
    try:
        fallback_solver = getattr(cp, fallback, cp.SCS)
        problem.solve(solver=fallback_solver, verbose=verbose)
        return problem.status or "unknown", fallback
    except Exception as e:
        return f"failed: {e}", None


# Per-call Cholesky cache
# Cached safe_cholesky for hot loops. Keyed by id(Sigma) so untouched
# Sigma arrays (re-used between HO ticks via the channel_predictions
# dict) skip the O(N_tx^3) factorisation. Falls through to safe_cholesky
# when the cache misses.
def cached_safe_cholesky(Sigma: np.ndarray, cache: dict) -> np.ndarray:
    key = id(Sigma)
    L = cache.get(key)
    if L is not None and L.shape[0] == Sigma.shape[0]:
        return L
    L = safe_cholesky(Sigma)
    cache[key] = L
    return L


# Geometry helpers
# Single source of truth lives in ntnconnect.registry.channels._steering.
# These wrappers preserve back-compat for the dozens of existing radio
# algorithm import sites.
from ntnconnect.registry.channels._steering import (
    off_nadir_rad as _off_nadir_rad,
    steering_upa_kron as _steering_upa_kron,
)


def off_nadir_rad(elev_deg: float, sat_alt_m: float) -> float:
    return _off_nadir_rad(elev_deg, sat_alt_m)


def steering_vector(theta_rad: float, phi_rad: float,
                    n_x: int, n_y: int,
                    d_over_lambda: float = 0.5) -> np.ndarray:
    """UPA Kronecker steering. Back-compat alias for canonical helper."""
    return _steering_upa_kron(theta_rad, phi_rad, n_x, n_y, d_over_lambda)


def steering_vector_for(cfg, theta_rad: float, phi_rad: float,
                        n_x: int, n_y: int,
                        dist_m: float = 1.0,
                        d_over_lambda: float = 0.5) -> np.ndarray:
    """Dispatcher that honours cfg.defaults_channel.steering_strategy.

    Used by per-sample MC builders (perfect_csi_mc, stat_csi_mc,
    wasserstein_dt) so a single config switch changes the steering
    strategy EVERYWHERE - channel pipeline AND radio algorithms.
    """
    from ntnconnect.registry.channels._steering import (
        steering_upa_kron, steering_ula, steering_upa_geometric_los,
    )
    strategy = "upa_kron"
    if hasattr(cfg, "defaults_channel"):
        strategy = str(getattr(cfg.defaults_channel, "steering_strategy",
                                "upa_kron")).lower()
    if strategy == "ula":
        return steering_ula(theta_rad, n_x * n_y, d_over_lambda)
    if strategy == "upa_geometric_los":
        carrier_hz = 20.0e9
        if hasattr(cfg, "operators") and cfg.operators:
            try:
                carrier_hz = float(cfg.operators[0].radio.carrier_freq_hz)
            except Exception:
                pass
        return steering_upa_geometric_los(theta_rad, phi_rad, n_x, n_y,
                                          float(dist_m), carrier_hz,
                                          d_over_lambda)
    return steering_upa_kron(theta_rad, phi_rad, n_x, n_y, d_over_lambda)


# Channel matrix builder
def build_channel_matrix(out: pd.DataFrame, rows: dict, user_list: list,
                         n_x: int, n_y: int, d_lambda: float,
                         sat_alt_m: float) -> np.ndarray:
    """Build H ∈ C^{N_t × k} for one sat.

    Each column is the per-antenna channel vector for one user:
        h_{s,u} = sqrt(chan_gain_lin * 10^{-PL/10}) · steering(theta, phi)

    Args:
        out:        the `links` frame (mutated nowhere).
        rows:       {user_id: row index in `out`} for users on this sat.
        user_list:  ordered user ids, len(user_list) == k.
        n_x, n_y:   UPA dimensions.
        d_lambda:   antenna spacing in wavelengths (0.5 by default).
        sat_alt_m:  satellite altitude (used for off-nadir angle).
    """
    n_tx = n_x * n_y
    elev = out.get("elev_deg", pd.Series(np.zeros(len(out)))).astype(float).to_numpy()
    # BUG #6 FIX: geometry writes "azimuth_deg"; fall back to "azimuth"
    # for back-compat, and to zeros only if neither exists.
    if "azimuth_deg" in out.columns:
        az = out["azimuth_deg"].astype(float).to_numpy()
    elif "azimuth" in out.columns:
        az = out["azimuth"].astype(float).to_numpy()
    else:
        az = np.zeros(len(out), dtype=float)
    g    = out.get("chan_gain_lin", pd.Series(np.ones(len(out)))).astype(float).to_numpy()
    pl   = out.get("path_loss_db",  pd.Series(np.zeros(len(out)))).astype(float).to_numpy()

    H = np.zeros((n_tx, len(user_list)), dtype=complex)
    for col, u in enumerate(user_list):
        r = rows[u]
        theta_rad = off_nadir_rad(elev[r], sat_alt_m)
        phi_rad   = float(np.radians(az[r]))
        a_vec     = steering_vector(theta_rad, phi_rad, n_x, n_y, d_lambda)
        amp       = float(np.sqrt(max(g[r] * 10.0 ** (-pl[r] / 10.0), 1e-30)))
        H[:, col] = amp * a_vec
    return H


# Per-sat grouping (a tiny iterator for radio algorithms)
def per_sat_groups(out: pd.DataFrame, assigns: dict):
    """Yield (sat_id, rows_per_user, user_list) for every sat serving >=1 user.

    rows_per_user : {user_id -> row index in `out`}
    user_list     : ordered list of user ids on this sat
    """
    n_arr = out["node_id"].astype(int).to_numpy()
    u_arr = out["user_id"].astype(int).to_numpy()
    served: dict = {}
    for u, s in assigns.items():
        if int(s) >= 0:
            served.setdefault(int(s), []).append(int(u))
    for sat_id, uids in served.items():
        rows = {}
        for u in uids:
            m   = (n_arr == sat_id) & (u_arr == u)
            idx = np.where(m)[0]
            if len(idx): rows[u] = int(idx[0])
        if rows:
            yield sat_id, rows, list(rows.keys())


# Final radio-column writer
def finalize_radio_columns(out: pd.DataFrame,
                           tx_pow: np.ndarray,
                           sig_w: np.ndarray,
                           intra_w: np.ndarray,
                           status: np.ndarray,
                           cfg,
                           fixed_ici_w: float = 0.0,
                           interference_model_name: str = "intra_only") -> pd.DataFrame:
    """Compute SINR/rate from (signal, intra, noise, fixed_ici_w) and write
    every radio column on `out`.  Returns the same `out` (mutated)."""
    cfg_op  = cfg.operators[0]
    noise_w = (KB
               * float(cfg.defaults_radio.system_temperature_k)
               * 10.0 ** (cfg.defaults_radio.noise_figure_db / 10.0)
               * float(cfg_op.radio.bandwidth_hz))
    N_bar    = noise_w + fixed_ici_w
    pl       = out.get("path_loss_db", pd.Series(np.zeros(len(out)))).astype(float).to_numpy()
    g        = out.get("chan_gain_lin", pd.Series(np.ones(len(out)))).astype(float).to_numpy()
    rx_lin   = 10.0 ** (float(cfg_op.radio.antenna_gain_rx_db) / 10.0)
    bw       = float(cfg_op.radio.bandwidth_hz)

    sinr_lin = sig_w / np.maximum(N_bar + intra_w, 1e-30)
    snr_lin  = sig_w / np.maximum(noise_w, 1e-30)
    out["tx_power_w"]                 = tx_pow
    out["signal_w"]                   = sig_w
    out["noise_w"]                    = noise_w
    out["interference_w"]             = intra_w + fixed_ici_w
    out["interference_intra_w"]       = intra_w
    out["interference_ext_w"]         = fixed_ici_w
    out["snr_db"]                     = 10.0 * np.log10(np.maximum(snr_lin, 1e-30))
    out["sinr_db"]                    = 10.0 * np.log10(np.maximum(sinr_lin, 1e-30))
    out["spectral_efficiency_bps_hz"] = np.log2(1.0 + sinr_lin)
    out["link_rate_bps"]              = out["spectral_efficiency_bps_hz"] * bw
    out["interference_model"]         = interference_model_name
    out["radio_status"]               = status
    # Effective BF gain (array gain over a single isotropic element).
    # Reference (single-antenna received power for same tx_pow) is
    #  P_single = tx_pow · chan_gain_lin · 10^(-PL/10) · rx_gain_lin
    # so bf_gain_db = 10·log10(signal_w / P_single). This matches the
    # conventional MU-MIMO array-gain definition (positive when the
    # precoder steers energy into the user's direction).
    single_ant_rx_w = (tx_pow
                       * np.maximum(g, 1e-30)
                       * 10.0 ** (-pl / 10.0)
                       * rx_lin)
    out["bf_gain_db"] = 10.0 * np.log10(
        np.maximum(sig_w / np.maximum(single_ant_rx_w, 1e-30), 1e-30)
    )
    return out
