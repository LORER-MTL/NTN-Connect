# Steering-vector math (single source). Unit-magnitude elements: ||a||^2 = N_t.
from __future__ import annotations
import numpy as np

EARTH_R_M = 6_371_000.0


def off_nadir_rad(elev_deg, sat_alt_m):
    """Spherical-Earth off-nadir angle at the satellite, in radians.
    Array-safe: returns a scalar for scalar input, an ndarray for array input."""
    denom = max(float(EARTH_R_M + sat_alt_m), 1.0)
    return np.arcsin(
        EARTH_R_M * np.cos(np.radians(np.clip(elev_deg, 0.0, 90.0))) / denom)


# Strategies
def steering_upa_kron(theta_rad: float, phi_rad: float,
                      n_x: int, n_y: int,
                      d_over_lambda: float = 0.5) -> np.ndarray:
    """UPA Kronecker steering, unit-magnitude elements (||a||^2 = n_x*n_y).

    a[kx, ky] = exp(j 2π d/λ sin θ (kx cos φ + ky sin φ))
    """
    kx = np.arange(n_x); ky = np.arange(n_y)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    phase = 2.0 * np.pi * d_over_lambda * np.sin(theta_rad) * (
        KX * np.cos(phi_rad) + KY * np.sin(phi_rad))
    return np.exp(1j * phase).ravel()


def steering_ula(theta_rad: float, n_x: int,
                 d_over_lambda: float = 0.5) -> np.ndarray:
    """Uniform Linear Array steering, unit-magnitude elements.

    a[k] = exp(j 2π d/λ sin θ · k)
    """
    k = np.arange(n_x)
    phase = 2.0 * np.pi * d_over_lambda * np.sin(theta_rad) * k
    return np.exp(1j * phase)


def steering_upa_geometric_los(theta_rad: float, phi_rad: float,
                               n_x: int, n_y: int,
                               dist_m: float, carrier_hz: float,
                               d_over_lambda: float = 0.5) -> np.ndarray:
    """UPA Kronecker steering. Identical to `steering_upa_kron`.

    The per-link distance->carrier phase exp(-j 2 pi f d / c) is owned by
    PropagationDelayComponent (the single, correctly-signed source); no bulk
    phase is applied here. `dist_m`/`carrier_hz` are kept only for signature
    back-compat. Enable propagation delay for the distance phase.
    """
    return steering_upa_kron(theta_rad, phi_rad, n_x, n_y, d_over_lambda)


# Registry - picks the strategy at runtime from cfg
_REG: dict[str, callable] = {
    "upa_kron":             steering_upa_kron,
    "ula":                  steering_ula,
    "upa_geometric_los":    steering_upa_geometric_los,
}


def list_steering_strategies() -> list[str]:
    return sorted(_REG.keys())


def get_steering_strategy(name: str) -> callable:
    if name not in _REG:
        raise KeyError(
            f"Unknown steering strategy '{name}'. "
            f"Available: {sorted(_REG)}")
    return _REG[name]


# Per-antenna channel helpers - used directly by models that produce
# their own per-antenna h (produces_per_antenna_h = True).
# Avoids the wasted "scalar Rician + steering component" double-draw.
def per_antenna_rician(K_dB: float, theta_rad: float, phi_rad: float,
                       n_x: int, n_y: int,
                       *,
                       strategy: str = "upa_kron",
                       dist_m: float = 1.0, carrier_hz: float = 20e9,
                       d_over_lambda: float = 0.5,
                       los_amp: complex = 1.0,        # scalar LoS amplitude
                       nlos_amp: float = 1.0,         # extra NLoS amplitude scale
                       rng=None) -> np.ndarray:
    """Per-antenna Rician channel for one link:

        h = sqrt(K/(K+1)) · a_strategy + sqrt(1/(K+1)) · CN(0, I/N_t)

    Single draw of NLoS per call. Strategy comes from cfg.steering_strategy
    via `get_steering_strategy`. Used by models that produce per-antenna h
    directly (Professional_NTN, 3GPP, Spatial_Geometric_Rician, etc).
    """
    if rng is None:
        rng = np.random
    n_tx = n_x * n_y
    K_lin  = 10.0 ** (K_dB / 10.0)
    los_w  = float(np.sqrt(K_lin / (K_lin + 1.0)))
    nlos_w = float(np.sqrt(1.0   / (K_lin + 1.0)))
    strat_fn = get_steering_strategy(strategy)
    if strategy == "upa_kron":
        a = strat_fn(theta_rad, phi_rad, n_x, n_y, d_over_lambda)
    elif strategy == "ula":
        a = strat_fn(theta_rad, n_tx, d_over_lambda)
    elif strategy == "upa_geometric_los":
        a = strat_fn(theta_rad, phi_rad, n_x, n_y,
                     float(dist_m), float(carrier_hz), d_over_lambda)
    else:
        a = strat_fn(theta_rad, phi_rad, n_x, n_y, d_over_lambda)
    nlos = (rng.standard_normal(n_tx) if hasattr(rng, "standard_normal")
            else rng.randn(n_tx)) \
         + 1j * (rng.standard_normal(n_tx) if hasattr(rng, "standard_normal")
                 else rng.randn(n_tx))
    nlos = nlos / np.sqrt(2.0 * n_tx)
    return los_w * los_amp * a + nlos_w * nlos_amp * nlos


def steering_matrix_batch(theta_rad, phi_rad, n_x: int, n_y: int,
                          *, strategy: str = "upa_kron",
                          dist_m=None, carrier_hz: float = 20e9,
                          d_over_lambda: float = 0.5) -> np.ndarray:
    """Just the LoS steering matrix A for all links → (n_links, n_tx), with the
    same phase convention as per_antenna_rician_batch. Factored out so models
    that generate their own (e.g. time-correlated) diffuse term can reuse the
    exact steering math. Unit-magnitude elements (||row||^2 = n_x*n_y)."""
    theta = np.asarray(theta_rad, dtype=float)
    phi   = np.asarray(phi_rad, dtype=float)
    n = theta.shape[0]; n_tx = n_x * n_y
    if strategy in ("upa_kron", "upa_geometric_los"):
        # upa_geometric_los == upa_kron for the array response. The distance phase
        # is owned by PropagationDelayComponent; no bulk phase here.
        KX, KY = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
        KXf = KX.ravel()[None, :]; KYf = KY.ravel()[None, :]
        phase = (2.0*np.pi*d_over_lambda) * np.sin(theta)[:, None] * (
            KXf*np.cos(phi)[:, None] + KYf*np.sin(phi)[:, None])
        A = np.exp(1j*phase)
    elif strategy == "ula":
        A = np.exp(1j*(2.0*np.pi*d_over_lambda)
                   * np.sin(theta)[:, None] * np.arange(n_tx)[None, :])
    else:
        A = np.empty((n, n_tx), dtype=complex)
        sf = get_steering_strategy(strategy)
        for r in range(n):
            A[r, :] = sf(float(theta[r]), float(phi[r]), n_x, n_y, d_over_lambda)
    return A


def per_antenna_rician_batch(K_dB, theta_rad, phi_rad, n_x: int, n_y: int,
                             *, strategy: str = "upa_kron",
                             dist_m=None, carrier_hz: float = 20e9,
                             d_over_lambda: float = 0.5, rng=None,
                             unit_mean: bool = True) -> np.ndarray:
    """Vectorised per-antenna Rician for all links at once → (n_links, n_tx).

    Identical result to calling `per_antenna_rician` in a Python loop - the
    NLoS draw uses one standard_normal((n, 2·n_tx)) whose row-major order
    reproduces the per-link [re(n_tx), im(n_tx)] sequence exactly (bit-exact),
    and the LoS steering phases are closed-form. Replaces the O(n) Python loop
    in the channel models with batched numpy."""
    if rng is None:
        rng = np.random
    theta = np.asarray(theta_rad, dtype=float)
    phi   = np.asarray(phi_rad, dtype=float)
    n = theta.shape[0]; n_tx = n_x * n_y
    K_lin  = 10.0 ** (np.asarray(K_dB, dtype=float) / 10.0)
    los_w  = np.sqrt(K_lin / (K_lin + 1.0))[:, None]
    nlos_w = np.sqrt(1.0   / (K_lin + 1.0))[:, None]
    if strategy in ("upa_kron", "upa_geometric_los"):
        # upa_geometric_los == upa_kron; distance phase owned by
        # PropagationDelayComponent. No bulk phase here.
        KX, KY = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")
        KXf = KX.ravel()[None, :]; KYf = KY.ravel()[None, :]
        phase = (2.0*np.pi*d_over_lambda) * np.sin(theta)[:, None] * (
            KXf*np.cos(phi)[:, None] + KYf*np.sin(phi)[:, None])
        A = np.exp(1j*phase)
    elif strategy == "ula":
        A = np.exp(1j*(2.0*np.pi*d_over_lambda)
                   * np.sin(theta)[:, None] * np.arange(n_tx)[None, :])
    else:
        A = np.empty((n, n_tx), dtype=complex)
        sf = get_steering_strategy(strategy)
        for r in range(n):
            A[r, :] = sf(float(theta[r]), float(phi[r]), n_x, n_y, d_over_lambda)
    # One draw, row-major → matches the per-link [re, im] interleaving exactly.
    buf = (rng.standard_normal((n, 2*n_tx)) if hasattr(rng, "standard_normal")
           else rng.randn(n, 2*n_tx))
    # NLoS normalisation. unit_mean=True (default) gives E[||nlos||^2] = N_t (each
    # element unit variance), so with ||A||^2 = N_t the Rician channel has
    # E[||h||^2] = N_t exactly and the REALIZED per-element K equals the configured
    # K (correct array-Rician convention). unit_mean=False restores the legacy
    # E[||nlos||^2] = 1 convention, which under-powers NLoS by N_t and inflates the
    # realized K by N_t (kept only for bit-exact reproduction of old runs).
    _denom = np.sqrt(2.0) if unit_mean else np.sqrt(2.0 * n_tx)
    nlos = (buf[:, :n_tx] + 1j*buf[:, n_tx:]) / _denom
    return los_w * A + nlos_w * nlos


def per_antenna_rayleigh(theta_rad: float, phi_rad: float,
                         n_x: int, n_y: int,
                         *,
                         amp: float = 1.0,             # total amplitude scale
                         rng=None) -> np.ndarray:
    """Pure Rayleigh per-antenna fading (no LoS direction).

        h = amp · CN(0, I/N_t)

    Used by NTN_TDL/SV in narrowband-collapse mode. The steering vector
    parameters are accepted for API symmetry but unused - Rayleigh has
    no LoS direction to steer.
    """
    if rng is None:
        rng = np.random
    n_tx = n_x * n_y
    re = rng.standard_normal(n_tx) if hasattr(rng, "standard_normal") else rng.randn(n_tx)
    im = rng.standard_normal(n_tx) if hasattr(rng, "standard_normal") else rng.randn(n_tx)
    return amp * (re + 1j * im) / np.sqrt(2.0 * n_tx)


def per_antenna_los_only(theta_rad: float, phi_rad: float,
                         n_x: int, n_y: int,
                         *,
                         strategy: str = "upa_kron",
                         dist_m: float = 1.0, carrier_hz: float = 20e9,
                         d_over_lambda: float = 0.5) -> np.ndarray:
    """Pure-LoS per-antenna channel (no fading). Used by Geometric_Nominal."""
    strat_fn = get_steering_strategy(strategy)
    if strategy == "upa_kron":
        return strat_fn(theta_rad, phi_rad, n_x, n_y, d_over_lambda)
    if strategy == "ula":
        return strat_fn(theta_rad, n_x * n_y, d_over_lambda)
    if strategy == "upa_geometric_los":
        return strat_fn(theta_rad, phi_rad, n_x, n_y,
                        float(dist_m), float(carrier_hz), d_over_lambda)
    return strat_fn(theta_rad, phi_rad, n_x, n_y, d_over_lambda)
