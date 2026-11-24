"""Canonical link-metric math - the single source of truth.

Previously the SINR / spectral-efficiency / rate / energy-efficiency formulas
were copy-pasted across `sinr_calc`, `link_budget`, `paper_sinr`, and
`scn_radio_link`. They now all call these functions, so a change to the physics
is made in one place and cannot drift between stages.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

_EPS = 1e-30
# Floor used when converting SINR(dB) back to linear for log2(1+·): keeps
# spectral efficiency finite for deeply-faded links (matches the historical
# `.clip(min=-30)` behaviour shared by the stages).
_SINR_DB_FLOOR = -30.0


_C_M_PER_S = 299_792_458.0     # speed of light (exact, SI)


def propagation_delay_ns(distance_m, round_trip: bool = False) -> np.ndarray:
    """Physical propagation delay as INTEGER nanoseconds: tau = distance / c
    (round_trip=True doubles it for a feedback RTT). This is the real-physics
    timing that drives CSI availability - sub-us for UAV/ground, ~ms for LEO,
    ~119 ms one-way for GEO. Integer ns so it composes exactly with the sim clock."""
    d = np.asarray(distance_m, dtype=float)
    one_way_s = d / _C_M_PER_S
    return np.round((2.0 if round_trip else 1.0) * one_way_s * 1e9).astype(np.int64)


def channel_power_gain(links, operators) -> np.ndarray:
    """Per-link linear channel power gain ||h||^2 per unit Tx power:

           ||h||^2 = 10^(G_db/10) * 10^(-PL_db/10) * fading_norm

    the single definition of the effective channel gain used by both the power
    allocator (min_power_miso) and the SINR calculator (sinr_calc). Because both
    stages consume the exact same number, the achieved SINR equals the allocation
    target gamma to machine precision.

    General for ANY antenna configuration - SISO, MISO, MU-MIMO - because the
    normalisation is driven purely by the CONFIGURED array size, not by any
    magnitude heuristic:

      * The channel stage writes ``chan_gain_lin`` = Σ|h|^2 over the N_t transmit
        elements. For an N_t-element array this ≈ N_t (the coherent array/
        beamforming gain is baked in); for SISO (N_t = 1) it is just |h|^2.
      * ``antenna_gain_db`` already carries the boresight array gain, so we divide
        ``chan_gain_lin`` by the configured N_t = n_tx_x·n_tx_y to recover the
        NORMALISED small-scale fading (mean ≈ 1). SISO keeps |h|^2 intact
        (including deep fades - no thresholding); a 64-element UPA divides out the
        64× array gain. No fading pipeline (column absent) → fading = 1.

    Rows carrying the channel stage's default ``chan_gain_lin = 1.0`` (no
    realisation, e.g. non-served rows in served-only mode) are left at fading = 1
    so they are never spuriously attenuated by the array-size divisor.
    """
    import pandas as _pd
    n = len(links)
    g_db  = links.get("antenna_gain_db", _pd.Series(np.zeros(n))).astype(float).to_numpy()
    pl_db = links.get("path_loss_db",    _pd.Series(np.zeros(n))).astype(float).to_numpy()
    gain = 10.0 ** (g_db / 10.0) * 10.0 ** (-pl_db / 10.0)
    if "chan_gain_lin" in links.columns:
        cg = links["chan_gain_lin"].astype(float).to_numpy()
        op_ids = links.get("operator_id",
                           _pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
        ntx = {}
        for op in operators:
            cm = getattr(op, "channel_model", None)
            ntx[int(op.id)] = (float(int(getattr(cm, "n_tx_x", 1))
                                     * int(getattr(cm, "n_tx_y", 1)))
                               if cm is not None else 1.0)
        n_tx = np.array([ntx.get(int(o), 1.0) for o in op_ids], dtype=float)
        # Normalise by configured N_t; leave the default-1.0 (no-realisation) rows
        # untouched so they are not divided down by the array size.
        fading = np.where(cg == 1.0, 1.0, cg / np.maximum(n_tx, 1.0))
        gain = gain * fading
    return gain


def sinr_db_from_powers(signal_w, noise_w, i_ext_w=0.0) -> np.ndarray:
    """SINR(dB) = 10·log10( S / (N + I_ext) )."""
    signal_w = np.asarray(signal_w, dtype=float)
    denom = np.maximum(np.asarray(noise_w, dtype=float)
                       + np.asarray(i_ext_w, dtype=float), _EPS)
    return 10.0 * np.log10(np.maximum(signal_w / denom, _EPS))


def snr_db_from_powers(signal_w, noise_w) -> np.ndarray:
    """SNR(dB) = 10·log10( S / N )  (interference excluded)."""
    signal_w = np.asarray(signal_w, dtype=float)
    noise_w = np.maximum(np.asarray(noise_w, dtype=float), _EPS)
    return 10.0 * np.log10(np.maximum(signal_w / noise_w, _EPS))


def spectral_efficiency_bps_hz(sinr_db) -> np.ndarray:
    """Shannon SE = log2(1 + SINR_lin), with the shared low-SINR floor."""
    sinr_db = np.clip(np.asarray(sinr_db, dtype=float), _SINR_DB_FLOOR, None)
    return np.log2(1.0 + 10.0 ** (sinr_db / 10.0))


def mimo_spectral_efficiency(sinr_db, n_streams=1) -> np.ndarray:
    """SE for a link with N_s spatial streams (MIMO spatial multiplexing):

        SE = N_s * log2(1 + SINR / N_s)

    the equal-power parallel-stream capacity - total power split across N_s
    streams, each seeing SINR/N_s. Correct reductions across antenna configs:
      * N_s = 1  ->  log2(1 + SINR)  = exactly spectral_efficiency_bps_hz.
        This is SISO, MISO (Tx beamforming gain already in SINR) and SIMO (Rx
        combining gain already in SINR) - all single-stream.
      * N_s > 1  ->  MIMO multiplexing gain (only when spatial_streams is set).
    n_streams may be a scalar or a per-link array; it is floored at 1."""
    sinr_db = np.clip(np.asarray(sinr_db, dtype=float), _SINR_DB_FLOOR, None)
    sinr_lin = 10.0 ** (sinr_db / 10.0)
    ns = np.maximum(np.asarray(n_streams, dtype=float), 1.0)
    return ns * np.log2(1.0 + sinr_lin / ns)


def spatial_streams(links, operators) -> np.ndarray:
    """Per-link MIMO multiplexing order N_s, derived purely from the configured
    antenna array + mode (never a magnitude heuristic):

        N_s = clip(operator.spatial_streams, 1, min(N_t, N_r))

    where N_t = n_tx_x*n_tx_y, N_r = n_rx. Defaults to 1 (single-stream) for
    every operator that does not set spatial_streams, so SISO/MISO/SIMO and any
    MIMO array without explicit multiplexing keep the log2(1+SINR) SE and results
    are unchanged. antenna_mode="siso"/"miso"/"simo" also forces N_s = 1."""
    import pandas as _pd
    n = len(links)
    op_ids = links.get("operator_id",
                       _pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
    ns_by_op = {}
    for op in operators:
        cm = getattr(op, "channel_model", None)
        if cm is None:
            ns_by_op[int(op.id)] = 1.0
            continue
        n_t = max(1, int(getattr(cm, "n_tx_x", 1)) * int(getattr(cm, "n_tx_y", 1)))
        n_r = max(1, int(getattr(cm, "n_rx", 1)))
        mode = str(getattr(cm, "antenna_mode", "auto")).lower()
        req = int(getattr(cm, "spatial_streams", 1))
        if mode in ("siso", "miso", "simo"):
            ns = 1
        else:                                    # auto / mimo
            ns = max(1, min(req, n_t, n_r))
        ns_by_op[int(op.id)] = float(ns)
    return np.array([ns_by_op.get(int(o), 1.0) for o in op_ids], dtype=float)


def served_mask_from_assignments(links, assigns) -> Optional[np.ndarray]:
    """Boolean mask of rows whose (user_id,node_id) is the served assignment.
    Returns None if the columns/assignments are unavailable."""
    if not assigns or not {"user_id", "node_id"}.issubset(getattr(links, "columns", [])):
        return None
    uid = links["user_id"].astype(int).to_numpy()
    nid = links["node_id"].astype(int).to_numpy()
    served_id = np.array([assigns.get(int(u), -1) for u in uid])
    return served_id == nid


def served_link_metrics(signal_w, noise_w, i_ext_w, bw, served_mask,
                        *, unserved_sinr_db: float = -300.0,
                        n_streams=None) -> dict:
    """Compute SINR/SE/rate/SNR, masking non-served rows to ``unserved_sinr_db``.

    Returns a dict of np.ndarray columns: sinr_db, sinr_db_full,
    spectral_efficiency_bps_hz, link_rate_bps, snr_db.

    n_streams (optional, per-link or scalar) is the MIMO spatial-multiplexing
    order N_s. Default None == 1 == single-stream, so SE = log2(1+SINR) and
    SISO/MISO/SIMO results are unchanged; N_s>1 gives the MIMO multiplexing SE.
    """
    signal_w = np.asarray(signal_w, dtype=float)
    n = signal_w.shape[0]
    sinr_full = sinr_db_from_powers(signal_w, noise_w, i_ext_w)
    if served_mask is None:
        served_mask = np.ones(n, dtype=bool)
    sinr_col = np.full(n, float(unserved_sinr_db))
    sinr_col[served_mask] = sinr_full[served_mask]
    se = (spectral_efficiency_bps_hz(sinr_col) if n_streams is None
          else mimo_spectral_efficiency(sinr_col, n_streams))
    bw = np.asarray(bw, dtype=float)
    return {
        "sinr_db": sinr_col,
        "sinr_db_full": sinr_full,
        "spectral_efficiency_bps_hz": se,
        "link_rate_bps": se * bw,
        "snr_db": snr_db_from_powers(signal_w, noise_w),
    }


def energy_kpis(p_tx, rate_bps, signal_w, spectral_eff, served_mask,
                *, eta_pa: float = 0.35, p_circ_w: float = 10.0,
                se_ref_bps_hz: float = 6.0) -> dict:
    """Generalised radio KPIs (served links only; NaN elsewhere).

    power_consumption_w = P_tx/eta_pa + P_circuit; energy_efficiency = rate/P.
    """
    p_tx = np.asarray(p_tx, dtype=float)
    rate_bps = np.asarray(rate_bps, dtype=float)
    signal_w = np.asarray(signal_w, dtype=float)
    spectral_eff = np.asarray(spectral_eff, dtype=float)
    p_cons = p_tx / max(eta_pa, 1e-3) + p_circ_w
    nanmask = np.where(served_mask, 1.0, np.nan) if served_mask is not None else 1.0
    return {
        "power_consumption_w": p_cons * nanmask,
        "energy_efficiency_bpj": (rate_bps / np.maximum(p_cons, 1e-9)) * nanmask,
        "tx_power_dbw": (10.0 * np.log10(np.maximum(p_tx, 1e-12))) * nanmask,
        "rx_signal_dbw": (10.0 * np.log10(np.maximum(signal_w, _EPS))) * nanmask,
        "throughput_mbps": (rate_bps / 1e6) * nanmask,
        "spectral_eff_util_pct": (np.clip(spectral_eff / se_ref_bps_hz, 0, 1) * 100.0) * nanmask,
    }
