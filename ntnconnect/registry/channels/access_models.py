"""Access path-loss models by node_type (registry). Returns total PL [dB];
the channel stage adds it as excess over FSPL. satellite->pipeline,
base_station->TR 38.901 UMa/RMa, haps/uav->TR 38.811 aerial, ground_station->FSPL."""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np

_C = 299_792_458.0

# name -> fn(dist_m, elev_deg, f_hz) -> total_pl_db (np.ndarray)
_ACCESS_MODELS: dict[str, Callable] = {}

# Which model each node_type uses by default. None => use the channel pipeline
# (satellites) / leave at plain FSPL. A scenario can override via
# ChannelConfig.access_models.
DEFAULT_ACCESS_MODEL_BY_NODE_TYPE: dict[str, Optional[str]] = {
    "satellite":      None,
    "base_station":   "terrestrial_uma",
    "haps":           "aerial_38811",
    "uav":            "aerial_38811",
    "ground_station": None,
}


def register_access_model(name: str):
    def deco(fn: Callable) -> Callable:
        _ACCESS_MODELS[name] = fn
        return fn
    return deco


def get_access_model(name: str) -> Optional[Callable]:
    return _ACCESS_MODELS.get(name)


def list_access_models() -> list[str]:
    return sorted(_ACCESS_MODELS)


def fspl_db(dist_m, f_hz) -> np.ndarray:
    d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    f = np.asarray(f_hz, dtype=float)
    return (20.0 * np.log10(d) + 20.0 * np.log10(f)
            + 20.0 * np.log10(4.0 * np.pi / _C))


# 3GPP TR 38.901 terrestrial (simplified single-slope, hbs=25 m, hut=1.5 m).
# fc in GHz, distances in m. These are the standard UMa/RMa large-scale forms.
@register_access_model("terrestrial_uma")
def terrestrial_uma(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Urban Macro NLoS (38.901 Table 7.4.1-1) - conservative urban BS."""
    d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    fc = np.asarray(f_hz, dtype=float) / 1e9
    return 13.54 + 39.08 * np.log10(d) + 20.0 * np.log10(fc)


@register_access_model("terrestrial_uma_los")
def terrestrial_uma_los(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Urban Macro LoS (38.901), pre-breakpoint slope."""
    d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    fc = np.asarray(f_hz, dtype=float) / 1e9
    return 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(fc)


@register_access_model("terrestrial_rma")
def terrestrial_rma(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Rural Macro NLoS (38.901), simplified - sparser, lower clutter."""
    d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    fc = np.asarray(f_hz, dtype=float) / 1e9
    h = 5.0   # avg building height (m), rural
    return (161.04 - 7.1 * np.log10(20.0) + 7.5 * np.log10(h)
            - (24.37 - 3.7 * (h / 35.0) ** 2) * np.log10(35.0)
            + (43.42 - 3.1 * np.log10(35.0)) * (np.log10(d) - 3.0)
            + 20.0 * np.log10(fc) - (3.2 * (np.log10(11.75 * 1.5)) ** 2 - 4.97))


# 3GPP TR 38.901 UMa - full form: LoS probability, breakpoint dual-slope LoS,
# NLoS, optional O2I building penetration. Heights: h_bs=25 m, h_ut=1.5 m.
# These are *expected* (probability-blended) path losses - deterministic and
# array-safe, so runs stay reproducible (no per-link LoS coin-flip).
_H_BS_M = 25.0
_H_UT_M = 1.5


def _d2d_d3d(dist_m, h_bs=_H_BS_M, h_ut=_H_UT_M):
    """Split slant range into (d2D, d3D). dist_m is treated as the 3D distance;
    d2D is recovered from the BS/UT height difference (clamped non-negative)."""
    d3d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    dh = abs(h_bs - h_ut)
    d2d = np.sqrt(np.maximum(d3d ** 2 - dh ** 2, 1.0))
    return d2d, d3d


def _uma_los_prob(d2d, h_ut=_H_UT_M):
    """38.901 Table 7.4.2-1 UMa LoS probability."""
    d = np.asarray(d2d, dtype=float)
    cprime = np.where(h_ut <= 13.0, 0.0, ((h_ut - 13.0) / 10.0) ** 1.5)
    base = (18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d))
    factor = (1.0 + cprime * (5.0 / 4.0) * (d / 100.0) ** 3 * np.exp(-d / 150.0))
    p = np.where(d <= 18.0, 1.0, base * factor)
    return np.clip(p, 0.0, 1.0)


def _uma_pl_los(d2d, d3d, fc_ghz, h_bs=_H_BS_M, h_ut=_H_UT_M):
    """38.901 UMa LoS dual-slope with breakpoint distance d_BP."""
    he = 1.0   # effective environment height (m), 38.901 note 1 (UMa)
    d_bp = 4.0 * (h_bs - he) * (h_ut - he) * (fc_ghz * 1e9) / _C
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.0 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def _uma_pl_nlos(d3d, fc_ghz, h_ut=_H_UT_M):
    """38.901 UMa NLoS (already the max with LoS per the standard)."""
    return 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)


@register_access_model("terrestrial_uma_full")
def terrestrial_uma_full(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Full 38.901 UMa: LoS-probability-blended dual-slope LoS + NLoS."""
    fc = np.asarray(f_hz, dtype=float) / 1e9
    d2d, d3d = _d2d_d3d(dist_m)
    p_los = _uma_los_prob(d2d)
    pl_los = _uma_pl_los(d2d, d3d, fc)
    pl_nlos = np.maximum(_uma_pl_nlos(d3d, fc), pl_los)
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


@register_access_model("terrestrial_uma_o2i")
def terrestrial_uma_o2i(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Full 38.901 UMa + low-loss O2I building penetration (38.901 §7.4.3.1)."""
    fc = np.asarray(f_hz, dtype=float) / 1e9
    base = terrestrial_uma_full(dist_m, elev_deg, f_hz)
    # Low-loss model PL_tw, plus indoor distance loss with d_2D_in ~ 12.5 m.
    pl_tw = 5.0 - 10.0 * np.log10(0.3 * 10.0 ** (-2.0 * fc / 10.0)
                                  + 0.7 * 10.0 ** (-0.2 / 10.0))
    pl_in = 0.5 * 12.5
    return base + pl_tw + pl_in


# 3GPP TR 38.811 aerial (HAPS / UAV): essentially free-space plus an
# elevation-dependent shadow-fading margin (more shadowing at low elevation).
# Shadow-fading std σ (dB) vs elevation (deg), NTN suburban (38.811 Table 6.7.2).
_SF_ELEV_DEG = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
_SF_SIGMA_DB = np.array([3.6, 2.9, 2.5, 2.2, 1.9, 1.6, 1.3, 1.0, 0.8], dtype=float)


@register_access_model("aerial_38811")
def aerial_38811(dist_m, elev_deg, f_hz) -> np.ndarray:
    """FSPL + elevation-dependent shadow-fading margin (38.811 aerial)."""
    base = fspl_db(dist_m, f_hz)
    elev = np.asarray(elev_deg, dtype=float)
    elev = np.clip(np.nan_to_num(elev, nan=45.0), 0.0, 90.0)
    sigma = np.interp(elev, _SF_ELEV_DEG, _SF_SIGMA_DB)
    return base + sigma   # add σ as a deterministic link margin


@register_access_model("free_space")
def free_space(dist_m, elev_deg, f_hz) -> np.ndarray:
    """Plain FSPL (excess = 0)."""
    return fspl_db(dist_m, f_hz)


def excess_loss_db(node_types, dist_m, elev_deg, f_hz,
                   mapping: Optional[dict] = None) -> np.ndarray:
    """Per-link EXCESS loss over FSPL, chosen by node_type.

    Returns an array the length of node_types. Satellite / unmapped node types
    get 0 (their loss comes from the channel pipeline). For mapped node types,
    excess = max(0, model_total - fspl), applied row-wise. ``f_hz`` may be a
    scalar or a per-row array.
    """
    nt = np.asarray(node_types, dtype=object)
    n = len(nt)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    dist = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
    elev = (np.asarray(elev_deg, dtype=float)
            if elev_deg is not None else np.full(n, 45.0))
    fhz = np.asarray(f_hz, dtype=float)
    if fhz.ndim == 0:
        fhz = np.full(n, float(fhz))
    table = dict(DEFAULT_ACCESS_MODEL_BY_NODE_TYPE)
    if mapping:
        table.update(mapping)
    base_fspl = fspl_db(dist, fhz)
    for t in np.unique(nt):
        model_name = table.get(str(t))
        if not model_name:
            continue
        fn = _ACCESS_MODELS.get(model_name)
        if fn is None:
            continue
        sel = (nt == t)
        total = np.asarray(fn(dist[sel], elev[sel], fhz[sel]), dtype=float)
        out[sel] = np.maximum(0.0, total - base_fspl[sel])
    return out
