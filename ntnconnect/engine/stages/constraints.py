# Stage: per-tick invariant checks; recompute SINR/SE, write VALIDATION_CHECKS.

from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.common.schemas import VALIDATION_CHECKS

# Use a class attribute (set by engine.setup) for engine ref - hooks
# can't write to state.data["validation_rows"] cleanly across ticks.

@register_stage("val_constraints", internal=True)
class ValConstraints(Stage):
    reads  = ["links"]
    writes = ["validation_rows"]
    thread_safe = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self._tol = 1e-3   # 0.1% relative tolerance on linear quantities

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            state.data["validation_rows"] = []
            return

        rows = []
        def _check(name, category, passed, measured=None, expected=None,
                   tolerance=None, notes=None):
            rows.append({
                "t_idx":          state.t_idx,
                "check_name":     name,
                "category":       category,
                "passed":         int(bool(passed)),
                "measured_value": float(measured) if measured is not None else None,
                "expected_value": float(expected) if expected is not None else None,
                "tolerance":      float(tolerance) if tolerance is not None else None,
                "notes":          notes,
            })

        # Bounds checks (cheap, per-link aggregates)
        if "elev_deg" in links.columns:
            mn, mx = float(links["elev_deg"].min()), float(links["elev_deg"].max())
            _check("elev_in_range", "bounds", -1e-3 <= mn and mx <= 90 + 1e-3,
                   measured=mx, expected=90, tolerance=1e-3,
                   notes=f"min={mn}, max={mx}")
        if "dist_m" in links.columns:
            mn = float(links["dist_m"].min())
            _check("dist_positive", "bounds", mn > 0,
                   measured=mn, expected=0, tolerance=0,
                   notes=f"min={mn}")
        if "chan_gain_lin" in links.columns:
            mn = float(links["chan_gain_lin"].min())
            _check("chan_gain_positive", "physical", mn >= 0,
                   measured=mn, expected=0, tolerance=0)

        # SINR consistency: re-derive sinr_db from signal/interference/noise
        if {"signal_w","interference_w","noise_w","sinr_db"}.issubset(links.columns):
            sig = links["signal_w"].values
            inter = links["interference_w"].values
            noise = links["noise_w"].values
            re_sinr = sig / np.maximum(inter + noise, 1e-30)
            re_sinr_db = 10 * np.log10(np.maximum(re_sinr, 1e-30))
            diff = np.abs(re_sinr_db - links["sinr_db"].values)
            max_diff = float(np.nanmax(diff))
            _check("sinr_db_reproducible", "equivalence", max_diff < 1e-3,
                   measured=max_diff, expected=0, tolerance=1e-3,
                   notes="max |sinr_db_recomputed - sinr_db_engine|")

        # SNR consistency
        if {"signal_w","noise_w","snr_db"}.issubset(links.columns):
            sig = links["signal_w"].values
            noise = links["noise_w"].values
            re_snr_db = 10 * np.log10(np.maximum(sig / np.maximum(noise, 1e-30), 1e-30))
            diff = np.abs(re_snr_db - links["snr_db"].values)
            _check("snr_db_reproducible", "equivalence",
                   float(np.nanmax(diff)) < 1e-3,
                   measured=float(np.nanmax(diff)), expected=0, tolerance=1e-3)

        # Spectral efficiency from SINR
        if {"sinr_db","spectral_efficiency_bps_hz"}.issubset(links.columns):
            sinr_lin = 10 ** (links["sinr_db"].values / 10.0)
            se_re = np.log2(1.0 + sinr_lin)
            diff = np.abs(se_re - links["spectral_efficiency_bps_hz"].values)
            _check("se_shannon_match", "equivalence",
                   float(np.nanmax(diff)) < 1e-3,
                   measured=float(np.nanmax(diff)), expected=0, tolerance=1e-3)

        # Power budget: P_rx <= P_tx * G_tx * G_rx (no path-loss gain)
        # i.e. linear chan_gain_lin * pl_lin <= 1 (no amplification beyond gain).
        if {"path_loss_db","chan_gain_lin"}.issubset(links.columns):
            pl_lin = 10 ** (-links["path_loss_db"].values / 10.0)
            ratio = links["chan_gain_lin"].values * pl_lin
            max_r = float(np.nanmax(ratio))
            _check("no_unphysical_gain", "physical", max_r <= 1.0 + 1e-6,
                   measured=max_r, expected=1.0, tolerance=1e-6,
                   notes="(chan_gain_lin * 10^(-PL/10)) must be <= 1")

        state.data["validation_rows"] = rows
