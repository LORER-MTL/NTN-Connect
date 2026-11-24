"""Built-in physical-sanity validators (registered on import).

Every check is independent and selectable by name. Each takes the served-link
snapshot DataFrame ``df`` and an ``expected`` dict, and returns a CheckResult.
A scenario picks the subset it cares about:

    run_validators(["served_links_present", "noise_floor", "sinr_range"], df, exp)

`expected` keys consumed (all optional - a validator that lacks its reference is
reported as passed=True with detail {"skipped": "..."} so it never blocks):
    noise_dbw, noise_tol_db, gain_db, gain_tol_db, pl_range, sinr_range
"""
from __future__ import annotations
import numpy as np
from ntnconnect.common.validation.registry import register_validator, ok


def _med(df, col):
    import pandas as pd
    if col not in getattr(df, "columns", []) or len(df) == 0:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").median())


@register_validator("served_links_present", category="coverage")
def served_links_present(df, expected):
    """At least one served link survived (catches a dead pipeline)."""
    n = int(len(df))
    return ok("served_links_present", n > 0, {"n_served": n})


@register_validator("noise_floor", category="radio")
def noise_floor(df, expected):
    """Median noise power matches kTB·NF from the link budget (catches the
    1 Hz-noise operator-id bug)."""
    if "noise_dbw" not in expected:
        return ok("noise_floor", True, {"skipped": "no expected noise_dbw"})
    n_w = _med(df, "noise_w")
    if not np.isfinite(n_w) or n_w <= 0:
        return ok("noise_floor", False, {"error": "no noise_w column / data"})
    nd = float(10.0 * np.log10(n_w))
    tol = float(expected.get("noise_tol_db", 1.5))
    return ok("noise_floor", abs(nd - expected["noise_dbw"]) <= tol,
              {"noise_dbw": round(nd, 1), "expected_dbw": round(float(expected["noise_dbw"]), 1),
               "tol_db": tol})


@register_validator("antenna_gain", category="radio")
def antenna_gain(df, expected):
    """Median served antenna gain equals G_tx + G_rx from the link budget."""
    if "gain_db" not in expected:
        return ok("antenna_gain", True, {"skipped": "no expected gain_db"})
    g = _med(df, "antenna_gain_db")
    if not np.isfinite(g):
        return ok("antenna_gain", True, {"skipped": "no antenna_gain_db column"})
    tol = float(expected.get("gain_tol_db", 0.5))
    return ok("antenna_gain", abs(g - expected["gain_db"]) <= tol,
              {"gain_db": round(g, 1), "expected_db": round(float(expected["gain_db"]), 1),
               "tol_db": tol})


@register_validator("path_loss_range", category="radio")
def path_loss_range(df, expected):
    """Median path loss lies in a physical band (catches 0 dB / runaway PL)."""
    lo, hi = expected.get("pl_range", (120.0, 210.0))
    pl = _med(df, "path_loss_db")
    if not np.isfinite(pl):
        return ok("path_loss_range", True, {"skipped": "no path_loss_db column"})
    return ok("path_loss_range", lo <= pl <= hi,
              {"path_loss_db": round(pl, 1), "range": [lo, hi]})


@register_validator("sinr_range", category="radio")
def sinr_range(df, expected):
    """Median served SINR lies in a sane range."""
    lo, hi = expected.get("sinr_range", (-30.0, 45.0))
    sv = _med(df, "sinr_db")
    if not np.isfinite(sv):
        return ok("sinr_range", True, {"skipped": "no sinr_db column"})
    return ok("sinr_range", lo <= sv <= hi,
              {"sinr_db": round(sv, 1), "range": [lo, hi]})
