"""
Figure catalog for the SkyNTNPro analysis  (additive, standalone).
============================================================================
Tags every built-in analysis figure with a professional 3-axis category
following the 3GPP TR 38.913 / ITU-R KPI families, so figures can be selected
by category instead of by hand-listing names.

  kpi    : coverage | capacity | latency | reliability | mobility | energy |
           connection | system
  scope  : link | user | satellite | network | system
  stage  : geometry (dataset-only, NO simulation) | sim (stage-2) | meta

This module is purely additive. It only RESOLVES category selectors down to the
existing figure NAMES that `analysis.core.run_analysis` already accepts, so
nothing in the analysis pipeline OR the scenario studies needs to change.

Usage
-----
    from ntnconnect.analysis.figure_catalog import list_figures, resolve, groups
    list_figures(kpi="coverage")                  # -> ['fig1_visibility_ccdf', ...]
    list_figures(stage="geometry")                # dataset-only figures (no run)
    list_figures(scope="link", kpi="capacity")    # combine filters
    resolve(["kpi:coverage", "fig13_sinr_ccdf"])  # expand selectors + keep names

Note: `energy` and `connection` KPI families are defined for completeness; their
system-level figures (total transmit power, admission rate, aggregate throughput)
are produced by the study's own paper plotter, not the built-in figure set.
"""
from __future__ import annotations

# name -> (kpi, scope, stage). Handover/mobility is a user/network sub-family of
# the sim stage -- it is not a top-level class; the pipeline runs without it.
FIGURE_CATALOG = {
    # COVERAGE (visibility / availability)
    "fig1_visibility_ccdf":              {"kpi": "coverage",    "scope": "user",    "stage": "geometry"},
    "fig2_coverage_bar":                 {"kpi": "coverage",    "scope": "network", "stage": "geometry"},
    "fig3_mean_nvis_vs_lat":             {"kpi": "coverage",    "scope": "network", "stage": "geometry"},
    "fig4_visibility_gap_cdf":           {"kpi": "coverage",    "scope": "user",    "stage": "geometry"},
    "fig11_service_gap_cdf":             {"kpi": "coverage",    "scope": "user",    "stage": "geometry"},
    "fig19_availability_bar":            {"kpi": "coverage",    "scope": "network", "stage": "sim"},
    "fig24_visgap_ccdf_per_user_overlay":{"kpi": "coverage",    "scope": "user",    "stage": "sim"},
    "fig_elev_visibility_ccdf":          {"kpi": "coverage",    "scope": "user",    "stage": "geometry"},
    "fig_elev_coverage_gap_cdf":         {"kpi": "coverage",    "scope": "network", "stage": "geometry"},

    # CAPACITY (SINR / spectral efficiency / rate)
    "fig13_sinr_ccdf":                   {"kpi": "capacity",    "scope": "link",    "stage": "sim"},
    "fig14_sinr_vs_elev":                {"kpi": "capacity",    "scope": "link",    "stage": "sim"},
    "fig19_rate_ccdf":                   {"kpi": "capacity",    "scope": "link",    "stage": "sim"},
    "fig20_sinr_ccdf_per_user_overlay":  {"kpi": "capacity",    "scope": "user",    "stage": "sim"},
    "fig21_sinr_ccdf_per_user_grid":     {"kpi": "capacity",    "scope": "user",    "stage": "sim"},
    "fig27_rate_ccdf_per_user_overlay":  {"kpi": "capacity",    "scope": "user",    "stage": "sim"},
    "fig28_avg_rate_bar":                {"kpi": "capacity",    "scope": "user",    "stage": "sim"},
    "fig29_avg_rate_bar_norm":           {"kpi": "capacity",    "scope": "network", "stage": "sim"},

    # LATENCY (access / propagation delay)
    "fig25_access_delay_ccdf":           {"kpi": "latency",     "scope": "link",    "stage": "geometry"},

    # RELIABILITY (outage / interference-limited)
    "fig16_ion_ccdf":                    {"kpi": "reliability", "scope": "link",    "stage": "sim"},
    "fig22_ion_ccdf_per_user_overlay":   {"kpi": "reliability", "scope": "user",    "stage": "sim"},
    "fig17_visibility_outage_ccdf":      {"kpi": "reliability", "scope": "network", "stage": "geometry"},
    "fig18_service_outage_ccdf":         {"kpi": "reliability", "scope": "network", "stage": "geometry"},

    # MOBILITY (contact duration + handover)
    "fig5_contact_time_cdf":             {"kpi": "mobility",    "scope": "link",    "stage": "geometry"},
    "fig6_contact_vs_peakelev":          {"kpi": "mobility",    "scope": "link",    "stage": "geometry"},
    "fig7_intercontact_gap_cdf":         {"kpi": "mobility",    "scope": "link",    "stage": "geometry"},
    "fig_elev_contact_cdf":              {"kpi": "mobility",    "scope": "link",    "stage": "geometry"},
    "fig10_pingpong_bar":                {"kpi": "mobility",    "scope": "user",    "stage": "sim"},
    "fig15_sinr_drop_around_ho":         {"kpi": "mobility",    "scope": "link",    "stage": "sim"},
    "fig23_ihi_ccdf_per_user_overlay":   {"kpi": "mobility",    "scope": "user",    "stage": "sim"},
    "fig12_ho_geo_map":                  {"kpi": "mobility",    "scope": "network", "stage": "sim"},
    "ihi_ccdf":                          {"kpi": "mobility",    "scope": "user",    "stage": "sim"},
    "handover_pareto":                   {"kpi": "mobility",    "scope": "network", "stage": "sim"},

    # ENERGY (transmit power / efficiency) -- ntnconnect.analysis.extra_kpis
    "energy_power_ccdf":                 {"kpi": "energy",      "scope": "network", "stage": "sim"},
    "energy_efficiency_bar":             {"kpi": "energy",      "scope": "network", "stage": "sim"},

    # CONNECTION (admission / blocking) -- ntnconnect.analysis.extra_kpis
    "connection_admission_bar":          {"kpi": "connection",  "scope": "network", "stage": "sim"},
    "connection_blocking_ccdf":          {"kpi": "connection",  "scope": "network", "stage": "sim"},

    # FAIRNESS (per-user throughput equity) -- ntnconnect.analysis.extra_kpis
    "fairness_jain_bar":                 {"kpi": "fairness",    "scope": "network", "stage": "sim"},

    # DOPPLER (serving-link Doppler shift) -- ntnconnect.analysis.extra_kpis
    "doppler_ccdf":                      {"kpi": "doppler",     "scope": "link",    "stage": "sim"},

    # SYSTEM / META (run cost, not a network metric)
    "resource_cpu":                      {"kpi": "system",      "scope": "system",  "stage": "meta"},
    "resource_memory":                   {"kpi": "system",      "scope": "system",  "stage": "meta"},
    "resource_cpu_ihaap_hungarian":      {"kpi": "system",      "scope": "system",  "stage": "meta"},
    "resource_memory_ihaap_hungarian":   {"kpi": "system",      "scope": "system",  "stage": "meta"},
}

# recognised category values
KPIS    = ["coverage", "capacity", "latency", "reliability", "mobility",
           "energy", "connection", "fairness", "doppler", "system"]
SCOPES  = ["link", "user", "satellite", "network", "system"]
STAGES  = ["geometry", "sim", "meta"]
AXES    = {"kpi": KPIS, "scope": SCOPES, "stage": STAGES}

# KPI families the simulator does not measure (out of scope for a system-level
# tool), documented so the taxonomy is explicit about the gaps -- no figures.
# These need lower-layer/protocol modelling (the packet-level emulator roadmap).
NOT_COVERED = ["handover_interruption_time", "handover_success_rate",
               "control_plane_latency", "beam_gateway_utilisation"]


def list_figures(kpi: str = None, scope: str = None, stage: str = None) -> list[str]:
    """Figure names matching all of the given (kpi, scope, stage) filters."""
    out = []
    for name, m in FIGURE_CATALOG.items():
        if kpi is not None and m["kpi"] != kpi:
            continue
        if scope is not None and m["scope"] != scope:
            continue
        if stage is not None and m["stage"] != stage:
            continue
        out.append(name)
    return out


def resolve(selectors) -> list[str]:
    """Expand a mixed list of explicit figure names and 'axis:value' selectors
    ('kpi:coverage', 'scope:link', 'stage:geometry') into a de-duplicated list of
    explicit figure names. Explicit names pass through unchanged, so this can be
    dropped in front of the existing `figures=[...]` selection without breaking it.
    Unknown 'axis:value' selectors raise ValueError; unknown plain names pass
    through (the caller/analysis validates them as before)."""
    names = []
    for s in selectors:
        if isinstance(s, str) and ":" in s and s.split(":", 1)[0] in AXES:
            axis, val = s.split(":", 1)
            if val not in AXES[axis]:
                raise ValueError(f"unknown {axis}: '{val}' (valid: {AXES[axis]})")
            names.extend(list_figures(**{axis: val}))
        else:
            names.append(s)                       # explicit figure name -> keep
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def groups(by: str = "kpi") -> dict:
    """Catalog grouped by one axis ('kpi' | 'scope' | 'stage') -> {value: [names]}."""
    g = {}
    for name, m in FIGURE_CATALOG.items():
        g.setdefault(m[by], []).append(name)
    return g


def describe() -> str:
    """Human-readable summary of the catalog grouped by KPI family."""
    lines = [f"{len(FIGURE_CATALOG)} figures across {len(groups('kpi'))} KPI families:"]
    for kpi, names in sorted(groups("kpi").items()):
        lines.append(f"  {kpi:12s} ({len(names)}): {', '.join(sorted(names))}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print("\ngeometry-only (dataset, no simulation):", list_figures(stage="geometry"))
    print("\nresolve(['kpi:mobility','stage:geometry']):",
          resolve(["kpi:mobility", "stage:geometry"]))
