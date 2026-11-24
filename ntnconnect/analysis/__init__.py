"""ntnconnect.analysis - scenario-independent results analysis.

Computes metrics and emits figures/tables from the engine's STANDARD per-run
Parquet outputs (sinr_snapshot, handover/ho_events, contact_events,
link_visibility, per_user/per_operator_kpis, ...). No scenario-specific
dependencies: a scenario calls `run_analysis(out_dir, ...)` and supplies its own
output style, ephemeris path and labels.

    from ntnconnect.analysis import run_analysis, list_figure_names
    run_analysis(out_dir, figures=["sinr_ccdf"], show_titles=False,
                 eph_dir=my_eph_dir)
"""
from __future__ import annotations
from ntnconnect.analysis.core import (
    run_analysis, resolve_figures, list_figure_names, set_eph_dir,
    elevation_sweep,
    _FIG_REGISTRY, _DEFAULT_FIGS,
)
# The three config-selectable analysis families (scenario-independent).
from ntnconnect.analysis.coverage import run_coverage_analysis
from ntnconnect.analysis.handover import run_handover_comparison
from ntnconnect.analysis.processing import run_processing_comparison

__all__ = ["run_analysis", "resolve_figures", "list_figure_names", "set_eph_dir",
           "elevation_sweep",
           "run_coverage_analysis", "run_handover_comparison",
           "run_processing_comparison", "run_analyses", "ANALYSES"]

# Registry of selectable analysis families -> callable. A scenario selects a
# subset through configuration; it never reimplements them.
ANALYSES = {
    "coverage":   run_coverage_analysis,
    "handover":   run_handover_comparison,
    "processing": run_processing_comparison,
}


def run_analyses(select, params: dict):
    """Run the selected analysis families.

    select : iterable of family names (subset of ANALYSES); () or ("all",) runs all.
    params : dict family_name -> kwargs for that family's function.
    """
    names = list(ANALYSES) if (not select or "all" in select) else list(select)
    return {n: ANALYSES[n](**params.get(n, {})) for n in names if n in ANALYSES}
