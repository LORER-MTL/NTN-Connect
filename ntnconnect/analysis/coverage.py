# Coverage analysis: standard per-run figures (coverage, visibility, outage, rate/SINR).
# Figures selected via figures=[...] (list_figure_names()).
from __future__ import annotations
from ntnconnect.analysis.core import run_analysis


def run_coverage_analysis(out_dir, **kwargs):
    """Constellation coverage / availability analysis (geometry-driven).

    Thin wrapper over the standard figure/table engine; `figures=[...]` selects
    which coverage figures to emit. See analysis.list_figure_names().
    """
    return run_analysis(out_dir, **kwargs)


__all__ = ["run_coverage_analysis"]
