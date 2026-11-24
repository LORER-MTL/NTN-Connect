"""Command-line front-end for the simulator's built-in analysis figures.

Lets a user choose which figures to ACTIVATE without writing any Python:

    python -m ntnconnect.analysis <out_dir> [--runs-dir DIR] [options]

Selection options (default = the curated default set):
    --figures a,b,c   activate only these. Names are figure names (e.g.
                      sinr_ccdf, rate_ccdf) OR category selectors resolved
                      through figure_catalog: kpi:<family>, scope:<level>,
                      stage:<geometry|sim|meta>  (e.g. --figures kpi:coverage).
    --all             activate every registered figure.
    --exclude a,b     deactivate these from the current selection
                      (combine with --all for "everything except ...").
    --list            print the figure registry (default set + all names,
                      grouped by KPI family when the catalog is available)
                      and exit without rendering.

Rendering options: --no-titles, --dpi N, --format {png,pdf,svg}, --tex.
"""
from __future__ import annotations
import argparse
import sys

from ntnconnect.analysis.core import run_analysis, list_figure_names, figure_categories


def _print_catalog() -> None:
    default_set, all_names = list_figure_names()
    cat, axes = figure_categories()
    print(f"default set ({len(default_set)} figures):")
    for n in default_set:
        print(f"    {n}")
    print(f"\nall registered figures ({len(all_names)}), by KPI family:")
    for kpi in axes["kpi"]:
        names = sorted(n for n in all_names if cat.get(n, (None,))[0] == kpi)
        if names:
            print(f"  {kpi:11s}: {', '.join(names)}")
    print(f"\nselectors:  kpi:{{{'|'.join(axes['kpi'])}}}")
    print(f"            scope:{{{'|'.join(axes['scope'])}}}")
    print(f"            stage:{{{'|'.join(axes['stage'])}}}")
    try:
        from ntnconnect.analysis.figure_catalog import NOT_COVERED
        print(f"\nnot covered (out of scope for this simulator, no figure): "
              f"{', '.join(NOT_COVERED)}")
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ntnconnect.analysis",
        description="Render the simulator's built-in analysis figures; "
                    "activate/deactivate figures by selection.")
    ap.add_argument("out_dir", nargs="?",
                    help="analysis output directory (also the runs dir unless --runs-dir)")
    ap.add_argument("--runs-dir", default=None,
                    help="directory holding the run manifest.csv (defaults to out_dir)")
    ap.add_argument("--figures", default=None,
                    help="comma list of figure names and/or category selectors "
                         "(kpi:/scope:/stage:) to activate")
    ap.add_argument("--all", dest="all_figures", action="store_true",
                    help="activate every registered figure")
    ap.add_argument("--exclude", default=None,
                    help="comma list of figures to deactivate from the selection")
    ap.add_argument("--no-titles", dest="show_titles", action="store_false",
                    help="render without figure titles (publication mode)")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--format", dest="fig_format", default="png",
                    choices=["png", "pdf", "svg"])
    ap.add_argument("--tex", dest="use_tex", action="store_true",
                    help="use a real LaTeX engine (falls back to mathtext if absent)")
    ap.add_argument("--bandwidth-hz", dest="bandwidth_hz", type=float, default=None,
                    help="channel bandwidth for extra-KPI rate/energy figures [Hz]")
    ap.add_argument("--carrier-hz", dest="carrier_hz", type=float, default=None,
                    help="carrier frequency for the extra-KPI Doppler figure [Hz]")
    ap.add_argument("--auto", action="store_true",
                    help="auto mode: drop selected figures whose data this "
                         "experiment did not produce (e.g. handover/routing figures "
                         "when no handover/routing ran)")
    ap.add_argument("--list", action="store_true",
                    help="print available figures and exit")
    a = ap.parse_args(argv)

    if a.list:
        _print_catalog()
        return 0
    if not a.out_dir:
        ap.error("out_dir is required unless --list is given")

    run_analysis(a.out_dir, runs_dir=a.runs_dir,
                 figures=a.figures, all_figures=a.all_figures, exclude=a.exclude,
                 show_titles=a.show_titles, dpi=a.dpi, fig_format=a.fig_format,
                 use_tex=a.use_tex,
                 bandwidth_hz=a.bandwidth_hz, carrier_hz=a.carrier_hz, auto=a.auto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
