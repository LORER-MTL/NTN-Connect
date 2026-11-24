# Catalog inspector: python scripts/catalog.py [tables|kpis|stages|hooks|channels|plots]

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def print_tables():
    from ntnconnect.common.schemas import ALL_TABLES
    print("\n=== OUTPUT TABLES ===")
    print(f"{'name':<22} {'granularity':<22} {'cols':>4} {'partition':<24} {'default'}")
    for n, s in ALL_TABLES.items():
        part = ",".join(s.partition_cols) or "-"
        print(f"{n:<22} {s.granularity:<22} {len(s.columns):>4} {part:<24} {s.default_format}")


def print_kpis():
    # Ensure all KPI modules are imported (they self-register)
    from ntnconnect.analysis.metrics import (
        link_kpis, user_kpis, operator_kpis,
        network_kpis, compute_kpis, system_kpis,
    )
    from ntnconnect.analysis.metrics.base import list_kpis
    print("\n=== KPIs ===")
    print(f"{'name':<34} {'granularity':<14} {'units':<10} {'default_plot':<12} description")
    for name, k in sorted(list_kpis().items()):
        print(f"{name:<34} {k.granularity:<14} {k.units:<10} {k.default_plot:<12} {k.description}")


def print_stages():
    # Import to trigger registration
    from ntnconnect.engine import engine  # noqa: F401
    for m in engine.DEFAULT_STAGE_MODULES:
        try:
            __import__(m)
        except Exception:
            pass
    from ntnconnect.registry.registry import list_stages
    print("\n=== STAGES ===")
    print(f"{'name':<30} {'reads':<35} {'writes':<35} thread_safe")
    for n, cls in sorted(list_stages().items()):
        r = ",".join(cls.reads)[:34]
        w = ",".join(cls.writes)[:34]
        print(f"{n:<30} {r:<35} {w:<35} {cls.thread_safe}")


def print_hooks():
    from ntnconnect.engine import engine  # noqa: F401
    for m in engine.DEFAULT_HOOK_MODULES:
        try: __import__(m)
        except Exception: pass
    from ntnconnect.analysis.hooks.registry import list_hooks
    print("\n=== HOOKS ===")
    for n in sorted(list_hooks()): print(f"  {n}")


def print_channels():
    from ntnconnect.registry.channels.pipeline import list_models
    # Trigger registration
    from ntnconnect.registry.channels.models import (
        three_gpp, loo, lutz, perez_fontan, ntn_tdl,
        saleh_valenzuela, spatial_geometric_rician,
    )
    print("\n=== CHANNEL MODELS ===")
    for n in sorted(list_models()): print(f"  {n}")


def print_plots():
    from ntnconnect.analysis.plots import (
        base, line, bar, cdf, histogram, heatmap, scatter, network_graph,
    )
    from ntnconnect.analysis.plots.base import list_plots
    print("\n=== PLOT TYPES ===")
    for n in sorted(list_plots()): print(f"  {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all","tables","kpis","stages","hooks","channels","plots"])
    args = ap.parse_args()
    do_all = (args.what == "all")
    if do_all or args.what == "tables":    print_tables()
    if do_all or args.what == "kpis":      print_kpis()
    if do_all or args.what == "stages":    print_stages()
    if do_all or args.what == "hooks":     print_hooks()
    if do_all or args.what == "channels":  print_channels()
    if do_all or args.what == "plots":     print_plots()


if __name__ == "__main__":
    main()
