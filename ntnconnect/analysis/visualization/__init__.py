"""ntnconnect.analysis.visualization - project-wide visualization package.

Reusable 3D + slot-by-slot visualization for the whole SkyNTNPro project.
Import it from anywhere that has the project package on its path:

    from ntnconnect.analysis.visualization import build_czml_json, build_tick_stats

Replaces the old per-scenario modules (_czml_export.py, make_cesium_view.py,
live_dashboard.py) and the empty ntnconnect/visualization stub.

Modules
-------
czml         : build CZML packets + per-tick HUD statistics from an ephemeris
               dataset and Phase-2 run streams.
cesium_view  : generate a STANDALONE, self-contained CesiumJS .html (with
               in-view constellation / handover-algorithm selectors) and serve
               it over http. Run:
                   python -m ntnconnect.analysis.visualization.cesium_view --open
dashboard    : Streamlit live dashboard. Run:
                   streamlit run ntnconnect/visualization/dashboard.py

Both runnable modules also work when launched directly as scripts (they fall
back to absolute sibling imports), so `streamlit run dashboard.py` and
`python cesium_view.py` keep working.
"""
from __future__ import annotations

from ntnconnect.analysis.visualization.czml import build_czml, build_czml_json, build_tick_stats
from ntnconnect.analysis.visualization import kpi_engine

__all__ = ["build_czml", "build_czml_json", "build_tick_stats", "kpi_engine"]
