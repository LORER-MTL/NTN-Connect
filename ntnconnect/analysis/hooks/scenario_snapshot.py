# Hook: scenario_snapshot
#
# At end-of-simulation, render one PNG of the static scenario over
# Quebec / Canada:
#  * gateway sites (squares, per-operator colour)
#  * user terminals (X markers)
#  * satellite sub-points at a representative tick (small dots)
#  * served-sat connections (light grey lines)
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook

_log = logging.getLogger(__name__)


@register_hook("scenario_snapshot")
def _build(engine):
    return ScenarioSnapshot(engine)


class ScenarioSnapshot(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self._last_links: pd.DataFrame | None = None
        self._last_assigns: dict = {}

    def on_step_end(self, state):
        # Keep last tick's view; rendered at end-of-sim
        df = state.data.get("links")
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._last_links = df.copy()
            self._last_assigns = dict(state.data.get("assignments", {}) or {})

    def on_simulation_end(self, state):
        if self._last_links is None or self._last_links.empty:
            _log.warning("scenario_snapshot: no links to render"); return
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self._last_links

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Scenario snapshot — last tick")
        ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")

        # Plot users
        if "user_lat" in df.columns and "user_lon" in df.columns:
            us = df[["user_id", "user_lat", "user_lon"]].drop_duplicates("user_id")
            ax.scatter(us["user_lon"], us["user_lat"], marker="x",
                        s=60, color="black", label=f"users ({len(us)})", zorder=3)

        # Plot gateways from engine.gateways
        gws = getattr(self.engine, "gateways", {}) or {}
        if gws:
            glon = [g.lon for g in gws.values()]
            glat = [g.lat for g in gws.values()]
            ax.scatter(glon, glat, marker="s", s=120, color="red",
                        edgecolor="black", label=f"gateways ({len(gws)})", zorder=4)
            for g in gws.values():
                ax.annotate(g.name, (g.lon, g.lat), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")

        # Plot sat sub-points (satellite rows only, deduped per node)
        if "node_type" in df.columns:
            sats = df[df["node_type"] == "satellite"].drop_duplicates("node_id")
            if not sats.empty and "node_lat" in sats.columns:
                # Colour per operator
                for op_id, sub in sats.groupby("operator_id"):
                    op_lbl = {1: "OneWeb", 2: "Starlink", 3: "Kuiper"}.get(int(op_id), f"Op{op_id}")
                    col   = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}.get(int(op_id), "gray")
                    ax.scatter(sub["node_lon"], sub["node_lat"], s=6,
                                color=col, alpha=0.4, label=f"{op_lbl} ({len(sub)})",
                                zorder=2)

        # Served-sat connection lines (very faint)
        if self._last_assigns and "user_lat" in df.columns and "node_lat" in df.columns:
            for uid, sid in self._last_assigns.items():
                if sid < 0: continue
                row = df[(df["user_id"] == uid) & (df["node_id"] == sid)]
                if row.empty: continue
                r = row.iloc[0]
                ax.plot([r["user_lon"], r["node_lon"]],
                         [r["user_lat"], r["node_lat"]],
                         color="gray", alpha=0.3, lw=0.5, zorder=1)

        ax.set_xlim(-145, -50); ax.set_ylim(40, 75)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_p = out_dir / "fig_scenario_snapshot.png"
        fig.savefig(out_p, dpi=140); plt.close(fig)
        _log.info("scenario_snapshot -> %s", out_p)
