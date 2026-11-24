# Hook: user_trace
#
# At end-of-simulation, for the first three users:
#  * served-sat elevation timeline
#  * served-link SINR timeline
#  * remaining_visibility_s timeline (if present)
#  * handover events overlaid as coloured vertical markers
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


@register_hook("user_trace")
def _build(engine):
    return UserTrace(engine)


_TRIGGER_COLORS = {
    "init":       "gray",
    "sinr_gain":  "tab:blue",
    "elev_drop":  "tab:red",
    "load":       "tab:purple",
    "leave":      "black",
    "strategy":   "tab:green",
}


class UserTrace(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        try:
            self._dt_s = float(self.engine.cfg.env.step_size_s)
        except Exception:
            self._dt_s = 1.0

    def on_simulation_end(self, state):
        rd = self.engine.sim_dir
        sin = self._read_concat(rd, "sinr_snapshot")
        vis = self._read_concat(rd, "visibility_per_tick")
        lvi = self._read_concat(rd, "link_visibility")
        ho  = self._read_concat(rd, "handover_events")
        if sin.empty:
            _log.warning("user_trace: no sinr_snapshot to render"); return
        uids = sorted(sin["user_id"].astype(int).unique())[:3]
        if not uids:
            _log.warning("user_trace: no served users"); return

        fig, axes = plt.subplots(len(uids), 1, figsize=(12, 3.0 * len(uids)),
                                  sharex=True, squeeze=False)
        for ax, uid in zip(axes[:, 0], uids):
            sub = sin[sin["user_id"] == uid].sort_values("t_idx")
            tt  = sub["t_idx"].to_numpy() * self._dt_s
            ax.set_title(f"User {uid}")
            ax.set_xlabel("time [s]")
            ax2 = ax.twinx()

            if "serving_sat_elev_deg" in sub.columns:
                ax.plot(tt, sub["serving_sat_elev_deg"], color="tab:blue",
                         label="served-sat elev [deg]")
            ax.set_ylabel("elev [deg]", color="tab:blue")
            ax.set_ylim(0, 90); ax.grid(alpha=0.3)

            ax2.plot(tt, sub["sinr_db"], color="tab:red",
                      label="served SINR [dB]")
            ax2.set_ylabel("SINR [dB]", color="tab:red")

            # remaining_visibility_s overlay (per served sat per tick)
            if not lvi.empty and "remaining_visibility_s" in lvi.columns:
                # Pull only this user's served-sat rows
                msv = lvi[lvi["user_id"] == uid]
                # Plot as a thin grey curve normalised against the same elev axis
                # by clipping at 600 s for readability
                if not msv.empty:
                    ttv = msv["t_idx"].to_numpy() * self._dt_s
                    rv  = np.clip(msv["remaining_visibility_s"].to_numpy(),
                                   0, 600) / 10.0  # 600s -> 60 elev scale
                    ax.plot(ttv, rv, color="tab:green", lw=0.4, alpha=0.5,
                             label="remaining_vis [s] / 10")

            # Handover markers
            if not ho.empty:
                h = ho[ho["user_id"] == uid]
                for r in h.itertuples(index=False):
                    col = _TRIGGER_COLORS.get(
                        str(getattr(r, "trigger_type", "strategy")), "tab:green")
                    ax.axvline(r.t_idx * self._dt_s, color=col, alpha=0.5,
                                lw=0.8, ls="--")

        # Legend on the first axis
        h, l = axes[0, 0].get_legend_handles_labels()
        h2, l2 = axes[0, 0].twinx().get_legend_handles_labels()
        # already have ax2 plotted above
        axes[0, 0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out_p = rd / "fig_user_trace.png"
        fig.savefig(out_p, dpi=140); plt.close(fig)
        _log.info("user_trace -> %s", out_p)

    def _read_concat(self, rd: Path, stream: str) -> pd.DataFrame:
        files = sorted(rd.glob(f"{stream}*.parquet"))
        if not files: return pd.DataFrame()
        return pd.concat([pd.read_parquet(p) for p in files],
                          ignore_index=True)
