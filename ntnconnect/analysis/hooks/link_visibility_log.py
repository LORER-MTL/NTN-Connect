# Hook: link_visibility_log - per-tick remaining-visibility predictions from
# stage link_visibility_predict, written to link_visibility.parquet.
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook

_log = logging.getLogger(__name__)


@register_hook("link_visibility_log")
def _build(engine):
    return LinkVisibilityLog(engine)


class LinkVisibilityLog(SimulationHook):
    """Captures remaining-visibility predictions per (user, sat, tick)."""

    def __init__(self, engine):
        self.engine = engine
        self._rows: list = []
        try:
            self._elev_min = float(
                getattr(self.engine.cfg.constraints, "min_elevation_deg",
                  getattr(self.engine.cfg.handover, "elev_min_deg", 20.0)))
        except Exception:
            self._elev_min = 10.0
        try:
            self._batch_ticks = int(
                getattr(self.engine.cfg.outputs,
                        "batch_log_every_n_ticks", 0) or 0)
        except Exception:
            self._batch_ticks = 0
        self._batch_idx = 0

    def on_step_end(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        if "elev_deg" not in links.columns or "user_id" not in links.columns:
            return
        if "node_type" not in links.columns:
            return

        # Only record satellite links above the visibility threshold.
        is_sat   = (links["node_type"].astype(str) == "satellite")
        vis_mask = is_sat & (links["elev_deg"] >= self._elev_min)
        if not vis_mask.any():
            return

        cols = [c for c in
                ("user_id", "node_id", "elev_deg",
                 "elev_rate_deg_s",
                 "remaining_visibility_s",
                 "remaining_visibility_ticks",
                 "dist_m", "range_rate_km_s")
                if c in links.columns]
        sub = links.loc[vis_mask, cols].copy()
        sub = sub.rename(columns={"node_id": "sat_id"})
        sub["t_idx"] = int(state.t_idx)
        front = ["t_idx", "user_id", "sat_id"]
        rest  = [c for c in sub.columns if c not in front]
        self._rows.append(sub[front + rest])

        if self._batch_ticks > 0 and ((state.t_idx + 1) % self._batch_ticks) == 0:
            self._flush_batch()

    def _flush_batch(self):
        if not self._rows: return
        from ntnconnect.common.utils.run_io import batch_table_path
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.concat(self._rows, ignore_index=True)
        if not df.empty:
            dst = (batch_table_path(out_dir, self._batch_idx, "link_visibility")
                   if self._batch_ticks > 0 else out_dir / "link_visibility.parquet")
            df.to_parquet(dst, index=False)
            _log.info("link_visibility_log: %d rows -> %s", len(df), dst)
        self._rows.clear()
        self._batch_idx += 1

    def on_simulation_end(self, state):
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._rows: return
        if self._batch_ticks > 0:
            self._flush_batch()
        else:
            df = pd.concat(self._rows, ignore_index=True)
            if not df.empty:
                df.to_parquet(out_dir / "link_visibility.parquet", index=False)
                _log.info("link_visibility_log: %d rows -> %s",
                          len(df), out_dir / "link_visibility.parquet")
            self._rows.clear()
