# Hook: users_per_tick
# Persists the per-tick users_t frame -- which carries the freshly
# recomputed cluster_id, prev_cluster_id, cluster_changed columns
# from the clustering stage -- into a single parquet file at
# simulation end:
#  <run_dir>/users_per_tick.parquet
# Enable via cfg.outputs.toggles.users_per_tick = true.
from __future__ import annotations
import logging
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry              import register_hook

_log = logging.getLogger(__name__)


@register_hook("users_per_tick")
def _build(engine):
    return UsersPerTick(engine)


class UsersPerTick(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.frames: list[pd.DataFrame] = []

    def on_step_end(self, state):
        u = state.data.get("users_t")
        if isinstance(u, pd.DataFrame) and not u.empty:
            self.frames.append(u.copy())

    def on_simulation_end(self, state):
        if not self.frames: return
        df = pd.concat(self.frames, ignore_index=True)
        out = self.engine.sim_dir / "users_per_tick.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        _log.info("users_per_tick: %d rows -> %s", len(df), out)
