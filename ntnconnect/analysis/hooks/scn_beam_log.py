# Persists per-tick beam assignments (user_id -> beam_id) into one parquet
# at simulation end. Empty when capacity_mode != 'beam_tagged'.
from __future__ import annotations
import logging
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry              import register_hook

_log = logging.getLogger(__name__)


@register_hook("scn_beam_log")
def _build(engine):
    return ScnBeamLog(engine)


class ScnBeamLog(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.rows: list[dict] = []

    def on_step_end(self, state):
        beams = state.data.get("beam_assignments") or {}
        assigns = state.data.get("assignments") or {}
        for u, b in beams.items():
            self.rows.append({"t_idx": state.t_idx,
                              "user_id": int(u),
                              "serving_node_id": int(assigns.get(int(u), -1)),
                              "beam_id": int(b)})

    def on_simulation_end(self, state):
        if not self.rows: return
        out = self.engine.sim_dir / "scn_beam_log.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(out, index=False)
        _log.info("scn_beam_log: %d (user,tick) beam rows -> %s",
                  len(self.rows), out)
