# Lightweight per-scenario handover counter hook.
#  - Records every (t_idx, user_id, prev_node_id, new_node_id) transition.
#  - Flushes a parquet file at simulation end.
#  - Enable via cfg.outputs.toggles.scn_ho_counter = true.
from __future__ import annotations
import logging
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry              import register_hook

_log = logging.getLogger(__name__)


@register_hook("scn_ho_counter")
def _build(engine):
    return ScnHOCounter(engine)


class ScnHOCounter(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self._prev: dict[int, int] = {}
        self.rows: list[dict] = []
        self.n_total = 0

    def on_step_end(self, state):
        a = state.data.get("assignments", {}) or {}
        for u, s in a.items():
            p = int(self._prev.get(int(u), -1))
            if int(s) != p:
                self.rows.append({"t_idx": state.t_idx, "user_id": int(u),
                                  "prev_node_id": p, "new_node_id": int(s)})
                self.n_total += 1
        self._prev = {int(u): int(s) for u, s in a.items()}

    def on_simulation_end(self, state):
        if not self.rows: return
        out = self.engine.sim_dir / "scn_ho_log.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(out, index=False)
        _log.info("scn_ho_counter: %d HOs -> %s", self.n_total, out)
