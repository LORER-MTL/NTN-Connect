# Hook: persists validation_rows produced by stages/val_constraints.py.

from __future__ import annotations
import logging
from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook
from ntnconnect.common.schemas import VALIDATION_CHECKS

_log = logging.getLogger(__name__)


class ValidationChecksHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = VALIDATION_CHECKS.collector(engine)
        self._fail = 0
        self._pass = 0

    def on_step_end(self, state):
        if not VALIDATION_CHECKS.is_enabled(self.engine.cfg):
            return
        for r in state.data.get("validation_rows", []):
            self.c.add(r)
            if r.get("passed", 0): self._pass += 1
            else:                  self._fail += 1

    def on_simulation_end(self, state):
        self.c.flush()
        total = self._pass + self._fail
        if total:
            _log.info("Validation: %d/%d checks passed (%.1f%%)",
                      self._pass, total, 100.0 * self._pass / total)


@register_hook("validation_checks")
def _b(engine): return ValidationChecksHook(engine)
