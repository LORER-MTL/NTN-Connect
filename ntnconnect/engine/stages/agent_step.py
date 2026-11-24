# Stage: agent_step - call.step() on every User/Gateway each tick (no-op if static).
# Publishes user_objs / gateway_objs to state.data.
from __future__ import annotations
import logging
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_log = logging.getLogger(__name__)


@register_stage("agent_step", internal=True)
class AgentStep(Stage):
    reads  = []
    writes = ["user_objs", "gateway_objs"]
    thread_safe = False

    def run(self, state):
        eng = getattr(self, "engine", None)
        if eng is None: return
        try:
            dt_s = float(self.cfg.env.step_size_s)
        except Exception:
            dt_s = 1.0
        t_idx = int(getattr(state, "t_idx", 0))

        users    = getattr(eng, "users", {})    or {}
        gateways = getattr(eng, "gateways", {}) or {}

        # Per-tick step on every agent. Default User/Gateway.step is a
        # no-op so this loop is sub-millisecond at scenario scale.
        for u in users.values():
            try:
                u.step(t_idx, dt_s, state)
            except Exception as e:
                _log.warning("user %d step failed: %s", u.user_id, e)
        for g in gateways.values():
            try:
                g.step(t_idx, dt_s, state)
            except Exception as e:
                _log.warning("gateway %d step failed: %s", g.gw_id, e)

        # Publish for downstream stages / hooks
        state.data["user_objs"]    = users
        state.data["gateway_objs"] = gateways
