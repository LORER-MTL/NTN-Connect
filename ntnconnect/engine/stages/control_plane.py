# Stage: build the ControlPlane from cfg.control_plane at the first tick
# (cached thereafter). Stores it in state.data["control_plane"].

from __future__ import annotations
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.control.plane import (
    ControlPlane, Controller, Scope, Arbiter, DECISION_DOMAINS,
)
from ntnconnect.common.utils.rng import make_rng
from ntnconnect.registry.agents.handover_strategy import get_handover_strategy
from ntnconnect.registry.agents.isl_strategy import get_isl_strategy
from ntnconnect.registry.agents.offload_strategy import (
    get_offload_strategy, get_scheduler_strategy, get_cache_strategy,
)


def _strategy_factory(domain: str, name: str, cfg, controller):
    """Build the strategy instance for one controller's decision domain.

    A controller's `decisions[domain]` value is the *name* of the strategy.
    We currently route through the existing factories by temporarily
    overriding cfg.<domain>.mode/policy on a shallow clone.
    """
    # Centralized vs. distributed selection still goes through the
    # existing factories; the controller just specifies which one.
    if domain == "handover":
        # Override mode for this lookup
        prev = cfg.handover.mode
        cfg.handover.mode = name
        s = get_handover_strategy(cfg)
        cfg.handover.mode = prev
        return s
    if domain == "isl":
        # ISL strategy uses cfg.isl on whatever object is passed; we wrap an
        # adapter inside get_isl_strategy. For per-controller override, we
        # call the function with a tiny adapter.
        class _IslCfg: pass
        ic = _IslCfg(); ic.isl = cfg.operators[0].isl if cfg.operators else None
        if ic.isl is not None:
            ic.isl.strategy = name
        return get_isl_strategy(ic) if ic.isl is not None else None
    if domain == "offload":
        prev = cfg.offload.target_policy
        cfg.offload.target_policy = name
        s = get_offload_strategy(cfg)
        cfg.offload.target_policy = prev
        return s
    if domain == "scheduling":
        prev = cfg.scheduler.policy
        cfg.scheduler.policy = name
        s = get_scheduler_strategy(cfg)
        cfg.scheduler.policy = prev
        return s
    if domain == "cache":
        return get_cache_strategy(cfg)
    if domain == "routing":
        # Routing strategy interface to be added in the next slice;
        # store the name so the dispatch stage can resolve it later.
        return ("routing", name)
    return None


@register_stage("control_plane")
class ControlPlaneStage(Stage):
    reads = []
    writes = ["control_plane", "digital_twin"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cached = None
        self._dt = None

    def _publish_dt(self, state):
        if self._dt is not None:
            state.data["digital_twin"] = self._dt

    def run(self, state):
        if self._cached is not None:
            state.data["control_plane"] = self._cached
            self._publish_dt(state)
            return

        cp_cfg = self.cfg.control_plane
        arb = Arbiter(mode=cp_cfg.arbiter)
        controllers = []
        seed = int(self.cfg.experiment.random_seed)
        for spec in cp_cfg.controllers:
            c = Controller(
                id=spec.id,
                priority=spec.priority,
                scope=Scope(
                    operator_ids=list(spec.scope.operator_ids),
                    node_types=list(spec.scope.node_types),
                ),
                visibility=spec.visibility,
                decisions=dict(spec.decisions),
                rng=make_rng(spec.id, seed),
            )
            for domain in DECISION_DOMAINS:
                name = spec.decisions.get(domain)
                if name in (None, "", "~", "none"):
                    continue
                try:
                    c.strategies[domain] = _strategy_factory(domain, name, self.cfg, c)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Controller '%s' could not build %s strategy '%s': %s",
                        c.id, domain, name, e)
                    c.strategies[domain] = None
            controllers.append(c)

        plane = ControlPlane(arbiter=arb, controllers=controllers)
        self._cached = plane
        state.data["control_plane"] = plane

        # Digital-twin foundation: built once here when activated. Off -> nothing
        # is created and the physical pipeline is untouched. The parallel twin
        # lane is added later (see DIGITAL_TWIN_PLAN.md).
        if getattr(self.cfg, "dt", None) is not None and self.cfg.dt.enabled:
            from ntnconnect.control.digital_twin import DigitalTwin
            self._dt = DigitalTwin(self.cfg)
        self._publish_dt(state)
