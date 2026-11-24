# Stage registry: stages register via @register_stage; engine builds groups from it.

from __future__ import annotations
from typing import Callable, Dict, Type
from ntnconnect.engine.stage_base import Stage

# Three SEPARATE registries, never mixed:
#   _REGISTRY         - the main simulation stages (the ~14 top-level stages the
#                       pipeline is built from, plus their internal sub-stages).
#   _DT_REGISTRY      - digital-twin lane stages (own registry).
#   _COMPUTE_REGISTRY - compute / offloading lane stages (own registry).
# get_stage_class resolves a name across all three so a pipeline can reference any
# stage, but each lane is listed on its own (no cross-contamination in the menus).
_REGISTRY: Dict[str, Type[Stage]] = {}
_DT_REGISTRY: Dict[str, Type[Stage]] = {}
_COMPUTE_REGISTRY: Dict[str, Type[Stage]] = {}


def register_stage(name: str, internal: bool = False):
    """Register a MAIN simulation stage. internal=True marks a sub-stage (a
    building block invoked by a top-level stage) so list_stages(False) hides it."""
    def deco(cls):
        cls.name = name
        cls.internal = bool(internal)
        cls.lane = "main"
        _REGISTRY[name] = cls
        return cls
    return deco


def register_dt_stage(name: str):
    """Register a DIGITAL-TWIN stage in the DT registry (separate lane)."""
    def deco(cls):
        cls.name = name
        cls.internal = False
        cls.lane = "dt"
        _DT_REGISTRY[name] = cls
        return cls
    return deco


def register_compute_stage(name: str):
    """Register a COMPUTE/offloading stage in the compute registry (separate lane)."""
    def deco(cls):
        cls.name = name
        cls.internal = False
        cls.lane = "compute"
        _COMPUTE_REGISTRY[name] = cls
        return cls
    return deco


def get_stage_class(name: str) -> Type[Stage]:
    """Resolve a stage name across all three registries (main -> dt -> compute)."""
    for reg in (_REGISTRY, _DT_REGISTRY, _COMPUTE_REGISTRY):
        if name in reg:
            return reg[name]
    raise KeyError(
        f"Unknown stage '{name}'. Registered: "
        f"main={sorted(_REGISTRY)} dt={sorted(_DT_REGISTRY)} "
        f"compute={sorted(_COMPUTE_REGISTRY)}"
    )


def list_stages(include_internal: bool = True) -> Dict[str, Type[Stage]]:
    """MAIN simulation stages. include_internal=False drops the sub-stages
    (building blocks invoked by a top-level stage), leaving the ~14 top-level
    stages. DT and compute stages live in their own registries (see
    list_dt_stages / list_compute_stages)."""
    if include_internal:
        return dict(_REGISTRY)
    return {n: c for n, c in _REGISTRY.items() if not getattr(c, "internal", False)}


def list_dt_stages() -> Dict[str, Type[Stage]]:
    """Digital-twin lane stages (separate registry)."""
    return dict(_DT_REGISTRY)


def list_compute_stages() -> Dict[str, Type[Stage]]:
    """Compute / offloading lane stages (separate registry)."""
    return dict(_COMPUTE_REGISTRY)


def all_stages() -> Dict[str, Type[Stage]]:
    """Union across the three registries (for tooling / validation)."""
    out: Dict[str, Type[Stage]] = {}
    out.update(_REGISTRY); out.update(_DT_REGISTRY); out.update(_COMPUTE_REGISTRY)
    return out


# consolidated registry inspection
# These helpers re-export the per-subsystem registries from one place so
# tooling (CLI, tests, docs generators) doesn't have to know which file
# each registry lives in. They lazy-import to avoid circular dependencies.
def list_radio_algorithms() -> Dict[str, type]:
    from ntnconnect.registry.agents.radio_algorithm import list_radio_algorithms as _f
    return _f()


def list_channel_predictors() -> Dict[str, type]:
    from ntnconnect.registry.channels.predictor import list_channel_predictors as _f
    return _f()


def list_channel_models() -> Dict[str, type]:
    try:
        from ntnconnect.registry.channels.pipeline import list_models as _f
        return _f()
    except Exception:
        return {}


def list_hooks() -> Dict[str, Callable]:
    from ntnconnect.analysis.hooks.registry import list_hooks as _f
    return _f()


def list_ext_interference_modes() -> list:
    try:
        from ntnconnect.engine.stages.external_interference import list_ext_interference_modes as _f
        return _f()
    except Exception:
        return []


def list_ext_colouring_algorithms() -> list:
    try:
        from ntnconnect.engine.stages.external_interference import list_colouring_algorithms as _f
        return _f()
    except Exception:
        return []


def list_validators() -> list:
    try:
        from ntnconnect.common.validation import list_validators as _f
        return _f()
    except Exception:
        return []


def list_access_models() -> list:
    try:
        from ntnconnect.registry.channels.access_models import list_access_models as _f
        return _f()
    except Exception:
        return []


def list_steering_strategies() -> list:
    try:
        from ntnconnect.registry.channels._steering import list_steering_strategies as _f
        return _f()
    except Exception:
        return []


def list_isl_strategies() -> Dict[str, type]:
    try:
        from ntnconnect.registry.agents.isl_strategy import list_isl_strategies as _f
        return _f()
    except Exception:
        return {}


def describe_registries() -> Dict[str, list]:
    """One-shot inspection: returns name lists for every registry. This is the
    menu a scenario author picks from - every selectable component the general
    simulator offers, surfaced from one call."""
    return {
        "stages":             sorted(list_stages(include_internal=False).keys()),
        "stage_building_blocks": sorted(n for n, c in list_stages().items()
                                        if getattr(c, "internal", False)),
        "dt_stages":          sorted(list_dt_stages().keys()),
        "compute_stages":     sorted(list_compute_stages().keys()),
        "validators":         sorted(list_validators()),
        "radio_algorithms":   sorted(list_radio_algorithms().keys()),
        "channel_predictors": sorted(list_channel_predictors().keys()),
        "channel_models":     sorted(list_channel_models().keys()),
        "access_models":      sorted(list_access_models()),
        "steering_strategies": sorted(list_steering_strategies()),
        "isl_strategies":     sorted(list_isl_strategies().keys()),
        "hooks":              sorted(list_hooks().keys()),
        "ext_interference":   sorted(list_ext_interference_modes()),
        "ext_colouring":      sorted(list_ext_colouring_algorithms()),
    }
