# SkyNTNPro v1 - Hook registry
#
# Hooks register themselves at import time via @register_hook.
# The engine picks which ones to attach based on
# cfg.outputs.toggles[<name>] and cfg.outputs.format(s).

from __future__ import annotations
from typing import Callable, Dict

_REGISTRY: Dict[str, Callable] = {}


def register_hook(name: str):
    """Decorator: register a build_hook callable under ``name``.

    The callable takes (engine) and returns a SimulationHook instance.
    """
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_hook_builder(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown hook '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_hooks() -> Dict[str, Callable]:
    return dict(_REGISTRY)
