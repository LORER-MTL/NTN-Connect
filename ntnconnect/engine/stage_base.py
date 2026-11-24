# Stage base: subclass declares read/write state keys + thread-safety for group validation.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List
from ntnconnect.orchestration.orchestrator import SimulationState


class Stage(ABC):
    """Base class for a Phase-2 stage.
    """

    name: str = "stage"
    reads: List[str] = []
    writes: List[str] = []
    thread_safe: bool = False
    internal: bool = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = None  # set by SimulationEngine._build_groups

    @abstractmethod
    def run(self, state: SimulationState) -> None: ...

    # The orchestrator calls Stage instances as callables.
    def __call__(self, state: SimulationState) -> None:
        self.run(state)
        # Expose stage identity for the orchestrator's group introspection.
        # (orchestrator reads __name__ as a heuristic; we set it once.)


def stage_callable(stage: Stage):
    """Adapter: turn a Stage instance into a callable with a useful __name__..
    """
    def _fn(state: SimulationState):
        stage.run(state)
    _fn.__name__ = stage.name
    _fn.stage = stage
    return _fn
