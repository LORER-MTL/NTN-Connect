# SkyNTNPro v1 - Control plane
#
# A scenario is a list of Controllers, each with:
#  * scope - which nodes it owns
#  * visibility - what state it can read
#  * decisions - which strategy classes it runs for each domain
#  * priority - tie-breaker for cross-controller arbitration
#  * rng - deterministic per-controller RNG
#
# Stages (handover, isl, routing, offload,...) consult the ControlPlane
# instead of holding strategy instances directly. This separates the
# data plane (what is computed) from the control plane (who decides).

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

_log = logging.getLogger(__name__)


# Scope + visibility
@dataclass
class Scope:
    """Which nodes a controller owns.

    Wildcard ``*`` (or empty list) matches all operators / all node types.
    """
    operator_ids: List[int] = field(default_factory=list)
    node_types:   List[str] = field(default_factory=list)

    def matches(self, op_id: int, node_type: str) -> bool:
        op_ok = (not self.operator_ids) or (op_id in self.operator_ids)
        nt_ok = (not self.node_types)   or (node_type in self.node_types)
        return op_ok and nt_ok

    def is_universal(self) -> bool:
        return not self.operator_ids and not self.node_types

    def overlaps(self, other: "Scope") -> bool:
        if self.is_universal() or other.is_universal(): return True
        op_overlap = (
            (not self.operator_ids) or (not other.operator_ids) or
            bool(set(self.operator_ids) & set(other.operator_ids))
        )
        nt_overlap = (
            (not self.node_types) or (not other.node_types) or
            bool(set(self.node_types) & set(other.node_types))
        )
        return op_overlap and nt_overlap


# Visibility levels - what state a controller can read.
VISIBILITY_LEVELS = {"own_node", "own_scope", "own_operator", "peers", "global"}


# Decision domains
DECISION_DOMAINS = ("handover", "isl", "routing",
                    "offload", "scheduling", "cache")


# Controller
@dataclass
class Controller:
    id: str
    priority: int
    scope: Scope
    visibility: str
    decisions: Dict[str, Optional[str]] = field(default_factory=dict)
    rng: Any = None             # numpy Generator, built by control_plane
    strategies: Dict[str, Any] = field(default_factory=dict)  # built strategy instances

    def owns(self, domain: str) -> bool:
        return self.decisions.get(domain) not in (None, "", "~", "none")

    def strategy(self, domain: str):
        return self.strategies.get(domain)


# Arbiter
class Arbiter:
    """Resolves cross-controller conflicts.

    Implementations:
      * priority_based - controller with the higher priority wins
      * first_wins     - iteration order decides (stable: by controller.id)
      * none           - raise if a conflict is detected
    """

    def __init__(self, mode: str = "priority_based"):
        self.mode = mode

    def resolve(self, candidates: List[Tuple[Controller, Any]]):
        """Pick one (controller, decision) tuple from a list.

        Caller is responsible for only calling this when a true conflict exists.
        Returns the winning (controller, decision).
        """
        if not candidates:
            return None
        if self.mode == "priority_based":
            return max(candidates, key=lambda x: x[0].priority)
        if self.mode == "first_wins":
            return min(candidates, key=lambda x: x[0].id)
        if self.mode == "none":
            raise RuntimeError(
                f"Arbiter mode 'none' but conflict among: "
                f"{[c.id for c, _ in candidates]}"
            )
        # Unknown - default to priority_based behavior
        _log.warning("Unknown arbiter mode '%s'; falling back to priority_based.",
                     self.mode)
        return max(candidates, key=lambda x: x[0].priority)


# ControlPlane
@dataclass
class ControlPlane:
    arbiter: Arbiter
    controllers: List[Controller] = field(default_factory=list)

    def controllers_owning(self, domain: str) -> List[Controller]:
        return [c for c in self.controllers if c.owns(domain)]

    def controller_for(self, domain: str, op_id: int, node_type: str) -> Optional[Controller]:
        """Find the highest-priority controller whose scope matches (op_id, node_type)
        and that owns the given decision domain."""
        cands = [c for c in self.controllers
                 if c.owns(domain) and c.scope.matches(op_id, node_type)]
        if not cands:
            return None
        return max(cands, key=lambda c: c.priority)

    def by_scope(self, domain: str, op_id_list: List[int], node_type_list: List[str]):
        """Iterate (controller, scope_slice) groups for parallel dispatch."""
        # Group nodes by their winning controller for this domain
        buckets: Dict[str, List[Tuple[int, str]]] = {}
        for op_id in op_id_list:
            for nt in node_type_list:
                c = self.controller_for(domain, op_id, nt)
                if c is None:
                    continue
                buckets.setdefault(c.id, []).append((op_id, nt))
        for c in self.controllers:
            if c.id in buckets:
                yield c, buckets[c.id]
