"""Control-plane presets - declare a decision stage CENTRALIZED or DISTRIBUTED
in one line, without ever touching the stage's code.

A stage's centralized/distributed behaviour is determined ENTIRELY by the
controllers in `cfg.control_plane.controllers` (their scope + visibility). These
helpers build those controllers from a high-level intent so a scenario author
writes, e.g.:

    from ntnconnect.control.presets import centralized, distributed, control_plane

    # one global controller decides handover for everyone → CENTRALIZED
    cfg.control_plane.controllers = centralized("handover", "max_sinr")

    # one controller per operator, each sees only its own scope → DISTRIBUTED
    cfg.control_plane.controllers = distributed(
        "handover", "max_sinr", operator_ids=[1, 2, 3], per="operator")

    # mixed, several domains at once:
    cfg.control_plane.controllers = control_plane({
        "handover":  ("centralized", "max_sinr"),
        "offload":   ("distributed", "nearest_edge", dict(operator_ids=[1,2,3])),
    })

The stage (`handover`, `offload`, `isl`, `scheduling`, `cache`) resolves its
controller via `ControlPlane.controller_for(domain, op_id, node_type)` and runs
the same strategy on whatever scope/visibility the controller carries - so
flipping centralized↔distributed is purely a config change.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
from ntnconnect.experiment.config import ControllerSpec, ScopeSpec

# Visibility used for each mode (see control.plane.VISIBILITY_LEVELS).
_CENTRAL_VIS = "global"      # sees all state → global optimum
_DISTRIB_VIS = "own_node"    # sees only its own node's state → local decisions
_SCOPE_VIS   = "own_scope"   # per-operator/per-scope slice


def centralized(domain: str, strategy: str, *,
                operator_ids: Optional[Sequence[int]] = None,
                node_types: Optional[Sequence[str]] = None,
                priority: int = 100, cid: Optional[str] = None) -> List[ControllerSpec]:
    """one controller with global visibility over a universal (or given) scope
    → the domain decides centrally. Returns a 1-element list."""
    return [ControllerSpec(
        id=cid or f"central_{domain}",
        priority=priority,
        scope=ScopeSpec(operator_ids=list(operator_ids or []),
                        node_types=list(node_types or [])),
        visibility=_CENTRAL_VIS,
        decisions={domain: strategy},
    )]


def distributed(domain: str, strategy: str, *,
                operator_ids: Sequence[int],
                node_types: Sequence[str] = ("satellite",),
                per: str = "operator", priority: int = 50) -> List[ControllerSpec]:
    """one controller per (operator) or per (operator, node_type), each seeing
    only its own slice → the domain decides locally. `per` ∈ {operator, node_type}."""
    specs: List[ControllerSpec] = []
    for op in operator_ids:
        if per == "node_type":
            for nt in node_types:
                specs.append(ControllerSpec(
                    id=f"dist_{domain}_op{op}_{nt}", priority=priority,
                    scope=ScopeSpec(operator_ids=[int(op)], node_types=[nt]),
                    visibility=_DISTRIB_VIS, decisions={domain: strategy}))
        else:  # per-operator
            specs.append(ControllerSpec(
                id=f"dist_{domain}_op{op}", priority=priority,
                scope=ScopeSpec(operator_ids=[int(op)], node_types=list(node_types)),
                visibility=_SCOPE_VIS, decisions={domain: strategy}))
    return specs


def control_plane(domain_modes: dict) -> List[ControllerSpec]:
    """Build a controllers list for several domains at once from a dict:

        {domain: ("centralized"|"distributed", strategy, opts?)}

    `opts` (optional dict) is passed through to the helper (operator_ids,
    node_types, per, priority, ...)."""
    out: List[ControllerSpec] = []
    for domain, spec in domain_modes.items():
        mode, strategy = spec[0], spec[1]
        opts = spec[2] if len(spec) > 2 else {}
        if mode == "centralized":
            out += centralized(domain, strategy, **opts)
        elif mode == "distributed":
            out += distributed(domain, strategy, **opts)
        else:
            raise ValueError(f"unknown control mode {mode!r} for {domain!r} "
                             "(expected 'centralized' or 'distributed')")
    return out
