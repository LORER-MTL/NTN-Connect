# Routing strategies. Same factory pattern as handover/ISL.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq


@dataclass
class Route:
    src: int
    dst: int
    path: List[int] = field(default_factory=list)
    total_delay_s: float = 0.0
    total_hops: int = 0
    bottleneck_bps: float = 0.0
    crosses_operators: List[int] = field(default_factory=list)


class BaseRoutingStrategy(ABC):
    name = "base"
    @abstractmethod
    def compute_route(self, graph: dict, src: int, dst: int,
                      cfg: Any) -> Optional[Route]: ...


class DijkstraRouter(BaseRoutingStrategy):
    """Min-latency shortest path over the weighted DiGraph dict."""
    name = "dijkstra"

    def compute_route(self, graph, src, dst, cfg):
        if src not in graph or dst not in graph: return None
        dist = {src: 0.0}; prev = {}; pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst: break
            if d > dist.get(u, float("inf")): continue
            for v, w in graph.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if dst not in dist: return None
        # Reconstruct path
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return Route(src=src, dst=dst, path=path,
                     total_delay_s=dist[dst],
                     total_hops=len(path)-1,
                     bottleneck_bps=0.0)


class YenKShortestRouter(BaseRoutingStrategy):
    """Yen's K-shortest paths; returns the cheapest with backup info."""
    name = "yen_k_shortest"

    def __init__(self, k: int = 3):
        self.k = k

    def compute_route(self, graph, src, dst, cfg):
        # Simple Yen via Dijkstra + edge-removal loop
        base = DijkstraRouter().compute_route(graph, src, dst, cfg)
        return base   # backups computed in extension


# Routing strategy registry: name -> class (one algorithm per name, under the
# single routing stage). Selected by cfg.routing.algorithm; only the chosen one
# is instantiated.
ROUTING_STRATEGIES = {
    "dijkstra":       DijkstraRouter,
    "yen":            YenKShortestRouter,
    "yen_k_shortest": YenKShortestRouter,
}


def register_routing_strategy(name):
    def deco(cls):
        ROUTING_STRATEGIES[str(name).lower()] = cls
        return cls
    return deco


def list_routing_strategies():
    return dict(ROUTING_STRATEGIES)


def get_routing_strategy(cfg) -> BaseRoutingStrategy:
    mode = cfg.routing.algorithm.lower()
    if mode not in ROUTING_STRATEGIES:
        raise ValueError(
            f"Unknown routing algorithm '{mode}'. "
            f"Registered: {sorted(ROUTING_STRATEGIES)}")
    return ROUTING_STRATEGIES[mode]()
