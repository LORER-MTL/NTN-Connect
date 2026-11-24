# Routing stages: build_graph, route_isl, route_e2e, load_balance.

from __future__ import annotations
import json
from typing import Dict
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.registry.agents.routing_strategy import get_routing_strategy

C_LIGHT = 299_792_458.0


@register_stage("net_build_graph", internal=True)
class NetBuildGraph(Stage):
    """Build a unified weighted DiGraph from access + ISL + feeder + backhaul."""
    reads = ["links", "isl_active", "feeder_active", "backhaul_active",
             "satellites_t"]
    writes = ["network_graph"]
    thread_safe = False

    def run(self, state):
        graph: Dict[int, Dict[int, float]] = {}

        def _add(u, v, w):
            graph.setdefault(int(u), {})[int(v)] = float(w)
            graph.setdefault(int(v), {})[int(u)] = float(w)

        # Access links: weight = propagation delay (vectorized fill)
        links = state.data.get("links")
        if isinstance(links, pd.DataFrame) and not links.empty:
            u_arr = links["user_id"].astype(int).to_numpy()
            n_arr = links["node_id"].astype(int).to_numpy()
            d_arr = (links["dist_m"].astype(float).to_numpy() / C_LIGHT)
            for u, n, d in zip(u_arr.tolist(), n_arr.tolist(), d_arr.tolist()):
                _add(u, n, d)

        # ISL edges
        sat_pos = {}
        sats = state.data.get("satellites_t")
        if isinstance(sats, pd.DataFrame) and not sats.empty:
            # Dedupe defensively - read_geometry may not always filter to one tick
            sats_one = sats.drop_duplicates(subset=["node_id"], keep="first")
            sat_pos = sats_one.set_index("node_id")[["lat","lon","alt_m"]].to_dict("index")
        for (a, b) in state.data.get("isl_active", set()):
            pa, pb = sat_pos.get(int(a)), sat_pos.get(int(b))
            if pa is None or pb is None:
                _add(a, b, 1e-3); continue
            d = ((pa["alt_m"]-pb["alt_m"])**2 +
                 (1e5*(pa["lat"]-pb["lat"]))**2 +
                 (1e5*(pa["lon"]-pb["lon"]))**2) ** 0.5
            _add(a, b, d / C_LIGHT)

        # Feeder + backhaul: small fixed weight
        for (a, b) in state.data.get("feeder_active", set()):
            _add(a, b, 1e-3)
        for (a, b) in state.data.get("backhaul_active", set()):
            _add(a, b, 1e-4)

        state.data["network_graph"] = graph


@register_stage("net_route_isl", internal=True)
class NetRouteISL(Stage):
    """ISL-only routes between satellites (precomputed for analysis).
    Stored as state.data['isl_routes'] = {(sat_a, sat_b): Route}."""
    reads = ["network_graph", "satellites_t"]
    writes = ["isl_routes"]
    thread_safe = False

    def run(self, state):
        g = state.data.get("network_graph") or {}
        sats = state.data.get("satellites_t")
        if not isinstance(sats, pd.DataFrame) or sats.empty or not g:
            state.data["isl_routes"] = {}; return
        strat = get_routing_strategy(self.cfg)
        ids = list(sats["node_id"].astype(int).unique())
        out = {}
        # Compute shortest path from each sat to the next-clockwise sat as a
        # representative (full pairwise is N^2 - too heavy for large fleets).
        for i, a in enumerate(ids[:-1]):
            b = ids[(i + 1) % len(ids)]
            r = strat.compute_route(g, a, b, self.cfg)
            if r is not None: out[(a, b)] = r
        state.data["isl_routes"] = out


@register_stage("net_route_e2e", internal=True)
class NetRouteE2E(Stage):
    """For each (user, serving_node, sink_class), compute the end-to-end route."""
    reads = ["network_graph", "assignments", "ground_stations_t"]
    writes = ["routes"]
    thread_safe = False

    def run(self, state):
        g = state.data.get("network_graph") or {}
        assigns = state.data.get("assignments", {}) or {}
        gss = state.data.get("ground_stations_t")
        if not g or not assigns:
            state.data["routes"] = []; return
        strat = get_routing_strategy(self.cfg)
        # Default sink class: pick the nearest GS by node_id
        gs_ids = (list(gss["node_id"].astype(int).unique())
                  if isinstance(gss, pd.DataFrame) and not gss.empty else [])
        routes = []
        for u, s in assigns.items():
            if int(s) < 0: continue
            sink = gs_ids[0] if gs_ids else int(s)
            r = strat.compute_route(g, int(u), int(sink), self.cfg)
            if r is None: continue
            routes.append({
                "user_id": int(u),
                "traffic_class": "internet",
                "path": r.path,
                "total_delay_s": r.total_delay_s,
                "total_hops": r.total_hops,
                "bottleneck_bps": r.bottleneck_bps,
            })
        state.data["routes"] = routes


@register_stage("net_load_balance", internal=True)
class NetLoadBalance(Stage):
    """Placeholder: leaves routes as-is. Real ECMP would split flows across
    equal-cost paths."""
    reads  = ["routes"]
    writes = ["routes"]
    thread_safe = False
    def run(self, state):
        return


@register_stage("routing")
class Routing(Stage):
    """Consolidated routing stage: builds the network graph (access + ISL +
    feeder + backhaul), then computes the routing table with the selected
    routing method (cfg.routing.algorithm via the routing_strategy registry:
    dijkstra | yen_k_shortest). Produces ISL routes + end-to-end user->gateway
    routes, then load-balances. Optional (network lane; single-hop access
    studies skip it)."""
    reads  = ["links", "isl_active", "feeder_active", "backhaul_active",
              "satellites_t", "assignments", "ground_stations_t"]
    writes = ["network_graph", "isl_routes", "routes"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._graph = NetBuildGraph(cfg)
        self._isl   = NetRouteISL(cfg)
        self._e2e   = NetRouteE2E(cfg)
        self._lb    = NetLoadBalance(cfg)

    def run(self, state):
        eng = getattr(self, "engine", None)
        for sub in (self._graph, self._isl, self._e2e, self._lb):
            sub.engine = eng
            sub.run(state)
