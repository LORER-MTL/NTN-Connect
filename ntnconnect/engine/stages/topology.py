# Feeder + backhaul topology stages.

from __future__ import annotations
import numpy as np
import pandas as pd
from math import radians
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@register_stage("topo_feeder", internal=True)
class TopoFeeder(Stage):
    """Feeder link (nearest satellite):
      * per-operator gateway (operator_id >= 0) -> its NEAREST visible satellite
        of its OWN operator: one link.
      * shared gateway (no operator_id, or operator_id < 0) -> the nearest
        visible satellite of EACH operator: one link per operator."""
    reads  = ["satellites_t", "ground_stations_t"]
    writes = ["feeder_active"]
    thread_safe = True

    def run(self, state):
        sats = state.data.get("satellites_t")
        gss  = state.data.get("ground_stations_t")
        if not isinstance(sats, pd.DataFrame) or sats.empty \
                or not isinstance(gss, pd.DataFrame) or gss.empty:
            state.data["feeder_active"] = set(); return
        min_elev = float(self.cfg.handover.min_elevation_deg)

        g_lat = gss["lat"].astype(float).to_numpy()
        g_lon = gss["lon"].astype(float).to_numpy()
        g_ids = gss["node_id"].astype(int).to_numpy()
        s_lat = sats["lat"].astype(float).to_numpy()
        s_lon = sats["lon"].astype(float).to_numpy()
        s_alt = sats["alt_m"].astype(float).to_numpy()
        s_ids = sats["node_id"].astype(int).to_numpy()
        g_op = gss["operator_id"].astype(int).to_numpy() if "operator_id" in gss.columns else None
        s_op = sats["operator_id"].astype(int).to_numpy() if "operator_id" in sats.columns else None

        # N_gs x N_sat geometry, then per gateway pick the nearest VISIBLE
        # same-operator satellite.
        d = np.maximum(_haversine_m(g_lat[:, None], g_lon[:, None],
                                    s_lat[None, :], s_lon[None, :]), 1.0)
        elev = np.degrees(np.arctan2(s_alt[None, :], d))
        slant = np.sqrt(d ** 2 + s_alt[None, :] ** 2)
        valid_elev = elev >= min_elev
        ops = np.unique(s_op) if s_op is not None else np.array([None])
        active = set()
        for i in range(len(g_ids)):
            shared = (g_op is None) or (int(g_op[i]) < 0)
            targets = ops if shared else np.array([g_op[i]])
            for op in targets:                          # one nearest sat per op
                m = valid_elev[i]
                if s_op is not None and op is not None:
                    m = m & (s_op == int(op))
                if not m.any():
                    continue
                j = int(np.argmin(np.where(m, slant[i], np.inf)))
                if np.isfinite(slant[i, j]) and m[j]:
                    active.add((int(g_ids[i]), int(s_ids[j])))
        state.data["feeder_active"] = active


@register_stage("topo_backhaul", internal=True)
class TopoBackhaul(Stage):
    """BS <-> GS pairs within reach via terrestrial fiber (placeholder = all)."""
    reads  = ["base_stations_t", "ground_stations_t"]
    writes = ["backhaul_active"]
    thread_safe = True

    def run(self, state):
        bss = state.data.get("base_stations_t")
        gss = state.data.get("ground_stations_t")
        if not (isinstance(bss, pd.DataFrame) and not bss.empty
                and isinstance(gss, pd.DataFrame) and not gss.empty):
            state.data["backhaul_active"] = set(); return
        # Full cross-product (placeholder semantics: all BS reach all GS)
        b_ids = bss["node_id"].astype(int).to_numpy()
        g_ids = gss["node_id"].astype(int).to_numpy()
        bi, gi = np.meshgrid(np.arange(len(b_ids)),
                             np.arange(len(g_ids)), indexing="ij")
        state.data["backhaul_active"] = {
            (int(b_ids[i]), int(g_ids[j]))
            for i, j in zip(bi.ravel(), gi.ravel())
        }


@register_stage("topology")
class Topology(Stage):
    """Consolidated network-topology stage: builds the backbone connectivity in
    one pass - ISL (sat<->sat: feasibility + activation via the isl strategy
    registry), feeder (sat<->gateway), backhaul (base-station<->gateway).
    Optional (network lane; single-hop access studies skip it). ISL activation
    can be load-aware via a strategy registered in registry.agents.isl_strategy.

    TODO: aerial relay/backhaul for HAPS and UAV (HAPS<->sat, UAV<->sat/gateway)
    - aerial nodes are not yet modeled as relays. When added, write
    `aerial_active` and include them in the network graph (net_build_graph)."""
    reads  = ["satellites_t", "ground_stations_t", "base_stations_t",
              "haps_t", "uavs_t"]
    writes = ["isl_active", "isl_feasibility", "feeder_active",
              "backhaul_active", "aerial_active"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        from ntnconnect.engine.stages.isl import ISLStage
        self._isl = ISLStage(cfg)
        self._feeder = TopoFeeder(cfg)
        self._backhaul = TopoBackhaul(cfg)

    def run(self, state):
        eng = getattr(self, "engine", None)
        for sub in (self._isl, self._feeder, self._backhaul):
            sub.engine = eng
            sub.run(state)
        # TODO: HAPS/UAV aerial relay links (see class docstring).
        state.data.setdefault("aerial_active", set())
