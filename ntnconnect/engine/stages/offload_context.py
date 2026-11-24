# Unified offload-input context: the SINGLE input every offload strategy reads.
#
# Task offloading is the edge/MEC orchestration layer: it sits ON TOP of the
# radio+network layers and consumes their state (like ETSI-MEC / EdgeCloudSim).
# So this context gathers, per task, the reachable compute CANDIDATES and the
# NETWORK STATE the orchestrator needs to choose one: the achieved uplink rate
# (from sinr_calc), each candidate's compute capacity + current load, and the
# propagation latency. Strategies differ only in how they pick among candidates.
#
# Decision vs calculation: offload OUTPUT is a decision (task -> compute node);
# the latency/energy CALCULATIONS happen later in work_execute.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


# node-type of each per-node table published to shared state
_NODE_TABLES = {
    "satellites_t":      "satellite",
    "haps_t":            "haps",
    "uavs_t":            "uav",
    "ground_stations_t": "ground_station",
    "base_stations_t":   "base_station",
}


@dataclass
class OffloadCandidate:
    node_id: int
    tier: str              # local | edge | cloud
    rate_bps: float        # achieved uplink rate to reach this node (0 = local)
    capacity_flops: float  # compute capability of the node
    queue_len: int         # current backlog at the node (load)
    latency_ms: float      # one-way propagation to the node


@dataclass
class OffloadContext:
    """Per-task offload inputs. tasks[i] is a task dict; candidates[user_id] is
    the list of reachable compute options for that task's user, each carrying the
    network state (rate/latency) + node state (capacity/load) the strategy needs."""
    tasks: List[dict]
    candidates: Dict[int, List[OffloadCandidate]]
    step_s: float
    local_flops: float = 0.0          # device compute (0 = no local execution)

    def __len__(self):
        return len(self.tasks)


def _node_compute_map(state, cfg):
    """node_id -> (node_type, operator_id) from the per-node tables in state."""
    out = {}
    for key, ntype in _NODE_TABLES.items():
        df = state.data.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty or "node_id" not in df.columns:
            continue
        ids = df["node_id"].astype(int).to_numpy()
        ops = (df["operator_id"].astype(int).to_numpy()
               if "operator_id" in df.columns else [-1] * len(ids))
        for nid, oid in zip(ids, ops):
            out[int(nid)] = (ntype, int(oid))
    return out


def _capability(cfg, node_type, op_id):
    """(compute_role, flops_peak) for a node's type under its operator."""
    op = cfg.operator_by_id(op_id) if op_id >= 0 else (
        cfg.operators[0] if cfg.operators else None)
    if op is None:
        return "none", 0.0
    cap = (op.compute or {}).get(node_type)
    if cap is None:
        return "none", 0.0
    return str(getattr(cap, "compute_role", "none")), float(getattr(cap, "flops_peak", 0.0))


def build_offload_context(state, cfg):
    """Assemble the offload context from shared state. Read-only. Returns None
    when there is nothing to offload."""
    tasks = state.data.get("new_tasks", []) or []
    if not tasks:
        return None

    assigns = state.data.get("assignments", {}) or {}
    links = state.data.get("links")

    # achieved uplink rate per user, from the SERVED link (sinr_calc output)
    user_rate: Dict[int, float] = {}
    if isinstance(links, pd.DataFrame) and not links.empty \
            and "link_rate_bps" in links.columns:
        u = links["user_id"].astype(int).to_numpy()
        nid = links["node_id"].astype(int).to_numpy()
        rate = links["link_rate_bps"].astype(float).to_numpy()
        for k in range(len(links)):
            if assigns.get(int(u[k]), -1) == int(nid[k]):     # served link
                user_rate[int(u[k])] = float(rate[k])

    node_map = _node_compute_map(state, cfg)
    node_load = state.data.get("node_queue_len", {}) or {}     # prev-tick backlog
    # a cloud (ground) compute node, if any: first compute-capable ground station
    cloud_id = cloud_cap = None
    for nid, (ntype, oid) in node_map.items():
        if ntype in ("ground_station", "base_station"):
            role, flops = _capability(cfg, ntype, oid)
            if role in ("mec", "both") and flops > 0:
                cloud_id, cloud_cap = int(nid), flops
                break

    off = cfg.offload
    backbone_rate = float(getattr(off, "backbone_rate_bps", 1.0e9))
    cloud_latency = float(getattr(off, "cloud_latency_ms", 40.0))
    edge_latency  = float(getattr(off, "edge_latency_ms", 5.0))
    local_flops   = float(getattr(off, "local_flops", 0.0))

    candidates: Dict[int, List[OffloadCandidate]] = {}
    for t in tasks:
        uid = int(t["user_id"])
        cand: List[OffloadCandidate] = []
        # local (device) - no transmission
        if local_flops > 0:
            cand.append(OffloadCandidate(uid, "local", 0.0, local_flops, 0, 0.0))
        # edge = serving access node, reached over the achieved uplink rate
        snode = int(assigns.get(uid, -1))
        if snode >= 0 and snode in node_map:
            ntype, oid = node_map[snode]
            role, flops = _capability(cfg, ntype, oid)
            if role in ("mec", "both") and flops > 0:
                cand.append(OffloadCandidate(
                    snode, "edge", float(user_rate.get(uid, 0.0)), flops,
                    int(node_load.get(snode, 0)), edge_latency))
        # cloud = ground compute node, reached over uplink then backbone
        if cloud_id is not None:
            eff = min(float(user_rate.get(uid, backbone_rate)), backbone_rate)
            cand.append(OffloadCandidate(
                cloud_id, "cloud", eff, float(cloud_cap),
                int(node_load.get(cloud_id, 0)), cloud_latency))
        candidates[uid] = cand

    return OffloadContext(tasks=list(tasks), candidates=candidates,
                          step_s=float(cfg.env.step_size_s), local_flops=local_flops)
