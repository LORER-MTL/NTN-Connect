# Compute execution + state telemetry stages.

from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_compute_stage


@register_compute_stage("work_execute")
class WorkExecute(Stage):
    """Advance running tasks each tick by node CPU capacity.

    Tasks track remaining FLOPS on `engine.task_state`. When remaining <= 0,
    status flips to 'done'.
    """
    reads  = ["task_events", "node_state_now"]
    writes = ["task_events"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._task_remaining: dict = {}     # task_id -> remaining flops
        self._task_assigned: dict = {}      # task_id -> node_id

    def run(self, state):
        events = state.data.get("task_events", [])
        if not events: return
        cfg = self.cfg
        step_s = float(cfg.env.step_size_s)
        # Per-node available FLOPS (= peak / # tasks running there this tick)
        per_node_tasks: dict = {}
        for t in events:
            if t.get("status") == "running":
                per_node_tasks.setdefault(int(t.get("assigned_node_idx",
                                                    t.get("assigned_node_id", -1))), []).append(t)

        # Map node_id -> capacity
        op_caps = {}
        for op in cfg.operators:
            for ntype, cap in op.compute.items():
                op_caps[(op.id, ntype)] = cap
        # We'll look up node_type/op_id via assignments path or rely on
        # node_state_now.
        ns = state.data.get("node_state_now")
        nid_to_op = {}
        if isinstance(ns, pd.DataFrame) and not ns.empty:
            nid_to_op = ns.set_index("node_id" if "node_id" in ns.columns else "node_idx")[
                ["operator_id","node_type"]].to_dict("index")

        out = []
        for t in events:
            tid = int(t.get("task_id", -1))
            status = t.get("status", "queued")
            nid = int(t.get("assigned_node_idx", t.get("assigned_node_id", -1)))
            if status == "dropped":
                out.append(t); continue
            if status == "queued":
                # Move to running
                t = dict(t); t["status"] = "running"
            if tid not in self._task_remaining:
                self._task_remaining[tid] = float(t.get("cpu_required_flops", 0.0))
                self._task_assigned[tid] = nid
            # Spend flops this tick
            cap = None
            meta = nid_to_op.get(nid)
            if meta:
                cap = op_caps.get((int(meta["operator_id"]), str(meta["node_type"])))
            if cap is not None:
                # Share node's capacity across running tasks
                n_share = max(1, len(per_node_tasks.get(nid, [])))
                flops_this_tick = (float(cap.flops_peak) * step_s) / n_share
                self._task_remaining[tid] -= flops_this_tick
            else:
                self._task_remaining[tid] -= float(t.get("cpu_required_flops", 0)) / 2.0
            if self._task_remaining[tid] <= 0:
                t = dict(t); t["status"] = "done"
            out.append(t)
        state.data["task_events"] = out


@register_compute_stage("state_energy")
class StateEnergy(Stage):
    """Per-node energy accounting per tick. Writes node_state_now rows
    augmented with power_w + energy_joules_tick."""
    reads  = ["node_state_now", "links", "assignments"]
    writes = ["node_state_now"]
    thread_safe = False

    def run(self, state):
        ns = state.data.get("node_state_now")
        if not isinstance(ns, pd.DataFrame) or ns.empty: return
        cfg = self.cfg
        step_s = float(cfg.env.step_size_s)
        op_caps = {}
        for op in cfg.operators:
            for ntype, cap in op.compute.items():
                op_caps[(op.id, ntype)] = cap
        # P_total = P_idle + util * peak_flops * P_per_flop_j (vectorized)
        out = ns.copy()
        ntype_arr = out.get("node_type", pd.Series([""]*len(out))).astype(str).to_numpy()
        op_arr    = out.get("operator_id", pd.Series([-1]*len(out))).fillna(-1).astype(int).to_numpy()
        n_run     = out.get("n_tasks_running", pd.Series([0]*len(out))).fillna(0).astype(int).to_numpy()

        p_idle   = np.zeros(len(out), dtype=float)
        p_comp_f = np.zeros(len(out), dtype=float)  # peak_flops * power_per_flop_j
        for i, (oid, nt) in enumerate(zip(op_arr.tolist(), ntype_arr.tolist())):
            cap = op_caps.get((int(oid), nt))
            if cap is None: continue
            p_idle[i]   = float(cap.power_idle_w)
            p_comp_f[i] = float(cap.flops_peak) * float(cap.power_per_flop_j)
        util = np.minimum(1.0, n_run * 0.3)
        powers = p_idle + util * p_comp_f
        out["power_w"]      = powers
        out["energy_j_tick"] = powers * step_s
        state.data["node_state_now"] = out


@register_compute_stage("state_link")
class StateLink(Stage):
    """Per-link utilization snapshot. Writes link_state_now."""
    reads  = ["links", "rb_alloc_map", "assignments"]
    writes = ["link_state_now"]
    thread_safe = True

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            state.data["link_state_now"] = pd.DataFrame(); return
        rb = state.data.get("rb_alloc_map", {}) or {}
        if not rb:
            state.data["link_state_now"] = pd.DataFrame(); return
        # Vectorized lookup: build pair keys once, filter to those in rb.
        nid = links["node_id"].astype(int).to_numpy()
        uid = links["user_id"].astype(int).to_numpy()
        keys = list(zip(nid.tolist(), uid.tolist()))
        mask = np.array([k in rb for k in keys])
        if not mask.any():
            state.data["link_state_now"] = pd.DataFrame(); return
        sel_keys = [keys[i] for i in np.where(mask)[0]]
        entries  = [rb[k] for k in sel_keys]
        n_rbs       = np.array([e["n_rbs"]          for e in entries], dtype=int)
        bw_alloc    = np.array([e["bw_alloc_hz"]    for e in entries], dtype=float)
        n_rbs_total = np.array([max(e["n_rbs_total"], 1) for e in entries], dtype=int)
        state.data["link_state_now"] = pd.DataFrame({
            "t_idx":       np.full(mask.sum(), state.t_idx, dtype=int),
            "user_id":     uid[mask],
            "node_id":     nid[mask],
            "n_rbs_alloc": n_rbs,
            "bw_alloc_hz": bw_alloc,
            "utilization": n_rbs / n_rbs_total,
        })
