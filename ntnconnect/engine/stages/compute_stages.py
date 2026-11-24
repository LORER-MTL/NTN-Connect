# Compute/edge lane + per-tick state stages:
#   workload   - task arrivals
#   offload    - placement DECISION (registry-driven; reads achieved rate)
#   scheduling - per-node queue + policy (persists across ticks)
#   cache      - per-node LRU/LFU/FIFO
#   node_state - per-node load snapshot (node_state_now)
#   ho_classify- handover-type classification
#   rrm        - LEGACY name: a per-tick KPI-row aggregator (kpi_row), NOT
#                resource management. The real RRM is the `radio` stage; kept
#                under this name for config back-compat.

from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage, register_compute_stage
from ntnconnect.registry.agents.handover_strategy import classify_ho


@register_compute_stage("workload")
class WorkloadStage(Stage):
    reads = []
    writes = ["new_tasks"]
    thread_safe = False
    def run(self, state):
        if not self.cfg.workload.enabled:
            state.data["new_tasks"] = []
            return
        # Poisson arrivals per user (stub)
        lam = self.cfg.workload.arrival_rate_per_user_per_s * self.cfg.env.step_size_s
        n = np.random.poisson(lam)
        state.data["new_tasks"] = [
            {"task_id": int(state.t_idx * 10_000 + i),
             "user_id": int(1 + (i % 10)),
             "arrival_t_idx": state.t_idx,
             "cpu_required_flops": float(self.cfg.workload.cpu_required_flops_mean),
             "data_in_mb": float(self.cfg.workload.data_in_mb_mean),
             "data_out_mb": float(self.cfg.workload.data_out_mb_mean),
             "deadline_ms": float(self.cfg.workload.deadline_ms_mean),
             "priority": "normal"}
            for i in range(n)
        ]


@register_compute_stage("offload")
class OffloadStage(Stage):
    """Offload/placement DECISION: choose the compute node for each task. Reads
    the network state produced by the radio+network lanes (achieved uplink rate
    from sinr_calc, node compute capacity + load, propagation latency) via the
    unified OffloadContext, then dispatches to the offload strategy registry
    (cfg.offload.target_policy). Output is a decision only - task -> node; the
    latency/energy CALCULATIONS are done later in work_execute."""
    reads = ["new_tasks", "links", "assignments", "satellites_t",
             "ground_stations_t", "base_stations_t", "haps_t", "uavs_t"]
    writes = ["task_assignments"]
    thread_safe = False

    def run(self, state):
        from ntnconnect.engine.stages.offload_context import build_offload_context
        from ntnconnect.registry.agents.offload_strategy import get_offload_strategy
        if not self.cfg.workload.enabled:
            state.data["task_assignments"] = []
            return
        ctx = build_offload_context(state, self.cfg)
        if ctx is None:
            state.data["task_assignments"] = []
            return
        placement = get_offload_strategy(self.cfg).assign(ctx)   # {task_id: node_id}
        state.data["task_assignments"] = [
            {**t, "assigned_node_idx": int(placement.get(int(t["task_id"]), -1))}
            for t in ctx.tasks
        ]


@register_compute_stage("scheduling")
class SchedulingStage(Stage):
    """F7 - Real per-node queue + policy.

    Maintains a FIFO/EDF/SRTF queue per assigned node with a hard
    `cfg.scheduler.queue_capacity`. On each tick:
      1. Admit newly arrived task_assignments to their node's queue.
         If the queue is full, mark the task 'dropped' (overflow).
      2. Sort the queue per policy:
           fifo - arrival order
           edf  - earliest deadline first  (deadline_ms - elapsed_ms)
           srtf - shortest remaining time  (cpu_required_flops)
      3. Run K tasks per node where K = max parallel slots (here 1; the
         compute_run stage shares capacity equally among running tasks).
      4. Emit task_events with status in {running, queued, dropped}.

    Latency is tracked from arrival_t_idx -> first 'running' tick on the
    task itself ('queue_wait_ticks') so KPIs can use it.
    """
    reads  = ["task_assignments"]
    writes = ["task_events"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        # node_id -> list of pending task dicts (admitted, not yet 'running')
        self._queues: dict = {}

    def _sort_key(self, task: dict, now_t: int):
        pol = (self.cfg.scheduler.policy or "fifo").lower()
        if pol == "edf":
            # smaller "slack" wins; slack = deadline_ms - elapsed_ms
            elapsed_ms = (now_t - int(task.get("arrival_t_idx", now_t))) * \
                         float(self.cfg.env.step_size_s) * 1000.0
            return float(task.get("deadline_ms", 1e12)) - elapsed_ms
        if pol == "srtf":
            return float(task.get("cpu_required_flops", 0.0))
        return int(task.get("arrival_t_idx", now_t))  # fifo

    def run(self, state):
        assigns = state.data.get("task_assignments", []) or []
        cap = max(1, int(self.cfg.scheduler.queue_capacity))
        events = []

        # 1) Admit newcomers
        for t in assigns:
            nid = int(t.get("assigned_node_idx", -1))
            if nid < 0:
                events.append({**t, "status": "dropped",
                               "drop_reason": "no_assignment",
                               "queue_wait_ticks": 0})
                continue
            q = self._queues.setdefault(nid, [])
            if len(q) >= cap:
                events.append({**t, "status": "dropped",
                               "drop_reason": "queue_overflow",
                               "queue_wait_ticks": 0})
                continue
            q.append(dict(t))

        # 2) Dispatch per node, ordered by policy
        for nid, q in self._queues.items():
            if not q: continue
            q.sort(key=lambda r: self._sort_key(r, state.t_idx))
            # Slot 0 -> running; rest -> queued (single-slot scheduler;
            # compute_run.WorkExecute shares the node capacity across all
            # tasks it sees as 'running').
            head = q[0]
            head["status"] = "running"
            head["queue_wait_ticks"] = state.t_idx - \
                                       int(head.get("arrival_t_idx", state.t_idx))
            events.append(head)
            for tail in q[1:]:
                tail["status"] = "queued"
                events.append(tail)

        state.data["task_events"] = events
        # Remove tasks that the WorkExecute will mark done; we keep them in
        # the queue for now and prune on next tick if their status flipped.
        for nid in list(self._queues):
            self._queues[nid] = [t for t in self._queues[nid]
                                 if t.get("status") != "done"]


@register_compute_stage("cache")
class CacheStage(Stage):
    """F4 - Real per-node LRU/LFU/FIFO cache.

    Cache key = `data_in_mb` size bucket (10 MB granularity) of each
    task seen by this node. The cache is touched when the task arrives
    and a hit/miss decision is emitted into `state.data["cache_state"]`
    so the audit hook and KPI (`cache_hit_rate`) can read it.
    """
    reads  = ["task_assignments"]
    writes = ["cache_state"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._caches: dict = {}                  # node_id -> OrderedDict (LRU)
        self._lfu_counts: dict = {}              # node_id -> {key: count}
        self._stats: dict = {}                   # node_id -> {hit, miss}

    def _ensure_cache(self, nid: int):
        import collections
        if nid not in self._caches:
            self._caches[nid] = collections.OrderedDict()
            self._lfu_counts[nid] = {}
            self._stats[nid] = {"hit": 0, "miss": 0}

    def _touch(self, nid: int, key: str, capacity: int, policy: str):
        c = self._caches[nid]
        counts = self._lfu_counts[nid]
        stats = self._stats[nid]
        if key in c:
            stats["hit"] += 1
            counts[key] = counts.get(key, 0) + 1
            if policy == "lru":
                c.move_to_end(key)
            return True
        # miss → admit
        stats["miss"] += 1
        c[key] = True
        counts[key] = counts.get(key, 0) + 1
        if len(c) > capacity:
            if policy == "lfu":
                victim = min(counts, key=counts.get)
            elif policy == "fifo":
                victim = next(iter(c))
            else:  # lru
                victim = next(iter(c))
            c.pop(victim, None); counts.pop(victim, None)
        return False

    def run(self, state):
        cfg = self.cfg.cache
        cap = max(1, int(cfg.capacity_items))
        policy = (cfg.policy or "lru").lower()
        if not bool(cfg.populate_on_task):
            state.data["cache_state"] = {}
            return
        for t in state.data.get("task_assignments", []) or []:
            nid = int(t.get("assigned_node_idx", -1))
            if nid < 0: continue
            key = f"sz_{int(float(t.get('data_in_mb', 0.0)) // 10)}"
            self._ensure_cache(nid)
            self._touch(nid, key, cap, policy)
        state.data["cache_state"] = {
            "by_node": {int(n): {"n_items": len(c),
                                 "hits":  self._stats[n]["hit"],
                                 "misses": self._stats[n]["miss"]}
                        for n, c in self._caches.items()}
        }


@register_compute_stage("node_state")
class NodeStateStage(Stage):
    reads = ["task_events", "links"]
    writes = ["node_state_now"]
    thread_safe = True
    def run(self, state):
        # Build a node_id -> (node_type, operator_id) lookup from the links
        # table so downstream stages (state_energy) can resolve capabilities.
        links = state.data.get("links")
        nid_meta = {}
        if isinstance(links, pd.DataFrame) and not links.empty \
           and {"node_id","node_type","operator_id"}.issubset(links.columns):
            nid_meta = (links[["node_id","node_type","operator_id"]]
                              .drop_duplicates(subset=["node_id"])
                              .set_index("node_id")
                              .to_dict("index"))

        events = state.data.get("task_events", [])
        load = {}
        for e in events:
            nid = e.get("assigned_node_idx", -1)
            if nid >= 0: load[nid] = load.get(nid, 0) + 1

        rows = []
        for nid, n in load.items():
            meta = nid_meta.get(int(nid), {})
            rows.append({
                "timestamp":       state.timestamp,
                "t_idx":           state.t_idx,
                "node_id":         int(nid),    # canonical FK
                "node_idx":        int(nid),    # legacy alias for older readers
                "node_type":       meta.get("node_type"),
                "operator_id":     int(meta["operator_id"])
                                    if "operator_id" in meta else None,
                "n_tasks_running": int(n),
                "queue_length":    int(n),
            })
        state.data["node_state_now"] = pd.DataFrame(rows)


@register_compute_stage("rrm")
class RRMStage(Stage):
    """Aggregate KPI scalars per tick from the per-link table."""
    reads = ["links"]
    writes = ["kpi_row"]
    thread_safe = True
    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            state.data["kpi_row"] = {"timestamp": state.timestamp,
                                     "t_idx": state.t_idx,
                                     "n_links": 0}
            return
        row = {
            "timestamp": state.timestamp,
            "t_idx": state.t_idx,
            "n_links": int(len(links)),
            "snr_db":  float(links["snr_db"].mean()) if "snr_db" in links else float("nan"),
            "sinr_db": float(links["sinr_db"].mean()) if "sinr_db" in links else float("nan"),
            "path_loss_db": float(links["path_loss_db"].mean()) if "path_loss_db" in links else float("nan"),
        }
        state.data["kpi_row"] = row


@register_stage("ho_classify", internal=True)
class HOClassifyStage(Stage):
    reads = ["assignments", "prev_assign_snapshot", "links"]
    writes = ["ho_classified"]
    thread_safe = False
    def run(self, state):
        ass  = state.data.get("assignments", {})
        prev = state.data.get("prev_assign_snapshot", {})
        links = state.data.get("links")
        # node_idx -> (node_type, op_id)
        type_op = {}
        if isinstance(links, pd.DataFrame) and not links.empty:
            dd = links[["node_id", "node_type", "operator_id"]] \
                    .drop_duplicates()
            type_op = {int(nid): (str(nt), int(op)) for nid, nt, op in
                       zip(dd["node_id"].to_numpy(),
                           dd["node_type"].to_numpy(),
                           dd["operator_id"].to_numpy())}

        classified = []
        for u, s in ass.items():
            p = prev.get(u, -1)
            new_type, new_op = type_op.get(s, (None, -1)) if s >= 0 else (None, -1)
            prev_type, prev_op = type_op.get(p, (None, -1)) if p >= 0 else (None, -1)
            ho = classify_ho(prev_type, prev_op, new_type, new_op)
            classified.append({
                "user_id": int(u),
                "serving_node_idx": int(s), "prev_node_idx": int(p),
                "serving_node_type": new_type, "prev_node_type": prev_type,
                "serving_operator_id": int(new_op), "prev_operator_id": int(prev_op),
                "ho_type": ho,
            })
        state.data["ho_classified"] = classified
