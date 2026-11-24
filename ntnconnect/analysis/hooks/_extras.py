# Output hooks - dim/fact normalized.
#
# Hooks write FK-only fact rows. Dimension tables (time, nodes,
# operators, users, ho_type) are written ONCE by stages.dimensions.
# Downstream.read() joins dims transparently.

from __future__ import annotations
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook
from ntnconnect.common.schemas import (
    PER_USER_KPIS, HO_EVENTS, ACTIVE_ISL, NODE_STATE, TASKS, PER_OPERATOR_KPIS,
)
from ntnconnect.common.schemas.dims import encode_ho_type


class PerUserKPIsHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = PER_USER_KPIS.collector(engine)

    def on_step_end(self, state):
        assigns = state.data.get("assignments", {}) or {}
        ho = {r["user_id"]: r for r in state.data.get("ho_classified", [])}
        for u, s in assigns.items():
            cls = ho.get(int(u), {})
            self.c.add({
                "t_idx":           state.t_idx,
                "user_id":         int(u),
                "serving_node_id": int(s),
                "prev_node_id":    int(cls.get("prev_node_idx", -1)),
                "ho_type_id":      encode_ho_type(cls.get("ho_type", "none")),
                "is_in_outage":    int(int(s) < 0),
            })

    def on_simulation_end(self, state):
        self.c.flush()


class NodeStateHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = NODE_STATE.collector(engine)

    def on_step_end(self, state):
        df = state.data.get("node_state_now")
        if not isinstance(df, pd.DataFrame) or df.empty: return
        for _, r in df.iterrows():
            self.c.add({
                "t_idx":           state.t_idx,
                "node_id":         int(r.get("node_id", r.get("node_idx", -1))),
                "cpu_util_pct":    None,
                "ram_used_gb":     None,
                "storage_used_gb": None,
                "queue_length":    int(r.get("queue_length", 0)),
                "power_w":         None,
                "n_tasks_running": int(r.get("n_tasks_running", 0)),
                "n_tasks_queued":  None,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class TasksHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = TASKS.collector(engine)
        self.bucket_size = max(1, self.engine.cfg.storage.tick_bucket_size)

    def on_step_end(self, state):
        for t in state.data.get("task_events", []):
            self.c.add({
                "t_idx":              state.t_idx,
                "task_id":            int(t.get("task_id", -1)),
                "user_id":            int(t.get("user_id", -1)),
                "arrival_t_idx":      int(t.get("arrival_t_idx", state.t_idx)),
                "cpu_required_flops": float(t.get("cpu_required_flops", 0.0)),
                "data_in_mb":         float(t.get("data_in_mb", 0.0)),
                "data_out_mb":        float(t.get("data_out_mb", 0.0)),
                "deadline_ms":        float(t.get("deadline_ms", 0.0)),
                "priority":           t.get("priority", "normal"),
                "assigned_node_id":   int(t.get("assigned_node_idx",
                                                t.get("assigned_node_id", -1))),
                "status":             t.get("status", "queued"),
                "tick_bucket":        state.t_idx // self.bucket_size,
            })

    def on_simulation_end(self, state):
        self.c.flush()


class HOEventsHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = HO_EVENTS.collector(engine)

    def on_step_end(self, state):
        for r in state.data.get("ho_classified", []):
            ho_name = r.get("ho_type") or "none"
            if ho_name == "none": continue
            self.c.add({
                "t_idx":           state.t_idx,
                "user_id":         int(r["user_id"]),
                "serving_node_id": int(r.get("serving_node_idx", -1)),
                "prev_node_id":    int(r.get("prev_node_idx", -1)),
                "ho_type_id":      encode_ho_type(ho_name),
            })

    def on_simulation_end(self, state):
        self.c.flush()


class ActiveISLHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = ACTIVE_ISL.collector(engine)

    def on_step_end(self, state):
        for (a, b) in state.data.get("isl_active", set()):
            self.c.add({
                "t_idx": state.t_idx,
                "sat_a": int(a),
                "sat_b": int(b),
            })

    def on_simulation_end(self, state):
        self.c.flush()


class PerOperatorKPIHook(SimulationHook):
    def __init__(self, engine):
        self.engine = engine
        self.c = PER_OPERATOR_KPIS.collector(engine)

    def on_step_end(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        for op_id, g in links.groupby("operator_id", observed=False):
            self.c.add({
                "t_idx":        state.t_idx,
                "operator_id":  int(op_id),
                "n_links":      int(len(g)),
                "mean_snr_db":  float(g.get("snr_db",  pd.Series([float("nan")])).mean()),
                "mean_sinr_db": float(g.get("sinr_db", pd.Series([float("nan")])).mean()),
            })

    def on_simulation_end(self, state):
        self.c.flush()


# Registration: stable hook names referenced by cfg.outputs.toggles
@register_hook("per_user_kpis")
def _b1(engine): return PerUserKPIsHook(engine)
@register_hook("node_state")
def _b2(engine): return NodeStateHook(engine)
@register_hook("tasks")
def _b3(engine): return TasksHook(engine)
@register_hook("ho_events")
def _b4(engine): return HOEventsHook(engine)
@register_hook("active_isl")
def _b5(engine): return ActiveISLHook(engine)
@register_hook("per_operator_kpis")
def _b6(engine): return PerOperatorKPIHook(engine)
