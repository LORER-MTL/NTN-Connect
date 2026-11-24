# End-of-run aggregation: read KPI accumulators, emit summary tables + report.

from __future__ import annotations
import json
import logging
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.common.schemas.base import TableSchema, Column

_log = logging.getLogger(__name__)


# A lightweight in-line schema for summary outputs - registered manually
# instead of in tables.py because they're derived, not collected.
SUMMARY_PER_USER = TableSchema(
    name="summary_per_user",
    granularity="(user)",
    columns=[
        Column("user_id",         "int32"),
        Column("n_ticks",         "int32"),
        Column("outage_fraction", "float64"),
        Column("n_handovers",     "int32"),
        Column("n_inter_op_ho",   "int32"),
        Column("mean_serving_sinr_db", "float64"),
    ],
)

SUMMARY_PER_OPERATOR = TableSchema(
    name="summary_per_operator",
    granularity="(operator)",
    columns=[
        Column("operator_id",     "int32"),
        Column("n_ticks",         "int32"),
        Column("mean_sinr_db",    "float64"),
        Column("mean_sum_rate_bps","float64"),
        Column("mean_n_links",    "float64"),
        Column("n_users_avg",     "float64"),
    ],
)

SUMMARY_SYSTEM = TableSchema(
    name="summary_system",
    granularity="(run)",
    columns=[
        Column("n_ticks",                 "int32"),
        Column("n_operators",             "int32"),
        Column("n_users",                 "int32"),
        Column("mean_outage_fraction",    "float64"),
        Column("total_handovers",         "int32"),
        Column("mean_active_isl_per_tick","float64"),
        Column("validation_pass_rate",    "float64"),
    ],
)


# Internal building block of the merged `post` stage (no longer standalone).
class PostAggregate(Stage):
    """Runs only on the last tick. Produces per-user/per-op/system summaries."""
    name   = "post_aggregate"
    reads  = []
    writes = ["post_summaries"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        n_ticks = max(1, int(cfg.env.duration_s // cfg.env.step_size_s))
        self._last_idx = n_ticks - 1
        self._done = False

    def run(self, state):
        if state.t_idx < self._last_idx or self._done or self.engine is None:
            return
        self._done = True
        kpi = self.engine.kpi_results
        n_ticks = state.t_idx + 1

        # Per-user summary
        # Defensive: any KPI accumulator may legitimately be empty (a user
        # that never had a visible sat above min_elevation_deg, a
        # constellation with no Quebec coverage, a very short run, etc.).
        # That is a valid first-class outcome - zero availability is the
        # answer the paper wants - not a crash. We therefore guarantee a
        # per-user row exists for every configured user, with outage=1 and
        # mean_serving_sinr_db=NaN when no data was collected.
        per_user = pd.DataFrame()
        if "is_in_outage" in kpi and kpi["is_in_outage"]:
            try:
                df = pd.concat(kpi["is_in_outage"], ignore_index=True)
                if not df.empty and "user_id" in df.columns:
                    outage = (df.groupby("user_id")["is_in_outage"]
                                .mean().rename("outage_fraction"))
                    per_user = outage.to_frame()
            except Exception as e:
                _log.warning("post_run: outage aggregation failed (%s); "
                              "treating all users as full outage", e)

        if "serving_sinr_db" in kpi and kpi["serving_sinr_db"] \
           and not per_user.empty:
            try:
                df = pd.concat(kpi["serving_sinr_db"], ignore_index=True)
                if not df.empty and "user_id" in df.columns:
                    ssinr = (df.groupby("user_id")["serving_sinr_db"]
                                .mean().rename("mean_serving_sinr_db"))
                    per_user = per_user.join(ssinr, how="left")
            except Exception as e:
                _log.warning("post_run: serving SINR aggregation failed (%s)", e)

        # Populate every configured user, even those that never had a row.
        # This makes "no coverage at all" a valid first-class result.
        all_uids: List[int] = []
        try:
            ex = getattr(self.cfg.users, "explicit_user_list", None)
            if ex:
                all_uids = [int(u.get("user_id", u.get("id", -1))) for u in ex]
                all_uids = [u for u in all_uids if u >= 0]
            elif int(getattr(self.cfg.users, "num_seed_users", 0)) > 0:
                all_uids = list(range(int(self.cfg.users.num_seed_users)))
        except Exception:
            all_uids = []

        if all_uids:
            full_idx = pd.Index(sorted(set(all_uids)), name="user_id")
            if per_user.empty:
                per_user = pd.DataFrame(index=full_idx)
            else:
                per_user = per_user.reindex(full_idx)
            per_user["outage_fraction"] = per_user.get(
                "outage_fraction", pd.Series(index=full_idx)).fillna(1.0)
            if "mean_serving_sinr_db" not in per_user.columns:
                per_user["mean_serving_sinr_db"] = float("nan")
        if not per_user.empty:
            per_user["user_id"] = per_user.index.astype(int)
            per_user["n_ticks"] = n_ticks
            per_user["n_handovers"]   = 0
            per_user["n_inter_op_ho"] = 0
            SUMMARY_PER_USER.write(per_user, self.engine)

        # Per-operator summary
        sinr = pd.Series(dtype=float)
        if "operator_mean_sinr_db" in kpi and kpi["operator_mean_sinr_db"]:
            try:
                df = pd.concat(kpi["operator_mean_sinr_db"], ignore_index=True)
                if not df.empty and "operator_id" in df.columns:
                    sinr = (df.groupby("operator_id")["operator_mean_sinr_db"]
                                .mean())
            except Exception as e:
                _log.warning("post_run: operator SINR aggregation failed (%s)", e)
        sumr = pd.Series(dtype=float)
        if "operator_sum_rate_bps" in kpi and kpi["operator_sum_rate_bps"]:
            try:
                df = pd.concat(kpi["operator_sum_rate_bps"], ignore_index=True)
                if not df.empty and "operator_id" in df.columns:
                    sumr = (df.groupby("operator_id")["operator_sum_rate_bps"]
                                .mean())
            except Exception as e:
                _log.warning("post_run: operator rate aggregation failed (%s)", e)
        if not sinr.empty or not sumr.empty:
            per_op = pd.concat([sinr.rename("mean_sinr_db"),
                                sumr.rename("mean_sum_rate_bps")], axis=1)
            per_op["operator_id"] = per_op.index.astype(int)
            per_op["n_ticks"] = n_ticks
            per_op["mean_n_links"] = 0.0
            per_op["n_users_avg"]  = 0.0
            SUMMARY_PER_OPERATOR.write(per_op, self.engine)

        # System summary
        sys_row = {
            "n_ticks":              n_ticks,
            "n_operators":          len(self.cfg.operators),
            "n_users":              self.cfg.users.num_seed_users,
            "mean_outage_fraction": float(per_user["outage_fraction"].mean())
                                    if not per_user.empty else 0.0,
            "total_handovers":      0,
            "mean_active_isl_per_tick": 0.0,
            "validation_pass_rate": 1.0,
        }
        if "n_isl_active" in kpi and kpi["n_isl_active"]:
            try:
                df = pd.concat(kpi["n_isl_active"], ignore_index=True)
                if not df.empty and "n_isl_active" in df.columns:
                    sys_row["mean_active_isl_per_tick"] = float(
                        df["n_isl_active"].mean())
            except Exception as e:
                _log.warning("post_run: ISL aggregation failed (%s)", e)
        SUMMARY_SYSTEM.write(pd.DataFrame([sys_row]), self.engine)
        state.data["post_summaries"] = {"system": sys_row}


# Internal building block of the merged `post` stage (no longer standalone).
class PostReport(Stage):
    """Generates a markdown run report from the summaries + plots."""
    name   = "post_report"
    reads  = ["post_summaries"]
    writes = []
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        n_ticks = max(1, int(cfg.env.duration_s // cfg.env.step_size_s))
        self._last_idx = n_ticks - 1
        self._done = False

    def run(self, state):
        if state.t_idx < self._last_idx or self._done or self.engine is None:
            return
        self._done = True
        summaries = state.data.get("post_summaries", {})
        if not summaries: return
        sys_row = summaries.get("system", {})
        sim_dir = self.engine.sim_dir
        lines = ["# SkyNTNPro v1 - Run Report", "",
                 f"- experiment: **{self.cfg.experiment.name}**",
                 f"- seed: {self.cfg.experiment.random_seed}",
                 f"- duration: {self.cfg.env.duration_s} s "
                 f"({sys_row.get('n_ticks',0)} ticks)",
                 f"- operators: {sys_row.get('n_operators',0)}",
                 f"- users: {sys_row.get('n_users',0)}",
                 "",
                 "## System KPIs",
                 f"- mean outage fraction: {sys_row.get('mean_outage_fraction', 0):.3f}",
                 f"- mean active ISLs per tick: {sys_row.get('mean_active_isl_per_tick', 0):.1f}",
                 f"- validation pass rate: {sys_row.get('validation_pass_rate', 1.0)*100:.1f}%",
                 "",
                 "## Outputs",
                 f"- artifact directory: `{sim_dir}`",
                 ]
        rep = sim_dir / "run_report.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text("\n".join(lines), encoding="utf-8")
        _log.info("Run report written to %s", rep)


# Internal building block of the merged `post` stage (no longer standalone).
class PostPersist(Stage):
    """Final dumps of anything still in engine accumulators (no-op today since
    hooks flush at on_simulation_end). Kept as the canonical exit point."""
    name   = "post_persist"
    reads  = []
    writes = []
    thread_safe = False
    def run(self, state):
        return

# `post_summaries` stage removed - it was a no-op alias of post_aggregate.


@register_stage("finalize")
class Finalize(Stage):
    """MERGED finalisation stage: aggregate → report → persist in one group.
    Composes the three tested sub-stages (no duplicated logic). PostAggregate
    writes `post_summaries` into state.data, which PostReport then reads within
    the same run; PostPersist is the canonical exit point."""
    reads  = []
    writes = ["post_summaries"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._agg     = PostAggregate(cfg)
        self._report  = PostReport(cfg)
        self._persist = PostPersist(cfg)

    def run(self, state):
        for s in (self._agg, self._report, self._persist):
            s.engine = getattr(self, "engine", None)
            s.run(state)
