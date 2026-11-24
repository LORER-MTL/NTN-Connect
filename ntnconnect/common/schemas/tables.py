# SkyNTNPro v1 - Output table catalog
#
# Every table the simulator can produce is declared here, once.
# Hooks reference these schemas; the user toggles them in YAML.
# To activate a new column: add a Column row here, update the hook
# that fills it. The schema validates at write time.

from __future__ import annotations
from typing import Dict
from ntnconnect.common.schemas.base import Column, TableSchema, DimJoin


# Phase-1 node tables (written by the geometry generators)
# Already partitioned by operator_id via cfg.storage.partition_by.
def _node_columns():
    return [
        Column("timestamp",    "datetime64[ns]"),
        Column("t_idx",        "int32"),
        Column("node_id",      "int64"),
        Column("node_type",    "string",
               enum=["satellite", "haps", "uav", "ground_station", "base_station"]),
        Column("operator_id",  "int32"),
        Column("name",         "string", nullable=True),
        Column("lat",          "float64", units="deg"),
        Column("lon",          "float64", units="deg"),
        Column("alt_m",        "float64", units="m"),
        Column("vx_mps",       "float32", units="m/s", nullable=True),
        Column("vy_mps",       "float32", units="m/s", nullable=True),
        Column("vz_mps",       "float32", units="m/s", nullable=True),
        # Static compute capability
        Column("cpu_cores",        "int32",   nullable=True),
        Column("cpu_clock_ghz",    "float32", nullable=True),
        Column("flops_peak",       "float64", nullable=True),
        Column("ram_gb",           "float32", nullable=True),
        Column("storage_gb",       "float32", nullable=True),
        Column("gpu_present",      "bool",    nullable=True),
        Column("power_idle_w",     "float32", nullable=True),
        Column("power_per_flop_j", "float64", nullable=True),
        Column("compute_role",     "string", nullable=True,
               enum=["none", "mec", "cache", "both"]),
    ]


SATELLITES = TableSchema(
    name="satellites",
    granularity="(satellite, tick)",
    columns=_node_columns(),
    partition_cols=["operator_id"],
)
HAPS = TableSchema(
    name="haps",
    granularity="(haps, tick)",
    columns=_node_columns(),
    partition_cols=["operator_id"],
)
UAVS = TableSchema(
    name="uavs",
    granularity="(uav, tick)",
    columns=_node_columns(),
    partition_cols=["operator_id"],
)
GROUND_STATIONS = TableSchema(
    name="ground_stations",
    granularity="(gs, tick)",
    columns=_node_columns(),
    partition_cols=["operator_id"],
)
BASE_STATIONS = TableSchema(
    name="base_stations",
    granularity="(bs, tick)",
    columns=_node_columns(),
    partition_cols=["operator_id"],
)


# Phase-2 derived tables (written by hooks)
PER_USER_KPIS = TableSchema(
    name="per_user_kpis",
    granularity="(user, tick)",
    columns=[
        Column("t_idx",           "int32",
               description="FK -> time_dim"),
        Column("user_id",         "int32",
               description="FK -> users_dim"),
        Column("serving_node_id", "int64",
               description="FK -> nodes_dim. -1 means outage."),
        Column("prev_node_id",    "int64", nullable=True,
               description="FK -> nodes_dim of the previous tick"),
        Column("ho_type_id",      "int8",
               description="FK -> ho_type_dim"),
        Column("is_in_outage",    "int8"),
    ],
    dim_joins=[
        DimJoin("time_dim",    on="t_idx",           dim_on="t_idx",
                add_columns=["timestamp"]),
        DimJoin("nodes_dim",   on="serving_node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"],
                rename_to={"node_type":"serving_node_type",
                           "operator_id":"serving_operator_id",
                           "node_id":"serving_node_id"}),
        DimJoin("nodes_dim",   on="prev_node_id",    dim_on="node_id",
                add_columns=["node_type","operator_id"],
                rename_to={"node_type":"prev_node_type",
                           "operator_id":"prev_operator_id",
                           "node_id":"prev_node_id"}),
        DimJoin("ho_type_dim", on="ho_type_id",      dim_on="ho_type_id",
                add_columns=["name"],
                rename_to={"name":"ho_type"}),
    ],
    description="One row per user per timestep. Joins time_dim, nodes_dim, "
                "ho_type_dim transparently on .read().",
)


HO_EVENTS = TableSchema(
    name="ho_events",
    granularity="(user, ho_event)",
    columns=[
        Column("t_idx",           "int32"),
        Column("user_id",         "int32"),
        Column("serving_node_id", "int64"),
        Column("prev_node_id",    "int64", nullable=True),
        Column("ho_type_id",      "int8"),
    ],
    dim_joins=[
        DimJoin("time_dim",    on="t_idx",           dim_on="t_idx",
                add_columns=["timestamp"]),
        DimJoin("nodes_dim",   on="serving_node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"],
                rename_to={"node_type":"serving_node_type",
                           "operator_id":"serving_operator_id",
                           "node_id":"serving_node_id"}),
        DimJoin("nodes_dim",   on="prev_node_id",    dim_on="node_id",
                add_columns=["node_type","operator_id"],
                rename_to={"node_type":"prev_node_type",
                           "operator_id":"prev_operator_id",
                           "node_id":"prev_node_id"}),
        DimJoin("ho_type_dim", on="ho_type_id",      dim_on="ho_type_id",
                add_columns=["name"],
                rename_to={"name":"ho_type"}),
    ],
    description="One row per actual HO transition (ho_type != none).",
)


ACTIVE_ISL = TableSchema(
    name="active_isl",
    granularity="(edge, tick)",
    columns=[
        Column("t_idx", "int32"),
        Column("sat_a", "int64"),
        Column("sat_b", "int64"),
    ],
    dim_joins=[
        DimJoin("time_dim", on="t_idx", dim_on="t_idx", add_columns=["timestamp"]),
    ],
    description="One row per active ISL edge per tick (canonical sat_a<sat_b).",
)


NODE_STATE = TableSchema(
    name="node_state",
    granularity="(node, tick)",
    columns=[
        Column("t_idx",           "int32"),
        Column("node_id",         "int64",
               description="FK -> nodes_dim"),
        Column("cpu_util_pct",    "float32", units="%",   nullable=True),
        Column("ram_used_gb",     "float32", units="GB",  nullable=True),
        Column("storage_used_gb", "float32", units="GB",  nullable=True),
        Column("queue_length",    "int32",   nullable=True),
        Column("power_w",         "float32", units="W",   nullable=True),
        Column("n_tasks_running", "int32",   nullable=True),
        Column("n_tasks_queued",  "int32",   nullable=True),
    ],
    # note: partition_cols dropped here because node_type/operator_id are no
    # longer columns of this fact table. Partition can be added back at
    # write time using a join against nodes_dim if needed.
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id","name"]),
    ],
    description="Per-tick per-node resource snapshot. Joins time_dim + nodes_dim on read.",
)


TASKS = TableSchema(
    name="tasks",
    granularity="(task, lifetime)",
    columns=[
        Column("t_idx",              "int32"),
        Column("task_id",            "int64"),
        Column("user_id",            "int32"),
        Column("arrival_t_idx",      "int32"),
        Column("cpu_required_flops", "float64"),
        Column("data_in_mb",         "float32", units="MB"),
        Column("data_out_mb",        "float32", units="MB"),
        Column("deadline_ms",        "float32", units="ms"),
        Column("priority",           "string", enum=["low","normal","high"]),
        Column("assigned_node_id",   "int64", nullable=True),
        Column("status",             "string", enum=["queued","running","done","dropped","failed"]),
        Column("tick_bucket",        "int32"),
    ],
    partition_cols=["tick_bucket"],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",            dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="assigned_node_id", dim_on="node_id",
                add_columns=["node_type","operator_id","name"],
                rename_to={"node_type":"assigned_node_type",
                           "operator_id":"assigned_operator_id",
                           "name":"assigned_node_name",
                           "node_id":"assigned_node_id"}),
    ],
)


PER_OPERATOR_KPIS = TableSchema(
    name="per_operator_kpis",
    granularity="(operator, tick)",
    columns=[
        Column("t_idx",         "int32"),
        Column("operator_id",   "int32"),
        Column("n_links",       "int32"),
        Column("mean_snr_db",   "float64", units="dB"),
        Column("mean_sinr_db",  "float64", units="dB"),
    ],
    dim_joins=[
        DimJoin("time_dim",      on="t_idx",       dim_on="t_idx",       add_columns=["timestamp"]),
        DimJoin("operators_dim", on="operator_id", dim_on="operator_id", add_columns=["name"],
                rename_to={"name":"operator_name"}),
    ],
)


# Audit / validation tables - let reviewers reproduce KPIs by hand
LINK_AUDIT = TableSchema(
    name="link_audit",
    granularity="(user, candidate_node, tick)",
    columns=[
        Column("t_idx",                "int32"),
        Column("user_id",              "int32"),
        Column("node_id",              "int64"),
        Column("is_serving",           "int8",
               description="1 if this link is the user's serving access link."),
        # Geometry inputs
        Column("dist_m",               "float64",  units="m"),
        Column("elev_deg",             "float32",  units="deg"),
        Column("azimuth_deg",          "float32",  units="deg", nullable=True),
        Column("range_rate_mps",       "float32",  units="m/s", nullable=True),
        Column("doppler_hz",           "float64",  units="Hz"),
        # RF inputs (Tx side)
        Column("tx_power_w",           "float32",  units="W"),
        Column("antenna_gain_tx_db",   "float32",  units="dB"),
        Column("antenna_gain_rx_db",   "float32",  units="dB"),
        Column("beamforming_gain_db",  "float32",  units="dB", nullable=True),
        # Channel
        Column("path_loss_db",         "float64",  units="dB"),
        Column("channel_gain_lin",     "float64",
               description="||H||_F^2 in linear; squared magnitude of complex channel."),
        Column("carrier_hz",           "float64",  units="Hz"),
        Column("bandwidth_hz",         "float64",  units="Hz"),
        # Linear powers (the inputs to SINR)
        Column("received_power_w",     "float64",  units="W",
               description="P_rx = P_tx * G_tx * G_rx * chan_gain_lin * 10^(-PL/10)"),
        Column("noise_power_w",        "float64",  units="W",
               description="N = k_B * T_sys * B"),
        Column("intra_interf_w",       "float64",  units="W"),
        Column("inter_interf_w",       "float64",  units="W"),
        Column("total_interf_w",       "float64",  units="W"),
        # KPIs derived from the above
        Column("snr_db",               "float64",  units="dB"),
        Column("sinr_db",              "float64",  units="dB"),
        Column("spectral_efficiency_bps_hz", "float64", units="bps/Hz"),
        Column("link_rate_bps",        "float64",  units="bps"),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"]),
    ],
    description="Full per-link audit trail per tick. Lets a reviewer recompute "
                "SNR/SINR/SE/rate from raw inputs and verify the engine's outputs.",
)


POWER_ALLOC = TableSchema(
    name="power_alloc",
    granularity="(node, tick)",
    columns=[
        Column("t_idx",                  "int32"),
        Column("node_id",                "int64"),
        Column("total_tx_power_w",       "float32",  units="W"),
        Column("n_served_users",         "int32"),
        Column("power_per_user_w",       "float32",  units="W"),
        Column("equal_split",            "int8",
               description="1 if power was divided equally among served users."),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id","name"]),
    ],
    description="Per-node per-tick transmit power budget breakdown.",
)


BEAMFORMING_AUDIT = TableSchema(
    name="beamforming_audit",
    granularity="(node, user, tick)",
    columns=[
        Column("t_idx",            "int32"),
        Column("node_id",          "int64"),
        Column("user_id",          "int32"),
        Column("bf_mode",          "string", enum=["analog","digital","hybrid","none"]),
        Column("bf_gain_db",       "float32", units="dB"),
        Column("n_rx",             "int32"),
        Column("n_tx",             "int32"),
        Column("precoder_norm",    "float32", nullable=True,
               description="||w||_2 for digital/hybrid precoders."),
        Column("steering_az_deg",  "float32", units="deg", nullable=True),
        Column("steering_el_deg",  "float32", units="deg", nullable=True),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"]),
    ],
    description="Per-link beamforming details (mode, gain, geometry, precoder).",
)


VALIDATION_CHECKS = TableSchema(
    name="validation_checks",
    granularity="(check, tick)",
    columns=[
        Column("t_idx",          "int32"),
        Column("check_name",     "string"),
        Column("category",       "string",
               enum=["physical","stochastic","equivalence","budget","bounds"]),
        Column("passed",         "int8"),
        Column("measured_value", "float64", nullable=True),
        Column("expected_value", "float64", nullable=True),
        Column("tolerance",      "float64", nullable=True),
        Column("notes",          "string", nullable=True),
    ],
    dim_joins=[
        DimJoin("time_dim", on="t_idx", dim_on="t_idx", add_columns=["timestamp"]),
    ],
    description="Per-tick automated invariant checks: SINR consistency, "
                "energy conservation, range bounds, elevation bounds, etc.",
)


# Compute resource audit (per node per tick + per task)
COMPUTE_RESOURCE = TableSchema(
    name="compute_resource",
    granularity="(node, tick)",
    columns=[
        Column("t_idx",                "int32"),
        Column("node_id",              "int64"),
        Column("cpu_used_flops",       "float64", units="FLOPS"),
        Column("cpu_capacity_flops",   "float64", units="FLOPS"),
        Column("cpu_util_pct",         "float32", units="%"),
        Column("ram_used_gb",          "float32", units="GB"),
        Column("ram_capacity_gb",      "float32", units="GB"),
        Column("storage_used_gb",      "float32", units="GB"),
        Column("storage_capacity_gb",  "float32", units="GB"),
        Column("power_compute_w",      "float32", units="W"),
        Column("power_idle_w",         "float32", units="W"),
        Column("power_total_w",        "float32", units="W"),
        Column("energy_joules_tick",   "float64", units="J",
               description="Energy spent during this tick (P_total * step_size_s)."),
        Column("energy_joules_cum",    "float64", units="J",
               description="Cumulative energy since simulation start."),
        Column("n_tasks_admitted",     "int32"),
        Column("n_tasks_rejected",     "int32"),
        Column("n_tasks_completed",    "int32"),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id","flops_peak","ram_gb","storage_gb"]),
    ],
    description="Per-node per-tick compute + storage + energy accounting.",
)


TASK_AUDIT = TableSchema(
    name="task_audit",
    granularity="(task, lifetime)",
    columns=[
        Column("task_id",            "int64"),
        Column("user_id",            "int32"),
        Column("arrival_t_idx",      "int32"),
        Column("queue_t_idx",        "int32", nullable=True),
        Column("start_t_idx",        "int32", nullable=True),
        Column("finish_t_idx",       "int32", nullable=True),
        Column("assigned_node_id",   "int64", nullable=True),
        Column("queue_latency_ms",   "float32", units="ms", nullable=True),
        Column("exec_time_ms",       "float32", units="ms", nullable=True),
        Column("e2e_latency_ms",     "float32", units="ms", nullable=True),
        Column("cpu_consumed_flops", "float64", units="FLOPS"),
        Column("energy_joules",      "float64", units="J"),
        Column("data_in_mb",         "float32", units="MB"),
        Column("data_out_mb",        "float32", units="MB"),
        Column("deadline_ms",        "float32", units="ms"),
        Column("deadline_met",       "int8"),
        Column("final_status",       "string", enum=["done","dropped","failed","timeout"]),
        Column("failure_reason",     "string", nullable=True),
    ],
    description="Per-task full lifetime audit: arrival to completion (or drop).",
)


CACHE_AUDIT = TableSchema(
    name="cache_audit",
    granularity="(node, tick)",
    columns=[
        Column("t_idx",            "int32"),
        Column("node_id",          "int64"),
        Column("cache_capacity_mb","float32", units="MB"),
        Column("cache_used_mb",    "float32", units="MB"),
        Column("n_items",          "int32"),
        Column("n_hits",           "int32"),
        Column("n_misses",         "int32"),
        Column("n_evictions",      "int32"),
        Column("hit_rate",         "float32"),
        Column("policy",           "string", enum=["lru","lfu","arc","fifo","disabled"]),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"]),
    ],
)


# Radio resource audit (RB allocation, MCS, spectrum, beams)
RB_ALLOCATION = TableSchema(
    name="rb_allocation",
    granularity="(user, node, tick)",
    columns=[
        Column("t_idx",                "int32"),
        Column("user_id",              "int32"),
        Column("node_id",              "int64"),
        Column("n_rbs_allocated",      "int32",
               description="Number of resource blocks assigned to this UE."),
        Column("n_rbs_total",          "int32",
               description="Total RBs available on the serving node."),
        Column("bw_allocated_hz",      "float64", units="Hz"),
        Column("mcs_index",            "int8",   nullable=True),
        Column("modulation",           "string", nullable=True,
               enum=["BPSK","QPSK","16QAM","64QAM","256QAM"]),
        Column("coding_rate",          "float32", nullable=True),
        Column("target_bler",          "float32", nullable=True),
        Column("achievable_rate_bps",  "float64", units="bps"),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"]),
    ],
    description="Per-user resource-block grid and MCS chosen by the scheduler.",
)


BEAM_BUDGET = TableSchema(
    name="beam_budget",
    granularity="(node, beam, tick)",
    columns=[
        Column("t_idx",            "int32"),
        Column("node_id",          "int64"),
        Column("beam_idx",         "int32"),
        Column("n_beams_total",    "int32"),
        Column("power_per_beam_w", "float32", units="W"),
        Column("steering_az_deg",  "float32", units="deg", nullable=True),
        Column("steering_el_deg",  "float32", units="deg", nullable=True),
        Column("n_users_on_beam",  "int32"),
        Column("active",           "int8"),
    ],
    dim_joins=[
        DimJoin("time_dim",  on="t_idx",   dim_on="t_idx",   add_columns=["timestamp"]),
        DimJoin("nodes_dim", on="node_id", dim_on="node_id",
                add_columns=["node_type","operator_id"]),
    ],
)


SPECTRUM_USAGE = TableSchema(
    name="spectrum_usage",
    granularity="(operator, tick)",
    columns=[
        Column("t_idx",                  "int32"),
        Column("operator_id",            "int32"),
        Column("bw_total_hz",            "float64", units="Hz"),
        Column("bw_used_hz",             "float64", units="Hz"),
        Column("bw_used_pct",            "float32", units="%"),
        Column("frequency_reuse_factor", "float32"),
        Column("activity_factor",        "float32"),
    ],
    dim_joins=[
        DimJoin("time_dim",      on="t_idx",       dim_on="t_idx",       add_columns=["timestamp"]),
        DimJoin("operators_dim", on="operator_id", dim_on="operator_id", add_columns=["name"]),
    ],
)


# Future tables (declared now so consumers can be coded against them)
ROUTES = TableSchema(
    name="routes",
    granularity="(user, traffic_class, tick)",
    columns=[
        Column("timestamp",          "datetime64[ns]"),
        Column("t_idx",              "int32"),
        Column("user_id",            "int32"),
        Column("traffic_class",      "string", enum=["internet","mec_task","peer"]),
        Column("path_node_ids",      "string", description="JSON-encoded list of node_ids"),
        Column("total_delay_s",      "float64", units="s"),
        Column("total_hops",         "int32"),
        Column("bottleneck_bps",     "float64", units="bps"),
        Column("crosses_operators",  "string", description="JSON list of operator_ids crossed"),
    ],
    description="End-to-end routes per (user, traffic_class) per tick.",
)


# Canonical catalog
ALL_TABLES: Dict[str, TableSchema] = {
    # Phase 1
    "satellites":          SATELLITES,
    "haps":                HAPS,
    "uavs":                UAVS,
    "ground_stations":     GROUND_STATIONS,
    "base_stations":       BASE_STATIONS,
    # Phase 2 - main outputs
    "per_user_kpis":       PER_USER_KPIS,
    "ho_events":           HO_EVENTS,
    "active_isl":          ACTIVE_ISL,
    "node_state":          NODE_STATE,
    "tasks":               TASKS,
    "per_operator_kpis":   PER_OPERATOR_KPIS,
    "routes":              ROUTES,
    # Phase 2 - audit / validation (default OFF in master.yaml)
    # Radio
    "link_audit":          LINK_AUDIT,
    "power_alloc":         POWER_ALLOC,
    "beamforming_audit":   BEAMFORMING_AUDIT,
    "rb_allocation":       RB_ALLOCATION,
    "beam_budget":         BEAM_BUDGET,
    "spectrum_usage":      SPECTRUM_USAGE,
    # Compute
    "compute_resource":    COMPUTE_RESOURCE,
    "task_audit":          TASK_AUDIT,
    "cache_audit":         CACHE_AUDIT,
    # Validation
    "validation_checks":   VALIDATION_CHECKS,
}


def describe_all() -> str:
    """Generate a markdown reference for every output table."""
    parts = ["# Output table reference\n"]
    for name, s in ALL_TABLES.items():
        parts.append(s.describe())
        parts.append("")
    return "\n".join(parts)
