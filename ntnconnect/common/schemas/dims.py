# SkyNTNPro v1 - Dimension tables (written ONCE per run)
#
# Dims hold metadata that doesn't change tick-to-tick:
#  time_dim - t_idx <-> timestamp mapping
#  operators_dim- operator metadata (name, freq, bandwidth)
#  nodes_dim - node metadata + static capability
#  users_dim - UE metadata (initial position, preference)
#  ho_type_dim - HO_TYPE enum encoding
#
# Fact tables reference these via foreign keys (t_idx, node_id,
# operator_id, user_id, ho_type_id) instead of duplicating strings
# and timestamps on every row.

from __future__ import annotations
from typing import Dict, List, Tuple
from ntnconnect.common.schemas.base import Column, TableSchema


TIME_DIM = TableSchema(
    name="time_dim",
    granularity="(tick)",
    columns=[
        Column("t_idx",     "int32",
               description="Foreign key from every fact table."),
        Column("timestamp", "datetime64[ns]",
               description="UTC time of this tick."),
    ],
    description="Maps t_idx -> timestamp. Written once at simulation start.",
)


OPERATORS_DIM = TableSchema(
    name="operators_dim",
    granularity="(operator)",
    columns=[
        Column("operator_id",     "int32"),
        Column("name",            "string"),
        Column("sat_id_lo",       "int64"),
        Column("sat_id_hi",       "int64"),
        Column("carrier_freq_hz", "float64", units="Hz"),
        Column("bandwidth_hz",    "float64", units="Hz"),
        Column("power_max_w",     "float32", units="W"),
        Column("channel_model",   "string",
               description="Selected channel model name."),
    ],
    description="Per-operator metadata; one row per operator. Written once.",
)


NODES_DIM = TableSchema(
    name="nodes_dim",
    granularity="(node)",
    columns=[
        Column("node_id",          "int64"),
        Column("node_type",        "string",
               enum=["satellite","haps","uav","ground_station","base_station"]),
        Column("operator_id",      "int32"),
        Column("name",             "string", nullable=True),
        # Static capabilities (do not change over time)
        Column("cpu_cores",        "int32",   nullable=True),
        Column("cpu_clock_ghz",    "float32", nullable=True),
        Column("flops_peak",       "float64", nullable=True),
        Column("ram_gb",           "float32", nullable=True),
        Column("storage_gb",       "float32", nullable=True),
        Column("gpu_present",      "bool",    nullable=True),
        Column("power_idle_w",     "float32", nullable=True),
        Column("power_per_flop_j", "float64", nullable=True),
        Column("compute_role",     "string",  nullable=True,
               enum=["none","mec","cache","both"]),
    ],
    description="One row per node, ever. Carries identity + static capability. "
                "Position is dynamic and lives in the per-tick node tables.",
)


USERS_DIM = TableSchema(
    name="users_dim",
    granularity="(user)",
    columns=[
        Column("user_id",              "int32"),
        Column("init_lat",             "float64", units="deg"),
        Column("init_lon",             "float64", units="deg"),
        Column("mobility_type",        "string",  nullable=True),
        Column("preferred_operator_id","int32",   nullable=True),
    ],
    description="One row per UE. Position evolves in users_state (future).",
)


HO_TYPE_DIM = TableSchema(
    name="ho_type_dim",
    granularity="(enum value)",
    columns=[
        Column("ho_type_id", "int8"),
        Column("name",       "string"),
    ],
    description="Integer encoding of the ho_type enum used by fact tables.",
)


# HO_TYPE enum encoding - imported by hooks/stages that write ho_type_id
HO_TYPE_VALUES: List[Tuple[int, str]] = [
    (0, "none"),
    (1, "initial"),
    (2, "intra-operator-intra-type"),
    (3, "intra-operator-inter-type"),
    (4, "inter-operator"),
    (5, "outage-enter"),
    (6, "outage-recover"),
]
HO_TYPE_ID: Dict[str, int] = {name: i for i, name in HO_TYPE_VALUES}
HO_TYPE_NAME: Dict[int, str] = {i: name for i, name in HO_TYPE_VALUES}


def encode_ho_type(name: str) -> int:
    return HO_TYPE_ID.get(name, 0)


def decode_ho_type(i: int) -> str:
    return HO_TYPE_NAME.get(i, "none")


ALL_DIMS: Dict[str, TableSchema] = {
    "time_dim":      TIME_DIM,
    "operators_dim": OPERATORS_DIM,
    "nodes_dim":     NODES_DIM,
    "users_dim":     USERS_DIM,
    "ho_type_dim":   HO_TYPE_DIM,
}
