"""Schema layer: one source of truth for every output table.

Fact tables hold FK-only rows. Dimension tables (time, operators, nodes,
users, ho_type) are written once per run by stages/dimensions.py. The
.read() method on every fact table transparently joins all declared dims
so analysis code sees denormalised dataframes.
"""
from ntnconnect.common.schemas.base import Column, TableSchema, TableCollector, DimJoin
from ntnconnect.common.schemas.tables import (
    ALL_TABLES,
    SATELLITES, HAPS, UAVS, GROUND_STATIONS, BASE_STATIONS,
    PER_USER_KPIS, HO_EVENTS, ACTIVE_ISL, NODE_STATE, TASKS,
    PER_OPERATOR_KPIS, ROUTES,
    LINK_AUDIT, POWER_ALLOC, BEAMFORMING_AUDIT, VALIDATION_CHECKS,
    RB_ALLOCATION, BEAM_BUDGET, SPECTRUM_USAGE,
    COMPUTE_RESOURCE, TASK_AUDIT, CACHE_AUDIT,
    describe_all,
)
from ntnconnect.common.schemas.dims import (
    ALL_DIMS,
    TIME_DIM, OPERATORS_DIM, NODES_DIM, USERS_DIM, HO_TYPE_DIM,
    HO_TYPE_VALUES, HO_TYPE_ID, HO_TYPE_NAME,
    encode_ho_type, decode_ho_type,
)

__all__ = [
    "Column", "TableSchema", "TableCollector", "DimJoin",
    "ALL_TABLES", "ALL_DIMS",
    "SATELLITES", "HAPS", "UAVS", "GROUND_STATIONS", "BASE_STATIONS",
    "PER_USER_KPIS", "HO_EVENTS", "ACTIVE_ISL", "NODE_STATE", "TASKS",
    "PER_OPERATOR_KPIS", "ROUTES",
    "LINK_AUDIT", "POWER_ALLOC", "BEAMFORMING_AUDIT", "VALIDATION_CHECKS",
    "RB_ALLOCATION", "BEAM_BUDGET", "SPECTRUM_USAGE",
    "COMPUTE_RESOURCE", "TASK_AUDIT", "CACHE_AUDIT",
    "TIME_DIM", "OPERATORS_DIM", "NODES_DIM", "USERS_DIM", "HO_TYPE_DIM",
    "HO_TYPE_VALUES", "HO_TYPE_ID", "HO_TYPE_NAME",
    "encode_ho_type", "decode_ho_type",
    "describe_all",
]
