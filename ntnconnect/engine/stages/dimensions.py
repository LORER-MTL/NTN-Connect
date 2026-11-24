# Stage: write the five dimension tables once (first tick), then no-op.

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.common.schemas.dims import (
    TIME_DIM, OPERATORS_DIM, NODES_DIM, USERS_DIM, HO_TYPE_DIM,
    HO_TYPE_VALUES,
)


def _build_time_dim(cfg) -> pd.DataFrame:
    env = cfg.env
    try:
        start = datetime.fromisoformat(env.start_utc.replace("Z", "+00:00"))
    except Exception:
        start = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
    n = max(1, int(env.duration_s // env.step_size_s))
    rows = [{"t_idx": i,
             "timestamp": start + timedelta(seconds=i * env.step_size_s)}
            for i in range(n)]
    return pd.DataFrame(rows)


def _build_operators_dim(cfg) -> pd.DataFrame:
    rows = []
    for op in cfg.operators:
        rows.append({
            "operator_id":     int(op.id),
            "name":            op.name,
            "sat_id_lo":       int(op.sat_id_range[0]),
            "sat_id_hi":       int(op.sat_id_range[1]),
            "carrier_freq_hz": float(op.radio.carrier_freq_hz),
            "bandwidth_hz":    float(op.radio.bandwidth_hz),
            "power_max_w":     float(op.radio.power_max_w),
            "channel_model":   op.channel_model.model_type,
        })
    return pd.DataFrame(rows)


def _build_nodes_dim(state) -> pd.DataFrame:
    """Harvest unique nodes from each per-type per-tick frame.

    Static columns (capability, name) are read once - we keep first occurrence.
    """
    frames = []
    for key in ("satellites_t", "haps_t", "uavs_t",
                "ground_stations_t", "base_stations_t"):
        df = state.data.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty: continue
        keep = [c for c in (
            "node_id","node_type","operator_id","name",
            "cpu_cores","cpu_clock_ghz","flops_peak","ram_gb","storage_gb",
            "gpu_present","power_idle_w","power_per_flop_j","compute_role",
        ) if c in df.columns]
        frames.append(df[keep].drop_duplicates(subset=["node_id"], keep="first"))
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["node_id"], keep="first")
    return full


def _build_users_dim(cfg, state) -> pd.DataFrame:
    """Best-effort: use users_t if present, else synthesize the seed list."""
    df = state.data.get("users_t")
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.drop_duplicates(subset=["user_id"], keep="first")
        cols = {
            "user_id":               df["user_id"].astype(int),
            "init_lat":              df.get("lat",   pd.Series([None]*len(df))),
            "init_lon":              df.get("lon",   pd.Series([None]*len(df))),
            "mobility_type":         pd.Series([cfg.users.mobility.model]*len(df)),
            "preferred_operator_id": pd.Series(
                [cfg.users.preferences.preferred_operator_id]*len(df)),
        }
        return pd.DataFrame(cols)
    # Fallback: emit empty - downstream joins will gracefully no-op
    return pd.DataFrame()


def _build_ho_type_dim() -> pd.DataFrame:
    return pd.DataFrame(HO_TYPE_VALUES, columns=["ho_type_id", "name"])


@register_stage("dimensions", internal=True)
class Dimensions(Stage):
    reads  = ["satellites_t","haps_t","uavs_t",
              "ground_stations_t","base_stations_t","users_t"]
    writes = []
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._written = False

    def run(self, state):
        if self._written or self.engine is None:
            return
        # Time-dim doesn't depend on state - we can write it immediately
        TIME_DIM.write(_build_time_dim(self.cfg), self.engine)
        OPERATORS_DIM.write(_build_operators_dim(self.cfg), self.engine)
        HO_TYPE_DIM.write(_build_ho_type_dim(), self.engine)

        # Nodes / users need the first tick's data to be loaded
        nodes_df = _build_nodes_dim(state)
        if not nodes_df.empty:
            NODES_DIM.write(nodes_df, self.engine)

        users_df = _build_users_dim(self.cfg, state)
        if not users_df.empty:
            USERS_DIM.write(users_df, self.engine)

        self._written = True
