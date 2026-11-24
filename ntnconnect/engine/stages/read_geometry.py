# Stage: read Phase-1 tables for the current tick, streaming one batch at a time.
# Phase-1 writes per-batch files (<table>/<stem>__batch_NN.parquet); these stages
# keep only the current batch window resident and load the next lazily. Legacy
# single-file datasets still work (read once, cached, filter per tick).

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.common.utils.partition_writer import read_partitioned


def _ticks_per_batch(out_dir: Path) -> int | None:
    """From the dataset manifest: batch_size_s / step_s. None if unknown."""
    m = out_dir / "manifest.json"
    if not m.is_file():
        return None
    try:
        d = json.loads(m.read_text())
        step = float(d.get("step_s", 0) or 0)
        bs = float(d.get("batch_size_s", 0) or 0)
        return max(1, round(bs / step)) if step > 0 and bs > 0 else None
    except (ValueError, OSError):
        return None


class _BatchStreamCache:
    """Per-stage cache: holds the resident batch per table and serves one tick."""

    def __init__(self):
        self._tpb: int | None = None
        self._tpb_dir: Path | None = None
        self._loaded: dict[str, tuple] = {}   # table -> (key, df); key = batch idx or "*"

    def _tpb_for(self, out_dir: Path) -> int | None:
        if self._tpb_dir != out_dir:
            self._tpb = _ticks_per_batch(out_dir)
            self._tpb_dir = out_dir
        return self._tpb

    def tick(self, out_dir: Path, table: str, t_idx: int, ts) -> pd.DataFrame:
        folder = out_dir / table
        tpb = self._tpb_for(out_dir)
        batch_files = list(folder.glob("*__batch_*.parquet")) if folder.is_dir() else []

        if batch_files and tpb:
            b = int(t_idx) // tpb
            cur = self._loaded.get(table)
            if cur is None or cur[0] != b:
                paths = sorted(folder.glob(f"*__batch_{b:02d}.parquet"))
                df = (pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
                      if paths else pd.DataFrame())
                self._loaded[table] = (b, df)          # evicts the previous batch
                cur = self._loaded[table]
            df = cur[1]
        else:
            # legacy single-file: load once, cache whole, reuse
            cur = self._loaded.get(table)
            if cur is None or cur[0] != "*":
                self._loaded[table] = ("*", read_partitioned(out_dir, table))
                cur = self._loaded[table]
            df = cur[1]

        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        # filter to the tick: prefer t_idx, else timestamp
        if "t_idx" in df.columns:
            return df[df["t_idx"] == int(t_idx)]
        if "timestamp" in df.columns:
            return df[df["timestamp"] == ts]
        return df


_NODE_KEYS = [("satellites", "satellites_t"), ("haps", "haps_t"),
              ("uavs", "uavs_t"), ("ground_stations", "ground_stations_t"),
              ("base_stations", "base_stations_t")]


@register_stage("read")
class Read(Stage):
    """Top-level reader: all node frames + users for the current tick, one batch at a time."""
    reads = []
    writes = ["satellites_t", "haps_t", "uavs_t",
              "ground_stations_t", "base_stations_t", "users_t"]
    thread_safe = True

    def run(self, state):
        cache = getattr(self, "_bcache", None) or _BatchStreamCache()
        self._bcache = cache
        out_dir = Path(self.cfg.storage.data_dir) / f"run_{self.cfg.storage.run_id:03d}"
        for ntype, key in _NODE_KEYS + [("users", "users_t")]:
            state.data[key] = cache.tick(out_dir, ntype, state.t_idx, state.timestamp)


@register_stage("read_geometry", internal=True)
class ReadGeometry(Stage):
    reads = []
    writes = ["satellites_t", "haps_t", "uavs_t",
              "ground_stations_t", "base_stations_t"]
    thread_safe = True

    def run(self, state):
        cache = getattr(self, "_bcache", None) or _BatchStreamCache()
        self._bcache = cache
        out_dir = Path(self.cfg.storage.data_dir) / f"run_{self.cfg.storage.run_id:03d}"
        for ntype, key in _NODE_KEYS:
            state.data[key] = cache.tick(out_dir, ntype, state.t_idx, state.timestamp)


@register_stage("read_node_state_prev", internal=True)
class ReadNodeStatePrev(Stage):
    """Reads the previous tick's node_state (if any) for stateful schedulers."""
    reads = []
    writes = ["node_state_prev"]
    thread_safe = True

    def run(self, state):
        state.data["node_state_prev"] = None      # stub until the hook writes node_state


@register_stage("read_users", internal=True)
class ReadUsers(Stage):
    """Reads the Phase-1 users batches filtered to the current tick -> users_t."""
    reads = []
    writes = ["users_t"]
    thread_safe = True

    def run(self, state):
        cache = getattr(self, "_bcache", None) or _BatchStreamCache()
        self._bcache = cache
        out_dir = Path(self.cfg.storage.data_dir) / f"run_{self.cfg.storage.run_id:03d}"
        state.data["users_t"] = cache.tick(out_dir, "users", state.t_idx, state.timestamp)
