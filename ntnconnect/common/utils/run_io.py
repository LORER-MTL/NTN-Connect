"""Canonical reader for a phase-2 run's per-tick output tables.

Phase 2 writes each over-time table as per-batch shards; this is the one place that
knows that layout. Access modes:
  run_shards(rd, table)                 -> ordered shard paths (or a single file)
  iter_run_table(rd, table, cols, ...)  -> yield one shard DataFrame at a time
  read_run_table(rd, table, cols, ...)  -> union all shards

`table` is the stem (e.g. "sinr_snapshot"); a legacy single `<table>.parquet` works.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd


def batch_dir(sim_dir, batch_idx: int) -> Path:
    """The per-batch subfolder `batch_{NNNN}/` inside a run."""
    return Path(sim_dir) / f"batch_{int(batch_idx):04d}"


def batch_table_path(sim_dir, batch_idx: int, table: str) -> Path:
    """Writer path for one table's shard of one batch (folder created)."""
    d = batch_dir(sim_dir, batch_idx)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{table}.parquet"


def run_shards(run_dir, table: str = "sinr_snapshot") -> list:
    """Ordered shard files for one table: per-batch subfolder layout, else legacy
    flat shards, else a single `<table>.parquet`."""
    rd = Path(run_dir)
    nested = sorted(rd.glob(f"batch_*/{table}.parquet"))
    if nested:
        return nested
    flat = sorted(rd.glob(f"{table}*batch*.parquet"))
    if flat:
        return flat
    single = rd / f"{table}.parquet"
    return [single] if single.is_file() else []


def has_run_table(run_dir, table: str = "sinr_snapshot") -> bool:
    """True if the run has this output table (shards or single file)."""
    return bool(run_shards(run_dir, table))


def iter_run_table(run_dir, table: str = "sinr_snapshot", columns=None,
                   max_ticks=None):
    """Yield the table one shard at a time; read only `columns`, drop rows past
    `max_ticks` (by t_idx)."""
    for f in run_shards(run_dir, table):
        try:
            import pyarrow.parquet as pq
            avail = set(pq.read_schema(f).names)
            cols = [c for c in columns if c in avail] if columns else None
        except Exception:
            cols = columns
        try:
            d = pd.read_parquet(f, columns=cols)
        except Exception:
            continue
        if max_ticks is not None and "t_idx" in d.columns:
            d = d[d["t_idx"] <= max_ticks]
        if not d.empty:
            yield d


def read_run_table(run_dir, table: str = "sinr_snapshot", columns=None,
                   max_ticks=None) -> "pd.DataFrame":
    """Union every shard into one DataFrame; empty if the table is absent."""
    parts = list(iter_run_table(run_dir, table, columns, max_ticks))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
