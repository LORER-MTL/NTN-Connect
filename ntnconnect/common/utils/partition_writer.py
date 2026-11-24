# SkyNTNPro v1 - Partitioned parquet writer
#
# Writes a DataFrame as a small, human-readable dataset: one cleanly NAMED file
# per partition value (e.g. satellites/oneweb.parquet), NOT hive-style
# `col=value/<uuid>.parquet` directories. The partition columns are kept inside
# each file, so read_partitioned just unions the folder back to the full table.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd


def _slug(v) -> str:
    """Filesystem-safe label for a partition value."""
    s = str(v)
    return re.sub(r"[^0-9A-Za-z._=-]+", "_", s).strip("_") or "part"


def write_partitioned(df: pd.DataFrame, out_dir: Path, table_name: str,
                      partition_cols: List[str], compression: str = "zstd",
                      name_map: Optional[Dict[str, Dict]] = None) -> Path:
    """Write `df` under out_dir as a readable dataset.

    No partition cols -> a single file `<table_name>.parquet`.
    With partition cols -> a folder `<table_name>/` holding one file per unique
    partition combination, named by joining the (mapped) values, e.g.
    `satellites/oneweb.parquet`. name_map optionally relabels values, e.g.
    {"operator_id": {1: "oneweb", 2: "starlink"}}.
    """
    if df.empty:
        return out_dir
    out_dir = Path(out_dir)
    cols = [c for c in (partition_cols or []) if c in df.columns]
    name_map = name_map or {}

    if not cols:
        target = out_dir / f"{table_name}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, compression=compression, index=False)
        return target

    folder = out_dir / table_name
    folder.mkdir(parents=True, exist_ok=True)
    for keys, g in df.groupby(cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        parts = []
        for c, v in zip(cols, keys):
            label = name_map.get(c, {}).get(v)
            parts.append(_slug(label) if label is not None else f"{c}={_slug(v)}")
        fname = "__".join(parts) + ".parquet"
        g.to_parquet(folder / fname, compression=compression, index=False)
    return folder


def read_partitioned(out_dir: Path, table_name: str, filters=None) -> pd.DataFrame:
    """Union a dataset written by write_partitioned. Accepts the new clean layout
    (`<table>/` folder of named files) and legacy layouts (`<table>.parquet` as a
    single file or a hive directory)."""
    out_dir = Path(out_dir)
    folder = out_dir / table_name              # new clean layout
    legacy = out_dir / f"{table_name}.parquet" # single file or hive dir
    for p in (folder, legacy):
        if p.exists():
            return pd.read_parquet(p, filters=filters)
    return pd.DataFrame()
