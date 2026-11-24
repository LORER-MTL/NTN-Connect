# SkyNTNPro v1 - Output writers
#
# One factory: write_table(df, out_dir, name, format=...)
# Supported formats: parquet | csv | json | jsonl | parquet+csv
# Partitioned writes fall back to CSV/JSON when the format is non-parquet.
# Hooks use this exclusively - users never touch serialization code.

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from ntnconnect.common.utils.partition_writer import write_partitioned

_log = logging.getLogger(__name__)

ALLOWED_FORMATS = ("parquet", "csv", "json", "jsonl", "parquet+csv")


def _resolve_format(global_default: str,
                    per_table: dict,
                    table_name: str) -> str:
    fmt = (per_table.get(table_name) or global_default or "parquet").lower()
    if fmt not in ALLOWED_FORMATS:
        _log.warning("Unknown output format '%s' for table '%s' - using parquet",
                     fmt, table_name)
        fmt = "parquet"
    return fmt


def write_table(df: pd.DataFrame,
                out_dir: Path,
                table_name: str,
                global_default: str = "parquet",
                per_table: Optional[dict] = None) -> List[Path]:
    """Write a non-partitioned table in the format declared in config.

    Returns the list of files actually written (one or two paths).
    """
    if df is None or df.empty:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _resolve_format(global_default, per_table or {}, table_name)
    written: List[Path] = []

    def _pq(p):  df.to_parquet(p, index=False);                       written.append(p)
    def _cs(p):  df.to_csv(p,     index=False);                       written.append(p)
    def _js(p):  df.to_json(p,    orient="records", indent=2);        written.append(p)
    def _jl(p):  df.to_json(p,    orient="records", lines=True);      written.append(p)

    base = out_dir / table_name
    try:
        if fmt == "parquet":      _pq(base.with_suffix(".parquet"))
        elif fmt == "csv":        _cs(base.with_suffix(".csv"))
        elif fmt == "json":       _js(base.with_suffix(".json"))
        elif fmt == "jsonl":      _jl(base.with_suffix(".jsonl"))
        elif fmt == "parquet+csv":
            _pq(base.with_suffix(".parquet"))
            _cs(base.with_suffix(".csv"))
    except Exception as e:
        _log.warning("write_table('%s', %s) failed: %s - falling back to CSV",
                     table_name, fmt, e)
        try: _cs(base.with_suffix(".csv"))
        except Exception as e2:
            _log.error("CSV fallback also failed: %s", e2)
    return written


def write_partitioned_table(df: pd.DataFrame,
                            out_dir: Path,
                            table_name: str,
                            partition_cols: List[str],
                            global_default: str = "parquet",
                            per_table: Optional[dict] = None,
                            compression: str = "zstd") -> Path:
    """Partitioned write. Parquet is native; other formats degrade to a single
    flat file (we keep the schema columns so downstream filters still work)."""
    if df is None or df.empty:
        return out_dir
    fmt = _resolve_format(global_default, per_table or {}, table_name)
    if fmt.startswith("parquet"):
        return write_partitioned(df, out_dir, table_name,
                                 partition_cols=partition_cols,
                                 compression=compression)
    # non-parquet: flat fallback
    write_table(df, out_dir, table_name, fmt, per_table)
    return out_dir
