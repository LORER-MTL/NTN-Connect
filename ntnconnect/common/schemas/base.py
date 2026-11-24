# SkyNTNPro v1 - Schema layer (data ownership)
#
# Every output table is a TableSchema declared once in tables.py.
# Hooks and stages reference these schemas instead of hardcoding
# column names. Schemas validate columns at write time and dispatch
# to the right writer (parquet / csv / json / jsonl / parquet+csv).
#
# Users activate/deactivate tables in cfg.outputs.toggles.
# Per-table format override in cfg.outputs.formats.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import pandas as pd

from ntnconnect.common.utils.writers import write_table, write_partitioned_table

_log = logging.getLogger(__name__)


@dataclass
class Column:
    name: str
    dtype: str = "string"
    description: str = ""
    units: str = ""
    nullable: bool = True
    enum: Optional[List[str]] = None


@dataclass
class DimJoin:
    """Declarative join from this fact table to a dim table.

    `dim_schema_name` is looked up at read time in all_DIMS so we don't
    create circular imports between tables.py and dims.py.
    """
    dim_schema_name: str
    on:            str                              # fact column (FK)
    dim_on:        str                              # dim column (PK)
    add_columns:   List[str] = field(default_factory=list)
    rename_to:     Optional[Dict[str, str]] = None  # optional col renames
    how:           str = "left"                    # pandas merge how


@dataclass
class TableSchema:
    """Declarative schema for one output table.

    Stages and hooks reference an instance of this class to:
      * .collector(engine) - get a row-buffered collector
      * .write(df, engine) - validate + persist a DataFrame
      * .read(run_dir)     - load back for analysis
      * .describe()        - markdown doc generation
    """
    name: str
    granularity: str = ""
    columns: List[Column] = field(default_factory=list)
    partition_cols: List[str] = field(default_factory=list)
    default_format: str = "parquet"
    description: str = ""
    dim_joins: List["DimJoin"] = field(default_factory=list)
    write_once: bool = False

    # convenience
    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Optional[Column]:
        for c in self.columns:
            if c.name == name: return c
        return None

    # activation / format resolution
    def is_enabled(self, cfg) -> bool:
        toggles = (getattr(cfg.outputs, "tables", {}) or
                   getattr(cfg.outputs, "hooks", {}) or
                   getattr(cfg.outputs, "toggles", {}) or {})
        return bool(toggles.get(self.name, True))

    def resolved_format(self, cfg) -> str:
        per = getattr(cfg.outputs, "formats", {}) or {}
        return per.get(self.name) or getattr(cfg.outputs, "format", self.default_format)

    # validation
    def validate_df(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """Reorder + fill missing columns with NA.

        With strict=True, also rejects unexpected extra columns.
        """
        names = set(self.column_names)
        df_cols = set(df.columns)
        missing = [c for c in self.column_names if c not in df_cols]
        for col in missing:
            df[col] = None
        if strict:
            extra = df_cols - names
            if extra:
                raise ValueError(f"Table '{self.name}' has unexpected columns: {sorted(extra)}")
        # Order columns canonically
        keep = self.column_names + [c for c in df.columns if c not in names]
        return df[keep] if not strict else df[self.column_names]

    # collector helper
    def collector(self, engine) -> "TableCollector":
        """Return a row-buffering TableCollector bound to this schema."""
        return TableCollector(self, engine)

    # write
    def write(self, df: pd.DataFrame, engine) -> Optional[Path]:
        if df is None or df.empty:
            return None
        if not self.is_enabled(engine.cfg):
            return None
        df = self.validate_df(df, strict=False)
        fmt = self.resolved_format(engine.cfg)
        per = getattr(engine.cfg.outputs, "formats", {}) or {}
        if self.partition_cols:
            return write_partitioned_table(
                df, engine.sim_dir, self.name,
                partition_cols=self.partition_cols,
                global_default=fmt, per_table=per,
                compression=engine.cfg.storage.parquet_compression,
            )
        return write_table(df, engine.sim_dir, self.name, fmt, per)

    # raw read (no joins)
    def read_raw(self, run_dir: Path) -> pd.DataFrame:
        run_dir = Path(run_dir)
        for ext in ("parquet", "csv", "json", "jsonl"):
            p = run_dir / f"{self.name}.{ext}"
            if p.exists():
                if ext == "parquet":  return pd.read_parquet(p)
                if ext == "csv":      return pd.read_csv(p)
                if ext == "json":     return pd.read_json(p)
                if ext == "jsonl":    return pd.read_json(p, lines=True)
        d = run_dir / f"{self.name}.parquet"
        if d.is_dir():
            return pd.read_parquet(d)
        return pd.DataFrame()

    # high-level read (joins dims transparently)
    def read(self, run_dir: Path, denormalize: bool = True) -> pd.DataFrame:
        """Read the fact table and (by default) join every declared dim
        so the returned DataFrame has the human-readable columns the user
        expects (timestamp, node_type, operator name, ho_type string, ...).
        """
        df = self.read_raw(run_dir)
        if df.empty or not denormalize or not self.dim_joins:
            return df
        # Lazy import to avoid circular dep
        from ntnconnect.common.schemas import dims as _dims_mod
        for j in self.dim_joins:
            dim_schema = _dims_mod.ALL_DIMS.get(j.dim_schema_name)
            if dim_schema is None: continue
            dim_df = dim_schema.read_raw(run_dir)
            if dim_df.empty: continue
            cols = [j.dim_on] + (j.add_columns or
                                 [c.name for c in dim_schema.columns
                                  if c.name != j.dim_on])
            dim_slice = dim_df[cols].copy()
            if j.rename_to:
                dim_slice = dim_slice.rename(columns=j.rename_to)
                # Also rename the key if it appears in rename_to
                dim_on = j.rename_to.get(j.dim_on, j.dim_on)
            else:
                dim_on = j.dim_on
            df = df.merge(dim_slice, left_on=j.on, right_on=dim_on, how=j.how)
            if j.on != dim_on and dim_on in df.columns:
                df = df.drop(columns=[dim_on])
        return df

    # self-documentation
    def describe(self) -> str:
        lines = [f"## `{self.name}` ({self.granularity})", ""]
        if self.description:
            lines += [self.description, ""]
        lines += ["| column | dtype | units | enum | description |",
                  "|---|---|---|---|---|"]
        for c in self.columns:
            enum = ", ".join(c.enum) if c.enum else ""
            lines.append(f"| `{c.name}` | {c.dtype} | {c.units} | {enum} | {c.description} |")
        if self.partition_cols:
            lines += ["", f"Partitioned by: `{', '.join(self.partition_cols)}`"]
        return "\n".join(lines)


# TableCollector - row-buffered writer used by hooks
class TableCollector:
    """Buffers rows for a TableSchema, flushes on demand.

    Hooks call .add(row_dict) per tick and .flush() at simulation end.
    Column-name validation happens on every add() so typos die fast.
    """

    def __init__(self, schema: TableSchema, engine):
        self.schema = schema
        self.engine = engine
        self.rows: List[Dict[str, Any]] = []
        self._known = set(schema.column_names)

    def enabled(self) -> bool:
        return self.schema.is_enabled(self.engine.cfg)

    def add(self, row: Dict[str, Any], allow_extra: bool = False) -> None:
        if not self.enabled():
            return
        if not allow_extra:
            extra = set(row.keys()) - self._known
            if extra:
                raise ValueError(
                    f"Hook tried to add unknown columns {sorted(extra)} "
                    f"to table '{self.schema.name}'. Declared columns: "
                    f"{sorted(self._known)}"
                )
        self.rows.append(row)

    def add_many(self, rows: List[Dict[str, Any]],
                 allow_extra: bool = False) -> None:
        for r in rows:
            self.add(r, allow_extra=allow_extra)

    def add_df(self, df: pd.DataFrame, allow_extra: bool = False) -> None:
        """Batch-append rows from a DataFrame. Vectorized column check.
        Use this in hot paths instead of looping with .add(row_dict)."""
        if not self.enabled() or df is None or df.empty:
            return
        if not allow_extra:
            extra = set(df.columns) - self._known
            if extra:
                raise ValueError(
                    f"Hook tried to add unknown columns {sorted(extra)} "
                    f"to table '{self.schema.name}'."
                )
        self.rows.extend(df.to_dict("records"))

    def flush(self) -> Optional[Path]:
        if not self.rows or not self.enabled():
            return None
        df = pd.DataFrame(self.rows)
        out = self.schema.write(df, self.engine)
        if out:
            _log.info("Table '%s' saved: %d rows -> %s",
                      self.schema.name, len(df), out)
        self.rows.clear()
        return out
