# SkyNTNPro v1 - Legacy denormalizer
#
# Produces a "flat" copy of every fact table next to the original
# normalized files, with all dim columns inlined. Useful when an
# external tool can't follow joins.
#
#  python v1/scripts/denormalize.py outputs/run_001

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ntnconnect.common.schemas import ALL_TABLES


def denormalize_run(run_dir: Path, fmt: str = "parquet") -> None:
    run_dir = Path(run_dir)
    out_dir = run_dir / "_flat"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, schema in ALL_TABLES.items():
        df = schema.read(run_dir, denormalize=True)
        if df.empty: continue
        p = out_dir / f"{name}.{fmt}"
        if fmt == "parquet":  df.to_parquet(p, index=False)
        elif fmt == "csv":    df.to_csv(p, index=False)
        elif fmt == "json":   df.to_json(p, orient="records", indent=2)
        elif fmt == "jsonl":  df.to_json(p, orient="records", lines=True)
        else:                 df.to_parquet(p, index=False)
        print(f"  {name:20s} -> {p}  ({len(df)} rows, {len(df.columns)} cols)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--format", type=str, default="parquet",
                    choices=["parquet", "csv", "json", "jsonl"])
    args = ap.parse_args()
    denormalize_run(Path(args.run_dir), fmt=args.format)


if __name__ == "__main__":
    main()
