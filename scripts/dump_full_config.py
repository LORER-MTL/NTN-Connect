# Dump all defaults into one explicit YAML (for strict_config runs).
#  python scripts/dump_full_config.py [--base master.yaml --out full.yaml]

from __future__ import annotations
import argparse
import sys
from dataclasses import asdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ntnconnect.experiment.config import (
    SimulationConfig, load_simulation_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=None,
                    help="Load this YAML first, then fill in defaults for "
                         "everything else. Default: pure-defaults dump.")
    ap.add_argument("--out",  type=str,
                    default="v1/configs/master_full.yaml")
    ap.add_argument("--strict", action="store_true",
                    help="Set experiment.strict_config = true in output.")
    args = ap.parse_args()

    if args.base:
        cfg = load_simulation_config(args.base)
    else:
        cfg = SimulationConfig()      # all dataclass defaults

    if args.strict:
        cfg.experiment.strict_config = True

    raw = asdict(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)
    print(f"Full config written to {args.out}")
    print(f"Top-level sections: {sorted(raw.keys())}")


if __name__ == "__main__":
    main()
