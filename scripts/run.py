# Unified entry point. Params from --config YAML + USER_overrides block below.
#  python scripts/run.py [--config path.yaml]

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ntnconnect.experiment.config import load_simulation_config
from ntnconnect.scenario.phase1 import run_phase1, _build_timestamps
from ntnconnect.engine.engine import SimulationEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


# USER_overrides: the only place to override YAML values in code.
# Leave the body empty for a pure YAML run; use for sweeps/seeds.
def user_overrides(cfg):
    """Modify cfg fields before engine.setup() runs.

    Examples (commented out by default):

        # Override the random seed for this run only
        # cfg.experiment.random_seed = 123

        # Use a different handover mode without editing master.yaml
        # cfg.handover.mode = "nearest_sat"

        # Restrict to a subset of operators for a quick check
        # cfg.operators = [op for op in cfg.operators if op.id in (1, 2)]

        # Force a different output directory
        # cfg.storage.data_dir = "outputs/sweep_001"
    """
    # nothing here by default. Edit between this line
    pass
    # and this line


def main():
    parser = argparse.ArgumentParser(description="SkyNTNPro v1 runner")
    parser.add_argument("--config", default="v1/configs/master.yaml",
                        help="The single YAML config to use for this run.")
    args = parser.parse_args()

    # 1) Load the one YAML.
    cfg = load_simulation_config(args.config)

    # 2) Apply the only legitimate code-side overrides.
    user_overrides(cfg)

    # 3) Report what we have, then run.
    print(f"[v1] config: {args.config}")
    print(f"[v1] experiment: {cfg.experiment.name}, "
          f"strict_config: {cfg.experiment.strict_config}")
    print(f"[v1] operators: {[op.name for op in cfg.operators]}")
    print(f"[v1] handover.mode: {cfg.handover.mode}")
    print(f"[v1] pipeline groups: {[g.name for g in cfg.pipeline.groups]}")

    # 4) Phase 1.
    sim_dir = run_phase1(cfg)
    print(f"[v1] Phase 1 done -> {sim_dir}")

    # 5) Phase 2 (engine.setup() enforces strict_config if asked).
    engine = SimulationEngine(cfg, sim_dir=sim_dir)
    engine.setup()
    print("[v1] groups: " + ", ".join(
        f"{g.name}({'P' if g.parallel else 'S'},n={len(g.stages)})"
        for g in engine.orchestrator.groups))

    # 6) Run the time loop.
    timestamps = _build_timestamps(cfg.env)
    engine.run(timestamps)
    print(f"[v1] Done. Artifacts in {sim_dir}")


if __name__ == "__main__":
    main()
