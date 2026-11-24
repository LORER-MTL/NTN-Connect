#!/usr/bin/env python
"""WISEE - PHASE 2 : per-tick simulation  (EXPLICIT flow).

Runs ONE case (constellation, strategy, operating point, gamma) of the study
over a dataset built in phase 1. It shows the real phase-2 steps and calls the
shared engine primitives from ntnconnect (no duplication):

  1. resolve the case config (link budget + handover knobs + the stage pipeline)
                                                     -> phase2/study config
  2. point the READ stage at the pre-built dataset   -> cfg.storage.data_dir/run_001
  3. wire the SimulationEngine (sim_dir = where THIS case writes its outputs)
  4. engine.setup() builds the stage pipeline from cfg.pipeline and attaches hooks
  5. build the tick timestamps and run the per-tick orchestrator loop
                                                     -> engine.run(timestamps)

Each stage (mobility, geometry, handover, channel, interference, radio, sinr,
classify, metrics, finalize) reads/writes the shared SimulationState; the hooks
flush sinr_snapshot.parquet, handover_events.parquet and the summaries into
sim_dir.

Output: <out_dir>/sinr_snapshot.parquet, handover_events.parquet,
        summary_system.parquet, config_used.yaml, stage_timing.txt ...

Run one case directly:
  python scenarios/WISEE/scripts/phase2/simulate.py \
      -c starlink -s farthest_approaching --a1 0.1 --a2 0.0 --gamma 11.76 \
      --dataset outputs/datasets/wisee_starlink --out outputs/wisee_run/starlink__farthest_approaching
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import sys
_SCEN = Path(__file__).resolve().parents[2]        # scenarios/WISEE
_ROOT = Path(__file__).resolve().parents[4]        # SkyNTN_public_v3
for _p in (str(_ROOT), str(_SCEN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as P                                  # the faithful study config
from ntnconnect.engine.engine import SimulationEngine


def simulate_case(constellation: str, strategy: str, *, a1: float, a2: float,
                  g_db: float, psat: float = 100.0, minutes: float = 45.0,
                  step_s: float = 1.0, dataset_root: Path, out_dir: Path,
                  verbose: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. case config: link budget + handover knobs + the WISEE stage pipeline ----
    cfg = P.build_wisee_cfg(constellation, strategy, a1=a1, a2=a2, g_db=g_db,
                            psat_w=float(psat), minutes=minutes, step_s=step_s)

    # -- 2. the READ stage consumes the phase-1 dataset at data_dir/run_001 ------
    cfg.storage.data_dir = str(dataset_root)
    cfg.storage.run_id = 1

    # -- 3. wire the engine; sim_dir is where THIS case writes its outputs -------
    eng = SimulationEngine(cfg, sim_dir=out_dir)

    # -- 4. build the pipeline (stages from cfg.pipeline) + attach the hooks -----
    eng.setup()
    if verbose:
        print(f"[phase2] {constellation}/{strategy}  a1={a1} a2={a2} g={g_db}")
        print("  pipeline :", [g.name for g in eng.orchestrator.groups])
        print("  hooks    :", [type(h).__name__ for h in eng.orchestrator.hooks])

    # -- 5. tick timestamps + the per-tick orchestrator loop --------------------
    t0 = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    n = int(round(minutes * 60.0 / step_s))
    ts = [t0 + timedelta(seconds=i * step_s) for i in range(n)]
    eng.run(ts)

    if verbose:
        # the engine writes sinr_snapshot per 15-min batch under batch_<NNNN>/
        snaps = list(out_dir.glob("batch_*/sinr_snapshot.parquet")) or \
            ([out_dir / "sinr_snapshot.parquet"] if (out_dir / "sinr_snapshot.parquet").exists() else [])
        print(f"[phase2] done -> {out_dir}"
              f"  (sinr_snapshot: {len(snaps)} batch(es))" if snaps
              else f"[phase2] done -> {out_dir}  (sinr_snapshot: MISSING)")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="WISEE phase 2 - simulate one case")
    ap.add_argument("-c", "--constellation", default="starlink",
                    choices=["starlink", "oneweb"])
    ap.add_argument("-s", "--strategy", default="farthest_approaching")
    ap.add_argument("--a1", type=float, default=0.1)
    ap.add_argument("--a2", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=11.76, help="gamma_th [dB]")
    ap.add_argument("--psat", type=float, default=100.0)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--dataset", required=True, help="phase-1 dataset root (has run_001/)")
    ap.add_argument("--out", required=True, help="output dir for this case")
    a = ap.parse_args()
    simulate_case(a.constellation, a.strategy, a1=a.a1, a2=a.a2, g_db=a.gamma,
                  psat=a.psat, minutes=a.minutes, step_s=a.step,
                  dataset_root=Path(a.dataset), out_dir=Path(a.out))


if __name__ == "__main__":
    main()
