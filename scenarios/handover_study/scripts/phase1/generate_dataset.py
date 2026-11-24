#!/usr/bin/env python
"""WISEE - PHASE 1 : scenario / dataset generation  (EXPLICIT flow).

This file lays out the REAL phase-1 steps for the WISEE handover study, calling
the shared low-level primitives from ntnconnect (no engine code is duplicated).
Read top-to-bottom to see exactly what phase 1 does:

  1. resolve the WISEE study config           -> phase2.build_wisee_cfg()
  2. build the fine tick timestamps           -> _build_timestamps()
  3. generate the users (hotspot cluster)     -> generate_users()
  4. generate the ground stations (gateways)  -> GroundStationGenerator.generate()
  5. generate the satellites with a per-batch COARSE visibility pre-pass
     (propagate ALL sats coarsely, keep those visible to any USER or GATEWAY,
      then fine-propagate ONLY the kept set)  -> _generate_satellites_batched()
  6. write the partitioned parquet dataset + manifest.json  -> write_partitioned()

Output (the dataset the simulation phase reads):
  <out_dir>/run_001/
      satellites/<operator>.parquet     (t_idx, node_id, lat, lon, alt_m, x,y,z, v..)
      ground_stations/<operator>.parquet
      users.parquet
      manifest.json
      config_used.yaml

Run:
  python scenarios/WISEE/scripts/phase1/generate_dataset.py --constellation starlink
  python scenarios/WISEE/scripts/phase1/generate_dataset.py -c oneweb --minutes 45 --step 1.0
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

import pandas as pd

# --- make the scenario root (phase2.py) and the package root importable -------
import sys
_SCEN = Path(__file__).resolve().parents[2]        # scenarios/WISEE
_ROOT = Path(__file__).resolve().parents[4]        # SkyNTN_public_v3
for _p in (str(_ROOT), str(_SCEN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as P                                  # the faithful WISEE study config
from ntnconnect.experiment.config import DatasetTimingConfig
from ntnconnect.scenario.phase1 import (
    _build_timestamps,                              # step 2
    _generate_satellites_batched,                   # step 5 (coarse pre-pass)
    _dataset_manifest,
)
from ntnconnect.scenario.user_generator import generate_users          # step 3
from ntnconnect.scenario.fixed_generators import GroundStationGenerator  # step 4
from ntnconnect.scenario.sat_generator import SatelliteGenerator         # step 5
from ntnconnect.common.utils.partition_writer import write_partitioned   # step 6

# faithful build_ephemeris timing: 15-min batches, 10-s coarse visibility pre-pass
BATCH_S  = 900.0
COARSE_S = 10.0


def build_dataset(constellation: str, minutes: float = 45.0, step_s: float = 1.0,
                  out_root: Path | None = None) -> Path:
    # -- 1. resolve the WISEE study config (link budget, TLEs, users, gateways) --
    # a1/a2/gamma do not affect the geometry, so a fixed base point is fine here.
    cfg = P.build_wisee_cfg(constellation, "farthest_approaching", a1=0.1, a2=0.0,
                            g_db=11.76, psat_w=100.0, minutes=minutes, step_s=step_s)
    out_root = Path(out_root) if out_root else (_ROOT / "outputs" / "datasets"
                                                / f"wisee_{constellation}")
    cfg.storage.data_dir = str(out_root)
    cfg.storage.run_id = 1
    out_dir = out_root / "run_001"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_used(out_dir / "config_used.yaml")

    # timing for the STUDY layer: the env window, with the coarse pre-pass enabled.
    t = DatasetTimingConfig(utc=cfg.env.start_utc,
                            duration_s=float(cfg.env.duration_s),
                            step_s=float(cfg.env.step_size_s),
                            batch_size_s=BATCH_S,
                            coarse_step_s=COARSE_S)

    # -- 2. fine tick timestamps -------------------------------------------------
    timestamps = _build_timestamps(t.utc, t.duration_s, t.step_s)
    print(f"[phase1] {constellation}: {len(timestamps)} ticks "
          f"({minutes} min @ {step_s}s), batch {BATCH_S/60:.0f}min, coarse {COARSE_S}s")

    # -- 3. users (hotspot cluster) at every tick; t0 drives the pre-pass keep-set
    users_df = generate_users(cfg, timestamps)
    users_t0 = users_df[users_df["t_idx"] == 0] if not users_df.empty else pd.DataFrame()
    print(f"[phase1] users: {users_df['user_id'].nunique() if not users_df.empty else 0}")

    op = cfg.operators[0]                            # WISEE = one operator per run

    # -- 4. ground stations (gateways) ------------------------------------------
    gws_df = GroundStationGenerator(cfg, op).generate(timestamps)
    n_gw = gws_df["node_id"].nunique() if not gws_df.empty else 0
    print(f"[phase1] gateways: {n_gw}")

    # -- 5. satellites with the per-batch COARSE visibility pre-pass ------------
    gen = SatelliteGenerator(cfg, op)
    sats_df = _generate_satellites_batched(cfg, op, gen, timestamps, t,
                                           COARSE_S, users_t0)
    n_sat = sats_df["node_id"].nunique() if not sats_df.empty else 0
    print(f"[phase1] satellites (visible keep-set): {n_sat}")

    # -- 6. write the partitioned parquet dataset + manifest --------------------
    name_map = {"operator_id": {int(op.id): op.name}}
    if not sats_df.empty:
        write_partitioned(sats_df, out_dir, "satellites",
                          partition_cols=cfg.storage.partition_by["nodes"],
                          name_map=name_map)
    if not gws_df.empty:
        write_partitioned(gws_df, out_dir, "ground_stations",
                          partition_cols=cfg.storage.partition_by["nodes"],
                          name_map=name_map)
    if not users_df.empty:
        write_partitioned(users_df, out_dir, "users", partition_cols=[])
    (out_dir / "manifest.json").write_text(
        json.dumps(_dataset_manifest(cfg, "study", t), indent=2))

    print(f"[phase1] wrote dataset -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="WISEE phase 1 - dataset generation")
    ap.add_argument("-c", "--constellation", default="starlink",
                    choices=["starlink", "oneweb"])
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--step", type=float, default=1.0, help="tick step [s]")
    ap.add_argument("--out", default=None, help="output dataset root")
    a = ap.parse_args()
    build_dataset(a.constellation, a.minutes, a.step, a.out)


if __name__ == "__main__":
    main()
