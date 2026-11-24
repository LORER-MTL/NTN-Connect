#!/usr/bin/env python
"""NTN-Connect: a minimal, self-contained example of the full pipeline.

It walks through the three phases of the simulator on a SHORT run (5 min) so it
finishes in a couple of minutes on a laptop:

    Phase 1  generate_dataset  -> propagate the constellation, compute geometry
                                  and visibility, write the per-batch dataset
    Phase 2  simulate          -> run the stage pipeline (channel, Doppler,
                                  association/handover, min-power allocation)
    Phase 3  metrics           -> reduce the run to the paper KPIs
                                  (admission, outage, tx power, IHI, handovers)

Run it from anywhere:

    python examples/example.py

A longer study is the same three phases run at a finer step and over more
strategies and spectral-efficiency levels; edit the constants below to explore.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path

# ---- 0. make the package + the handover_study scenario importable -----------
ROOT = Path(__file__).resolve().parent.parent          # repo root
SCEN = ROOT / "scenarios" / "handover_study"
for p in (ROOT, SCEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCEN / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phase1 = _load("scripts/phase1/generate_dataset.py", "ntnc_p1")
phase2 = _load("scripts/phase2/simulate.py", "ntnc_p2")
phase3 = _load("scripts/phase3/metrics.py", "ntnc_p3")

# ---- example settings -------------------------------------------------------
CONSTELLATION = "starlink"   # or "oneweb"
STRATEGY      = "farthest_approaching"   # MST baseline; see README for the full set
SE_LEVEL      = 2            # spectral-efficiency target 1..6
GAMMA_DB      = 4.77         # SE2 rate threshold (Table I)
MINUTES       = 5.0          # short for the example; the paper uses 45
STEP_S        = 2.0          # coarse for speed; the paper uses 0.1

OUT = ROOT / "outputs" / "example"
ds_dir  = OUT / "dataset"
run_dir = OUT / "run" / f"{CONSTELLATION}__{STRATEGY}__SE{SE_LEVEL}" / "run_001"


def main() -> None:
    # ---- PHASE 1: constellation + propagation + visibility -> dataset -------
    print("\n[phase 1] generating the dataset (orbits, geometry, visibility) ...")
    phase1.build_dataset(CONSTELLATION, minutes=MINUTES, step_s=STEP_S, out_root=ds_dir)
    print(f"[phase 1] dataset written -> {ds_dir / 'run_001'}")

    # ---- PHASE 2: channel, Doppler, association/handover, min-power ---------
    print("\n[phase 2] simulating (channel, Doppler, association, min-power) ...")
    phase2.simulate_case(
        CONSTELLATION, STRATEGY, a1=0.1, a2=0.0, g_db=GAMMA_DB, psat=100.0,
        minutes=MINUTES, step_s=STEP_S, dataset_root=ds_dir, out_dir=run_dir,
        verbose=False,
    )
    print(f"[phase 2] run written -> {run_dir}")

    # ---- PHASE 3: reduce to the paper KPIs ---------------------------------
    print("\n[phase 3] computing metrics ...")
    m = phase3.analyze_case(run_dir, SE_LEVEL, STEP_S, write_json=False)
    print("\n================= RESULT =================")
    print(f"  constellation / strategy : {CONSTELLATION} / {STRATEGY} (SE{SE_LEVEL})")
    print(f"  admission                : {m['admission_pct']} %")
    print(f"  outage                   : {m['outage_pct']} %")
    print(f"  total tx power           : {m['total_tx_power_dbw']} dBW")
    print(f"  inter-handover interval  : {m['ihi_min']} min")
    print(f"  handovers                : {m['handovers']}")
    print("==========================================")


if __name__ == "__main__":
    main()
