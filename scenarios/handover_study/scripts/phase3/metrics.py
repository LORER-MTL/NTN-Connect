#!/usr/bin/env python
"""WISEE - PHASE 3 : metrics / KPIs  (EXPLICIT flow).

Reads a phase-2 run and computes the study KPIs with the SAME definitions as the
original paper_results.py (no engine duplication - pure pandas on the snapshot):

  * admission [%]        = 100 * mean(slot_admitted)
  * outage [%]           = 100 * mean( sinr_db_paper[admitted] < gamma_th(SE) )
  * IHI [min]            = len*STEP / n_ho / 60,
                           n_ho = sum_u max( (#serving-sat changes) - 2, 0 )
  * total Tx power [dBW] = 10*log10( sum(min_power_w[admitted]) / n_ticks )

analyze_case()  -> metrics for one run dir (also written to <run>/metrics.json)
analyze_point() -> a small CSV over the strategies of one operating point

Run one case:
  python scenarios/WISEE/scripts/phase3/metrics.py \
      --run outputs/wisee_run/starlink__farthest_approaching --se 4
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# SE index -> required SINR threshold gamma_th [dB]  (SE 1..6)
GAMMAS = [0.0, 4.77, 8.45, 11.76, 14.91, 18.0]
GTH = {se: GAMMAS[se - 1] for se in range(1, len(GAMMAS) + 1)}


def analyze_case(run_dir: Path, se: int, step_s: float = 1.0,
                 write_json: bool = True) -> dict:
    run_dir = Path(run_dir)
    # The engine writes the per-tick sinr_snapshot once per 15-min batch, under
    # batch_<NNNN>/. A run therefore has one snapshot per batch; load them all
    # (root-level snapshot, if present from a merge, is treated as a single seg).
    snaps = sorted(run_dir.glob("batch_*/sinr_snapshot.parquet"))
    root_snap = run_dir / "sinr_snapshot.parquet"
    if root_snap.is_file():
        snaps = [root_snap]
    if not snaps:
        return {"run_dir": str(run_dir), "error": "no sinr_snapshot"}
    parts = []
    for seg, sp in enumerate(snaps):
        d = pd.read_parquet(sp); d["_seg"] = seg
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    adm = df["slot_admitted"].to_numpy().astype(bool)
    # t_idx is GLOBAL across the run (batches are 0..899, 900..1799, 1800..2699),
    # i.e. one continuous series split into shards for I/O. So the shards concat
    # into the single series the original single-file metrics.py assumed.
    nt = int(df["t_idx"].nunique())

    m = {"run_dir": str(run_dir), "se": se, "gamma_th_db": GTH[se],
         "n_batches": len(snaps),
         "n_users": int(df["user_id"].nunique()), "n_ticks": nt}
    # admission
    m["admission_pct"] = round(100.0 * adm.mean(), 3) if len(adm) else np.nan
    # outage = admitted users below the SE threshold
    if "sinr_db_paper" in df.columns and adm.any():
        sp = df["sinr_db_paper"].to_numpy()
        m["outage_pct"] = round(100.0 * float(np.mean(sp[adm] < GTH[se])), 3)
    else:
        m["outage_pct"] = np.nan
    # total transmit power [dBW]
    if "min_power_w" in df.columns and adm.any() and nt:
        tx = float(df.loc[adm, "min_power_w"].sum()) / nt
        m["total_tx_power_dbw"] = round(10.0 * np.log10(max(tx, 1e-9)), 3)
        m["total_tx_power_w"] = round(tx, 4)
    else:
        m["total_tx_power_dbw"] = m["total_tx_power_w"] = np.nan
    # IHI [min] from serving-sat changes, minus 2 boundary transitions PER USER
    # (the first and last tenure are censored by the finite window). Counted over
    # the GLOBAL per-user series (sorted by global t_idx), identical to the engine-
    # native analysis/handover.py and the original single-file metrics.py. Batch
    # boundaries are NOT segment breaks: t_idx is continuous, so a handover across
    # a boundary is a real handover and must be counted.
    n_ho = 0
    for _u, g in df.sort_values("t_idx").groupby("user_id", sort=False):
        s = g["sat_id"].to_numpy()
        if len(s) > 1:
            n_ho += max(int((s[1:] != s[:-1]).sum()) - 2, 0)
    m["ihi_min"] = round(len(df) * step_s / n_ho / 60.0, 3) if n_ho else np.nan
    m["handovers"] = int(n_ho)

    if write_json:
        (run_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    return m


def analyze_point(point_sim_dir: Path, constellation: str, se: int,
                  step_s: float = 1.0) -> pd.DataFrame:
    """Every <cnst>__<strat>/run_001 under a point's simulation/ dir -> one table."""
    rows = []
    for run in sorted(Path(point_sim_dir).glob(f"{constellation}__*/run_001")):
        strat = run.parent.name.split("__", 1)[1]
        m = analyze_case(run, se, step_s)
        m["strategy"] = strat
        rows.append(m)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(Path(point_sim_dir) / "metrics.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="WISEE phase 3 - metrics")
    ap.add_argument("--run", required=True, help="a phase-2 run dir (has sinr_snapshot)")
    ap.add_argument("--se", type=int, default=4, help="spectral-efficiency index 1..6")
    ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    m = analyze_case(Path(a.run), a.se, a.step)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
