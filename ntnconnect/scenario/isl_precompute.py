# Phase-1 ISL feasibility precompute: per-tick sparse sat-sat distances.
# Stores raw distances (<= cap); Phase 2 applies each operator's max_range_km.
# Distances use the same spherical-ECEF formula as stages/isl.py so the
# masked result reproduces the recompute path exactly.
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

EARTH_R_M = 6_371_000.0
DEFAULT_CAP_KM = 6000.0
SIDECAR = "isl_precomp.json"


def _ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    r = EARTH_R_M + alt_m
    return np.stack([r * np.cos(lat) * np.cos(lon),
                     r * np.cos(lat) * np.sin(lon),
                     r * np.sin(lat)], axis=1)


def _pairs_for_group(df_op, cap_m):
    ids = df_op["node_id"].to_numpy(dtype=int)
    n = len(ids)
    if n < 2:
        return None
    pos = _ecef(df_op["lat"].to_numpy(float), df_op["lon"].to_numpy(float),
                df_op["alt_m"].to_numpy(float))
    diff = pos[:, None, :] - pos[None, :, :]
    dmat = np.linalg.norm(diff, axis=2)
    iu, ju = np.triu_indices(n, k=1)
    d = dmat[iu, ju]
    keep = d <= cap_m
    if not keep.any():
        return None
    return ids[iu[keep]], ids[ju[keep]], d[keep]


def build_isl_precomp(eph_dir: str | Path, cap_km: float = DEFAULT_CAP_KM,
                      source: str = "visible_sats") -> None:
    """Read an existing ephemeris dataset and write isl_links_NN.parquet
    next to each constellation's satellite batches. `source` picks the batch
    prefix ('visible_sats' = visible set, matches satellites_t; 'all_sats' = full)."""
    eph_dir = Path(eph_dir)
    man = json.loads((eph_dir / "manifest.json").read_text())
    cap_m = float(cap_km) * 1000.0
    n_batches = int(man["n_batches"])
    total = 0
    for cnst in man["constellations"]:
        cdir = eph_dir / cnst
        for b in range(n_batches):
            src = cdir / f"{source}_{b:02d}.parquet"
            if not src.exists():
                src = cdir / f"visible_sats_{b:02d}.parquet"
            if not src.exists():
                continue
            df = pd.read_parquet(src, columns=["t_idx", "node_id",
                                               "operator_id", "lat", "lon", "alt_m"])
            rows = []
            for (t_idx, op_id), g in df.groupby(["t_idx", "operator_id"], sort=False):
                res = _pairs_for_group(g, cap_m)
                if res is None:
                    continue
                a, bb, d = res
                rows.append(pd.DataFrame({
                    "t_idx": int(t_idx), "sat_a": a, "sat_b": bb,
                    "dist_m": d, "operator_id": int(op_id)}))
            out = (pd.concat(rows, ignore_index=True) if rows
                   else pd.DataFrame(columns=["t_idx", "sat_a", "sat_b",
                                              "dist_m", "operator_id"]))
            out.to_parquet(cdir / f"isl_links_{b:02d}.parquet", index=False)
            total += len(out)
    (eph_dir / SIDECAR).write_text(json.dumps(
        {"cap_km": cap_km, "source": source}, indent=2))
    _log.info("isl precompute: %d pairs over %s", total, eph_dir)


def read_cap_m(eph_dir: str | Path) -> float | None:
    p = Path(eph_dir) / SIDECAR
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["cap_km"]) * 1000.0


def main():
    ap = argparse.ArgumentParser(description="Precompute ISL sat-sat distances.")
    ap.add_argument("--eph-dir", required=True)
    ap.add_argument("--cap-km", type=float, default=DEFAULT_CAP_KM)
    ap.add_argument("--source", default="visible_sats", choices=["visible_sats", "all_sats"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_isl_precomp(a.eph_dir, a.cap_km, a.source)


if __name__ == "__main__":
    main()
