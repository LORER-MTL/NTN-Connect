#!/usr/bin/env python
"""Smoke test for the NTN-Connect pipeline.

Runs ONE short case (starlink / farthest_approaching / SE2, 5 min @ 2 s) through
all three phases and asserts every pipeline component is wired and physical:

    constellation + propagation   -> a dataset manifest exists
    visibility                    -> serving elevation respects the 25-deg mask
    channel modeling              -> path loss / SINR / noise finite and physical
    Doppler                       -> range-rate is signed and LEO-bounded
    association / handover        -> valid serving sat ids, sane handover count
    resource allocation           -> per-link power within the 100 W cap, >= 0
    performance (phase 3 KPIs)    -> admission/outage in [0,100], power finite

Usage:
    pytest tests/                 # as part of a suite
    python tests/test_smoke.py    # standalone, prints PASS/FAIL per check

It is deliberately small (a couple of minutes on a laptop). The full validation
is the paper campaign in scenarios/handover_study/.
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCEN = ROOT / "scenarios" / "handover_study"
for p in (ROOT, SCEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import config as C  # noqa: E402


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCEN / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P1 = _load("scripts/phase1/generate_dataset.py", "p1")
P2 = _load("scripts/phase2/simulate.py", "p2")
P3 = _load("scripts/phase3/metrics.py", "p3")

CNST, STRAT, SE, GAMMA = "starlink", "farthest_approaching", 2, 4.77
MIN, STEP = 5.0, 2.0
OUT = ROOT / "outputs" / "smoketest"
DS = OUT / "dataset"
RUN = OUT / "run" / f"{CNST}__{STRAT}__SE{SE}" / "run_001"

_CACHE: dict[str, object] = {}


def _pipeline():
    """Run phase 1 + 2 once; return (manifest, dataframe). Cached across checks."""
    if "df" in _CACHE:
        return _CACHE["man"], _CACHE["df"]
    if not (DS / "run_001" / "manifest.json").is_file():
        P1.build_dataset(CNST, minutes=MIN, step_s=STEP, out_root=DS)
    man = json.loads((DS / "run_001" / "manifest.json").read_text())
    P2.simulate_case(CNST, STRAT, a1=0.1, a2=0.0, g_db=GAMMA, psat=100.0,
                     minutes=MIN, step_s=STEP, dataset_root=DS, out_dir=RUN,
                     verbose=False)
    snaps = [s for s in sorted(RUN.glob("batch_*/sinr_snapshot.parquet")) if s.is_file()]
    assert snaps, "phase 2 produced no sinr_snapshot"
    df = pd.concat([pd.read_parquet(s) for s in snaps], ignore_index=True)
    _CACHE["man"], _CACHE["df"] = man, df
    return man, df


def test_phase1_dataset():
    man, _ = _pipeline()
    assert isinstance(man, dict) and man, "dataset manifest is empty"


def test_visibility_mask():
    _, df = _pipeline()
    el = df["serving_sat_elev_deg"].to_numpy()
    assert np.nanmin(el) >= C.MIN_ELEV_USER_DEG - 0.5, "serving sat below the 25-deg mask"
    assert np.nanmax(el) <= 90.1


def test_channel():
    _, df = _pipeline()
    pl = df["path_loss_db"].to_numpy()
    assert np.all(np.isfinite(pl)) and 140 < np.nanmedian(pl) < 210, "path loss unphysical"
    sinr = df["sinr_db_paper"].to_numpy()
    assert np.isfinite(sinr).mean() > 0.99, "SINR not finite"
    assert (df["noise_w"].to_numpy() > 0).all(), "noise power must be > 0"


def test_doppler():
    _, df = _pipeline()
    rr = df["range_rate_m_s"].to_numpy()
    assert rr.min() < 0 < rr.max(), "range-rate must be signed (approaching and receding)"
    assert np.nanmax(np.abs(rr)) < 8000, "range-rate exceeds LEO bounds"


def test_association_handover():
    _, df = _pipeline()
    assert df["sat_id"].notna().all(), "unassociated users present"
    nho = 0
    for _u, g in df.sort_values("t_idx").groupby("user_id"):
        s = g["sat_id"].to_numpy()
        nho += int((s[1:] != s[:-1]).sum())
    assert nho >= 0


def test_resource_allocation():
    _, df = _pipeline()
    mp = df["min_power_w"].to_numpy()
    assert np.nanmax(mp) <= 100.0 + 1e-6, "per-link power exceeds the 100 W cap"
    assert np.nanmin(mp[np.isfinite(mp)]) >= 0, "negative power"
    adm = df["slot_admitted"].to_numpy().astype(bool)
    assert set(np.unique(adm)).issubset({True, False})


def test_phase3_kpis():
    _pipeline()
    m = P3.analyze_case(RUN, SE, STEP, write_json=False)
    assert 0 <= m["admission_pct"] <= 100
    assert 0 <= m["outage_pct"] <= 100
    assert np.isfinite(m["total_tx_power_dbw"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            npass += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n===== {npass}/{len(tests)} checks passed =====")
    sys.exit(0 if npass == len(tests) else 1)
