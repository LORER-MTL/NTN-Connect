# Standard Phase-3 analysis: ONE analyser for every scenario. Reads the standard
# per-run tables written by Phase 2 (the metrics stage + hooks) - the same layout
# for every study, no per-study manifest or special path - and writes a standard
# analysis/ folder (summary.json + a SINR CDF figure). Returns the summary dict.

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# Standard SINR-bearing tables, in priority order (first present wins).
_SINR_TABLES = ("per_user_kpis", "sinr_snapshot", "link_audit", "kpi_sinr_db")


def _read(out_dir: Path, name: str):
    fs = list(out_dir.rglob(f"{name}*/**/*.parquet")) + list(out_dir.glob(f"{name}*.parquet"))
    if not fs:
        return None
    try:
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    except Exception as e:                       # noqa: BLE001
        _log.info("standard analysis: could not read %s (%s)", name, e)
        return None


def run_standard_analysis(out_dir) -> dict:
    """Uniform metrics for any scenario run. Tolerant: missing tables are simply
    skipped. Writes <out_dir>/analysis/summary.json and sinr_cdf.png."""
    out_dir = Path(out_dir)
    ana = out_dir / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    m: dict = {"run_dir": str(out_dir)}

    # --- served SINR / rate (from the first available standard SINR table) ---
    sinr_df = None
    for name in _SINR_TABLES:
        df = _read(out_dir, name)
        if df is not None and "sinr_db" in df.columns:
            sinr_df = df
            m["sinr_source"] = name
            break
    if sinr_df is not None:
        served = sinr_df[sinr_df["sinr_db"] > -100.0]
        m["served_links"] = int(len(served))
        m["median_sinr_db"] = round(float(served["sinr_db"].median()), 3) if len(served) else None
        m["mean_sinr_db"] = round(float(served["sinr_db"].mean()), 3) if len(served) else None
        for rc in ("link_rate_bps", "throughput_mbps"):
            if rc in sinr_df.columns and len(served):
                v = float(served[rc].mean())
                m["mean_rate_mbps"] = round(v / (1e6 if rc == "link_rate_bps" else 1.0), 3)
                break

    # --- handovers ---
    ho = _read(out_dir, "ho_events")
    if ho is not None:
        m["handovers"] = int(len(ho))

    # --- system summary row (if present) ---
    sysrow = _read(out_dir, "summary_system")
    if sysrow is not None and len(sysrow):
        m["system_summary"] = {k: (round(v, 4) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in sysrow.iloc[-1].to_dict().items()}

    # --- standard figure: served-SINR CDF ---
    if sinr_df is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            served = sinr_df[sinr_df["sinr_db"] > -100.0]["sinr_db"].to_numpy()
            if served.size:
                xs = np.sort(served)
                ys = np.arange(1, xs.size + 1) / xs.size
                plt.figure()
                plt.plot(xs, ys)
                plt.xlabel("served SINR [dB]"); plt.ylabel("CDF"); plt.grid(True, alpha=0.3)
                plt.title(f"SINR CDF ({out_dir.name})")
                plt.savefig(ana / "sinr_cdf.png", dpi=120, bbox_inches="tight")
                plt.close()
                m["figure"] = str(ana / "sinr_cdf.png")
        except Exception as e:                   # noqa: BLE001
            _log.info("standard analysis: figure skipped (%s)", e)

    (ana / "summary.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return m
