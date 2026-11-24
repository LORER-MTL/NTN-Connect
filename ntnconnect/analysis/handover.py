# Handover-strategy comparison: IHI (mean inter-handover interval), rate (median),
# balanced HEI = sqrt(rate_norm*ihi_norm), Pareto plot. Reads sinr_snapshot/handover_events.
from __future__ import annotations
import csv, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

_DEFAULT_LABELS = {
    "max_sinr": "MRSRP", "farthest_approaching": "MST",
    "max_rx_power": "MaxRxP", "load_aware": "LoadAware", "utility_based": "Utility",
    "persistent_nearest": "MEA", "gpp_a3": "3GPP-A3",
}


def _concat(rd: str, pat: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(rd, pat)))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _n_rows(rd: str, pat: str) -> int:
    return sum(len(pd.read_parquet(f)) for f in sorted(glob.glob(os.path.join(rd, pat))))


def _find_run(sim_dir: Path, cnst: str, strat: str) -> str | None:
    cell = Path(sim_dir) / f"{cnst}__{strat}"
    runs = sorted(cell.glob("run_*")) if cell.exists() else []
    return str(runs[-1]) if runs else None


def _serving_change_ticks(rd: str):
    """Per-user list of ticks at which the SERVING satellite changes, from
    sinr_snapshot (ground truth). Counts every handover - soft sinr-gain
    switches and forced break-before-make reconnections alike."""
    sn = _concat(rd, "sinr_snapshot*.parquet")
    if sn.empty or not {"user_id", "sat_id", "t_idx"}.issubset(sn.columns):
        return {}
    out = {}
    for uid, g in sn.sort_values("t_idx").groupby("user_id"):
        t = g["t_idx"].to_numpy(); sat = g["sat_id"].to_numpy()
        out[int(uid)] = t[1:][sat[1:] != sat[:-1]]
    return out


def _ihi_intervals(rd: str, step_s: float = 1.0) -> np.ndarray:
    """Per-user inter-handover intervals (seconds), for the IHI CCDF. An IHI is
    the time between consecutive changes of the serving satellite. np.diff gives
    the gap in TICKS, so multiply by step_s to get real seconds."""
    iv = []
    for _uid, chg in _serving_change_ticks(rd).items():
        if chg.size >= 2:
            iv.extend((np.diff(chg.astype(float)) * float(step_s)).tolist())
    return np.asarray(iv, float)


def _metrics(rd: str, bw_hz: float, duration_min: float | None, step_s: float = 1.0) -> dict:
    sn = _concat(rd, "sinr_snapshot*.parquet")
    scol = next((c for c in sn.columns if "sinr" in c.lower()), None) if not sn.empty else None
    if scol is None:
        return {}
    sinr_db = sn[scol].astype(float).to_numpy()
    sinr_db = sinr_db[np.isfinite(sinr_db)]
    sinr_lin = 10 ** (sinr_db / 10.0)
    rcol = next((c for c in sn.columns if c.lower() in
                 ("rate_mbps", "throughput_mbps")), None)
    rate_mbps = (float(np.nanmedian(sn[rcol].astype(float))) if rcol
                 else float(np.median(bw_hz * np.log2(1.0 + sinr_lin)) / 1e6))
    served_user_sec = len(sinr_db) * float(step_s)       # rows are step_s-apart -> real user-seconds
    # Handover = a change of the serving satellite (soft OR forced reconnection),
    # counted from sinr_snapshot so it matches the fig9 IHI definition.
    _chg = _serving_change_ticks(rd)
    n_ho_all = int(sum(c.size for c in _chg.values()))   # every handover (reported)
    # IHI / HO-rate EXCLUDE each user's first and last handover: the finite
    # observation window truncates the tenure before the first change and after
    # the last change, so those two boundary handovers per user are censored and
    # must not enter the inter-handover statistic (otherwise IHI is biased LOW).
    n_ho = int(sum(max(c.size - 2, 0) for c in _chg.values()))
    ihi_s = (served_user_sec / n_ho) if n_ho > 0 else float(served_user_sec)
    ho_per_min = (n_ho / (served_user_sec / 60.0)) if served_user_sec else 0.0
    return {
        "median_rate_mbps": round(rate_mbps, 2),
        "n_handovers":      n_ho_all,
        "ho_per_user_min":  round(ho_per_min, 3),
        "ihi_s":            round(ihi_s, 1),
        "ihi_min":          round(ihi_s / 60.0, 2),
    }


def _add_hei(rows: list[dict]) -> None:
    R = np.array([r["median_rate_mbps"] for r in rows], float)
    S = np.array([r["ihi_s"] for r in rows], float)
    Rtil = R / np.maximum(R.max(), 1e-9)
    Stil = S / np.maximum(S.max(), 1e-9)
    for r, rt, st in zip(rows, Rtil, Stil):
        r["HEI"] = round(float(np.sqrt(rt * st)), 3)


# emit CSV / LaTeX / figures
def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[out] {path}")


def _plots(rows, cnst: str, out_dir: Path, labels: dict):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skipped ({e})"); return
    names = [labels.get(r["strategy"], r["strategy"]) for r in rows]
    # Pareto: rate (y, higher better) vs IHI (x, higher better) -> top-right is best
    fig, ax = plt.subplots(figsize=(5, 4))
    for r, n in zip(rows, names):
        ax.scatter(r["ihi_min"], r["median_rate_mbps"], s=80)
        ax.annotate(n, (r["ihi_min"], r["median_rate_mbps"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean inter-handover interval (min)  →  more stable")
    ax.set_ylabel("Median rate (Mbps)  →  higher quality")
    ax.set_title(f"Rate vs. handover-stability trade-off — {cnst}")
    ax.grid(alpha=0.3); fig.tight_layout()
    p1 = out_dir / f"handover_pareto_{cnst}.png"; fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"[out] {p1}")
    # HEI is reported in the handover comparison TABLE (not a separate bar figure).


def _plot_ihi_ccdf(intervals, cnst, out_dir, labels):
    """CCDF of the inter-handover interval, overlaying the strategies."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skipped ({e})"); return
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted = False
    for s, iv in intervals.items():
        iv = np.asarray(iv, float)
        if iv.size == 0:
            continue
        x = np.sort(iv) / 60.0                              # minutes
        ccdf = 1.0 - np.arange(x.size) / x.size
        ax.step(x, ccdf, where="post", label=labels.get(s, s))
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("Inter-handover interval (min)")
    ax.set_ylabel(r"CCDF  $P[\mathrm{IHI} > x]$")
    ax.set_title(f"Inter-handover interval — {cnst}")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    p = out_dir / f"ihi_ccdf_{cnst}.png"; fig.savefig(p, dpi=140); plt.close(fig)
    print(f"[out] {p}")


def run_handover_comparison(sim_dir, analysis_dir, constellation="oneweb",
                            strategies=("max_sinr", "persistent_nearest",
                                        "farthest_approaching"),
                            bw_hz=10e6, duration_min=None, labels=None,
                            emit=True, step_s=None):
    """Compare handover strategies for one constellation on IHI + rate + HEI.

    sim_dir      : directory holding <constellation>__<strategy>/run_*/ outputs.
    analysis_dir : where CSV/TeX/figures are written.
    step_s       : simulation step [s]; if None, read from sim_dir/manifest.csv so
                   ticks convert to real seconds (t_idx is in step_s units, not 1 s).
    Returns the list of per-strategy metric dicts (with HEI).
    """
    sim_dir = Path(sim_dir); analysis_dir = Path(analysis_dir)
    labels = labels or _DEFAULT_LABELS
    if step_s is None:                                   # resolve the time base
        step_s = 1.0
        try:
            mf = pd.read_csv(sim_dir / "manifest.csv")
            if "step_s" in mf.columns and mf["step_s"].notna().any():
                step_s = float(mf["step_s"].dropna().iloc[0])
        except Exception:
            pass
    rows, intervals = [], {}
    for s in strategies:
        rd = _find_run(sim_dir, constellation, s)
        if rd is None:
            print(f"[skip] no run for {constellation}/{s}"); continue
        m = _metrics(rd, bw_hz, duration_min, step_s)
        if not m:
            print(f"[skip] no SINR data for {constellation}/{s}"); continue
        rows.append({"strategy": s, **m})
        intervals[s] = _ihi_intervals(rd, step_s)
    if not rows:
        print(f"[handover] no runs found under {sim_dir}"); return []
    _add_hei(rows)
    print(f"\n=== Handover comparison ({constellation}) ===")
    print("%-22s %10s %8s %9s %6s" % ("strategy", "rate_Mbps", "HO/min", "IHI_min", "HEI"))
    for r in rows:
        print("%-22s %10.2f %8.3f %9.2f %6.3f" %
              (r["strategy"], r["median_rate_mbps"], r["ho_per_user_min"],
               r["ihi_min"], r["HEI"]))
    if emit:
        _write_csv(rows, analysis_dir / f"handover_comparison_{constellation}.csv")
        _plots(rows, constellation, analysis_dir, labels)
        _plot_ihi_ccdf(intervals, constellation, analysis_dir, labels)
    return rows
