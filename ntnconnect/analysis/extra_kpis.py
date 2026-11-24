"""
Extra KPI figures - energy / connection / fairness  (additive, standalone).
============================================================================
Closes the 3GPP TR 38.913 / ITU-R M.2410 KPI gaps in the built-in figure set
that ARE measurable from what the simulator already records. Reads a run's
`sinr_snapshot.parquet` (the same artifact the study plotter reads) and writes
new figures, reusing the analysis output style. It does not modify core.py or
any scenario study.

Figures (registered in figure_catalog.py):
  energy_power_ccdf         CCDF of per-slot total transmit power [dBW]        (energy)
  energy_efficiency_bar     energy efficiency = served bits / Joule [Mbit/J]   (energy)
  connection_admission_bar  admission rate and blocking (1 - admission) [%]    (connection)
  connection_blocking_ccdf  CCDF of per-slot blocking probability              (connection)
  fairness_jain_bar         Jain's fairness index of per-user throughput       (fairness)

not added (the simulator does not measure these - out of scope for a
system-level tool): Doppler shift/rate, handover interruption time / HO
success-failure rate, control-plane/signalling latency, beam/gateway resource
utilisation. They are documented as reserved in figure_catalog.py.

Usage:
  from ntnconnect.analysis.extra_kpis import make_extra_kpi_figures
  make_extra_kpi_figures("<run_dir>/sinr_snapshot.parquet", out_dir, bandwidth_hz=62.5e6)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_C = 299792458.0
_RE = 6371e3

FIGURES = ["energy_power_ccdf", "energy_efficiency_bar",
           "connection_admission_bar", "connection_blocking_ccdf",
           "fairness_jain_bar", "doppler_ccdf"]


def _ccdf(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    return x, 1.0 - np.arange(1, x.size + 1) / max(x.size, 1)


def _slant_range_m(ground_dist_m, elev_deg):
    """Slant range user->satellite from ground distance + elevation (spherical Earth)."""
    g = np.asarray(ground_dist_m, float) / _RE
    th = np.radians(np.asarray(elev_deg, float))
    return _RE * np.sin(g) / np.maximum(np.cos(g + th), 1e-6)


def _style(dpi):
    """Reuse the enhanced analysis style (mathtext, no hard LaTeX dep)."""
    try:
        from ntnconnect.analysis.core import _apply_output_style
        _apply_output_style(show_titles=True, dpi=dpi, fmt="png",
                            paper_style=True, use_tex=False, base_fontsize=9)
    except Exception:
        plt.rcParams.update({"font.family": "serif", "axes.grid": True,
                             "grid.alpha": 0.35, "grid.linestyle": ":"})


def make_extra_kpi_figures(snapshot, out_dir, bandwidth_hz: float = 62.5e6,
                           carrier_hz: float = 12e9, step_s: float = 0.1,
                           dpi: int = 140, figures=None) -> list[str]:
    """Read a run snapshot and write the extra-KPI figures. Returns the paths
    written. `figures` optionally restricts to a subset of FIGURES."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    want = set(figures) if figures else set(FIGURES)
    df = pd.read_parquet(snapshot, columns=["user_id", "t_idx", "slot_admitted",
                                            "sinr_db_paper", "min_power_w"])
    _style(dpi)
    made = []

    def _save(fig, name):
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        made.append(str(p)); print(f"[extra-kpi] {p}")

    adm = df["slot_admitted"].to_numpy().astype(bool)
    d = df[adm].copy()
    d["rate_mbps"] = bandwidth_hz * np.log2(1.0 + 10 ** (d["sinr_db_paper"] / 10.0)) / 1e6
    p_tick = d.groupby("t_idx")["min_power_w"].sum()                 # total power per slot [W]

    # ENERGY: per-slot total transmit power (dBW) CCDF
    if "energy_power_ccdf" in want and len(p_tick):
        x, y = _ccdf(10 * np.log10(np.maximum(p_tick.to_numpy(), 1e-9)))
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=1.8)
        ax.set_xlabel("Total transmit power [dBW]"); ax.set_ylabel("CCDF")
        ax.set_ylim(0, 1.02); ax.grid(True, alpha=.35)
        _save(fig, "energy_power_ccdf")

    # ENERGY: energy efficiency = served bits / Joule [Mbit/J]
    if "energy_efficiency_bar" in want and d["min_power_w"].sum() > 0:
        ee = d["rate_mbps"].sum() / d["min_power_w"].sum()          # Mbit/s / W = Mbit/J
        fig, ax = plt.subplots()
        ax.bar(["Energy efficiency"], [ee], color="#1f77b4", edgecolor="black")
        ax.set_ylabel("Energy efficiency [Mbit/J]")
        ax.text(0, ee, f"{ee:.2f}", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=.35)
        _save(fig, "energy_efficiency_bar")

    # CONNECTION: admission rate + blocking [%]
    if "connection_admission_bar" in want:
        a = 100.0 * adm.mean(); b = 100.0 - a
        fig, ax = plt.subplots()
        ax.bar(["Admitted", "Blocked"], [a, b], color=["#2ca02c", "#d62728"], edgecolor="black")
        ax.set_ylabel("Traffic [%]"); ax.set_ylim(0, 105)
        for i, v in enumerate((a, b)):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=.35)
        _save(fig, "connection_admission_bar")

    # CONNECTION: per-slot blocking probability CCDF
    if "connection_blocking_ccdf" in want:
        block = 1.0 - df.groupby("t_idx")["slot_admitted"].mean().to_numpy()
        x, y = _ccdf(block)
        fig, ax = plt.subplots()
        ax.plot(x, y, lw=1.8, color="#d62728")
        ax.set_xlabel("Per-slot blocking probability"); ax.set_ylabel("CCDF")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(True, alpha=.35)
        _save(fig, "connection_blocking_ccdf")

    # FAIRNESS: Jain's index over per-user throughput
    if "fairness_jain_bar" in want:
        ru = d.groupby("user_id")["rate_mbps"].mean().to_numpy()
        jain = (ru.sum() ** 2 / (len(ru) * np.sum(ru ** 2))) if len(ru) and np.sum(ru ** 2) > 0 else np.nan
        fig, ax = plt.subplots()
        ax.bar(["Jain fairness"], [jain], color="#9467bd", edgecolor="black")
        ax.set_ylabel("Jain's fairness index"); ax.set_ylim(0, 1.05)
        ax.text(0, jain, f"{jain:.3f}", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=.35)
        _save(fig, "fairness_jain_bar")

    # DOPPLER: |Doppler shift| of the serving link = f_c/c * d(range)/dt
    if "doppler_ccdf" in want:
        dd = pd.read_parquet(snapshot, columns=["user_id", "sat_id", "t_idx",
             "ground_dist_m", "serving_sat_elev_deg"]).sort_values(["user_id", "t_idx"])
        dd = dd.assign(sl=_slant_range_m(dd["ground_dist_m"].to_numpy(),
                                         dd["serving_sat_elev_deg"].to_numpy()))
        fd = []
        for _u, gi in dd.groupby("user_id", sort=False):
            s = gi["sl"].to_numpy(); sat = gi["sat_id"].to_numpy(); t = gi["t_idx"].to_numpy()
            if len(s) < 2:
                continue
            dt = step_s * (t[1:] - t[:-1])
            same = (sat[1:] == sat[:-1]) & (dt > 0)          # same serving sat, contiguous ticks
            if same.any():
                rr = (s[1:] - s[:-1])[same] / dt[same]        # range rate [m/s]
                fd.append(np.abs(carrier_hz / _C * rr))       # |Doppler| [Hz]
        fd = np.concatenate(fd) if fd else np.array([])
        if fd.size:
            x, y = _ccdf(fd / 1e3)                             # kHz
            fig, ax = plt.subplots()
            ax.plot(x, y, lw=1.8, color="#ff7f0e")
            ax.set_xlabel("$|$Doppler shift$|$ [kHz]"); ax.set_ylabel("CCDF")
            ax.set_ylim(0, 1.02); ax.grid(True, alpha=.35)
            _save(fig, "doppler_ccdf")

    return made
