# Processing/scalability comparison: per-constellation build time, wall time,
# CPU time, peak memory (from manifests + resource_profile.json).
from __future__ import annotations
import csv, json
from pathlib import Path

_DEFAULT_LABELS = {"oneweb": "OneWeb", "starlink": "Starlink", "kuiper": "Kuiper"}
_IEEE_FIGSIZE = (3.5, 2.6)   # uniform IEEE single-column size (inches)


def _parse_build(dataset_dir: Path, cnst: str) -> dict:
    man = json.loads((dataset_dir / "manifest.json").read_text())
    c = man["constellations"][cnst]
    per_batch = c.get("per_batch", [])
    pre  = sum(b.get("pre_pass_s", 0.0)  for b in per_batch)
    fine = sum(b.get("fine_pass_s", 0.0) for b in per_batch)
    alls = c.get("all_sats", {}) or {}
    return {"n_sat_catalogue": c.get("n_sats_in_tle", 0),
            "build_wall_s": c.get("wall_time_s", 0.0),
            "pre_pass_s": round(pre, 1), "fine_pass_s": round(fine, 1),
            "allsat_rows": alls.get("n_rows", 0), "allsat_wall_s": alls.get("wall_s", 0.0)}


def _parse_sim_wall(sim_dir: Path, cnst: str, strategy: str) -> float | None:
    p = sim_dir / "manifest.csv"
    if not p.exists():
        return None
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["constellation"] == cnst and r["strategy"] == strategy:
                return float(r["wall_time_s"])
    return None


def _parse_resource(sim_dir: Path, cnst: str, strategy: str) -> dict:
    cell = sim_dir / f"{cnst}__{strategy}"
    runs = sorted(cell.glob("run_*")) if cell.exists() else []
    if not runs:
        return {}
    p = runs[-1] / "resource_profile.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {"sim_cpu_s": d.get("cpu_s"), "sim_peak_rss_mb": d.get("peak_rss_mb"),
            "sim_wall_profiled_s": d.get("wall_s")}


def _parse_mean_visible(analysis_dir: Path, cnst: str, strategy: str) -> float | None:
    p = analysis_dir / "strategy_summary.csv"
    if not p.exists():
        return None
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["constellation"] == cnst and r["strategy"] == strategy:
                return float(r["mean_n_visible"])
    return None


def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[out] {path}")


def _read_timeseries(sim_dir: Path, cnst: str, strategy: str):
    """Return (elapsed_s, rss_mb, cpu_pct) arrays from resource_timeseries.csv
    for the latest run of <cnst>__<strategy>, or None if unavailable."""
    cell = sim_dir / f"{cnst}__{strategy}"
    runs = sorted(cell.glob("run_*")) if cell.exists() else []
    if not runs:
        return None
    p = runs[-1] / "resource_timeseries.csv"
    if not p.exists():
        return None
    t, rss, cpu = [], [], []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                t.append(float(r["elapsed_s"]))
                rss.append(float(r["rss_mb"]) if r["rss_mb"] not in ("", "None") else float("nan"))
                cpu.append(float(r["cpu_pct"]) if r["cpu_pct"] not in ("", "None") else float("nan"))
            except (KeyError, ValueError):
                continue
    return (t, rss, cpu) if t else None


def _write_timeseries_figs(sim_dir: Path, analysis_dir: Path, constellations,
                           labels, strat_label=None):
    """two separate single-panel figures - memory over run time and CPU
    utilisation over run time - each with one line per (constellation, strategy),
    covering all strategies present in sim_dir. Constellation sets the colour,
    strategy is named in the legend."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skipped ({e})"); return
    import os
    strat_label = strat_label or (lambda s: s)
    from ntnconnect.analysis.core import _COLORS as colors  # unified constellation colours
    # The sampler records CPU as process-CPU-time / wall-time (a per-core SUM in
    # 0..N_cores*100); normalize by the core count so the figure reads as a
    # fraction of the whole machine (100% = every core saturated).
    n_cores = os.cpu_count() or 1
    figM, axM = plt.subplots(figsize=_IEEE_FIGSIZE)
    figC, axC = plt.subplots(figsize=_IEEE_FIGSIZE)
    any_data = False
    for c in constellations:
        strats = sorted({p.name.split("__", 1)[1] for p in sim_dir.glob(f"{c}__*")
                         if "__" in p.name})
        for s in strats:
            ts = _read_timeseries(sim_dir, c, s)
            if ts is None:
                continue
            t, rss, cpu = ts
            cpu_norm = [min(100.0, v / n_cores) if v == v else float("nan") for v in cpu]
            lab = f"{labels.get(c, c)}-{strat_label(s)}"
            col = colors.get(c, None)
            axM.plot(t, rss, color=col, label=lab, lw=1.3, alpha=0.9)
            axC.plot(t, cpu_norm, color=col, label=lab, lw=1.3, alpha=0.9)
            any_data = True
    if not any_data:
        print("[fig] no resource_timeseries.csv — skipping resource figures")
        plt.close(figM); plt.close(figC); return
    axM.set_xlabel("Run time (s)"); axM.set_ylabel("Resident memory (MB)")
    axM.set_title("Memory footprint over the run"); axM.grid(alpha=0.3)
    axM.legend(fontsize=7, ncol=2)
    figM.tight_layout(); figM.savefig(analysis_dir / "resource_memory.png", dpi=140)
    plt.close(figM); print(f"[out] {analysis_dir / 'resource_memory.png'}")
    axC.set_xlabel("Run time (s)"); axC.set_ylabel(f"CPU utilisation (%, {n_cores} cores)")
    axC.set_ylim(0, 100); axC.set_title("CPU utilisation over the run"); axC.grid(alpha=0.3)
    axC.legend(fontsize=7, ncol=2)
    figC.tight_layout(); figC.savefig(analysis_dir / "resource_cpu.png", dpi=140)
    plt.close(figC); print(f"[out] {analysis_dir / 'resource_cpu.png'}")


def _write_resource_fig(sim_dir: Path, analysis_dir: Path, constellations,
                        labels, strat="farthest_approaching"):
    """two figures for one strategy - resident memory over run time and CPU
    utilisation over run time, both constellations - to show the per-epoch
    assignment cost stays cheap."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skipped ({e})"); return
    import os
    from ntnconnect.analysis.core import _COLORS as colors  # unified constellation colours
    n_cores = os.cpu_count() or 1
    figM, axM = plt.subplots(figsize=_IEEE_FIGSIZE)
    figC, axC = plt.subplots(figsize=_IEEE_FIGSIZE)
    any_data = False
    for c in constellations:
        ts = _read_timeseries(sim_dir, c, strat)
        if ts is None:
            continue
        t, rss, cpu = ts
        cpu_norm = [min(100.0, v / n_cores) if v == v else float("nan") for v in cpu]
        col = colors.get(c, None)
        axM.plot(t, rss, color=col, label=labels.get(c, c), lw=1.6)
        axC.plot(t, cpu_norm, color=col, label=labels.get(c, c), lw=1.6)
        any_data = True
    if not any_data:
        print(f"[fig] no resource_timeseries — skipping resource figures")
        plt.close(figM); plt.close(figC); return
    axM.set_xlabel("Run time (s)"); axM.set_ylabel("Resident memory (MB)")
    axM.set_title("resident memory over the run")
    axM.grid(alpha=0.3); axM.legend(fontsize=9)
    figM.tight_layout()
    figM.savefig(analysis_dir / "resource_memory.png", dpi=140)
    plt.close(figM); print(f"[out] {analysis_dir / 'resource_memory.png'}")
    axC.set_xlabel("Run time (s)"); axC.set_ylabel(f"CPU utilisation (%, {n_cores} cores)")
    axC.set_ylim(0, 100); axC.set_title("CPU utilisation over the run")
    axC.grid(alpha=0.3); axC.legend(fontsize=9)
    figC.tight_layout()
    figC.savefig(analysis_dir / "resource_cpu.png", dpi=140)
    plt.close(figC); print(f"[out] {analysis_dir / 'resource_cpu.png'}")


def run_processing_comparison(dataset_dir, sim_dir, analysis_dir,
                              constellations=("oneweb", "starlink"),
                              strategy="max_sinr", duration_min=60.0,
                              labels=None, emit=True):
    """Compare per-constellation processing cost (build/sim time, CPU, memory).

    dataset_dir  : scenario-generation output (manifest.json).
    sim_dir      : simulation output (manifest.csv, run dirs with resource_profile.json).
    analysis_dir : where CSV/TeX/figure are written (and strategy_summary.csv read).
    Returns the list of per-constellation metric rows.
    """
    dataset_dir = Path(dataset_dir); sim_dir = Path(sim_dir); analysis_dir = Path(analysis_dir)
    labels = labels or _DEFAULT_LABELS
    if not (dataset_dir / "manifest.json").exists():
        print(f"[processing] missing {dataset_dir/'manifest.json'}"); return []
    rows = []
    for c in constellations:
        b = _parse_build(dataset_dir, c)
        wall = _parse_sim_wall(sim_dir, c, strategy)
        res = _parse_resource(sim_dir, c, strategy)
        if wall is None:
            wall = res.get("sim_wall_profiled_s")
        vis = _parse_mean_visible(analysis_dir, c, strategy)
        dur_s = duration_min * 60.0
        allsat_thru = (b["allsat_rows"] / b["allsat_wall_s"]) if b["allsat_wall_s"] else None
        rows.append({
            "constellation": c, "n_sat_catalogue": b["n_sat_catalogue"],
            "mean_visible": round(vis, 1) if vis is not None else None,
            "allsat_rows_M": round(b["allsat_rows"] / 1e6, 2),
            "build_wall_s": b["build_wall_s"],
            "allsat_throughput_krows_s": round(allsat_thru / 1e3, 1) if allsat_thru else None,
            "sim_wall_s": wall,
            "sim_throughput_ticks_s": round(dur_s / wall, 1) if wall else None,
            "sim_cpu_s": res.get("sim_cpu_s"), "sim_peak_rss_mb": res.get("sim_peak_rss_mb")})
    print("\n=== Processing comparison ===")
    for r in rows:
        print(f"  {r['constellation']:9s} catalogue={r['n_sat_catalogue']:>6} "
              f"build={r['build_wall_s']}s sim={r['sim_wall_s']}s "
              f"cpu={r['sim_cpu_s']} peakMB={r['sim_peak_rss_mb']}")
    if emit:
        _write_csv(rows, analysis_dir / "processing_comparison.csv")
        # IEEE figure style: no titles, LaTeX serif, black text, uniform size.
        try:
            from ntnconnect.analysis.core import _apply_output_style
            _apply_output_style(show_titles=False, dpi=300, fmt="png",
                                paper_style=True, use_tex=True, base_fontsize=8)
        except Exception as _e:
            print(f"[style] skipped ({_e})")
        # CPU/memory scale is reported in tab:processing; the traces are emitted
        # as two separate figures (memory, CPU) covering all strategies.
        _write_timeseries_figs(sim_dir, analysis_dir, constellations, labels)
        # Focused single resource figure for one strategy.
        _write_resource_fig(sim_dir, analysis_dir, constellations, labels)
    return rows
