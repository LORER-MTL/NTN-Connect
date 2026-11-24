# Copyright (c) 2026 Mohammed Al-Mekhlafi, Email: mhdalmekhlafi@gmail.com, mohammed.al-mekhlafi@polymtl.ca
"""Build the globe's interactive-KPI data file (analysis/interactive.json)."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# Cap stored curves to keep interactive.json small; shape/tails preserved
_MAX_CURVE_POINTS = 600


def _downsample_xy(x, y):
    """Thin a curve to <= _MAX_CURVE_POINTS points, keeping both endpoints."""
    n = min(len(x), len(y))
    if n <= _MAX_CURVE_POINTS:
        return x, y
    idx = np.unique(np.linspace(0, n - 1, _MAX_CURVE_POINTS).round().astype(int))
    return [x[i] for i in idx], [y[i] for i in idx]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from ntnconnect.analysis.visualization.cesium_view import _discover_runs, _OP_LABEL
except ImportError:
    from cesium_view import _discover_runs, _OP_LABEL

from ntnconnect.analysis.core import run_analysis

# Renderable figures referenced by the rail panels; e2e_rtt_ccdf excluded (slow)
_COVERAGE_FIGS = ["visibility_ccdf", "coverage_bar", "visibility_gap_cdf",
                  "contact_cdf", "intercontact_gap_cdf", "access_delay_ccdf",
                  "outage_ccdf"]
_STRATEGY_FIGS = ["sinr_ccdf", "ion_ccdf", "ho_rate_bar", "ihi_cdf",
                  "availability_bar", "pingpong_bar", "rate_ccdf", "avg_rate_bar"]
_PANEL_FIGURES = _COVERAGE_FIGS + _STRATEGY_FIGS
_RTT_FIGURE = "e2e_rtt_ccdf"
_CNST_SUFFIX = re.compile(r"_(" + "|".join(_OP_LABEL) + r")$")

# Group overrides for handover stems missing from the figure catalog
_STEM_GROUP_OVERRIDE = {
    "fig8_ho_rate_bar": "strategy", "fig9_ihi_cdf": "strategy",
    "fig10_pingpong_bar": "strategy", "fig28_avg_rate_bar": "strategy",
    "fig19_rate_ccdf": "strategy",
}
# Clean labels for per-constellation figures whose base alias inherits a bare name
_STEM_LABEL = {
    "fig8_ho_rate_bar": "Handover rate (per user-min)",
    "fig9_ihi_cdf": "Inter-handover interval CDF",
}


def _figure_index(inter: dict) -> list:
    """Index of renderable figures tagged coverage|strategy for the Analytics panel."""
    try:
        from ntnconnect.analysis.figure_catalog import FIGURE_CATALOG
    except Exception:
        FIGURE_CATALOG = {}
    seen, out = set(), []
    for stem, rec in inter.items():
        if stem.startswith("_") or not isinstance(rec, dict):
            continue
        base = _CNST_SUFFIX.sub("", stem)                 # drop _oneweb/_starlink suffix
        if base in seen:
            continue
        grp = _STEM_GROUP_OVERRIDE.get(base)
        if grp is None:
            meta = FIGURE_CATALOG.get(base) or FIGURE_CATALOG.get(stem)
            if meta:
                grp = "coverage" if meta.get("stage") == "geometry" else "strategy"
        if grp is None:
            grp = "strategy"                              # sim is the safe default
        seen.add(base)
        label = _STEM_LABEL.get(base) or rec.get("label", base)
        label = _CNST_SUFFIX.sub("", str(label)).replace(" ()", "").strip()
        out.append({"stem": base, "label": label,
                    "group": grp, "kind": rec.get("kind")})
    out.sort(key=lambda d: (d["group"] != "coverage", d["label"]))
    return out


def _write_manifest(runs: dict, out_dir: Path, step_s: float) -> Path:
    """Synthesize the manifest.csv run_analysis expects from discovered runs."""
    rows = [dict(constellation=c, strategy=s, status="ok",
                 run_dir=str(rd), step_s=step_s)
            for (c, s), rd in sorted(runs.items())]
    man = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    return man


def _sidecar_to_record(d: dict, stem: str) -> dict | None:
    """Convert one analysis sidecar to the globe schema."""
    kind = d.get("kind")
    if kind == "curves":
        series = []
        for c in d.get("curves", []):
            xs, ys = _downsample_xy(c.get("x", []), c.get("y", []))
            series.append(dict(x=xs, y=ys, name=c.get("label", ""),
                               color=c.get("color")))
        return dict(label=d.get("title", stem), xlabel=d.get("xlabel", ""),
                    ylabel=d.get("ylabel", ""), kind="ccdf", series=series)
    if kind == "bars":
        bars = [dict(label=b.get("label", ""), value=b.get("value"),
                     lo=b.get("ci_lo", b.get("value")),
                     hi=b.get("ci_hi", b.get("value")), color=b.get("color"))
                for b in d.get("bars", [])]
        return dict(label=d.get("title", stem), xlabel=d.get("xlabel", ""),
                    ylabel=d.get("ylabel", ""), kind="bar", bars=bars)
    return None                       # scatter / overlay / hexbin not shown here


# Reports dashboard KPI metadata: key -> (label, unit, better direction, quantile)
_MATRIX_STRAT_KPIS = [
    ("sinr_p50",   "SINR (p50)",       "dB",    "hi", "fig13_sinr_ccdf",           0.50),
    ("ho_rate",    "Handover rate",    "/min",  "lo", "fig8_ho_rate_bar",          None),
    ("ihi_p50",    "Median HO gap",    "min",   "hi", "fig9_ihi_cdf",              0.50),
    ("access_p90", "Access delay (p90)","ms",   "lo", "fig25_access_delay_ccdf",   0.90),
    ("outage_p90", "Outage gap (p90)", "s",     "lo", "fig17_visibility_outage_ccdf", 0.90),
]
_MATRIX_CTX_KPIS = [
    ("availability", "Availability",     "%",    "hi", "fig19_availability_bar", None),
    ("visible_p50",  "Visible sats (p50)","sats","hi", "fig1_visibility_ccdf",   0.50),
    ("contact_p50",  "Contact time (p50)","s",   "hi", "fig5_contact_time_cdf",  0.50),
    ("ion_p50",      "I/N (p50)",        "dB",   "lo", "fig16_ion_ccdf",         0.50),
]
_OPS = {"oneweb": "OneWeb", "starlink": "Starlink"}


def _actual_step_s(runs: dict, fallback: float) -> float:
    """True seconds-per-tick from a run's resource_profile (step_size_s)."""
    for (_c, _s), rd in (runs or {}).items():
        rp = Path(rd) / "resource_profile.json"
        if rp.is_file():
            try:
                v = float(json.loads(rp.read_text()).get("step_size_s") or 0.0)
                if v > 0:
                    return v
            except Exception:
                pass
    return float(fallback)


def _norm_key(s) -> str:
    """Join key across differing strategy spellings: alphanumerics only, upper-cased."""
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def _curve_quantile(series: dict, q: float):
    """Value x_q with P(X<=x_q)=q, from a curve stored as either CDF or CCDF."""
    x = np.asarray(series.get("x", []), float)
    y = np.asarray(series.get("y", []), float)
    n = min(x.size, y.size)
    if n < 2:
        return None
    x, y = x[:n], y[:n]
    p = y if y[0] <= y[-1] else 1.0 - y            # -> P(X<=x)
    order = np.argsort(p)
    return float(np.interp(q, p[order], x[order]))


def _kpi_matrix(inter: dict) -> dict:
    """Reduce curve/bar records to a per-(constellation, strategy) scalar matrix."""
    # Strategy display names, joined by normalised key
    disp: dict[str, str] = {}
    for cnst in _OPS:
        r = inter.get(f"fig8_ho_rate_bar_{cnst}") or inter.get("fig8_ho_rate_bar")
        for b in (r or {}).get("bars", []):
            disp.setdefault(_norm_key(b["label"]), b["label"])
    for s in (inter.get("fig13_sinr_ccdf") or {}).get("series", []):
        _, _, st = s["name"].partition("-")
        disp.setdefault(_norm_key(st), st)

    def _strat_scalar(cnst_label, key, stem, q):
        """One scalar per (cnst_label, strategy-key) from KPI `stem`."""
        out = {}
        rec = inter.get(f"{stem}_{cnst_label.lower()}") or inter.get(stem)
        if not rec:
            return out
        per_cnst = bool(inter.get(f"{stem}_{cnst_label.lower()}"))
        if rec.get("kind") == "bar":                       # bar per strategy
            for b in rec.get("bars", []):
                out[_norm_key(b["label"])] = b.get("value")
        else:                                              # curves: series per strategy
            known = set(_OPS.values())
            for s in rec.get("series", []):
                nm = s["name"]
                if per_cnst:                               # name is the strategy
                    st = nm
                else:                                      # name is "Cnst-Strat"
                    prefix = nm.split("-", 1)[0]
                    if prefix in known:
                        if not nm.startswith(cnst_label + "-"):
                            continue                       # different constellation
                        st = nm[len(cnst_label) + 1:]
                    else:
                        st = nm
                v = _curve_quantile(s, q if q is not None else 0.5)
                if v is not None:
                    out[_norm_key(st)] = v
        return out

    def _ctx_scalar(cnst_label, stem, q):
        rec = inter.get(stem)
        if not rec:
            return None
        if rec.get("kind") == "bar":
            for b in rec.get("bars", []):
                if _norm_key(b["label"]) == _norm_key(cnst_label):
                    return b.get("value")
            return None
        for s in rec.get("series", []):
            if _norm_key(s["name"]) == _norm_key(cnst_label):
                return _curve_quantile(s, q if q is not None else 0.5)
        return None

    data = {}
    for cnst_label in _OPS.values():
        per_strat = {k: {} for k in disp}
        for kkey, _lab, _u, _b, stem, q in _MATRIX_STRAT_KPIS:
            vals = _strat_scalar(cnst_label, kkey, stem, q)
            for nk, v in vals.items():
                per_strat.setdefault(nk, {})[kkey] = v
        strat_out = {disp.get(nk, nk): sc for nk, sc in per_strat.items() if sc}
        if not strat_out:
            continue
        ctx = {}
        for kkey, _lab, _u, _b, stem, q in _MATRIX_CTX_KPIS:
            v = _ctx_scalar(cnst_label, stem, q)
            if v is not None:
                ctx[kkey] = v
        data[cnst_label] = {"context": ctx, "strategies": strat_out}

    return {
        "constellations": [c for c in _OPS.values() if c in data],
        "strat_kpis": [dict(key=k, label=l, unit=u, better=b)
                       for k, l, u, b, _s, _q in _MATRIX_STRAT_KPIS],
        "ctx_kpis":   [dict(key=k, label=l, unit=u, better=b)
                       for k, l, u, b, _s, _q in _MATRIX_CTX_KPIS],
        "data": data,
    }


# Extra leaderboard columns from sinr_snapshot (key, label, unit, better-direction)
_EXTRA_LEADER_KPIS = [
    ("sum_rate", "Sum rate",      "Gbps",   "hi"),
    ("se",       "Spectral eff.", "b/s/Hz", "hi"),
    ("power_w",  "Power",         "W",      "lo"),
    ("p5",       "Robustness",    "dB",     "hi"),
    ("dist_km",  "Serving dist.", "km",     "lo"),
    ("hops",     "Hops to GW",    "",       "lo"),
]


def _serving_dist_km(fp: Path, max_ticks) -> float | None:
    """Mean serving-satellite slant range (km)."""
    lvps = list(fp.parent.glob("link_visibility*.parquet"))
    if not lvps:
        return None
    try:
        key = ["t_idx", "user_id", "sat_id"]
        lv = pd.concat([pd.read_parquet(x, columns=key + ["dist_m"]) for x in lvps],
                       ignore_index=True)
        sk = pd.read_parquet(fp, columns=key)
        if max_ticks:
            lv = lv[lv["t_idx"] <= max_ticks]; sk = sk[sk["t_idx"] <= max_ticks]
        d = sk.merge(lv, on=key, how="left")["dist_m"].astype(float)
        d = d[np.isfinite(d)]
        return round(float(d.mean()) / 1000.0, 1) if len(d) else None
    except Exception:
        return None


def _iter_snapshot_shards(rd: Path, columns=None, max_ticks=None):
    """Stream a run's sinr_snapshot one shard at a time."""
    from ntnconnect.common.utils.run_io import iter_run_table
    return iter_run_table(rd, "sinr_snapshot", columns, max_ticks)


def _merge_extra_metrics(matrix: dict, runs: dict, max_ticks,
                         eph_dir: Path | None = None, step_s: float = 1.0) -> None:
    """Add power / sum-rate / SE / robustness columns to the leaderboard matrix."""
    from ntnconnect.analysis.core import short_label, _strat_abbrev
    want = ["t_idx", "sinr_db", "power_consumption_w",
            "throughput_mbps", "spectral_efficiency_bps_hz"]
    present = set()
    for (cnst, strat), rd in (runs or {}).items():
        cell = (matrix.get("data", {}).get(short_label(cnst), {})
                      .get("strategies", {}).get(_strat_abbrev(strat)))
        if cell is None:
            continue
        tp_per_tick: dict = {}          # t_idx -> summed throughput
        se_sum = se_n = pw_sum = pw_n = 0.0
        sinr_parts = []
        for df in _iter_snapshot_shards(rd, want, max_ticks):
            if "throughput_mbps" in df.columns and "t_idx" in df.columns:
                for ti, v in df.groupby("t_idx")["throughput_mbps"].sum().items():
                    tp_per_tick[int(ti)] = tp_per_tick.get(int(ti), 0.0) + float(v)
            if "spectral_efficiency_bps_hz" in df.columns:
                se_sum += float(df["spectral_efficiency_bps_hz"].sum()); se_n += len(df)
            if "power_consumption_w" in df.columns:
                pw_sum += float(df["power_consumption_w"].sum()); pw_n += len(df)
            if "sinr_db" in df.columns:
                sinr_parts.append(df["sinr_db"].to_numpy(dtype=float))
        if tp_per_tick:
            cell["sum_rate"] = round(float(np.mean(list(tp_per_tick.values()))) / 1000.0, 2)
            present.add("sum_rate")     # mean per-tick throughput [Gbps]
        if se_n:
            cell["se"] = round(se_sum / se_n, 2); present.add("se")
        if pw_n:
            cell["power_w"] = round(pw_sum / pw_n, 2); present.add("power_w")
        if sinr_parts:
            cell["p5"] = round(float(np.percentile(np.concatenate(sinr_parts), 5)), 1)
            present.add("p5")
        # Serving-sat distance + hops to nearest gateway
        try:
            from ntnconnect.analysis.core import _compute_e2e_rtt_for_cell, set_eph_dir
            keyc = ["t_idx", "user_id", "sat_id"]
            lvps = sorted(Path(rd).glob("link_visibility*.parquet"))
            lvi = (pd.concat([pd.read_parquet(x, columns=keyc + ["dist_m"]) for x in lvps],
                             ignore_index=True) if lvps else None)
            if lvi is not None and max_ticks:
                lvi = lvi[lvi["t_idx"] <= max_ticks]
            if lvi is not None and not lvi.empty:
                dsum = dn = 0.0; hops_parts = []
                if eph_dir is not None:
                    set_eph_dir(Path(eph_dir))
                for sk in _iter_snapshot_shards(rd, keyc, max_ticks):
                    d = sk.merge(lvi, on=keyc, how="left")["dist_m"].astype(float)
                    d = d[np.isfinite(d)]
                    dsum += float(d.sum()); dn += len(d)
                    try:
                        _rtt, _hops = _compute_e2e_rtt_for_cell(rd, sk, lvi, cnst,
                                                                float(step_s or 1.0))
                        _h = _hops[np.isfinite(_hops)]
                        if len(_h):
                            hops_parts.append(_h)
                    except Exception:
                        pass
                if dn:
                    cell["dist_km"] = round((dsum / dn) / 1000.0, 1); present.add("dist_km")
                if hops_parts:
                    cell["hops"] = round(float(np.concatenate(hops_parts).mean()), 2)
                    present.add("hops")
        except Exception:
            pass
    for k, lab, unit, better in _EXTRA_LEADER_KPIS:
        if k in present:
            matrix["strat_kpis"].append(dict(key=k, label=lab, unit=unit, better=better))


def _trend_series(runs: dict, step_s: float, max_ticks, nbins: int = 16) -> dict:
    """Per-(constellation, strategy) time trend of median SINR, binned into nbins."""
    from ntnconnect.analysis.core import short_label, _strat_abbrev
    tmp: dict[str, list] = {}
    hi: dict[str, float] = {}
    for (cnst, strat), rd in runs.items():
        parts = list(_iter_snapshot_shards(rd, ["t_idx", "sinr_db"], max_ticks))
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            continue
        t = df["t_idx"].to_numpy(float)
        cl = short_label(cnst)
        tmp.setdefault(cl, []).append((_strat_abbrev(strat), t, df["sinr_db"].to_numpy(float)))
        hi[cl] = max(hi.get(cl, 0.0), float(t.max()) if t.size else 0.0)
    out = {}
    for cl, lst in tmp.items():
        top = hi.get(cl, 0.0) or 1.0
        edges = np.linspace(0.0, top, nbins + 1)
        centers = [round(float(c) * step_s / 60.0, 2)          # bin centre [min]
                   for c in (edges[:-1] + edges[1:]) / 2.0]
        series = {}
        for disp, t, s in lst:
            idx = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
            med = [None] * nbins
            for b in range(nbins):
                m = s[idx == b]
                if m.size:
                    med[b] = round(float(np.median(m)), 3)
            series[disp] = med
        out[cl] = {"x": centers, "series": series}
    return out


def _radio_params(runs: dict) -> dict:
    """Representative PHY parameters per constellation, keyed by display name."""
    out, seen = {}, set()
    for (cnst, _strat), rd in (runs or {}).items():
        if cnst in seen:
            continue
        from ntnconnect.common.utils.run_io import run_shards
        _sh = run_shards(rd, "sinr_snapshot")
        if not _sh:
            continue
        fp = _sh[0]                                   # any shard carries the metadata
        try:
            df = pd.read_parquet(fp, columns=["antenna_gain_db", "tx_power_dbw", "noise_w"])
        except Exception:
            continue
        nw = float(df["noise_w"].median())
        out[_OPS.get(cnst.lower(), cnst)] = {
            "antenna_gain_db": round(float(df["antenna_gain_db"].median()), 1),
            "tx_power_dbw":    round(float(df["tx_power_dbw"].median()), 1),
            "noise_dbw":       round(float(10.0 * np.log10(max(nw, 1e-30))), 1),
        }
        seen.add(cnst)
    return out


def _scenario_summary(eph_dir: Path, matrix: dict, runs: dict | None = None) -> dict:
    """Simulation config + PHY parameters for the Overview 'run information' panel."""
    try:
        m = json.loads((eph_dir / "manifest.json").read_text())
    except Exception:
        m = {}
    radio = _radio_params(runs or {})
    cnsts = []
    for key, c in (m.get("constellations") or {}).items():
        disp = _OPS.get(key.lower(), key)
        entry = {"name": disp, "n_sats": c.get("n_sats_in_tle"),
                 "n_gateways": c.get("n_gateways")}
        entry.update(radio.get(disp, {}))
        cnsts.append(entry)
    strategies = sorted({s for cd in matrix.get("data", {}).values()
                         for s in cd.get("strategies", {})})
    return {
        "start_utc": m.get("start_utc"),
        "duration_h": m.get("duration_h"),
        "fine_step_s": m.get("fine_step_s"),
        "coarse_step_s": m.get("coarse_step_s"),
        "elev_min_user_deg": m.get("elev_min_user_deg"),
        "n_users": m.get("n_users"),
        "n_batches": m.get("n_batches"),
        "user_grid": m.get("user_grid"),
        "constellations": cnsts,
        "strategies": strategies,
    }


def _alerts_from(inter: dict) -> dict:
    """Alerts summary: cells below 99% availability."""
    rec = inter.get("fig19_availability_bar")
    below = []
    if rec and rec.get("kind") == "bar":
        for b in rec.get("bars", []):
            v = b.get("value")
            if v is not None and v < 99.0:
                below.append(f"{b.get('label', '?')} {v:.1f}%")
    return {"count": len(below),
            "message": (f"{len(below)} cell(s) below 99% availability: "
                        + ", ".join(below)) if below
                       else "all cells at/above 99% availability"}


# Fast-preview batch cap; None for the full run
DEFAULT_MAX_BATCHES = 2


def build_interactive(scan_roots, eph_dir, out_dir, step_s: float = 0.1,
                      include_rtt: bool = False, figures=None,
                      max_batches: int | None = DEFAULT_MAX_BATCHES) -> Path:
    """Discover runs, run the analysis, and write <out_dir>/interactive.json."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover_runs([Path(r) for r in scan_roots])
    if not runs:
        raise SystemExit("build_interactive: no runs with SINR data found under "
                         f"{[str(r) for r in scan_roots]}")
    # Use the true (sub-slot) tick duration so time-based metrics scale correctly
    real_step = _actual_step_s(runs, step_s)
    if abs(real_step - float(step_s)) > 1e-9:
        print(f"build_interactive: using actual tick step {real_step}s "
              f"(manifest/param was {step_s}s)")
    step_s = real_step
    figs = (list(figures) if figures is not None
            else list(_PANEL_FIGURES) + ([_RTT_FIGURE] if include_rtt else []))
    # Batch cap -> tick cap
    max_ticks = None
    if max_batches is not None:
        try:
            bm = float(json.loads((Path(eph_dir) / "manifest.json").read_text())
                       .get("batch_min", 15.0))
            max_ticks = int(round(max_batches * bm * 60.0 / float(step_s)))
        except Exception:
            max_ticks = None
    cap = f", first {max_batches} batch(es) (<{max_ticks} ticks)" if max_ticks else ""
    print(f"build_interactive: {len(runs)} run(s); running analysis -> {out_dir}{cap}")
    _write_manifest(runs, out_dir, step_s)
    # Render the panel figures; writes the *_data.json sidecars we convert
    run_analysis(out_dir, runs_dir=out_dir, figures=figs,
                 eph_dir=Path(eph_dir), write_tables=False, show_titles=True,
                 dpi=90, max_ticks=max_ticks, build_globe_json=False)  # avoid recursion

    inter: dict = {}
    for jp in sorted(out_dir.glob("*_data.json")):
        stem = jp.stem[:-5] if jp.stem.endswith("_data") else jp.stem
        try:
            rec = _sidecar_to_record(json.loads(jp.read_text()), stem)
        except Exception as e:
            print(f"  ! skip {jp.name}: {e}"); continue
        if rec:
            inter[stem] = rec
    # Base-name aliases so a single rail key resolves regardless of constellation
    for stem in list(inter):
        base = _CNST_SUFFIX.sub("", stem)
        if base != stem and base not in inter:
            inter[base] = inter[stem]

    inter["_alerts"] = _alerts_from(inter)
    inter["_matrix"] = _kpi_matrix(inter)          # Reports dashboard scalar table
    _merge_extra_metrics(inter["_matrix"], runs, max_ticks,   # power/rate/SE/robustness
                         eph_dir=Path(eph_dir), step_s=step_s)  # + distance + hops-to-GW
    inter["_matrix"]["trend"] = _trend_series(runs, float(step_s), max_ticks)
    inter["_figures"] = _figure_index(inter)       # Analytics coverage/strategy picker
    inter["_scenario"] = _scenario_summary(Path(eph_dir), inter["_matrix"], runs)
    outp = out_dir / "interactive.json"
    outp.write_text(json.dumps(inter))
    print(f"build_interactive: wrote {outp} "
          f"({len(inter)-1} KPIs, alerts={inter['_alerts']['count']})")
    return outp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ntnconnect.analysis.visualization.build_interactive",
        description="Build analysis/interactive.json for the 3D globe rail panels.")
    ap.add_argument("--scan-root", action="append", required=True, type=Path,
                    help="root(s) to search for <cnst>__<strat>/run_* (repeatable)")
    ap.add_argument("--eph-dir", required=True, type=Path,
                    help="dataset folder (manifest.json + per-constellation ephemeris)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir (write interactive.json here, e.g. outputs/analysis)")
    ap.add_argument("--step-s", type=float, default=0.1)
    ap.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES,
                    help="cap analysis to the first N dataset batches (fast preview; "
                         "0 or negative = full run)")
    ap.add_argument("--include-rtt", action="store_true",
                    help="also build the end-to-end RTT panel (much slower)")
    a = ap.parse_args(argv)
    build_interactive(a.scan_root, a.eph_dir, a.out, step_s=a.step_s,
                      include_rtt=a.include_rtt,
                      max_batches=(a.max_batches if a.max_batches and a.max_batches > 0 else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
