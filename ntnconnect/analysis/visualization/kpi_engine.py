"""KPI discovery + chart-recommendation engine - the UI-agnostic core.

This module knows nothing about Streamlit/Plotly. It inspects an arbitrary
DataFrame (any Phase-2 output, present or future) and answers:

    * which columns are KPIs (vs. identifiers / keys / time)?
    * what category does each KPI belong to (auto, by name)?
    * what chart types best fit each KPI?
    * summary statistics, CCDF, trend slope, and anomaly flags.

Adding a new KPI column anywhere in the pipeline needs NO change here: it is
discovered, categorised and charted automatically by type + name heuristics.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

# Colour-blind-safe palette (Okabe - Ito) + sequential map name
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
SEQUENTIAL = "Viridis"      # colour-blind-friendly sequential (Plotly name)
DIVERGING  = "RdBu"

# Column-role detection
_ID_RE   = re.compile(r"(^|_)(id|uid|seed|run|operator)$|_id$|^node_id$|^sat_id$"
                      r"|^user_id$|^gw_id$|^batch$|^series$", re.I)
_TIME_RE = re.compile(r"^(t_idx|t_s|time|tick|timestamp|t_enter_idx|t_leave_idx)$", re.I)

# Ordered category rules - first match wins.
_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("SINR & Signal", re.compile(r"sinr|snr|cinr|rsrp|signal|i_over_n|i_ext|noise", re.I)),
    ("Throughput",    re.compile(r"rate|throughput|bps|spectral|capacity|\bse\b", re.I)),
    ("Handover",      re.compile(r"handover|\bho_|service_gap|trigger|from_sat|to_sat|churn|success", re.I)),
    ("Visibility",    re.compile(r"visib|n_visible|contact|coverage|remaining|outage", re.I)),
    ("Delay & RTT",   re.compile(r"delay|rtt|latency|hop|gap", re.I)),
    ("Power",         re.compile(r"power|eirp|watt|_w$|energy|consum", re.I)),
    ("Geometry",      re.compile(r"elev|range|dist|azimuth|doppler|alt|lat|lon|rate_deg|range_rate|^[xyz]$", re.I)),
]

# Unit hints by name fragment (best-effort labels for axes).
_UNIT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"_db$|_db_|sinr|snr|elev.*db", re.I), "dB"),
    (re.compile(r"elev|azimuth|deg", re.I), "deg"),
    (re.compile(r"bps$|rate.*bps", re.I), "bps"),
    (re.compile(r"_s$|duration|gap|delay|rtt|latency", re.I), "s"),
    (re.compile(r"_w$|power|noise_w|i_ext_w", re.I), "W"),
    (re.compile(r"dist_m|_m$", re.I), "m"),
    (re.compile(r"km_s|range_rate", re.I), "km/s"),
    (re.compile(r"hz$", re.I), "Hz"),
]


@dataclass
class KpiInfo:
    name: str
    kind: str                      # "numeric" | "categorical" | "temporal" | "id"
    category: str = "Other"
    unit: str = ""
    n_unique: int = 0
    n_missing: int = 0
    is_boolean: bool = False


def _guess_unit(name: str) -> str:
    for rx, u in _UNIT_RULES:
        if rx.search(name):
            return u
    return ""


def _categorize(name: str) -> str:
    for cat, rx in _CATEGORY_RULES:
        if rx.search(name):
            return cat
    return "Other"


def infer_kind(series: pd.Series, name: str) -> str:
    """Classify a column as time / id / numeric / categorical."""
    if _TIME_RE.search(name):
        return "temporal"
    if _ID_RE.search(name):
        return "id"
    if pd.api.types.is_bool_dtype(series):
        return "numeric"            # booleans are charted as rates
    if pd.api.types.is_numeric_dtype(series):
        nuniq = series.nunique(dropna=True)
        # Small-cardinality integer codes that look categorical
        if nuniq <= 12 and pd.api.types.is_integer_dtype(series) \
                and not _TIME_RE.search(name):
            # keep as numeric unless clearly a label (handled by name)
            return "numeric"
        return "numeric"
    return "categorical"


def discover(df: pd.DataFrame) -> dict[str, KpiInfo]:
    """Return {column -> KpiInfo} for every column, auto-classified."""
    out: dict[str, KpiInfo] = {}
    for col in df.columns:
        s = df[col]
        kind = infer_kind(s, col)
        out[col] = KpiInfo(
            name=col, kind=kind,
            category=_categorize(col),
            unit=_guess_unit(col),
            n_unique=int(s.nunique(dropna=True)),
            n_missing=int(s.isna().sum()),
            is_boolean=bool(pd.api.types.is_bool_dtype(s)),
        )
    return out


def kpi_columns(info: dict[str, KpiInfo]) -> list[str]:
    """Columns that are actual KPIs (numeric or categorical, not id/time)."""
    return [k for k, v in info.items() if v.kind in ("numeric", "categorical")]


def grouped_by_category(info: dict[str, KpiInfo]) -> dict[str, list[str]]:
    """{category -> [kpi names]} for KPI columns only, category order preserved."""
    order = [c for c, _ in _CATEGORY_RULES] + ["Other"]
    groups: dict[str, list[str]] = {c: [] for c in order}
    for name in kpi_columns(info):
        groups[info[name].category].append(name)
    return {c: v for c, v in groups.items() if v}


def time_column(info: dict[str, KpiInfo]) -> str | None:
    for pref in ("t_s", "t_idx", "timestamp", "tick"):
        if pref in info and info[pref].kind == "temporal":
            return pref
    for k, v in info.items():
        if v.kind == "temporal":
            return k
    return None


def key_columns(info: dict[str, KpiInfo]) -> list[str]:
    return [k for k, v in info.items() if v.kind == "id"]


# Chart recommendation
# Chart kinds the dashboard's factory understands.
CHART_KINDS = ("line_trend", "histogram", "ccdf", "box_by_group",
               "control_chart", "bar_counts", "scatter_corr", "heatmap_corr")


def recommend(df: pd.DataFrame, info: dict[str, KpiInfo], kpi: str,
              time_col: str | None = None,
              group_col: str | None = None) -> list[str]:
    """Ordered list of recommended chart kinds for one KPI."""
    meta = info[kpi]
    recs: list[str] = []
    if meta.kind == "categorical" or (meta.is_boolean):
        recs.append("bar_counts")
        if meta.is_boolean and time_col:
            recs.append("line_trend")     # rate over time
        return recs
    if meta.kind != "numeric":
        return recs
    # numeric KPI
    if time_col and time_col in df.columns:
        recs.append("line_trend")         # trend (aggregated over keys)
        recs.append("control_chart")      # anomaly detection
    recs.append("histogram")              # distribution
    recs.append("ccdf")                   # tail behaviour (SINR/rate/delay)
    if group_col and group_col in df.columns:
        recs.append("box_by_group")       # comparison across batches/groups
    return recs


def best_chart(df, info, kpi, time_col=None, group_col=None) -> str:
    r = recommend(df, info, kpi, time_col, group_col)
    return r[0] if r else "histogram"


# Statistics / insights / anomaly detection
def summary_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return {}
    return {
        "count": int(s.size),
        "mean":  float(s.mean()),
        "std":   float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min":   float(s.min()),
        "p05":   float(np.percentile(s, 5)),
        "median": float(s.median()),
        "p95":   float(np.percentile(s, 95)),
        "max":   float(s.max()),
    }


def ccdf(series: pd.Series, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Return (x sorted, P[X > x]) for a complementary CDF."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if s.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(s)
    cc = 1.0 - np.arange(1, xs.size + 1) / xs.size
    if xs.size > n:                                  # thin for plotting
        idx = np.linspace(0, xs.size - 1, n).astype(int)
        xs, cc = xs[idx], cc[idx]
    return xs, cc


def control_limits(series: pd.Series, k: float = 3.0) -> dict:
    """Shewhart-style mean ± kσ control limits + outlier mask."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    mu = float(s.mean()); sd = float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0
    ucl, lcl = mu + k * sd, mu - k * sd
    mask = (s > ucl) | (s < lcl)
    return {"mean": mu, "sd": sd, "ucl": ucl, "lcl": lcl,
            "outlier_mask": mask.fillna(False).to_numpy(),
            "n_outliers": int(mask.sum())}


def detect_anomalies(series: pd.Series, z: float = 3.0) -> dict:
    """IQR + z-score anomaly summary."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.size < 4:
        return {"n": 0, "iqr_outliers": 0, "z_outliers": 0}
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    iqr_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    sd = s.std(ddof=1)
    z_out = int((np.abs(s - s.mean()) > z * sd).sum()) if sd > 0 else 0
    return {"n": int(s.size), "iqr_outliers": iqr_out, "z_outliers": z_out}


def trend_slope(df: pd.DataFrame, kpi: str, time_col: str) -> float | None:
    """Least-squares slope of KPI vs time (per unit time). None if N/A."""
    if time_col not in df.columns or kpi not in df.columns:
        return None
    g = (df[[time_col, kpi]].apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna().astype(float))
    if len(g) < 3:
        return None
    x = g[time_col].to_numpy(); y = g[kpi].to_numpy()
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def auto_insight(df: pd.DataFrame, info: dict[str, KpiInfo], kpi: str,
                 time_col: str | None) -> str:
    """One-line natural-language insight for a KPI."""
    stats = summary_stats(df[kpi]) if info[kpi].kind == "numeric" else {}
    u = info[kpi].unit
    if not stats:
        if info[kpi].kind == "categorical":
            top = df[kpi].astype(str).value_counts()
            if not top.empty:
                return f"{kpi}: most common = '{top.index[0]}' ({top.iloc[0]} rows, {len(top)} categories)."
        return f"{kpi}: no numeric summary available."
    msg = (f"{kpi}: median {stats['median']:.2f}{u}, "
           f"5–95% [{stats['p05']:.2f}, {stats['p95']:.2f}]{u}.")
    an = detect_anomalies(df[kpi])
    if an.get("z_outliers", 0):
        msg += f" {an['z_outliers']} outliers (>3σ)."
    if time_col:
        sl = trend_slope(df, kpi, time_col)
        if sl is not None and stats["std"] > 0:
            direction = "rising" if sl > 0 else "falling" if sl < 0 else "flat"
            msg += f" Trend: {direction}."
    return msg


# Batch derivation (15-min slots) from a tick column
def add_batch_column(df: pd.DataFrame, time_col: str,
                     ticks_per_batch: int) -> pd.DataFrame:
    """Add an integer `batch` column derived from the tick index."""
    if time_col not in df.columns or ticks_per_batch <= 0:
        return df
    out = df.copy()
    out["batch"] = (pd.to_numeric(out[time_col], errors="coerce").astype(float)
                    // int(ticks_per_batch)).astype("Int64")
    return out


# PLUGIN ARCHITECTURE - KPI-category registry + visualization registry
#
# New KPI categories and new chart types are added by REGISTERING them here
# (or from any other module via the public decorators), with NO change to the
# dashboard core. The dashboard renders whatever the registries expose.
from dataclasses import dataclass as _dc
from typing import Callable as _Callable


@_dc
class CategoryRule:
    name: str
    pattern: "re.Pattern"
    color: str                 # seed colour for this KPI category
    higher_is_better: bool | None = None   # None = neutral
    priority: int = 50


# Registry is seeded from the legacy _CATEGORY_RULES, then extended.
_CATEGORY_REGISTRY: list[CategoryRule] = []


def register_category(name: str, pattern, color: str,
                      higher_is_better: bool | None = None,
                      priority: int = 50) -> None:
    """Register (or override) a KPI category. First match by priority wins."""
    rx = pattern if hasattr(pattern, "search") else re.compile(pattern, re.I)
    _CATEGORY_REGISTRY[:] = [c for c in _CATEGORY_REGISTRY if c.name != name]
    _CATEGORY_REGISTRY.append(CategoryRule(name, rx, color, higher_is_better, priority))
    _CATEGORY_REGISTRY.sort(key=lambda c: -c.priority)


# Per-category colour seeds (Okabe - Ito based, distinct + colour-blind safe).
_CAT_COLORS = {
    "SINR & Signal": "#0072B2", "Throughput": "#009E73", "Handover": "#D55E00",
    "Visibility": "#56B4E9", "Delay & RTT": "#CC79A7", "Power": "#E69F00",
    "Geometry": "#999999", "Network": "#117733", "Quality": "#882255",
    "Mobility": "#44AA99", "Other": "#666666",
}
# higher-is-better polarity per category (drives best/worst + ranking colour).
_CAT_POLARITY = {
    "SINR & Signal": True, "Throughput": True, "Visibility": True,
    "Quality": True, "Network": True,
    "Handover": False, "Delay & RTT": False,        # fewer / lower is better
    "Power": False, "Geometry": None, "Mobility": None, "Other": None,
}

# Seed the registry from the legacy rules + add new categories.
for _name, _rx in _CATEGORY_RULES:
    register_category(_name, _rx, _CAT_COLORS.get(_name, "#666"),
                      _CAT_POLARITY.get(_name), priority=50)
register_category("Network", r"isl|hop|route|rtt|throughput_e2e|backhaul|topology|degree",
                  _CAT_COLORS["Network"], True, priority=60)
register_category("Quality", r"\bber\b|\bbler\b|\bmos\b|outage|reliab|availab|success|drop",
                  _CAT_COLORS["Quality"], True, priority=60)
register_category("Mobility", r"speed|velocity|range_rate|doppler|approach|mobility",
                  _CAT_COLORS["Mobility"], None, priority=55)


def categorize2(name: str) -> str:
    """Registry-driven category lookup (superset of legacy _categorize)."""
    for c in _CATEGORY_REGISTRY:
        if c.pattern.search(name):
            return c.name
    return "Other"


def category_color(cat: str) -> str:
    for c in _CATEGORY_REGISTRY:
        if c.name == cat:
            return c.color
    return _CAT_COLORS.get(cat, "#666")


def is_higher_better(kpi: str) -> bool | None:
    cat = categorize2(kpi)
    for c in _CATEGORY_REGISTRY:
        if c.name == cat:
            return c.higher_is_better
    return None


# Visualization registry
@_dc
class ChartSpec:
    key: str                                  # builder id (dashboard maps it)
    label: str                                # human label
    applies: "_Callable"                      # fn(df, KpiInfo, ctx)->bool
    priority: int = 50                         # higher = preferred default
    scope: str = "single"                     # "single" | "multi" (≥2 KPIs)


_CHART_REGISTRY: dict[str, ChartSpec] = {}


def register_chart(key: str, label: str, applies, priority: int = 50,
                   scope: str = "single") -> None:
    _CHART_REGISTRY[key] = ChartSpec(key, label, applies, priority, scope)


def _has_time(ctx):    return bool(ctx.get("time_col"))
def _has_group(ctx):   return bool(ctx.get("group_col"))
def _is_num(meta):     return meta.kind == "numeric"
def _is_cat(meta):     return meta.kind == "categorical" or meta.is_boolean

# Default chart catalogue (single-KPI).
register_chart("line_trend",   "Trend (time)",   lambda d, m, c: _is_num(m) and _has_time(c), 90)
register_chart("histogram",    "Histogram",      lambda d, m, c: _is_num(m), 70)
register_chart("ccdf",         "CCDF",           lambda d, m, c: _is_num(m), 65)
register_chart("box_by_group", "Box by batch",   lambda d, m, c: _is_num(m) and _has_group(c), 60)
register_chart("violin",       "Violin by batch",lambda d, m, c: _is_num(m) and _has_group(c), 55)
register_chart("control_chart","SPC / control",  lambda d, m, c: _is_num(m), 58)
register_chart("outliers",     "Outlier scan",   lambda d, m, c: _is_num(m), 40)
register_chart("bar_counts",   "Category counts",lambda d, m, c: _is_cat(m), 90)
# Multi-KPI (cross-KPI) charts - applicability checked at the panel level.
register_chart("heatmap_corr",  "Correlation matrix", lambda d, m, c: True, 50, scope="multi")
register_chart("scatter_matrix","Scatter matrix",     lambda d, m, c: True, 45, scope="multi")
register_chart("radar",         "Radar comparison",   lambda d, m, c: _has_group(c), 48, scope="multi")
register_chart("ranking",       "KPI ranking",        lambda d, m, c: _has_group(c), 52, scope="multi")
register_chart("waterfall",     "Batch waterfall",    lambda d, m, c: _has_group(c), 42, scope="multi")


def recommend_registry(df, info, kpi, ctx) -> list[str]:
    """Registry-driven chart recommendation for one KPI (single-scope), ordered
    by priority. Backward-compatible superset of recommend()."""
    meta = info[kpi]
    out = [(s.priority, s.key) for s in _CHART_REGISTRY.values()
           if s.scope == "single" and _safe_pred(s.applies, df, meta, ctx)]
    return [k for _, k in sorted(out, key=lambda t: -t[0])]


def _safe_pred(fn, df, meta, ctx):
    try:
        return bool(fn(df, meta, ctx))
    except Exception:
        return False


def multi_charts(ctx) -> list[tuple[str, str]]:
    """Return [(key,label)] for cross-KPI charts available in this context."""
    return [(s.key, s.label) for s in sorted(_CHART_REGISTRY.values(),
            key=lambda s: -s.priority) if s.scope == "multi"
            and _safe_pred(s.applies, None, None, ctx)]


# Batch analytics
def _agg(series_groups, stat: str):
    if stat == "median":  return series_groups.median()
    if stat == "mean":    return series_groups.mean()
    if stat == "p95":     return series_groups.quantile(0.95)
    if stat == "p05":     return series_groups.quantile(0.05)
    if stat == "std":     return series_groups.std()
    return series_groups.median()


def rank_groups(df: pd.DataFrame, kpi: str, group_col: str,
                stat: str = "median") -> pd.DataFrame:
    """Rank groups (batches / series) by a KPI statistic, best first."""
    if group_col not in df.columns or kpi not in df.columns or kpi == group_col:
        return pd.DataFrame()
    g = _agg(pd.to_numeric(df[kpi], errors="coerce").astype(float).groupby(df[group_col]), stat)
    g.name = f"{stat}_{kpi}"                     # avoid index/value name clash
    hib = is_higher_better(kpi)
    asc = (hib is False)                      # lower-is-better → ascending
    out = g.sort_values(ascending=asc).reset_index()
    out.columns = [group_col, f"{stat}_{kpi}"]
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def best_worst(df, kpi, group_col, stat="median") -> dict:
    r = rank_groups(df, kpi, group_col, stat)
    if r.empty:
        return {}
    return {"best": r.iloc[0].to_dict(), "worst": r.iloc[-1].to_dict(),
            "higher_is_better": is_higher_better(kpi), "table": r}


def benchmark(df, kpi, group_col) -> pd.DataFrame:
    """Per-group benchmark table: mean/median/p5/p95/std + rank."""
    if group_col not in df.columns or kpi not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[kpi], errors="coerce").astype(float).groupby(df[group_col])
    tab = pd.DataFrame({
        "mean": s.mean(), "median": s.median(),
        "p05": s.quantile(0.05), "p95": s.quantile(0.95), "std": s.std(),
        "n": s.size(),
    }).reset_index()
    hib = is_higher_better(kpi)
    tab = tab.sort_values("median", ascending=(hib is False)).reset_index(drop=True)
    tab.insert(1, "rank", np.arange(1, len(tab) + 1))
    return tab


def batch_evolution(df, kpi, batch_col, stat="median") -> pd.DataFrame:
    """Per-batch statistic ordered by batch index (for trend evolution)."""
    if batch_col not in df.columns or kpi not in df.columns:
        return pd.DataFrame()
    g = _agg(pd.to_numeric(df[kpi], errors="coerce").astype(float).groupby(df[batch_col]), stat)
    out = g.reset_index().sort_values(batch_col)
    out.columns = [batch_col, kpi]
    return out


def significance(a: pd.Series, b: pd.Series) -> dict:
    """Mann - Whitney U test between two groups (scipy optional)."""
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(a) < 8 or len(b) < 8:
        return {"p_value": None, "significant": None, "method": "n/a"}
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return {"p_value": float(p), "significant": bool(p < 0.05),
                "method": "Mann–Whitney U",
                "delta_median": float(b.median() - a.median())}
    except Exception:
        # crude fallback: non-overlap of bootstrap-ish CI on the median
        return {"p_value": None, "significant": None, "method": "scipy missing"}


def _fmt(val: float, unit: str = "") -> str:
    """Format large numbers with SI suffixes for readability."""
    if val == 0 or pd.isna(val):
        return f"0.00{unit}"
    abs_val = abs(val)
    # Only apply SI prefixes to typical large-scale units
    if unit in ("bps", "Hz", "users", "bytes", "W", "s", "") and abs_val >= 1000:
        for suffix in ["", "k", "M", "G", "T"]:
            if abs_val < 1000:
                return f"{val:.2f}{suffix}{unit}"
            abs_val /= 1000
            val /= 1000
        return f"{val:.2f}P{unit}"
    return f"{val:.2f}{unit}"


def executive_summary(df, info, group_col=None, time_col=None,
                      max_kpis: int = 8) -> list[str]:
    """Headline insights across the dataset for an executive panel."""
    out: list[str] = []
    num = [k for k in kpi_columns(info) if info[k].kind == "numeric"]
    # Rank-worthy KPIs first (SINR/throughput/quality/handover/delay).
    pri = sorted(num, key=lambda k: 0 if categorize2(k) in
                 ("SINR & Signal", "Throughput", "Quality", "Handover", "Delay & RTT")
                 else 1)
    for kpi in pri[:max_kpis]:
        stats = summary_stats(df[kpi])
        if not stats:
            continue
        u = info[kpi].unit
        
        m_str = _fmt(stats['median'], u)
        p5_str = _fmt(stats['p05'])
        p95_str = _fmt(stats['p95'])
        
        line = f"**{kpi}** — median {m_str} (p5 {p5_str}, p95 {p95_str})"
        if group_col and group_col in df.columns:
            bw = best_worst(df, kpi, group_col)
            if bw:
                g = group_col
                b_val = _fmt(bw['best'][f'median_{kpi}'])
                w_val = _fmt(bw['worst'][f'median_{kpi}'])
                line += (f"; best {g}={bw['best'][group_col]} ({b_val}), "
                         f"worst {g}={bw['worst'][group_col]} ({w_val})")
        an = detect_anomalies(df[kpi])
        if an.get("z_outliers"):
            line += f"; {an['z_outliers']} anomalies"
        if time_col:
            sl = trend_slope(df, kpi, time_col)
            if sl is not None and stats["std"] > 0:
                line += f"; trend {'↑' if sl > 0 else '↓' if sl < 0 else '→'}"
        out.append(line)
    # Strongest correlation
    if len(num) >= 2:
        corr = df[num].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0.0)
        if corr.values.size:
            i, j = np.unravel_index(np.argmax(corr.values), corr.shape)
            r = df[num].corr(numeric_only=True).iloc[i, j]
            if abs(r) > 0.4:
                out.append(f"Strongest correlation: **{num[i]} ↔ {num[j]}** (r = {r:+.2f}).")
    return out


# Themes (light / dark) - consistent enterprise palette
THEMES = {
    "light": dict(
        paper="#ffffff", plot="#ffffff", font="#1d2433", grid="#e8edf3",
        axis="#aab4c2", title="#0b1f3a", accent="#0072B2",
        colorway=OKABE_ITO, sequential=SEQUENTIAL, diverging=DIVERGING),
    "dark": dict(
        paper="#0e1726", plot="#0e1726", font="#dbe6f5", grid="#22304a",
        axis="#3a4d6e", title="#bcd6ff", accent="#56B4E9",
        colorway=OKABE_ITO, sequential="Cividis", diverging="RdBu"),
}
