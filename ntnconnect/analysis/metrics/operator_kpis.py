# Per-operator KPIs (one row per operator per tick).

from __future__ import annotations
import pandas as pd
from ntnconnect.analysis.metrics.base import register_kpi


@register_kpi("operator_mean_sinr_db", granularity="per_operator", units="dB",
              default_plot="line", description="Mean SINR across operator's links.")
def operator_mean_sinr(state, links, history):
    if links is None or links.empty or "sinr_db" not in links.columns: return None
    g = links.groupby("operator_id", observed=False)["sinr_db"].mean().reset_index(
        name="operator_mean_sinr_db")
    g["timestamp"] = state.timestamp; g["t_idx"] = state.t_idx
    return g


@register_kpi("operator_sum_rate_bps", granularity="per_operator", units="bps",
              default_plot="bar", description="Sum of Shannon rates across operator's links.")
def operator_sum_rate(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    g = links.groupby("operator_id", observed=False)["link_rate_bps"].sum().reset_index(
        name="operator_sum_rate_bps")
    g["timestamp"] = state.timestamp; g["t_idx"] = state.t_idx
    return g


@register_kpi("operator_n_links", granularity="per_operator", units="count",
              default_plot="line", description="Number of access links per operator.")
def operator_nlinks(state, links, history):
    if links is None or links.empty: return None
    g = links.groupby("operator_id", observed=False).size().reset_index(name="operator_n_links")
    g["timestamp"] = state.timestamp; g["t_idx"] = state.t_idx
    return g


@register_kpi("operator_n_users_served", granularity="per_operator", units="count",
              default_plot="bar", description="Distinct users served by each operator.")
def operator_users(state, links, history):
    assigns = state.data.get("assignments", {})
    if not assigns or links is None or links.empty: return None
    # node_id -> operator_id lookup
    nid_to_op = (links[["node_id","operator_id"]].drop_duplicates()
                       .set_index("node_id")["operator_id"].to_dict())
    rows = {}
    for u, nid in assigns.items():
        op = int(nid_to_op.get(int(nid), -1))
        if op < 0: continue
        rows.setdefault(op, set()).add(int(u))
    out = pd.DataFrame([
        {"timestamp": state.timestamp, "t_idx": state.t_idx,
         "operator_id": op, "operator_n_users_served": len(users)}
        for op, users in rows.items()
    ])
    return out


@register_kpi("operator_throughput_bps", granularity="per_operator", units="bps",
              default_plot="bar",
              description="Per-operator delivered throughput "
                          "(sum of SERVING-link rates only).")
def operator_throughput(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    assigns = state.data.get("assignments", {}) or {}
    # Filter to serving links only
    serving_pairs = set((int(u), int(s)) for u, s in assigns.items() if int(s) >= 0)
    if not serving_pairs: return None
    mask = links.apply(lambda r: (int(r["user_id"]), int(r["node_id"]))
                       in serving_pairs, axis=1)
    sub = links[mask]
    if sub.empty: return None
    g = sub.groupby("operator_id")["link_rate_bps"].sum().reset_index(
        name="operator_throughput_bps")
    g["timestamp"] = state.timestamp; g["t_idx"] = state.t_idx
    return g


@register_kpi("operator_load_factor", granularity="per_operator", units="ratio",
              default_plot="line",
              description="Served users / max users-per-sat across operator's sats.")
def operator_load(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    if not assigns or links is None or links.empty: return None
    nid_to_op = (links[["node_id","operator_id"]].drop_duplicates()
                       .set_index("node_id")["operator_id"].to_dict())
    cap_per_sat = float(state.data.get("max_users_per_sat", 50))
    per_op_load = {}
    for u, s in assigns.items():
        if int(s) < 0: continue
        op = int(nid_to_op.get(int(s), -1))
        if op < 0: continue
        per_op_load.setdefault(op, {}).setdefault(int(s), 0)
        per_op_load[op][int(s)] += 1
    rows = []
    for op, sats in per_op_load.items():
        loads = list(sats.values())
        load_factor = (sum(loads) / (len(loads) * cap_per_sat)) if loads else 0.0
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "operator_id": int(op),
                     "operator_load_factor": float(load_factor)})
    return pd.DataFrame(rows)


@register_kpi("operator_blocking_probability", granularity="per_operator",
              units="ratio", default_plot="bar",
              description="Fraction of users this operator could not admit (assignment=-2).")
def operator_blocking(state, links, history):
    assigns = state.data.get("assignments", {}) or {}
    if not assigns or links is None or links.empty: return None
    # In our admission stage, -2 = rejected, -1 = no visible sat.
    per_op_total = {}
    per_op_blocked = {}
    for u, s in assigns.items():
        # We can't directly recover the operator from a blocked user, so we
        # approximate by attributing blocked users to operators with sats
        # visible to them (omitted here; fallback to system-level metric).
        if int(s) == -2:
            per_op_blocked["all"] = per_op_blocked.get("all", 0) + 1
        per_op_total["all"] = per_op_total.get("all", 0) + 1
    rows = []
    for op in self_operator_ids_from_links(links):
        tot = per_op_total.get("all", 0)
        blk = per_op_blocked.get("all", 0)
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "operator_id": int(op),
                     "operator_blocking_probability": (blk / tot) if tot else 0.0})
    return pd.DataFrame(rows)


def self_operator_ids_from_links(links):
    return sorted(set(int(o) for o in links["operator_id"].unique()))


def _per_op_serving_rates(state, links):
    """Return {operator_id: [serving rates of UEs served by this operator]}"""
    import numpy as np
    out: dict = {}
    if links is None or links.empty or "link_rate_bps" not in links.columns:
        return out
    assigns = state.data.get("assignments", {}) or {}
    nid_to_op = (links[["node_id","operator_id"]].drop_duplicates()
                       .set_index("node_id")["operator_id"].to_dict())
    for u, s in assigns.items():
        if int(s) < 0: continue
        op = int(nid_to_op.get(int(s), -1))
        if op < 0: continue
        m = (links["user_id"] == int(u)) & (links["node_id"] == int(s))
        if not m.any(): continue
        r = float(links.loc[m, "link_rate_bps"].iloc[0])
        out.setdefault(op, []).append(r)
    return out


@register_kpi("operator_min_rate_bps", granularity="per_operator", units="bps",
              default_plot="bar",
              description="Min UE rate within each operator this tick.")
def op_min_rate(state, links, history):
    per_op = _per_op_serving_rates(state, links)
    if not per_op: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "operator_id": int(op),
        "operator_min_rate_bps": float(min(rs)),
    } for op, rs in per_op.items() if rs])


@register_kpi("operator_max_rate_bps", granularity="per_operator", units="bps",
              default_plot="bar",
              description="Max UE rate within each operator this tick.")
def op_max_rate(state, links, history):
    per_op = _per_op_serving_rates(state, links)
    if not per_op: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "operator_id": int(op),
        "operator_max_rate_bps": float(max(rs)),
    } for op, rs in per_op.items() if rs])


@register_kpi("operator_mean_rate_bps", granularity="per_operator", units="bps",
              default_plot="bar",
              description="Mean UE rate within each operator this tick.")
def op_mean_rate(state, links, history):
    per_op = _per_op_serving_rates(state, links)
    if not per_op: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "operator_id": int(op),
        "operator_mean_rate_bps": float(sum(rs) / len(rs)),
    } for op, rs in per_op.items() if rs])


@register_kpi("operator_p5_rate_bps", granularity="per_operator", units="bps",
              default_plot="bar",
              description="5th percentile UE rate within each operator (cell-edge).")
def op_p5_rate(state, links, history):
    import numpy as np
    per_op = _per_op_serving_rates(state, links)
    if not per_op: return None
    return pd.DataFrame([{
        "timestamp": state.timestamp, "t_idx": state.t_idx,
        "operator_id": int(op),
        "operator_p5_rate_bps": float(np.percentile(rs, 5)),
    } for op, rs in per_op.items() if rs])


@register_kpi("operator_fairness_jain", granularity="per_operator", units="ratio",
              default_plot="bar",
              description="Jain's fairness index over per-user throughput within operator.")
def operator_fairness(state, links, history):
    if links is None or links.empty or "link_rate_bps" not in links.columns: return None
    assigns = state.data.get("assignments", {}) or {}
    nid_to_op = (links[["node_id","operator_id"]].drop_duplicates()
                       .set_index("node_id")["operator_id"].to_dict())
    # User throughput attributed to operator of its serving sat
    per_op_rates = {}
    for u, s in assigns.items():
        if int(s) < 0: continue
        op = int(nid_to_op.get(int(s), -1))
        if op < 0: continue
        m = (links["user_id"] == int(u)) & (links["node_id"] == int(s))
        if not m.any(): continue
        r = float(links.loc[m, "link_rate_bps"].iloc[0])
        per_op_rates.setdefault(op, []).append(r)
    rows = []
    for op, rates in per_op_rates.items():
        if not rates: continue
        s1 = sum(rates); s2 = sum(r*r for r in rates)
        j = (s1*s1) / (len(rates)*s2) if s2 > 0 else 0.0
        rows.append({"timestamp": state.timestamp, "t_idx": state.t_idx,
                     "operator_id": int(op),
                     "operator_fairness_jain": float(j)})
    return pd.DataFrame(rows)
