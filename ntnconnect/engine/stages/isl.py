# Stage: ISL feasibility + active set, fanned out per operator.
# Thread-safe by partitioning writes per operator_id.

from __future__ import annotations
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.registry.agents.isl_strategy import get_isl_strategy


@register_stage("isl", internal=True)
class ISLStage(Stage):
    reads = ["satellites_t"]
    writes = ["isl_active", "isl_feasibility"]
    thread_safe = True

    def run(self, state):
        sats = state.data.get("satellites_t")
        if not isinstance(sats, pd.DataFrame) or sats.empty:
            state.data["isl_active"] = set()
            state.data["isl_feasibility"] = pd.DataFrame()
            return

        # Per-operator parallel computation, then combine.
        results_active = []
        results_feas   = []
        ops = sorted(set(int(o) for o in sats["operator_id"].unique()))

        EARTH_R_M = 6_371_000.0

        # Fast path: use Phase-1 precomputed sat-sat distances when present and
        # the stored cap covers every enabled operator's max_range_km. Falls
        # back to full recompute otherwise so the result is identical.
        feas_pre = state.data.get("isl_feas_precomp")
        cap_m = state.data.get("isl_feas_cap_m")
        use_pre = isinstance(feas_pre, pd.DataFrame) and cap_m is not None
        if use_pre:
            for op_id in ops:
                op = self.cfg.operator_by_id(op_id)
                if op is not None and op.isl.enabled \
                        and float(op.isl.max_range_km) * 1000.0 > float(cap_m):
                    use_pre = False
                    break

        def _activate(op, sub, feas):
            strat = get_isl_strategy(op)
            # Adapter: strategy expects cfg.isl on the cfg object. We also
            # plant a `_grid_hint` so plus_grid can derive (plane, slot)
            # from sequential node IDs when the satellites_t frame doesn't
            # carry explicit `plane`/`slot` columns (the SGP4-TLE path).
            class _C: pass
            c = _C(); c.isl = op.isl
            try:
                shells = op.satellites.shells
                if shells:
                    sh = shells[0]
                    P = int(sh.num_planes)
                    S = int(sh.num_satellites) // max(int(sh.num_planes), 1)
                    base_id = int(op.sat_id_range[0]) if op.sat_id_range else 0
                    c._grid_hint = (P, S, base_id, int(op.id))
            except Exception:
                pass
            return strat.select(feas, c, sats=sub)

        def _per_op(op_id):
            op = self.cfg.operator_by_id(op_id)
            if op is None or not op.isl.enabled:
                return set(), pd.DataFrame()
            sub = sats[sats["operator_id"] == op_id]
            ids = sub["node_id"].to_numpy(dtype=int)
            n = len(ids)
            if n < 2:
                return set(), pd.DataFrame(
                    columns=["sat_a", "sat_b", "dist_m", "operator_id"])

            max_m = float(op.isl.max_range_km) * 1000.0
            empty = pd.DataFrame(
                columns=["sat_a", "sat_b", "dist_m", "operator_id"])

            if use_pre:
                # Mask precomputed pairs by this operator's range.
                pr = feas_pre[feas_pre["operator_id"] == op_id]
                pr = pr[pr["dist_m"] <= max_m]
                feas = (pr[["sat_a", "sat_b", "dist_m", "operator_id"]]
                        .reset_index(drop=True) if not pr.empty else empty)
                return _activate(op, sub, feas), feas

            # Recompute: vectorized lat/lon/alt -> ECEF, then pairwise distance.
            lat = np.radians(sub["lat"].to_numpy(dtype=float))
            lon = np.radians(sub["lon"].to_numpy(dtype=float))
            r   = EARTH_R_M + sub["alt_m"].to_numpy(dtype=float)
            x = r * np.cos(lat) * np.cos(lon)
            y = r * np.cos(lat) * np.sin(lon)
            z = r * np.sin(lat)
            pos = np.stack([x, y, z], axis=1)                    # (n,3)
            diff = pos[:, None, :] - pos[None, :, :]             # (n,n,3)
            dmat = np.linalg.norm(diff, axis=2)                  # (n,n)

            # Upper-triangle indices (i<j) within range
            iu, ju = np.triu_indices(n, k=1)
            d_ij = dmat[iu, ju]
            keep = d_ij <= max_m
            if not keep.any():
                feas = empty
            else:
                feas = pd.DataFrame({
                    "sat_a":       ids[iu[keep]],
                    "sat_b":       ids[ju[keep]],
                    "dist_m":      d_ij[keep],
                    "operator_id": np.full(int(keep.sum()), op_id, dtype=int),
                })

            return _activate(op, sub, feas), feas

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(ops)))) as ex:
            for a, f in ex.map(_per_op, ops):
                results_active.append(a)
                results_feas.append(f)

        # Combine
        active = set()
        for s in results_active: active |= s
        feas = pd.concat([f for f in results_feas if not f.empty],
                         ignore_index=True) if any(not f.empty for f in results_feas) \
               else pd.DataFrame()
        state.data["isl_active"] = active
        state.data["isl_feasibility"] = feas
