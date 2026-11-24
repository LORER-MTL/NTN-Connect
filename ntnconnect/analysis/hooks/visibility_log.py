# Hook: visibility_log - persists visibility_per_tick, contact_events, sinr_snapshot.
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook

_log = logging.getLogger(__name__)


@register_hook("visibility_log")
def _build(engine):
    return VisibilityLog(engine)


class VisibilityLog(SimulationHook):
    """Per-tick visibility + contact-event + SINR snapshot."""

    def __init__(self, engine):
        self.engine = engine
        self._vis_rows: list = []
        self._sinr_rows: list[dict] = []
        # contact-state machine: {(user_id, sat_id): t_enter_idx}
        self._open_contacts: dict[tuple[int, int], int] = {}
        self._contact_rows: list[dict] = []
        # min_elevation_deg, then elev_min_deg, then 20° default
        try:
            self._elev_min = float(getattr(
                self.engine.cfg.constraints, "min_elevation_deg",
                getattr(self.engine.cfg.handover, "elev_min_deg", 20.0)))
        except Exception:
            self._elev_min = 20.0
        try:
            self._dt_s = float(self.engine.cfg.env.step_size_s)
        except Exception:
            self._dt_s = 1.0
        # batch_log_every_n_ticks > 0 => flush per-batch parquets, bounded memory
        try:
            self._batch_ticks = int(
                getattr(self.engine.cfg.outputs,
                        "batch_log_every_n_ticks", 0) or 0)
        except Exception:
            self._batch_ticks = 0
        self._batch_idx = 0
        self._uidx_cache = None

    def _user_index(self, state, links):
        # Static per-scenario user table, built once.
        if self._uidx_cache is not None:
            return self._uidx_cache
        users_t = state.data.get("users_t")
        if isinstance(users_t, pd.DataFrame) and "user_id" in users_t.columns:
            ut = users_t.drop_duplicates("user_id")
            uids = ut["user_id"].astype(int).to_numpy()
            lat = (ut["lat"].to_numpy(float) if "lat" in ut.columns
                   else np.full(uids.shape, np.nan))
            lon = (ut["lon"].to_numpy(float) if "lon" in ut.columns
                   else np.full(uids.shape, np.nan))
        else:
            uids = np.unique(links["user_id"].astype(int).to_numpy())
            lat = np.full(uids.shape, np.nan); lon = np.full(uids.shape, np.nan)
        uid2idx = {int(u): i for i, u in enumerate(uids)}
        self._uidx_cache = (uids, lat, lon, uid2idx)
        return self._uidx_cache

    def on_step_end(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        if "elev_deg" not in links.columns or "user_id" not in links.columns:
            return
        t_idx = int(state.t_idx)

        # ---- visibility per tick ----
        # Every scenario user is recorded, incl. those with zero visible sats.
        elev_all = links["elev_deg"].to_numpy(dtype=float)
        uid_all  = links["user_id"].astype(int).to_numpy()
        vis_mask = elev_all >= self._elev_min

        uids, ulat, ulon, uid2idx = self._user_index(state, links)
        nU = uids.shape[0]
        counts = np.zeros(nU, dtype=int)
        # -inf, not nan, so np.maximum reduction works; unset -> NaN after
        best   = np.full(nU, -np.inf, dtype=float)
        if vis_mask.any():
            vis_uidx = np.fromiter((uid2idx.get(int(u), -1)
                                    for u in uid_all[vis_mask]),
                                   dtype=np.int64, count=int(vis_mask.sum()))
            ev = elev_all[vis_mask]
            ok = (vis_uidx >= 0) & np.isfinite(ev)
            np.add.at(counts, vis_uidx[ok], 1)
            np.maximum.at(best, vis_uidx[ok], ev[ok])
        best[~np.isfinite(best)] = np.nan
        df_v_full = pd.DataFrame({
            "t_idx":          t_idx,
            "user_id":        uids,
            "user_lat":       ulat,
            "user_lon":       ulon,
            "n_visible_sats": counts,
            "best_elev_deg":  best,
        })
        self._vis_rows.append(df_v_full)

        # ---- contact events ----
        cur_visible = set()
        if "node_id" in links.columns:
            vis_rows = links.loc[vis_mask, ["user_id", "node_id"]]
            uids = vis_rows["user_id"].astype(int).to_numpy()
            sids = vis_rows["node_id"].astype(int).to_numpy()
            cur_visible = set(zip(uids.tolist(), sids.tolist()))
        for pair in cur_visible:
            if pair not in self._open_contacts:
                self._open_contacts[pair] = t_idx
        gone = [p for p in self._open_contacts if p not in cur_visible]
        for pair in gone:
            t_enter = self._open_contacts.pop(pair)
            self._contact_rows.append({
                "user_id":              int(pair[0]),
                "sat_id":               int(pair[1]),
                "t_enter_idx":          int(t_enter),
                "t_leave_idx":          int(t_idx),
                "contact_duration_s":   float((t_idx - t_enter) * self._dt_s),
            })

        # ---- SINR snapshot for served links only ----
        assigns = state.data.get("assignments", {}) or {}
        if assigns and "sinr_db" in links.columns and "node_id" in links.columns:
            uid_arr = links["user_id"].astype(int).to_numpy()
            nid_arr = links["node_id"].astype(int).to_numpy()
            served_for_uid = np.array(
                [assigns.get(int(u), -1) for u in uid_arr], dtype=int)
            served_mask = (served_for_uid == nid_arr)
            # Keep identifiers + every numeric KPI column on `links`.
            _EXCLUDE = {
                "operator_id", "t_idx", "timestamp", "x", "y", "z",
                "vx", "vy", "vz", "carrier_hz", "bandwidth_hz", "chan_gain_lin",
                "k_factor_db", "doppler_hz", "signal_w", "tx_power_w",
                "node_lat", "node_lon", "user_lat", "user_lon", "alt_m",
                "dist_m", "azimuth_deg", "azimuth", "range_rate_km_s",
                "remaining_visibility_ticks", "n_interferers", "n_visible_sats",
            }
            kpi_cols = [c for c in links.columns
                        if c not in _EXCLUDE and c not in ("user_id", "node_id")
                        and pd.api.types.is_numeric_dtype(links[c])]
            keep_cols = ["user_id", "node_id", "elev_deg"] + \
                        [c for c in kpi_cols if c != "elev_deg"]
            keep_cols = [c for c in keep_cols if c in links.columns]
            df_s = links.loc[served_mask, keep_cols].copy()
            df_s = df_s.rename(columns={"node_id": "sat_id",
                                          "elev_deg": "serving_sat_elev_deg"})
            df_s["t_idx"] = t_idx
            self._sinr_rows.append(df_s)

        if self._batch_ticks > 0 and ((t_idx + 1) % self._batch_ticks) == 0:
            self._flush_batch()

    def _flush_batch(self):
        # Write buffers to batch_{N:04d}/<table>.parquet and clear.
        from ntnconnect.common.utils.run_io import batch_table_path
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for buf, table in [
            (self._vis_rows,     "visibility_per_tick"),
            (self._contact_rows, "contact_events"),
            (self._sinr_rows,    "sinr_snapshot"),
        ]:
            if not buf: continue
            if isinstance(buf[0], pd.DataFrame):
                df = pd.concat(buf, ignore_index=True)
            else:
                df = pd.DataFrame(buf)
            if not df.empty:
                dst = batch_table_path(out_dir, self._batch_idx, table)
                df.to_parquet(dst, index=False)
                _log.info("visibility_log: %d rows -> %s", len(df), dst)
        # _open_contacts kept across batches for continuity
        self._vis_rows.clear()
        self._contact_rows.clear()
        self._sinr_rows.clear()
        self._batch_idx += 1

    def on_simulation_end(self, state):
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Close contacts still open at end-of-sim.
        if self._open_contacts:
            t_idx = int(state.t_idx)
            for pair, t_enter in self._open_contacts.items():
                self._contact_rows.append({
                    "user_id":            int(pair[0]),
                    "sat_id":             int(pair[1]),
                    "t_enter_idx":        int(t_enter),
                    "t_leave_idx":        int(t_idx),
                    "contact_duration_s": float((t_idx - t_enter) * self._dt_s),
                    "open_at_end":        True,
                })
            self._open_contacts.clear()

        # Batched mode: partial last window -> own shard; else merge end-closed
        # contacts into the last batch so the batch count stays exact.
        if self._batch_ticks > 0:
            if self._vis_rows or self._sinr_rows:
                self._flush_batch()
            elif self._contact_rows and self._batch_idx > 0:
                from ntnconnect.common.utils.run_io import batch_dir
                dst = batch_dir(out_dir, self._batch_idx - 1) / "contact_events.parquet"
                extra = pd.DataFrame(self._contact_rows)
                if dst.exists():
                    extra = pd.concat([pd.read_parquet(dst), extra], ignore_index=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                extra.to_parquet(dst, index=False)
                self._contact_rows.clear()
            elif self._contact_rows:
                self._flush_batch()
            return

        def _to_df(rows):
            if not rows: return None
            if isinstance(rows[0], pd.DataFrame):
                return pd.concat(rows, ignore_index=True)
            return pd.DataFrame(rows)
        for rows, fname in [
            (self._vis_rows,     "visibility_per_tick.parquet"),
            (self._contact_rows, "contact_events.parquet"),
            (self._sinr_rows,    "sinr_snapshot.parquet"),
        ]:
            df = _to_df(rows)
            if df is None or df.empty: continue
            df.to_parquet(out_dir / fname, index=False)
            _log.info("visibility_log: %d rows -> %s", len(df), out_dir / fname)
