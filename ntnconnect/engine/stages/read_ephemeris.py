# Stage: load pre-computed ephemeris (batch parquet, LRU=1) instead of SGP4.
# Writes satellites_t in the SGP4 SatelliteGenerator schema.
from __future__ import annotations
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry   import register_stage

_log = logging.getLogger(__name__)


@register_stage("read_ephemeris", internal=True)
class ReadEphemeris12h(Stage):
    """Read pre-computed ephemeris from the 12-hour dataset."""
    reads  = []
    # Also publishes the precomputed per-(user,sat) link table for this tick
    # (when the dataset was built with one), so geometry can skip the
    # cartesian-product recompute. General: any ephemeris dataset that ships
    # user_links_NN.parquet activates this; others fall back transparently.
    writes = ["satellites_t", "links_precomp", "isl_feas_precomp"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._loaded_batch: dict[str, tuple[int, pd.DataFrame]] = {}
        self._loaded_links: dict[str, tuple[int, pd.DataFrame]] = {}
        self._loaded_feas:  dict[str, tuple[int, pd.DataFrame]] = {}
        self._feas_cap_m: Optional[float] = None
        self._manifest: Optional[dict] = None
        self._t0: Optional[datetime] = None
        self._dt_fine_s: float = 1.0
        self._dt_batch_s: float = 900.0

    def _load_manifest(self) -> None:
        if self._manifest is not None: return
        eph_dir = self._eph_dir()
        man_p = eph_dir / "manifest.json"
        if not man_p.exists():
            raise FileNotFoundError(f"ephemeris manifest not found: {man_p}")
        self._manifest = json.loads(man_p.read_text())
        self._t0 = datetime.fromisoformat(
            self._manifest["start_utc"].replace("Z", "+00:00"))
        self._dt_fine_s  = float(self._manifest["fine_step_s"])
        self._dt_batch_s = float(self._manifest["batch_min"]) * 60.0

    def _eph_dir(self) -> Path:
        # Configurable via cfg.storage.ephemeris_dir; otherwise default
        st = getattr(self.cfg, "storage", None)
        path = getattr(st, "ephemeris_dir", None) if st is not None else None
        if path is None:
            # Project default
            path = Path(__file__).resolve().parents[3] / "data" / "ephemeris_12h"
        return Path(path)

    def run(self, state):
        self._load_manifest()
        ts = state.timestamp
        if ts is None:
            state.data["satellites_t"] = pd.DataFrame()
            return
        # Normalise to naive UTC for arithmetic
        if hasattr(ts, "to_pydatetime"):
            ts_dt = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            ts_dt = ts
        else:
            ts_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if ts_dt.tzinfo is None:
            from datetime import timezone
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)

        secs_since = (ts_dt - self._t0).total_seconds()
        batch_idx  = int(secs_since // self._dt_batch_s)
        t_in_batch = int(round((secs_since - batch_idx * self._dt_batch_s)
                                 / self._dt_fine_s))
        if batch_idx < 0 or batch_idx >= int(self._manifest["n_batches"]):
            state.data["satellites_t"] = pd.DataFrame()
            return

        # Constellation-name filter: only load the constellation named by
        # cfg.storage.ephemeris_constellation (if set). This is the explicit
        # opt-in path used by the run_with_ephemeris driver, and avoids
        # the operator-id ambiguity (the ephemeris dataset assigns
        # oneweb=1, starlink=2, kuiper=3 unconditionally, but a scenario
        # may re-use op_id=1 for any constellation).
        st = getattr(self.cfg, "storage", None)
        want_cnst = getattr(st, "ephemeris_constellation", None) if st else None

        frames = []
        eph_dir = self._eph_dir()
        for cnst, cnst_info in self._manifest["constellations"].items():
            if want_cnst is not None and cnst != want_cnst:
                continue
            # LRU of size 1 per constellation. Pre-split the batch by t_idx ONCE
            # on load (dict {t_idx: subframe}) so the per-tick lookup is O(1) -
            # not an O(batch_rows) boolean scan repeated every tick.
            cached = self._loaded_batch.get(cnst)
            if cached is None or cached[0] != batch_idx:
                p = eph_dir / cnst / f"visible_sats_{batch_idx:02d}.parquet"
                if not p.exists():
                    continue
                df_b = pd.read_parquet(p)
                by_t = {int(t): g for t, g in df_b.groupby("t_idx", sort=False)}
                self._loaded_batch[cnst] = (batch_idx, by_t)
            else:
                by_t = cached[1]
            df_t = by_t.get(t_in_batch)
            if df_t is None or df_t.empty: continue
            # Project the columns the rest of the engine expects.
            out = pd.DataFrame({
                "timestamp":   df_t["timestamp"].to_numpy(),
                "t_idx":       df_t["t_idx"].to_numpy(),
                "node_id":     df_t["node_id"].to_numpy(),
                "node_type":   np.full(len(df_t), "satellite", dtype=object),
                "operator_id": df_t["operator_id"].to_numpy(),
                "name":        np.array([f"{cnst.upper()}_{int(n)}"
                                          for n in df_t["node_id"].to_numpy()]),
                "lat":         df_t["lat"].to_numpy(),
                "lon":         df_t["lon"].to_numpy(),
                "alt_m":       df_t["alt_m"].to_numpy(),
                "vx_mps":      df_t["vx"].to_numpy(),
                "vy_mps":      df_t["vy"].to_numpy(),
                "vz_mps":      df_t["vz"].to_numpy(),
                "x":           df_t["x"].to_numpy(),
                "y":           df_t["y"].to_numpy(),
                "z":           df_t["z"].to_numpy(),
            })
            frames.append(out)

        state.data["satellites_t"] = (pd.concat(frames, ignore_index=True)
                                       if frames else pd.DataFrame())

        # Precomputed per-(user, sat) link table for this tick (optional)
        # When the dataset ships user_links_NN.parquet, surface this tick's
        # rows so geometry can join instead of recomputing the geometry.
        # Same batch-LRU + t_in_batch slicing as the satellite read above.
        link_frames = []
        for cnst in self._manifest["constellations"]:
            if want_cnst is not None and cnst != want_cnst:
                continue
            cachedl = self._loaded_links.get(cnst)
            if cachedl is None or cachedl[0] != batch_idx:
                lp = eph_dir / cnst / f"user_links_{batch_idx:02d}.parquet"
                if not lp.exists():
                    self._loaded_links[cnst] = (batch_idx, None)
                    continue
                df_l = pd.read_parquet(lp)
                by_t = {int(t): g for t, g in df_l.groupby("t_idx", sort=False)}
                self._loaded_links[cnst] = (batch_idx, by_t)
            else:
                by_t = cachedl[1]
            if by_t is None:
                continue
            sl = by_t.get(t_in_batch)
            if sl is not None and not sl.empty:
                link_frames.append(sl)
        state.data["links_precomp"] = (pd.concat(link_frames, ignore_index=True)
                                       if link_frames else pd.DataFrame())

        # Precomputed sat-sat distances for this tick (optional). When the
        # dataset ships isl_links_NN.parquet, surface this tick's rows so
        # the isl stage masks by max_range_km instead of recomputing pairwise
        # geometry. isl_feas_cap_m lets isl fall back if a range exceeds it.
        feas_frames = []
        any_feas = False
        for cnst in self._manifest["constellations"]:
            if want_cnst is not None and cnst != want_cnst:
                continue
            cachedf = self._loaded_feas.get(cnst)
            if cachedf is None or cachedf[0] != batch_idx:
                fp = eph_dir / cnst / f"isl_links_{batch_idx:02d}.parquet"
                if not fp.exists():
                    self._loaded_feas[cnst] = (batch_idx, None)
                    continue
                df_f = pd.read_parquet(fp)
                by_t = {int(t): g for t, g in df_f.groupby("t_idx", sort=False)}
                self._loaded_feas[cnst] = (batch_idx, by_t)
            else:
                by_t = cachedf[1]
            if by_t is None:
                continue
            any_feas = True
            sl = by_t.get(t_in_batch)
            if sl is not None and not sl.empty:
                feas_frames.append(sl)
        if any_feas:
            state.data["isl_feas_precomp"] = (pd.concat(feas_frames, ignore_index=True)
                                              if feas_frames else pd.DataFrame(
                    columns=["t_idx", "sat_a", "sat_b", "dist_m", "operator_id"]))
            if self._feas_cap_m is None:
                from ntnconnect.scenario.isl_precompute import read_cap_m
                self._feas_cap_m = read_cap_m(eph_dir)
            state.data["isl_feas_cap_m"] = self._feas_cap_m


@register_stage("read_ephemeris_users", internal=True)
class ReadEphemerisUsers(Stage):
    """Reads users.parquet from the ephemeris dataset and writes it
    unchanged into state.data['users_t'] (static across ticks)."""
    reads  = []
    writes = ["users_t"]
    thread_safe = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self._users_df: Optional[pd.DataFrame] = None

    def run(self, state):
        if self._users_df is None:
            st = getattr(self.cfg, "storage", None)
            path = getattr(st, "ephemeris_dir", None) if st else None
            if path is None:
                path = Path(__file__).resolve().parents[3] / "data" / "ephemeris_12h"
            users_p = Path(path) / "users.parquet"
            if users_p.exists():
                df = pd.read_parquet(users_p)
            else:
                df = pd.DataFrame()
            # Honour an explicit user list (e.g. --max-users truncation) so
            # the driver's user-count control actually limits the engine.
            explicit = getattr(getattr(self.cfg, "users", None),
                                "explicit_user_list", None)
            if explicit and not df.empty:
                keep = {int(u.get("user_id", u.get("id", -1)))
                         for u in explicit}
                df = df[df["user_id"].astype(int).isin(keep)].reset_index(drop=True)
            self._users_df = df
        state.data["users_t"] = self._users_df
