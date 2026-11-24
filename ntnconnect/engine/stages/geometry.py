# Stage: build per-(user, access_node) link table from the per-type node frames.
# Series (writes state.data["links"]).

from __future__ import annotations
import numpy as np
import pandas as pd
from math import radians
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage


_R_EARTH = 6_371_000.0


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle SURFACE distance (ground track), metres."""
    R = _R_EARTH
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _slant_and_elev(ground_dist_m, sat_alt_m):
    """BUG #1 + #2 fix: spherical-Earth slant range and elevation angle.

        arc      = ground_dist / R_E
        slant    = sqrt(R_E^2 + (R_E+h)^2 - 2 R_E (R_E+h) cos(arc))
        elev_deg = arctan2((R_E+h) cos(arc) - R_E,  (R_E+h) sin(arc))

    Inputs are numpy arrays. Returns (slant_m, elev_deg) arrays.
    """
    R   = _R_EARTH
    h   = np.asarray(sat_alt_m, dtype=float)
    arc = np.asarray(ground_dist_m, dtype=float) / R
    Rh  = R + h
    cos_a, sin_a = np.cos(arc), np.sin(arc)
    slant = np.sqrt(np.maximum(R*R + Rh*Rh - 2.0*R*Rh*cos_a, 0.0))
    # Elevation (rad) - positive above horizon
    elev_rad = np.arctan2(Rh * cos_a - R, np.maximum(Rh * sin_a, 1e-9))
    elev_deg = np.degrees(elev_rad)
    elev_deg = np.clip(elev_deg, 0.0, 90.0)
    return slant, elev_deg


# ORACLE remaining-visibility (forward-scan of the ACTUAL ephemeris)
# Alternative to the causal kinematic prediction in compute_visibility. Reads the
# true future positions and counts consecutive seconds the link stays above the
# mask. Enable with cfg.handover.remaining_vis_oracle = True. Cached per (ephemeris dir, constellation, mask).
_ORACLE_CACHE: dict = {}


def build_oracle_remaining(cfg, elev_min, max_tick=None):
    """Return {(t_idx, user_id, node_id): remaining_visibility_s} by forward-
    scanning the ephemeris. Same geometry (_haversine_m + _slant_and_elev) the
    online stage uses, so the only difference vs the kinematic model is
    predict-vs-know-the-future. Vectorized over ticks; only ticks up to
    max_tick are STORED (a pass-length buffer beyond it is still scanned so the
    remaining at max_tick is complete). max_tick=None scans everything."""
    import json
    from pathlib import Path
    from ntnconnect.common.utils.partition_writer import read_partitioned
    R = _R_EARTH
    st = getattr(cfg, "storage", None)
    # Batched dataset layout: <data_dir>/run_<id>/ with per-batch shards for
    # users/ and satellites/ (read_partitioned unions them).
    run_dir = (Path(st.data_dir) / f"run_{int(getattr(st, 'run_id', 1)):03d}"
               if st else None)
    key = (str(run_dir), round(float(elev_min), 3), max_tick)
    if key in _ORACLE_CACHE:
        return _ORACLE_CACHE[key]
    users = read_partitioned(run_dir, "users")
    if users is None or users.empty:
        _ORACLE_CACHE[key] = {}; return {}
    users = users.drop_duplicates("user_id")            # static -> one row/user
    man = json.loads((run_dir / "manifest.json").read_text())
    dt = float(man.get("fine_step_s", man.get("step_s", 1.0)))
    u_lat = users["lat"].to_numpy(float); u_lon = users["lon"].to_numpy(float)
    u_id = users["user_id"].to_numpy(int); U = len(users)
    scan_to = None if max_tick is None else int(max_tick + 1200.0 / dt)  # +20 min pass buffer
    allsat = read_partitioned(run_dir, "satellites")
    if allsat is None or allsat.empty:
        _ORACLE_CACHE[key] = {}; return {}
    allsat = allsat[["t_idx", "node_id", "lat", "lon", "alt_m"]]
    if scan_to is not None:
        allsat = allsat[allsat["t_idx"] <= scan_to]
    remaining: dict = {}
    p2 = np.radians(u_lat)[None, :]; ulon = u_lon[None, :]; ulat = u_lat[None, :]
    for nid, g in allsat.groupby("node_id", sort=False):
        g = g.sort_values("t_idx")
        ts = g["t_idx"].to_numpy(int)
        la = g["lat"].to_numpy(float)[:, None]; lo = g["lon"].to_numpy(float)[:, None]
        al = g["alt_m"].to_numpy(float)[:, None]
        n = len(ts)
        # vectorized haversine + slant elevation over [n_ticks, U]
        dlat = np.radians(ulat - la); dlon = np.radians(ulon - lo)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(la))*np.cos(p2)*np.sin(dlon/2)**2
        ground = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        arc = ground / R; Rh = R + al
        elev = np.degrees(np.arctan2(Rh*np.cos(arc) - R, np.maximum(Rh*np.sin(arc), 1e-9)))
        vism = elev >= elev_min
        rem = np.zeros((n, U))
        for i in range(n - 1, -1, -1):                      # backward consecutive scan
            nxt = rem[i + 1] if (i + 1 < n and ts[i + 1] == ts[i] + 1) else 0.0
            rem[i] = np.where(vism[i], nxt + 1.0, 0.0)
        for i in range(n):
            if max_tick is not None and ts[i] > max_tick:
                continue
            for ui in np.where(vism[i])[0]:
                remaining[(int(ts[i]), int(u_id[ui]), int(nid))] = rem[i, ui] * dt
    _ORACLE_CACHE[key] = remaining
    return remaining


# Shared link-derived computations (single source of truth). Used by the
# merged `geometry` stage AND the standalone `geo_range_rate` /
# `link_visibility_predict` stages, so the logic exists in exactly one place.
def compute_range_rate(links, prev, t_idx, dt_s):
    """Per-link line-of-sight range-rate (m/s) = d(slant range)/dt, by finite
    difference of `dist_m` across ticks for the SAME (user, sat) pair.

    Sign convention: negative = approaching (slant range shrinking, sat rising),
    positive = receding (sat setting); it passes through ~0 near peak elevation.
    This is frame-independent (it uses the already-correct slant geometry, not the
    raw ECEF velocity vector) and so matches the elevation/visibility kinematics.
    The first tick of a contact has no predecessor, so its rate is 0. `prev` is a
    persistent per-(uid,sid) cache {(uid,sid): (t_idx, dist_m)} that this function
    updates and prunes, exactly like compute_visibility's elevation cache."""
    if not {"dist_m", "user_id", "node_id"}.issubset(links.columns):
        return None
    n = len(links)
    uid  = links["user_id"].astype(int).to_numpy()
    sid  = links["node_id"].astype(int).to_numpy()
    dist = links["dist_m"].astype(float).to_numpy()
    rate = np.zeros(n, dtype=float)
    for i in range(n):
        v = prev.get((int(uid[i]), int(sid[i])))
        if v is not None:
            dt = (t_idx - v[0]) * dt_s
            if dt > 0:
                rate[i] = (dist[i] - v[1]) / dt
    for i in range(n):
        prev[(int(uid[i]), int(sid[i]))] = (t_idx, float(dist[i]))
    if t_idx % 50 == 0:
        for k in [k for k, v in prev.items() if t_idx - v[0] > 5]:
            del prev[k]
    return rate


# Upper clamp for the remaining-visibility estimate [s]. No LEO/MEO pass above
# a 0deg mask exceeds this; it just guards the parabolic extrapolation against
# blow-up when the curvature estimate is near zero.
_MAX_REMAIN_S = 1800.0


def compute_visibility(links, t_idx, dt_s, elev_min, prev):
    """Cross-tick remaining-visibility prediction for every visible satellite,
    rising or setting, using a velocity+curvature (parabolic-pass) kinematic
    model. Mutates `prev` cache; returns {elev_rate_deg_s,
    remaining_visibility_s, remaining_visibility_ticks} or {}.

    Model: over a pass, elevation traces a smooth hump E(t) ~= Epk - a*(t-tpk)^2
    with a>0. From the current elevation E, its rate (1st finite difference) and
    its acceleration (2nd finite difference) we solve for the pass peak and the
    time at which E next falls back to `elev_min` (the SET instant):
        a    = -accel/2                       (concave hump => accel < 0 => a>0)
        Epk  = E + rate^2 / (4a)              (extrapolated peak elevation)
        rem  = sqrt((Epk - elev_min)/a) + rate/(2a)
    This yields a LONG remaining time for just-risen sats (rate>0, low E) and a
    SHORT one for sats near setting (rate<0, low E) -- exactly the signal
    `farthest_approaching` needs to ride the whole pass. When curvature is not
    yet observable (first two ticks of a contact) or the model is degenerate, it
    falls back to the linear setting-only estimate (E-elev_min)/(-rate)."""
    if ("elev_deg" not in links.columns or "user_id" not in links.columns):
        return {}
    n = len(links)
    uid = links["user_id"].astype(int).to_numpy()
    sid = links.get("node_id", pd.Series(np.zeros(n))).astype(int).to_numpy()
    elev = links["elev_deg"].astype(float).to_numpy()
    rate = np.zeros(n, dtype=float)
    accel = np.full(n, np.nan, dtype=float)
    remain = np.full(n, np.nan, dtype=float)
    prev_t    = np.full(n, -1, dtype=int)
    prev_elev = np.full(n, np.nan, dtype=float)
    prev_rate = np.full(n, np.nan, dtype=float)
    for i in range(n):
        v = prev.get((int(uid[i]), int(sid[i])))
        if v is not None:
            prev_t[i], prev_elev[i], prev_rate[i] = v
    mask = prev_t >= 0
    if mask.any():
        dt = (t_idx - prev_t[mask]) * dt_s
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dt > 0, (elev[mask] - prev_elev[mask]) / dt, 0.0)
            a = np.where((dt > 0) & np.isfinite(prev_rate[mask]),
                         (r - prev_rate[mask]) / dt, np.nan)
        rate[mask]  = r
        accel[mask] = a

        # Parabolic-pass remaining estimate (rising OR setting)
        m_par = mask.copy()
        m_par[mask] = np.isfinite(a) & (a < -1e-9)        # concave hump only
        if m_par.any():
            ai   = -accel[m_par] / 2.0                    # a > 0
            Epk  = elev[m_par] + (rate[m_par] ** 2) / (4.0 * ai)
            arg  = (Epk - elev_min) / ai
            ok   = arg > 0
            rem  = np.full(ai.shape, np.nan)
            rem[ok] = np.sqrt(arg[ok]) + rate[m_par][ok] / (2.0 * ai[ok])
            remain[np.where(m_par)[0]] = rem

        # Linear fallback for descending sats without valid curvature
        m_lin = mask.copy()
        m_lin[mask] = (r < -1e-6)
        need = m_lin & ~np.isfinite(remain)
        remain[need] = (elev[need] - elev_min) / (-rate[need])

    remain = np.clip(remain, 0.0, _MAX_REMAIN_S)
    for i in range(n):
        prev[(int(uid[i]), int(sid[i]))] = (t_idx, float(elev[i]), float(rate[i]))
    if t_idx % 50 == 0:
        for k in [k for k, v in prev.items() if t_idx - v[0] > 5]:
            del prev[k]
    return {
        "elev_rate_deg_s": rate,
        "remaining_visibility_s": remain,
        "remaining_visibility_ticks": remain / max(dt_s, 1e-9),
    }


@register_stage("geometry")
class Geometry(Stage):
    # MERGED stage: builds the link table AND computes range-rate AND
    # remaining-visibility in one pass (formerly geometry +
    # geo_range_rate + link_visibility_predict).
    reads = ["satellites_t", "haps_t", "uavs_t", "base_stations_t", "users_t"]
    writes = ["links", "cluster_links"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._vis_prev: dict = {}                      # cross-tick elevation cache
        self._vis_prev_cluster: dict = {}              # same, for cluster_links
        self._rng_prev: dict = {}                      # cross-tick slant-range cache (range-rate)
        self._rng_prev_cluster: dict = {}              # same, for cluster_links
        try:
            self._dt_s = float(cfg.env.step_size_s)
        except Exception:
            self._dt_s = 1.0
        try:
            self._elev_min = float(getattr(cfg.constraints, "min_elevation_deg", 20.0))
        except Exception:
            self._elev_min = 10.0

    def _apply_oracle(self, u, state):
        """If enabled, override remaining_visibility_s with the forward-scan
        (oracle) value from the actual ephemeris, instead of the kinematic
        prediction. Enable via cfg.handover.remaining_vis_oracle."""
        import os as _os
        on = bool(getattr(getattr(self.cfg, "handover", None), "remaining_vis_oracle", False))
        if not on or not isinstance(u, pd.DataFrame) or u.empty:
            return
        if "user_id" not in u.columns or "node_id" not in u.columns:
            return
        _dur = float(getattr(getattr(self.cfg, "env", None), "duration_s", 0.0) or 0.0)
        _mt = int(_dur / max(self._dt_s, 1e-9)) if _dur > 0 else None
        oracle = build_oracle_remaining(self.cfg, self._elev_min, max_tick=_mt)
        t_idx = int(getattr(state, "t_idx", 0))
        uid = u["user_id"].to_numpy(int); nid = u["node_id"].to_numpy(int)
        rem = np.array([oracle.get((t_idx, int(uid[i]), int(nid[i])), 0.0)
                        for i in range(len(u))], dtype=float)
        u["remaining_visibility_s"]     = rem
        u["remaining_visibility_ticks"] = rem / max(self._dt_s, 1e-9)

    def _run_precomputed(self, state, lp, sats):
        """Assemble the links frame from the precomputed Phase-1 link table
        (`lp`) by joining satellite state + users. Same output schema as the
        recompute path, plus `off_nadir_deg`. Cross-tick visibility
        (elev_rate / remaining_visibility) is still derived here."""
        # node side (drop operator_id to avoid colliding with lp's)
        nd = sats[["node_id", "node_type", "lat", "lon", "alt_m"]].rename(
            columns={"lat": "node_lat", "lon": "node_lon"})
        u = lp.merge(nd, on="node_id", how="left")
        users = state.data.get("users_t")
        if isinstance(users, pd.DataFrame) and "user_id" in users.columns:
            ucols = ["user_id"] + [c for c in ("lat", "lon", "cluster_id")
                                   if c in users.columns]
            us = users[ucols].rename(columns={"lat": "user_lat", "lon": "user_lon"})
            u = u.merge(us, on="user_id", how="left")
        # aliases + derived columns (match the recompute path)
        u["azimuth_deg"]      = u["az_deg"]
        u["azimuth"]          = u["az_deg"]
        u["link_type"]        = "access"
        u["range_rate_km_s"]  = u["range_rate_m_s"] / 1000.0
        # elev_rate / remaining_visibility: use the Phase-1 precomputed columns
        # if present (exact, full-pass); else fall back to the per-tick estimate.
        if "remaining_visibility_s" not in u.columns:
            for k, v in compute_visibility(u, int(getattr(state, "t_idx", 0)),
                                           self._dt_s, self._elev_min,
                                           self._vis_prev).items():
                u[k] = v
        self._apply_oracle(u, state)
        state.data["links"] = u
        state.data["_sat_mode"] = "visible_only"
        is_sat = (u["node_type"].to_numpy() == "satellite") if not u.empty \
                  else np.zeros(0, dtype=bool)
        state.data["_n_visible_links"] = int(is_sat.sum()) if is_sat.size else 0

    def _build_cluster_links(self, state):
        """Cluster-level geometry: per-(occupied Earth-cell, access node) link
        from the cell centroid. Always computed (both user and cluster links are
        available every tick). Geographic cells, independent of the granularity
        switch. Writes state.data['cluster_links']."""
        users = state.data.get("users_t")
        if not isinstance(users, pd.DataFrame) or users.empty \
                or not ({"lat", "lon"} <= set(users.columns)):
            state.data["cluster_links"] = pd.DataFrame(); return
        try:
            from ntnconnect.scenario.grid import lat_lon_to_cluster_id, cluster_centroid
            cells = np.unique(lat_lon_to_cluster_id(users["lat"].to_numpy(),
                                                    users["lon"].to_numpy(), self.cfg))
        except Exception:
            state.data["cluster_links"] = pd.DataFrame(); return
        frames = []
        for key in ("satellites_t", "haps_t", "uavs_t", "base_stations_t"):
            df = state.data.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df[["node_id", "node_type", "operator_id",
                                  "lat", "lon", "alt_m"]])
        if cells.size == 0 or not frames:
            state.data["cluster_links"] = pd.DataFrame(); return
        cen = [cluster_centroid(int(c), self.cfg) for c in cells]
        c_lat = np.array([p[0] for p in cen], float)
        c_lon = np.array([p[1] for p in cen], float)
        nodes = pd.concat(frames, ignore_index=True)
        n_nid = nodes["node_id"].to_numpy(); n_ntyp = nodes["node_type"].to_numpy()
        n_op = nodes["operator_id"].to_numpy(); n_lat = nodes["lat"].to_numpy(float)
        n_lon = nodes["lon"].to_numpy(float); n_alt = nodes["alt_m"].to_numpy(float)
        nC, nN = cells.shape[0], n_nid.shape[0]
        ci = np.repeat(np.arange(nC), nN); ni = np.tile(np.arange(nN), nC)
        clat = c_lat[ci]; clon = c_lon[ci]
        nlat = n_lat[ni]; nlon = n_lon[ni]; nalt = n_alt[ni]
        ground = _haversine_m(clat, clon, nlat, nlon)
        slant, elev = _slant_and_elev(ground, nalt)
        lat1 = np.radians(clat); lat2 = np.radians(nlat); dlon = np.radians(nlon - clon)
        ax = np.sin(dlon) * np.cos(lat2)
        ay = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        az = (np.degrees(np.arctan2(ax, ay)) + 360.0) % 360.0
        is_sat = (n_ntyp[ni] == "satellite")
        try:
            gcons = self.cfg.constraints.geometry
            sat_mode = str(getattr(gcons, "satellite_mode", "full"))
            elev_min = float(getattr(self.cfg.constraints, "min_elevation_deg", 20.0))
        except Exception:
            sat_mode, elev_min = "full", 10.0
        keep = ((~is_sat) | (elev >= elev_min)) \
            if sat_mode in ("visible_only", "visible_over_duration") and is_sat.any() \
            else np.ones(ci.shape[0], dtype=bool)
        ci, ni = ci[keep], ni[keep]
        cl = pd.DataFrame({
            "cluster_id": cells[ci], "user_id": cells[ci],
            "node_id": n_nid[ni], "node_type": n_ntyp[ni], "operator_id": n_op[ni],
            "cluster_lat": clat[keep], "cluster_lon": clon[keep],
            "node_lat": nlat[keep], "node_lon": nlon[keep], "alt_m": nalt[keep],
            "ground_dist_m": ground[keep], "dist_m": slant[keep],
            "elev_deg": elev[keep], "azimuth_deg": az[keep], "azimuth": az[keep],
            "link_type": "access",
        })
        rr = compute_range_rate(cl, self._rng_prev_cluster,
                                int(getattr(state, "t_idx", 0)), self._dt_s)
        if rr is not None:
            cl["range_rate_m_s"] = rr; cl["range_rate_km_s"] = rr / 1000.0
        for k, v in compute_visibility(cl, int(getattr(state, "t_idx", 0)),
                                       self._dt_s, self._elev_min,
                                       self._vis_prev_cluster).items():
            cl[k] = v
        state.data["cluster_links"] = cl

    def run(self, state):
        # FAST PATH: per-(user,sat) link geometry was precomputed in Phase 1
        # (elev/slant/azimuth/off-nadir/ground-dist at the fine step). JOIN it
        # with the satellite state + users instead of recomputing the cartesian
        # product. Only valid when both endpoints are Phase-1 known: if the users
        # move live in Phase 2 the precompute is stale, so recompute instead. A
        # link touching any Phase-2-mobile node is always computed here in Phase 2.
        precomp = state.data.get("links_precomp")
        sats0 = state.data.get("satellites_t")
        mob = getattr(self.cfg.users, "mobility", None)
        users_live = (str(getattr(mob, "model", "dataset"))
                      not in ("", "dataset", "static", "hover", "none")
                      and int(getattr(mob, "phase", 2)) != 1)
        if (not users_live
                and isinstance(precomp, pd.DataFrame) and not precomp.empty
                and isinstance(sats0, pd.DataFrame) and not sats0.empty):
            self._run_precomputed(state, precomp, sats0)
            self._build_cluster_links(state)
            return

        # Combine all UE-facing access nodes (sats + HAPS + UAVs + BS).
        # Ground stations are feeder-only; excluded from access links.
        frames = []
        for key in ("satellites_t", "haps_t", "uavs_t", "base_stations_t"):
            df = state.data.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df[["node_id", "node_type", "operator_id",
                                  "lat", "lon", "alt_m"]])
        if not frames:
            state.data["links"] = pd.DataFrame()
            return
        nodes = pd.concat(frames, ignore_index=True)

        users = state.data.get("users_t")
        if users is None or users.empty:
            # Stub: synthesize 1 user if no user pipeline yet (keeps demo alive)
            users = pd.DataFrame([{"user_id": 1, "lat": 50.0, "lon": -95.0}])
        # Project users onto (user_id, lat, lon) so the merge produces clean
        # _x / _y suffixes and `alt_m` unambiguously refers to the node side.
        # Carry cluster_id through to the links frame so HO strategies and
        # KPIs can see which Earth-cell each user is in this tick.
        u_keep = ["user_id", "lat", "lon", "cluster_id"]
        users = users[[c for c in u_keep if c in users.columns]]

        # Cartesian product via NUMPY broadcasting (was a pandas `_k=1`
        # merge + ~11 per-column DataFrame __setitem__). Build every column as
        # a numpy array first, CULL by elevation, then construct the links
        # DataFrame ONCE from a dict. Eliminates the merge, the per-column
        # setitem churn and the rename/assign/drop chain entirely.
        has_cluster = "cluster_id" in users.columns
        u_uid  = users["user_id"].to_numpy()
        u_lat  = users["lat"].to_numpy(dtype=float)
        u_lon  = users["lon"].to_numpy(dtype=float)
        u_clu  = users["cluster_id"].to_numpy() if has_cluster else None
        n_nid  = nodes["node_id"].to_numpy()
        n_ntyp = nodes["node_type"].to_numpy()
        n_op   = nodes["operator_id"].to_numpy()
        n_lat  = nodes["lat"].to_numpy(dtype=float)
        n_lon  = nodes["lon"].to_numpy(dtype=float)
        n_alt  = nodes["alt_m"].to_numpy(dtype=float)

        nU, nN = u_uid.shape[0], n_nid.shape[0]
        ui = np.repeat(np.arange(nU), nN)         # user index per product row
        ni = np.tile(np.arange(nN), nU)           # node index per product row

        user_lat = u_lat[ui]; user_lon = u_lon[ui]
        node_lat = n_lat[ni]; node_lon = n_lon[ni]; node_alt = n_alt[ni]

        # BUG #1 + #2 FIX: ground-track distance, then spherical-Earth geometry
        # for SLANT range and TRUE elevation.
        ground_m = _haversine_m(user_lat, user_lon, node_lat, node_lon)
        slant_m, elev_deg_arr = _slant_and_elev(ground_m, node_alt)
        # Azimuth from user toward node (deg, 0 = North, 90 = East).
        lat1 = np.radians(user_lat); lat2 = np.radians(node_lat)
        dlon = np.radians(node_lon - user_lon)
        ax = np.sin(dlon) * np.cos(lat2)
        ay = (np.cos(lat1)*np.sin(lat2)
              - np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
        azimuth = (np.degrees(np.arctan2(ax, ay)) + 360.0) % 360.0

        # Visibility culling (cfg.constraints.geometry.satellite_mode)
        try:
            cfg = getattr(state, "cfg", None) or getattr(self, "cfg", None)
            gcons = cfg.constraints.geometry  # type: ignore[union-attr]
            sat_mode = str(getattr(gcons, "satellite_mode", "full"))
            elev_min = float(getattr(cfg.constraints, "min_elevation_deg", 20.0))
        except Exception:
            sat_mode, elev_min = "full", 10.0
        is_sat = (n_ntyp[ni] == "satellite")
        if sat_mode in ("visible_only", "visible_over_duration") and is_sat.any():
            keep = (~is_sat) | (elev_deg_arr >= elev_min)
        else:
            keep = np.ones(ui.shape[0], dtype=bool)
        ui, ni = ui[keep], ni[keep]
        is_sat = is_sat[keep]

        # Build the links DataFrame in one shot from culled arrays.
        cols = {
            "user_id":      u_uid[ui],
            "node_id":      n_nid[ni],
            "node_type":    n_ntyp[ni],
            "operator_id":  n_op[ni],
            "user_lat":     user_lat[keep],
            "user_lon":     user_lon[keep],
            "node_lat":     node_lat[keep],
            "node_lon":     node_lon[keep],
            "alt_m":        node_alt[keep],
            "ground_dist_m": ground_m[keep],
            "dist_m":       slant_m[keep],          # OVERLOADED = SLANT
            "elev_deg":     elev_deg_arr[keep],
            "azimuth_deg":  azimuth[keep],
            "azimuth":      azimuth[keep],          # alias (steering lookup)
            "link_type":    "access",
        }
        if has_cluster:
            cols["cluster_id"] = u_clu[ui]
        u = pd.DataFrame(cols)

        # MERGED: range-rate + remaining-visibility in the same stage
        rr = compute_range_rate(u, self._rng_prev,
                                int(getattr(state, "t_idx", 0)), self._dt_s)
        if rr is not None:
            u["range_rate_m_s"]  = rr
            u["range_rate_km_s"] = rr / 1000.0
        for k, v in compute_visibility(u, int(getattr(state, "t_idx", 0)),
                                       self._dt_s, self._elev_min,
                                       self._vis_prev).items():
            u[k] = v
        self._apply_oracle(u, state)

        state.data["links"] = u
        state.data["_sat_mode"]        = sat_mode
        state.data["_n_visible_links"] = int(is_sat.sum()) if is_sat.size else 0
        self._build_cluster_links(state)
