# Stage: external interference I_ext as a constant (data, not optimizer var).
# ext_mode (7 modes), ext_knob, ext_K_r. Writes links["i_ext_w", "i_over_n_db",...].

from __future__ import annotations
import logging
from typing import Callable, Dict
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_log = logging.getLogger(__name__)
kB = 1.380649e-23
c0 = 299_792_458.0
_R_EARTH = 6_371_000.0


# Geometry helpers
def _lla_to_ecef(lat_deg, lon_deg, alt_m):
    """Spherical-Earth LLA -> ECEF (good to ~0.3% - sufficient for off-axis
    angles where we only need direction cosines). Returns arrays (x, y, z)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    r   = _R_EARTH + np.asarray(alt_m, dtype=float)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def _angle_between_deg(v1, v2):
    """Angle between two batched 3-vectors (each shape (N, 3)), in degrees."""
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    dot = np.einsum("ij,ij->i", v1, v2)
    cos = np.clip(dot / np.maximum(n1 * n2, 1e-30), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _haversine_deg(lat1, lon1, lat2, lon2):
    """Great-circle separation on unit sphere, in degrees."""
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


# Frequency-reuse colouring (applied to CELLS / served users)
#
# Conventional multibeam-sat reuse colours the beams/cells on the ground,
# not the satellites:
#  - Vázquez & Pérez-Neira 2016, "Precoding in Multibeam Satellite"
#  - 3GPP TR 38.821 §6.1.3.3 (K_r per cell)
#  - ITU-R S.1503 / S.1325 (spot beam colouring)
#
# Each active served cell (= served user this tick) gets a colour in
# [0, K_r-1]. Interferer sat s' hurts victim cell u iff
#  colour(j(s')) == colour(u)
# where j(s') is the cell currently served by s'.
#
# `_compute_cell_colouring()` is called ONCE per tick and returns
# {user_id -> colour_int}. The mode functions then look up rho per
# interferer in O(1).
def _cells_id_mod_kr(user_ids: np.ndarray, K_r: int, **_) -> np.ndarray:
    """user_id mod K_r."""
    K = max(int(K_r), 1)
    return (user_ids.astype(int) % K).astype(int)


def _cells_greedy_geo(user_ids: np.ndarray, K_r: int,
                      *, user_lat: np.ndarray, user_lon: np.ndarray,
                      **_) -> np.ndarray:
    """Sort by longitude then latitude; cycle through K_r colours so that
    geographically-adjacent cells receive distinct colours."""
    K = max(int(K_r), 1)
    order = np.lexsort((user_lat, user_lon))  # primary: lon, secondary: lat
    colours = np.empty(len(user_ids), dtype=int)
    colours[order] = np.arange(len(order)) % K
    return colours


def _build_cell_conflict_graph(user_lat: np.ndarray, user_lon: np.ndarray,
                               sep_thresh_deg: float) -> list[set]:
    """Adjacency list of the CELL conflict graph: cells whose centroids are
    within `sep_thresh_deg` great-circle degrees are considered to potentially
    share footprint/sidelobe interference if they used the same colour."""
    n = len(user_lat)
    if n == 0:
        return []
    LAT1 = np.radians(user_lat)[:, None]; LAT2 = np.radians(user_lat)[None, :]
    DLAT = LAT2 - LAT1
    DLON = np.radians(user_lon)[None, :] - np.radians(user_lon)[:, None]
    a = np.sin(DLAT/2.0)**2 + np.cos(LAT1)*np.cos(LAT2)*np.sin(DLON/2.0)**2
    sep = np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
    np.fill_diagonal(sep, np.inf)
    adj = [set() for _ in range(n)]
    pairs = np.argwhere(sep <= sep_thresh_deg)
    for i, j in pairs:
        if i < j:
            adj[i].add(int(j)); adj[j].add(int(i))
    return adj


def _cells_dsatur(user_ids: np.ndarray, K_r: int,
                  *, user_lat: np.ndarray, user_lon: np.ndarray,
                  sep_thresh_deg: float, **_) -> np.ndarray:
    """Brélaz DSATUR (1979) on the CELL conflict graph.

    Returns the colour assignment per cell.

    Algorithm:
      1. Build cell-conflict graph (edge iff cell-centroid separation
         ≤ sep_thresh_deg).
      2. Seed: max-degree cell gets colour 0.
      3. Loop: pick uncoloured cell with max 'saturation' (# distinct
         colours among coloured neighbours); break ties by degree.
         Assign smallest legal colour in [0, K_r-1]. If no legal colour
         exists (cell has K_r already-coloured neighbours of all colours),
         pick the colour least represented locally (collision tolerated).
      4. Repeat until all coloured.

    DSATUR is optimal on interval graphs and within ~1.1× of optimum
    on planar/sparse beam-conflict graphs typical for cell layouts.
    """
    K = max(int(K_r), 1)
    n = len(user_ids)
    if n == 0:
        return np.zeros(0, dtype=int)
    adj = _build_cell_conflict_graph(user_lat, user_lon, sep_thresh_deg)
    deg = np.array([len(a) for a in adj], dtype=int)
    colour    = np.full(n, -1, dtype=int)
    saturation = np.zeros(n, dtype=int)

    seed = int(np.argmax(deg))
    colour[seed] = 0
    for nb in adj[seed]:
        saturation[nb] += 1

    for _ in range(n - 1):
        mask = (colour < 0)
        if not mask.any(): break
        sat_view = np.where(mask, saturation, -1)
        max_sat  = sat_view.max()
        cand_mask = mask & (saturation == max_sat)
        deg_view  = np.where(cand_mask, deg, -1)
        v = int(np.argmax(deg_view))
        nb_colours = [colour[nb] for nb in adj[v] if colour[nb] >= 0]
        used = set(nb_colours)
        legal = [c for c in range(K) if c not in used]
        if legal:
            chosen = legal[0]
        else:
            # All K colours are already used by neighbours - collision must
            # happen. Pick the colour least frequent locally.
            counts = np.bincount(nb_colours, minlength=K)
            chosen = int(np.argmin(counts))
        colour[v] = chosen
        for nb in adj[v]:
            if colour[nb] < 0:
                cset = {colour[u] for u in adj[nb] if colour[u] >= 0}
                saturation[nb] = len(cset)
    return colour


_CELL_COLOURERS = {
    "id_mod_kr":   _cells_id_mod_kr,
    "greedy_geo":  _cells_greedy_geo,
    "dsatur":      _cells_dsatur,
    # "mean_field" is a special case: no per-cell colour is computed;
    # the modes apply a statistical 1/K_r scaling instead.
}


def list_colouring_algorithms() -> list[str]:
    return sorted(list(_CELL_COLOURERS.keys()) + ["mean_field"])


def _compute_cell_colouring(active_users: pd.DataFrame, algorithm: str,
                            K_r: int, sep_thresh_deg: float
                            ) -> dict[int, int]:
    """Return {user_id -> colour_int}. Empty dict for 'mean_field'."""
    if algorithm == "mean_field" or len(active_users) == 0:
        return {}
    fn = _CELL_COLOURERS.get(algorithm, _cells_id_mod_kr)
    uids = active_users["user_id"].astype(int).to_numpy()
    ulat = active_users["user_lat"].astype(float).to_numpy()
    ulon = active_users["user_lon"].astype(float).to_numpy()
    cols = fn(user_ids=uids, K_r=K_r,
              user_lat=ulat, user_lon=ulon,
              sep_thresh_deg=sep_thresh_deg)
    return {int(uid): int(c) for uid, c in zip(uids, cols)}


# Mode registry
ModeFn = Callable[[pd.DataFrame, dict], np.ndarray]
_EXT_MODES: Dict[str, ModeFn] = {}


def register_ext_mode(name: str):
    """Decorator: registers a mode under `name`. Signature:
       fn(links: pd.DataFrame, ctx: dict) -> np.ndarray of i_ext_w (size = len(links))
    `ctx` contains: knob, K_r, noise_w, c_nom_w, rx_gain_max_lin,
    freq_hz_array, n_visible_per_user, rng.
    """
    def deco(fn: ModeFn):
        _EXT_MODES[name] = fn
        return fn
    return deco


def list_ext_interference_modes() -> list[str]:
    return sorted(_EXT_MODES.keys())


# Helper: nominal received power C_nom,u (geometric, no h/w)
def _nominal_c_per_link(links: pd.DataFrame,
                        rx_gain_max_lin: np.ndarray,
                        op_power_max_w: dict[int, float] | None = None
                        ) -> np.ndarray:
    """C_nom = P_tx_max * G_tx_max * G_rx_max * (λ / 4π d)^2 / L_atm.

    Uses the *configured max EIRP* (operator.radio.power_max_w + antenna gain),
    not the per-row tx_power_w (which the SOCP often sets to 0 for non-served
    rows). This is the right convention for interferer-side modelling: the
    interferer is assumed to be radiating at full power toward its own served
    cell, with the sidelobe leakage hitting the victim.
    """
    n = len(links)
    p_tx_row = links.get("tx_power_w",
                         pd.Series(np.zeros(n))).astype(float).to_numpy()
    # Per-row max-power fallback by operator
    if op_power_max_w and "operator_id" in links.columns:
        op_ids = links["operator_id"].astype(int).to_numpy()
        p_tx_max = np.array([op_power_max_w.get(int(o), 1.0) for o in op_ids],
                            dtype=float)
    else:
        p_tx_max = np.ones(n, dtype=float)
    # Take max(row_tx_power, configured_max) so a non-zero per-row value wins,
    # but a zero is replaced by the configured max.
    p_tx = np.where(p_tx_row > 0.0, p_tx_row, p_tx_max)
    g_tx_db = links.get("antenna_gain_db",
                        pd.Series(np.zeros(n))).astype(float).to_numpy()
    pl_db   = links.get("path_loss_db",
                        pd.Series(np.zeros(n))).astype(float).to_numpy()
    g_tx_lin = 10.0 ** (g_tx_db / 10.0)
    pl_lin   = 10.0 ** (-pl_db / 10.0)        # already includes free-space + atmos
    return p_tx * g_tx_lin * rx_gain_max_lin * pl_lin


# O1. off
@register_ext_mode("off")
def _mode_off(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    return np.zeros(len(links), dtype=float)


# O2. fixed_in : I_ext = 10^(knob/10) * N0B
@register_ext_mode("fixed_in")
def _mode_fixed_in(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    return (10.0 ** (ctx["knob"] / 10.0)) * ctx["noise_w"]


# O3. fixed_ci_nominal : I_ext = C_nom * 10^(-knob/10)
@register_ext_mode("fixed_ci_nominal")
def _mode_fixed_ci(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    return ctx["c_nom_w"] * (10.0 ** (-ctx["knob"] / 10.0))


# O4. reuse_floor_const : C_nom * 10^(-SINR_max(K_r)/10) - N0B (>=0)
_SINR_MAX_DB_BY_KR = {1: 7.0, 2: 10.0, 3: 13.0, 4: 15.0}


@register_ext_mode("reuse_floor_const")
def _mode_reuse_floor(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    K_r = int(ctx["K_r"])
    sinr_max_db = _SINR_MAX_DB_BY_KR.get(K_r, 7.0)
    target_lin  = 10.0 ** (-sinr_max_db / 10.0)
    val = ctx["c_nom_w"] * target_lin - ctx["noise_w"]
    return np.clip(val, 0.0, None)


# O5. nvis_sidelobe_nominal : (N_vis-1)/K_r * <i_nom(s', u)>
# Implemented as: ((N_vis-1)/K_r) * <C_nom_per_visible_sat_at_u> * 10^(knob/10)
# where the average per-visible-sat nominal power at u is approximated by
# C_nom of the same row (served link) - i.e. assume interferers have similar
# geometry. This keeps the cost O(1) per link and stays purely geometric.
@register_ext_mode("nvis_sidelobe_nominal")
def _mode_nvis_sidelobe(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    """For colouring='mean_field' uses (N_vis-1)/K_r directly (cheap).
    For 'id_mod_kr'/'greedy_elev'/'dsatur' uses the per-user colouring
    result to count actual co-channel interferers."""
    K_r     = max(int(ctx["K_r"]), 1)
    beta_sl = 10.0 ** (ctx["knob"] / 10.0)
    xi_pol  = 10.0 ** (-float(ctx["pol_iso_db"]) / 10.0)

    if ctx["colouring"] == "mean_field":
        n_vis = ctx["n_visible_per_link"].astype(float)
        n_eff = np.maximum(n_vis - 1.0, 0.0) / K_r
        return n_eff * ctx["c_nom_w"] * beta_sl * xi_pol

    # Cell-colour-aware path. The per-tick CELL colouring was computed once
    # in the stage and passed in ctx["cell_colour"]: {user_id -> int}.
    # rho[s'→u] = 1 iff cell_colour[j(s')] == cell_colour[u], where j(s')
    # is the user currently served by interferer sat s'. Interferers whose
    # served cell has no colour entry (e.g. not in `assignments`) are
    # assumed to transmit on a uniformly random colour ⇒ mean weight 1/K_r.
    n   = len(links)
    out = np.zeros(n, dtype=float)
    if "user_id" not in links.columns or "node_type" not in links.columns:
        return out
    is_sat = (links["node_type"].astype(str) == "satellite").to_numpy()
    user   = links["user_id"].astype(int).to_numpy()
    sat_id = links["node_id"].astype(int).to_numpy()
    c_nom  = ctx["c_nom_w"]

    cell_colour    = ctx.get("cell_colour", {}) or {}
    sat_to_servee  = ctx.get("sat_to_servee_uid", {}) or {}

    # Vectorised: pandas groupby.groups is C-level → ~10× faster than
    # Python setdefault loop. Returns Index objects keyed by user_id.
    sat_rows = np.where(is_sat)[0]
    if len(sat_rows):
        sat_df = pd.DataFrame({"u": user[sat_rows]}, index=sat_rows)
        by_user = {int(k): list(v) for k, v
                   in sat_df.groupby("u", sort=False).groups.items()}
    else:
        by_user = {}

    for idx in range(n):
        if not is_sat[idx]: continue
        u = int(user[idx])
        idxs = by_user.get(u, [])
        if len(idxs) <= 1: continue
        u_colour = cell_colour.get(u, None)
        rho = np.zeros(len(idxs), dtype=float)
        for k, j in enumerate(idxs):
            if int(sat_id[j]) == int(sat_id[idx]):
                continue                              # exclude self
            j_uid = sat_to_servee.get(int(sat_id[j]), None)
            j_col = cell_colour.get(int(j_uid), None) if j_uid is not None else None
            if u_colour is None or j_col is None:
                rho[k] = 1.0 / K_r                    # mean-field fallback
            else:
                rho[k] = 1.0 if (j_col == u_colour) else 0.0
        out[idx] = float((c_nom[idxs] * rho).sum()) * beta_sl * xi_pol
    return out


# O6. s672_geom : per-interferer sum, ITU-R S.672 envelope
def _s672_envelope_db(phi_deg: np.ndarray,
                      theta_3db_deg: float,
                      L_S_db: float,
                      g_max_db: float = 0.0) -> np.ndarray:
    """ITU-R S.672-4 reference pattern (relative to G_max, in dB)."""
    a, b = 2.58, 6.32   # mid sidelobe class (-25 dB). Exact table-1 values vary.
    if L_S_db <= -29.0:
        a, b = 2.88, 7.94    # -30 dB class
    elif L_S_db >= -21.0:
        a, b = 2.28, 5.01    # -20 dB class
    x = np.maximum(phi_deg, 1e-9) / max(theta_3db_deg, 1e-9)
    out = np.full_like(x, g_max_db - 10.0)             # far floor at -10 dBi rel
    mask_main = x <= a
    mask_flat = (x > a) & (x <= 0.5 * b)
    mask_roll = (x > 0.5 * b) & (x <= b)
    out[mask_main] = g_max_db - 3.0 * (x[mask_main] ** 2)
    out[mask_flat] = g_max_db + L_S_db
    out[mask_roll] = g_max_db + L_S_db + 20.0 - 25.0 * np.log10(x[mask_roll])
    return out


@register_ext_mode("s672_geom")
def _mode_s672_geom(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    """Per-(user, served-sat) row, sum the nominal contribution of every OTHER
    visible sat of the same constellation through an S.672 envelope.

    Off-axis angle source is configurable:
      * "geom"       -> real ECEF unit-vector angle  s' -> j(s')   vs  s' -> u
                        where j(s') is the user CURRENTLY assigned to sat s'.
                        Falls back to |Δelev| if assignments are unavailable.
      * "elev_proxy" -> cheap |elev_served - elev_interferer| approximation.

    Frequency-reuse colouring chooses which interferers actually share u's RB:
      * "mean_field"  (statistical, divide aggregate by K_r)
      * "id_mod_kr"   (deterministic, sat_id mod K_r)
      * "greedy_elev" (highest-elev first, cycle through K_r colours)
      * "dsatur"      (Brélaz DSATUR on per-tick beam-conflict graph - optimal-
                       in-practice for sparse satellite footprint graphs)
    Polarisation isolation `xi_pol` is now a knob (cfg.ext_pol_iso_db).
    """
    L_S    = float(ctx["knob"])
    K_r    = max(int(ctx["K_r"]), 1)
    theta3 = float(ctx.get("theta_3db_deg", 4.0))
    xi_pol = 10.0 ** (-float(ctx["pol_iso_db"]) / 10.0)
    offax  = str(ctx.get("offaxis_mode", "geom")).lower()
    sep_thresh = float(ctx.get("conflict_sep_deg", 1.5))
    region_km  = float(ctx.get("region_radius_km", 0.0))   # 0 = no beam-region filter

    n   = len(links)
    out = np.zeros(n, dtype=float)
    if "user_id" not in links.columns or "node_type" not in links.columns:
        return out

    is_sat = (links["node_type"].astype(str) == "satellite").to_numpy()
    user   = links["user_id"].astype(int).to_numpy()
    sat_id = links["node_id"].astype(int).to_numpy()
    elev   = links.get("elev_deg",  pd.Series(np.zeros(n))).astype(float).to_numpy()
    ulat   = links.get("user_lat",  pd.Series(np.zeros(n))).astype(float).to_numpy()
    ulon   = links.get("user_lon",  pd.Series(np.zeros(n))).astype(float).to_numpy()
    nlat   = links.get("node_lat",  pd.Series(np.zeros(n))).astype(float).to_numpy()
    nlon   = links.get("node_lon",  pd.Series(np.zeros(n))).astype(float).to_numpy()
    nalt   = links.get("alt_m",     pd.Series(np.zeros(n))).astype(float).to_numpy()
    c_nom  = ctx["c_nom_w"]
    assigns = ctx.get("assignments", {}) or {}

    # Pre-compute ECEF for all rows once.
    ux, uy, uz = _lla_to_ecef(ulat, ulon, np.zeros_like(ulat))   # user on ground
    sx, sy, sz = _lla_to_ecef(nlat, nlon, nalt)                  # sat sub-point

    # Per-user index of all visible sat rows
    # Vectorised: pandas groupby.groups is C-level → ~10× faster than
    # Python setdefault loop. Returns Index objects keyed by user_id.
    sat_rows = np.where(is_sat)[0]
    if len(sat_rows):
        sat_df = pd.DataFrame({"u": user[sat_rows]}, index=sat_rows)
        by_user = {int(k): list(v) for k, v
                   in sat_df.groupby("u", sort=False).groups.items()}
    else:
        by_user = {}

    # Per-sat -> (user_id served by that sat) for the off-axis lookup
    sat_to_servee_uid = {int(sid): int(uid) for uid, sid in assigns.items()}

    # Pre-build {user_id -> first row index} once (vectorised)
    sentinel = np.iinfo(np.int64).max
    # np.minimum.reduceat would be ideal but `user` isn't sorted; build via
    # pandas, much faster than the Python setdefault loop.
    uid_first_row = (pd.Series(np.arange(n, dtype=np.int64))
                       .groupby(user, sort=False).min().to_dict())

    # Pre-extract ctx fields used inside the per-user loop
    colouring     = ctx["colouring"]
    cell_colour   = ctx.get("cell_colour", {}) or {}
    sat_to_servee = ctx.get("sat_to_servee_uid", {}) or {}

    # one PASS PER USER
    # The original code looped per ROW (per (user, served-sat) candidate).
    # For each victim user u the off-axis angles at the V interferers and
    # the per-interferer colour ρ are independent of WHICH served-sat row
    # within u we're computing for. So we compute them ONCE per user, then
    # broadcast across all V served-sat candidates via a V×V matrix.
    # Complexity drops from O(N·V) to O(N + V²·U) - at V=30, U=159 that is
    # 159·900 ≈ 143 k ops total instead of 159·30·30 ≈ 143 k *Python*
    # iterations. Net effect is ~10× speedup on the s672 path.
    max_int = int(ctx.get("max_interferers", 0) or 0)   # 0 = no cap

    # FULLY-VECTORISED FAST PATH (no per-user Python loop)
    # When there is no interferer cap and we use real geometry ("geom"), the
    # whole tick's interference is one pass over every satellite link-row:
    #  for row r = (victim user u, interferer sat j):
    #  phi_r = ∠ at j between (j→its served user) and (j→u)
    #  term_r = c_nom_r · g_S672(phi_r) · ρ_r · ξ_pol
    #  I(u) = Σ_{rows of u} term_r; out[r] = I(u_r) − term_r (self-exclude)
    # This is exactly the per-user loop's result (verified to machine epsilon),
    # but replaces ~U Python iterations/tick with vectorised numpy over N rows.
    if max_int == 0 and offax != "elev_proxy" and len(sat_rows) > 0:
        R = sat_rows
        spx, spy, spz = sx[R], sy[R], sz[R]
        # served user of each interferer sat j (its boresight target)
        served_uid = np.fromiter(
            (sat_to_servee_uid.get(int(s), -1) for s in sat_id[R]),
            dtype=np.int64, count=len(R))
        served_row = np.fromiter(
            (uid_first_row.get(int(su), -1) for su in served_uid),
            dtype=np.int64, count=len(R))
        has_t = served_row >= 0
        safe  = np.where(has_t, served_row, 0)
        tgt_x = ux[safe].copy(); tgt_y = uy[safe].copy(); tgt_z = uz[safe].copy()
        if (~has_t).any():
            fb = ~has_t
            fbx, fby, fbz = _lla_to_ecef(nlat[R][fb], nlon[R][fb],
                                         np.zeros(int(fb.sum())))
            tgt_x[fb] = fbx; tgt_y[fb] = fby; tgt_z[fb] = fbz
        v1 = np.stack([tgt_x - spx, tgt_y - spy, tgt_z - spz], axis=1)   # j→served
        v2 = np.stack([ux[R] - spx, uy[R] - spy, uz[R] - spz], axis=1)   # j→victim
        phi  = _angle_between_deg(v1, v2)
        glin = 10.0 ** (_s672_envelope_db(phi, theta3, L_S, 0.0) / 10.0)
        if colouring == "mean_field":
            rho = np.full(len(R), 1.0 / K_r, dtype=float)
        else:
            u_col = np.fromiter((cell_colour.get(int(u), -1) for u in user[R]),
                                dtype=np.int64, count=len(R))
            j_col = np.fromiter((cell_colour.get(int(su), -1) for su in served_uid),
                                dtype=np.int64, count=len(R))
            rho = np.where(u_col == -1, 1.0 / K_r,
                           np.where(j_col == -1, 1.0 / K_r,
                                    (j_col == u_col).astype(float)))
        term = c_nom[R] * glin * rho * xi_pol
        # Beam-in-region filter: a satellite interferes only if the beam it
        # serves lands within `region_km` of the victim user. Far beams (their
        # served user is elsewhere) are steered away and don't hit this region.
        # Cuts the over-counting that crushes dense-constellation SINR.
        if region_km > 0.0:
            duv = np.sqrt((ux[R]-tgt_x)**2 + (uy[R]-tgt_y)**2 + (uz[R]-tgt_z)**2)
            term = np.where(duv <= region_km*1000.0, term, 0.0)
        uniq, inv = np.unique(user[R], return_inverse=True)
        full = np.zeros(len(uniq), dtype=float)
        np.add.at(full, inv, term)
        out[R] = full[inv] - term            # I(u) − self (self j=i excluded)
        return out

    for u_id, idxs in by_user.items():
        V = len(idxs)
        if V <= 1:
            continue
        idxs_arr = np.asarray(idxs, dtype=int)

        # Performance + physics cap: a victim's interference is dominated by the
        # few strongest co-channel neighbours; the V×V cost is O(V²). Keep only
        # the top-`max_int` interferers by nominal power C_nom (the weak far
        # ones contribute negligibly). Bounds cost to O(max_int²) for dense
        # constellations (Starlink V≈97) with <0.1 dB error.
        if max_int and V > max_int:
            keep = np.argsort(c_nom[idxs_arr])[-max_int:]
            idxs_arr = idxs_arr[keep]
            V = len(idxs_arr)

        # Pick a row for the victim user u (used only for ECEF; all rows of
        # a user share the same user_lat/user_lon, so any row is fine).
        u_row = int(idxs_arr[0])

        # off-axis angle phi_j at each interferer j toward victim u
        if offax == "elev_proxy":
            # symmetric V×V matrix of elev differences
            elev_u   = elev[idxs_arr]                         # (V,)
            phi_mat  = np.abs(elev_u[:, None] - elev_u[None, :])
        else:
            # phi depends only on interferer j (not on the served-sat i),
            # so compute the V-vector once and broadcast across rows.
            sat_pos = np.stack([sx[idxs_arr], sy[idxs_arr],
                                 sz[idxs_arr]], axis=1)        # (V, 3)
            v_pos   = np.array([ux[u_row], uy[u_row], uz[u_row]])[None, :]
            # Target user position per interferer (j's served user; fallback
            # to sat sub-point on ground when assignment is unknown).
            served_uids = np.fromiter(
                (sat_to_servee.get(int(s), -1) for s in sat_id[idxs_arr]),
                dtype=np.int64, count=V)
            target_rows = np.fromiter(
                (uid_first_row.get(int(su), -1) for su in served_uids),
                dtype=np.int64, count=V)
            has_target = target_rows >= 0
            t_pos = np.empty((V, 3), dtype=float)
            if has_target.any():
                rs = target_rows[has_target]
                t_pos[has_target] = np.stack([ux[rs], uy[rs], uz[rs]], axis=1)
            need_fb = ~has_target
            if need_fb.any():
                fb_lat = nlat[idxs_arr[need_fb]]
                fb_lon = nlon[idxs_arr[need_fb]]
                fbx, fby, fbz = _lla_to_ecef(
                    fb_lat, fb_lon, np.zeros_like(fb_lat))
                t_pos[need_fb] = np.stack([fbx, fby, fbz], axis=1)
            v1_vec = t_pos - sat_pos                            # (V, 3)
            v2_vec = v_pos - sat_pos                            # (V, 3)
            phi_vec = _angle_between_deg(v1_vec, v2_vec)         # (V,)
            # (geom path uses phi_vec directly - see the factorised O(V) reduction
            #  below; no V×V matrix is built.)

        # colouring ρ (only j-dependent)
        if colouring == "mean_field":
            rho_vec = np.full(V, 1.0 / K_r, dtype=float)
        else:
            u_colour = cell_colour.get(int(u_id), -1)
            served_uids2 = np.fromiter(
                (sat_to_servee.get(int(s), -1) for s in sat_id[idxs_arr]),
                dtype=np.int64, count=V)
            j_cols = np.fromiter(
                (cell_colour.get(int(uu), -1) for uu in served_uids2),
                dtype=np.int64, count=V)
            if u_colour == -1:
                rho_vec = np.full(V, 1.0 / K_r, dtype=float)
            else:
                rho_vec = np.where(
                    j_cols == -1, 1.0 / K_r,
                    (j_cols == u_colour).astype(float))

        c_nom_u = c_nom[idxs_arr]                               # (V,)
        if offax == "elev_proxy":
            # phi depends on both i and j ⇒ a genuine V×V matrix is required.
            gdb_mat  = _s672_envelope_db(phi_mat, theta3, L_S, 0.0)
            glin_mat = 10.0 ** (gdb_mat / 10.0)
            np.fill_diagonal(glin_mat, 0.0)                      # self-mask
            contrib  = (c_nom_u[None, :] * glin_mat
                        * rho_vec[None, :]) * xi_pol
            out[idxs_arr] = contrib.sum(axis=1)
        else:
            # FACTORISED O(V²) → O(V). In "geom" mode phi depends only on the
            # interferer j, so the S.672 gain is a rank-1 vector and the V×V
            # interference sum collapses exactly to (Σ_j term_j) − term_i:
            #  term_j = c_nom_j · g(phi_j) · ρ_j · ξ_pol
            #  I(i) = Σ_j term_j − term_i (the −term_i drops self j=i,
            #  == the old fill_diagonal(0))
            # Also evaluates the S.672 envelope V times instead of V² times.
            gdb_vec  = _s672_envelope_db(phi_vec, theta3, L_S, 0.0)
            glin_vec = 10.0 ** (gdb_vec / 10.0)
            term     = c_nom_u * glin_vec * rho_vec * xi_pol      # (V,)
            out[idxs_arr] = term.sum() - term

    return out


# O7. lognormal : 10*log10(I/N) ~ N(knob, sigma=4 dB)
@register_ext_mode("lognormal")
def _mode_lognormal(links: pd.DataFrame, ctx: dict) -> np.ndarray:
    rng   = ctx["rng"]
    z     = rng.standard_normal(len(links))
    in_db = ctx["knob"] + 4.0 * z
    return ctx["noise_w"] * (10.0 ** (in_db / 10.0))


# The Stage
@register_stage("external_interference_const", internal=True)
class ExternalInterferenceConst(Stage):
    """Pre-solver external-interference computer.

    Place this stage after `geometry` + `channel` (so atmospheric
    columns exist) and before the optimizer / `interference` / `scn_radio`
    stages. The written `i_ext_w` is consumed by the SINR aggregator (and
    your CVX builder) as a constant Parameter.
    """
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        ic = self.cfg.defaults_interference
        mode = str(getattr(ic, "ext_mode", "off")).lower()
        if mode not in _EXT_MODES:
            _log.warning("ext_mode='%s' not registered. Using 'off'. "
                         "Available: %s", mode, list_ext_interference_modes())
            mode = "off"

        # Per-row noise N0B
        T_sys = (float(self.cfg.defaults_radio.system_temperature_k)
                 * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        bw = links.get("bandwidth_hz",
                       pd.Series(np.full(len(links), 20e6))
                       ).astype(float).to_numpy()
        noise_w = kB * T_sys * np.maximum(bw, 1.0)

        # Per-row nominal C (geometric link budget, no h/w)
        rx_gain_max_db = float(getattr(self.cfg.defaults_radio,
                                        "rx_antenna_gain_db", 0.0))
        rx_gain_max_lin = np.full(len(links), 10.0 ** (rx_gain_max_db / 10.0))
        # Interferer EIRP cap: use a realistic per-beam power for the interferers
        # (decoupled from the wanted-signal power_max_w, which a no-cap min-power
        # study may set huge). 0 = no separate cap.
        _icap = float(getattr(ic, "ext_interferer_pmax_w", 0.0) or 0.0)
        op_pw = {int(op.id): (min(float(getattr(op.radio, "power_max_w", 1.0)), _icap)
                              if _icap > 0.0 else float(getattr(op.radio, "power_max_w", 1.0)))
                 for op in self.cfg.operators}
        c_nom_w = _nominal_c_per_link(links, rx_gain_max_lin, op_pw)

        # Per-row #visible sats for the same user
        if "user_id" in links.columns and "node_type" in links.columns:
            is_sat = (links["node_type"].astype(str) == "satellite")
            vis_count = (links.loc[is_sat]
                         .groupby("user_id")["node_id"].transform("size"))
            n_vis = np.ones(len(links), dtype=float)
            n_vis[is_sat.to_numpy()] = vis_count.to_numpy()
        else:
            n_vis = np.ones(len(links), dtype=float)

        # Deterministic RNG (per-tick). global_seed may be None (unset) - fall
        # back to a fixed default so the RNG is still deterministic.
        seed = int(getattr(self.cfg.experiment, "global_seed", None) or 1234)
        try:
            t_idx = int(getattr(state, "t_idx", 0))
        except Exception:
            t_idx = 0
        rng = np.random.default_rng(seed + 7919 * t_idx)

        # Per-tick cell colouring (CELLS, not satellites!)
        # Build the active-users table from `assignments`: every user with a
        # served sat is an active cell this tick. Colour them once and pass
        # the {user_id -> colour} map into ctx for O(1) per-row lookup.
        assignments = state.data.get("assignments", {}) or {}
        colouring   = str(getattr(ic, "ext_colouring", "mean_field"))
        K_r         = int(getattr(ic, "ext_K_r", 1))
        sep_thresh  = float(getattr(ic, "ext_conflict_sep_deg", 1.5))
        cell_colour = {}
        if colouring != "mean_field" and assignments:
            # Pull lat/lon for the active users from any link row.
            uid_to_pos: dict[int, tuple[float, float]] = {}
            if {"user_id", "user_lat", "user_lon"}.issubset(links.columns):
                for r in links[["user_id", "user_lat", "user_lon"]].itertuples(index=False):
                    uid = int(r.user_id)
                    if uid not in uid_to_pos:
                        uid_to_pos[uid] = (float(r.user_lat), float(r.user_lon))
            active_uids = [int(u) for u in assignments.keys() if int(u) in uid_to_pos]
            if active_uids:
                au = pd.DataFrame({
                    "user_id":  active_uids,
                    "user_lat": [uid_to_pos[u][0] for u in active_uids],
                    "user_lon": [uid_to_pos[u][1] for u in active_uids],
                })
                cell_colour = _compute_cell_colouring(
                    au, colouring, K_r, sep_thresh)

        # Reverse lookup: served satellite -> served user
        sat_to_servee_uid = {int(sid): int(uid) for uid, sid in assignments.items()}

        ctx = dict(
            knob              = float(getattr(ic, "ext_knob", 0.0)),
            K_r               = K_r,
            noise_w           = noise_w,
            c_nom_w           = c_nom_w,
            n_visible_per_link= n_vis,
            rng               = rng,
            # Newly exposed knobs
            pol_iso_db        = float(getattr(ic, "ext_pol_iso_db", 12.0)),
            offaxis_mode      = str(getattr(ic, "ext_offaxis_mode", "geom")),
            theta_3db_deg     = float(getattr(ic, "ext_sat_theta_3db_deg", 4.0)),
            colouring         = colouring,
            conflict_sep_deg  = sep_thresh,
            max_interferers   = int(getattr(ic, "ext_max_interferers", 0)),
            region_radius_km  = float(getattr(ic, "ext_region_radius_km", 0.0)),
            # Assignments & cell colouring (CELLS, not sats)
            assignments       = assignments,
            cell_colour       = cell_colour,
            sat_to_servee_uid = sat_to_servee_uid,
        )

        i_ext_w = _EXT_MODES[mode](links, ctx)
        i_ext_w = np.clip(np.asarray(i_ext_w, dtype=float), 0.0, None)

        # Audit columns
        with np.errstate(divide="ignore", invalid="ignore"):
            ion_db = 10.0 * np.log10(
                np.where(noise_w > 0,
                         np.maximum(i_ext_w, 1e-300) / np.maximum(noise_w, 1e-300),
                         np.nan))
        out = links.copy()
        out["i_ext_w"]       = i_ext_w
        out["i_over_n_db"]   = ion_db
        out["i_ext_mode"]    = mode
        out["n_interferers"] = (n_vis - 1.0).clip(min=0).astype(int)
        state.data["links"]  = out

        # Interference TRACE (gated by SKYNTN_TRACE_INT=1)
        # Dumps the breakdown for the first few served ticks so the s672/reuse
        # model can be debugged: #interferers, colouring K_r, per-interferer
        # contribution, and how the aggregate compares to noise.
        import os as _os
        if _os.environ.get("SKYNTN_TRACE_INT") == "1":
            t_idx = int(getattr(state, "t_idx", 0))
            if t_idx < 3:
                served = out.loc[i_ext_w > 0] if (i_ext_w > 0).any() else out
                nint = out["n_interferers"]
                _log.warning(
                    "[INT-TRACE t=%d] mode=%s K_r=%s colouring=%s offaxis=%s | "
                    "links=%d  n_interferers(median/max)=%d/%d  "
                    "i_ext_w(median)=%.3e  noise_w=%.3e  I/N(median)=%.1f dB  "
                    "rows_with_iext=%d",
                    t_idx, mode, K_r, colouring,
                    getattr(ic, "ext_offaxis_mode", "?"),
                    len(out), int(nint.median()), int(nint.max()),
                    float(np.median(i_ext_w[i_ext_w > 0]) if (i_ext_w > 0).any() else 0.0),
                    float(np.median(noise_w)),
                    float(np.nanmedian(ion_db[np.isfinite(ion_db)]) if np.isfinite(ion_db).any() else float("nan")),
                    int((i_ext_w > 0).sum()))
