# Stage: handover dispatch. Per-scope controller via control_plane, else cfg.handover.mode.

from __future__ import annotations
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.registry.agents.handover_strategy import (
    get_handover_strategy, HandoverContext,
)


def _cell_center_links(links, cluster_links):
    """cell_center mode: replace each user's link geometry with its cell-centroid
    geometry from cluster_links (join on cluster_id + node_id). Users in a cell
    then share the cell-center geometry, so the cell is treated as one point."""
    if (not isinstance(cluster_links, pd.DataFrame) or cluster_links.empty
            or "cluster_id" not in links.columns):
        return links
    geo = ["dist_m", "ground_dist_m", "elev_deg", "azimuth", "azimuth_deg",
           "range_rate_m_s", "range_rate_km_s",
           "remaining_visibility_s", "remaining_visibility_ticks"]
    have = [c for c in geo if c in cluster_links.columns]
    cc = cluster_links[["cluster_id", "node_id"] + have].rename(
        columns={c: c + "__cc" for c in have})
    m = links.merge(cc, on=["cluster_id", "node_id"], how="left")
    for c in have:
        cc_col = c + "__cc"
        m[c] = m[cc_col].where(m[cc_col].notna(), m[c]) if c in m.columns else m[cc_col]
        m.drop(columns=[cc_col], inplace=True)
    return m


def _build_ctx(cfg, links, prev_assign_by_user_id, mask_rows=None) -> HandoverContext:
    """Build a HandoverContext.

    note: prev_assign_by_user_id maps user_id -> CATALOG node_id (what is
    stored in state.data["assignments"]). Strategies, however, treat
    prev_assign[uid] as a ROW INDEX into vis_mat. We translate here.
    """
    if mask_rows is not None:
        links = links[mask_rows]
    users = sorted(set(int(u) for u in links["user_id"].unique()))
    nodes = (links[["node_id", "node_type", "operator_id"]]
             .drop_duplicates().sort_values("node_id").reset_index(drop=True))
    node_ids   = nodes["node_id"].values.astype(int)
    node_types = nodes["node_type"].values.astype(object)
    node_ops   = nodes["operator_id"].values.astype(int)

    u_idx = {u: i for i, u in enumerate(users)}
    n_idx = {nid: i for i, nid in enumerate(node_ids)}
    n_user, n_node = len(users), len(node_ids)
    vis  = np.zeros((n_node, n_user))
    dist = np.full((n_node, n_user), np.inf)
    elev = np.zeros((n_node, n_user))
    rate = np.zeros((n_node, n_user))

    # Vectorized fill via fancy indexing
    ui_arr = links["user_id"].astype(int).map(u_idx).to_numpy()
    ni_arr = links["node_id"].astype(int).map(n_idx).to_numpy()
    vis[ni_arr,  ui_arr] = 1
    dist[ni_arr, ui_arr] = links["dist_m"].astype(float).to_numpy()
    if "elev_deg" in links.columns:
        elev[ni_arr, ui_arr] = links["elev_deg"].astype(float).to_numpy()
    if "range_rate_km_s" in links.columns:
        rate[ni_arr, ui_arr] = links["range_rate_km_s"].astype(float).to_numpy()

    # Predicted remaining visibility (seconds) per (node, user). NaN by
    # default. Populated only when the link_visibility_predict stage ran
    # earlier this tick and wrote the column to `links`.
    remain = np.full((n_node, n_user), np.nan, dtype=float)
    if "remaining_visibility_s" in links.columns:
        remain[ni_arr, ui_arr] = links["remaining_visibility_s"].astype(float).to_numpy()

    # Aggregate co-channel interference beta_{u,s} [W] per (node, user), if the
    # external_interference stage has already written `i_ext_w` this tick.
    beta = None
    if "i_ext_w" in links.columns:
        beta = np.zeros((n_node, n_user), dtype=float)
        beta[ni_arr, ui_arr] = links["i_ext_w"].astype(float).to_numpy()

    # Actual large-scale channel gain g_bar = 10^((G - PL)/10) per (node, user),
    # if the channel stage has already written path_loss_db this tick (channel-
    # first ordering). Atmospheric loss is included in path_loss_db.
    gain = None
    if "path_loss_db" in links.columns:
        gain = np.zeros((n_node, n_user), dtype=float)
        _gdb = links.get("antenna_gain_db", 0.0)
        _gdb = (_gdb.astype(float).to_numpy() if hasattr(_gdb, "astype")
                else np.zeros(len(links)))
        _pl = links["path_loss_db"].astype(float).to_numpy()
        gain[ni_arr, ui_arr] = 10.0 ** ((_gdb - _pl) / 10.0)

    # Translate catalog ids in prev_assign to row indices visible in vis_mat.
    # Users whose previous serving sat is not in this slice get -1 (no STAY).
    prev_idx = {int(u): int(n_idx.get(int(nid), -1))
                for u, nid in (prev_assign_by_user_id or {}).items()
                if int(nid) >= 0}

    # Optional: cluster_id per user (populated by clustering stage and
    # carried into the links frame by geometry). None when clustering
    # is not in use, in which case cluster-aware strategies degrade to
    # per-user behaviour.
    user_cluster_ids = None
    if "cluster_id" in links.columns:
        cid_map = {int(u): int(c) for u, c
                   in zip(links["user_id"].astype(int),
                          links["cluster_id"].astype(int))}
        user_cluster_ids = np.array([cid_map.get(int(u), -1) for u in users],
                                    dtype=np.int32)

    return HandoverContext(
        t_idx=0, timestamp=None,
        users=np.array(users, dtype=int),
        node_ids=node_ids, node_types=node_types, node_op_ids=node_ops,
        vis_mat=vis, dist_mat=dist, elev_mat=elev, rate_mat=rate,
        remain_mat=remain, beta_mat=beta, gain_mat=gain,
        prev_assign=prev_idx, cfg=cfg,
        user_cluster_ids=user_cluster_ids,
    )


@register_stage("handover")
class HandoverStage(Stage):
    reads  = ["links", "control_plane"]
    writes = ["assignments", "prev_assign_snapshot"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._prev = {}
        from ntnconnect.engine.stages.csi_buffer import CSIBuffer
        self._csi_buf = CSIBuffer(cfg)      # central-agent CSI aging (association)
        # Guaranteed/best-effort phase of the current (held) association, kept so
        # it persists across the iCSI sub-slots between handover decisions - the
        # association is held run_every_n_ticks ticks, so its phase flag must be
        # too, else assoc_guaranteed reads 0 on the non-decision ticks.
        self._prev_phase = {}
        # Legacy fallback strategy if no control_plane
        self._fallback = get_handover_strategy(cfg)
        # Elevation mask used to decide whether a HELD serving link is still
        # usable between periodic decisions (event-triggered handover).
        try:
            self._mask_deg = float(getattr(
                getattr(cfg, "constraints", None), "min_elevation_deg",
                getattr(cfg.handover, "min_elevation_deg", 20.0)))
        except Exception:
            self._mask_deg = 20.0

    def _assignment_stable(self, links) -> bool:
        """True iff NO re-decision is needed this tick. We re-decide (return
        False) on either event, so service is never left waiting for the next
        periodic decision:
          (a) a HELD serving satellite dropped below the mask (set / lost), OR
          (b) a user that has a satellite in view is currently UNSERVED (a
              reconnection opportunity). This is the key fix: once a user goes
              to -1 it is no longer a 'held' link, so (a) alone would leave it
              unserved until the next 5 s decision - the systemic gap seen
              across all strategies. (b) reconnects it the same tick."""
        if "elev_deg" not in links.columns or "user_id" not in links.columns \
           or "node_id" not in links.columns:
            return True
        ev = links["elev_deg"].astype(float).to_numpy()
        keep = ev >= self._mask_deg
        vis_users = links["user_id"].astype(int).to_numpy()[keep]
        vis_nodes = links["node_id"].astype(int).to_numpy()[keep]
        valid = set(zip(vis_users, vis_nodes))
        # (a) any held serving sat no longer visible?
        for u, s in self._prev.items():
            if int(s) >= 0 and (int(u), int(s)) not in valid:
                return False
        # (b) any user with a visible satellite that is not currently served?
        served = {int(u) for u, s in self._prev.items() if int(s) >= 0}
        for u in np.unique(vis_users):
            if int(u) not in served:
                return False
        return True

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            state.data["assignments"] = {}
            state.data["prev_assign_snapshot"] = dict(self._prev)
            return

        # cell_center mode: associate on the cell-centroid geometry (cluster_links)
        # instead of the per-user links. Default "users" leaves links unchanged.
        if str(getattr(self.cfg.users, "cluster_geometry", "users")).lower() == "cell_center":
            links = _cell_center_links(links, state.data.get("cluster_links"))

        # Online CSI-update timing (central agent): the association decides on the
        # realized channel (path_loss_db) aged by each user's report delay, unless
        # handover_assume_instant (agent omniscient). Local copy only - NOT stored,
        # so grading downstream uses the true channel. No-op at coarse slots.
        if not bool(getattr(self.cfg.csi_timing, "handover_assume_instant", False)) \
                and "path_loss_db" in links.columns:
            links = links.copy()
            links["path_loss_db"] = self._csi_buf.aged(
                links, int(getattr(state, "t_idx", 0)), "path_loss_db")

        # Three-time-scale support: skip HO unless this is a "HO tick"
        # Optimisation cadence is periodic (every run_every_n_ticks), BUT a
        # handover is also triggered immediately whenever a held serving link
        # drops below the mask. Holding a set satellite until the next periodic
        # decision would leave the user unserved for up to one period even
        # though other satellites are in view - a spurious service gap. With
        # coverage available, event-triggered re-decision closes that gap.
        run_every = int(getattr(self.cfg.handover, "run_every_n_ticks", 1) or 1)
        if run_every > 1 and (state.t_idx % run_every) != 0 \
           and self._assignment_stable(links):
            # Held links all still valid -> carry the assignment forward, AND
            # carry its guaranteed/best-effort phase forward too (same hold).
            state.data["assignments"]          = dict(self._prev)
            state.data["prev_assign_snapshot"] = dict(self._prev)
            state.data["beam_assignments"]     = {}
            state.data["assignment_phase"]     = dict(self._prev_phase)
            # GENIE bounds: the flag must also hold across the HELD sub-slots, else
            # min_power_miso re-optimizes on real CSI only at the slot boundary and
            # falls back to the outage reserve for the other (run_every-1) sub-slots.
            # state.data is fresh each tick, so re-assert it here every sub-slot.
            state.data["genie_power"] = bool(getattr(self._fallback, "genie_power", False))
            return
        # Otherwise (periodic HO tick, or a held link was lost) fall through and
        # re-decide this tick so every user is (re)assigned to a visible sat.

        plane = state.data.get("control_plane")
        prev_snapshot = dict(self._prev)
        assignments = {}
        beam_assignments: dict = {}     # user_id -> beam_id (when beam_tagged)
        assoc_phase: dict = {}          # user_id -> 1 (guaranteed) | 0 (best-effort/rectify)

        if plane is None or not plane.controllers_owning("handover"):
            # Legacy single-strategy path
            ctx = _build_ctx(self.cfg, links, self._prev)
            out = self._fallback.assign(ctx)
            if getattr(ctx, "out_assoc_phase", None):     # guaranteed/best-effort flag
                assoc_phase.update(ctx.out_assoc_phase)
            # `out[i]` is a ROW INDEX into ctx.node_ids (the vis_mat row), not
            # a catalog node_id. Translate to the catalog node_id so that
            # downstream consumers (paper_sinr served-mask, visibility_log
            # SINR snapshot, ISL feeder lookup) match on links["node_id"].
            for i, u in enumerate(ctx.users):
                sel = int(out[i])
                if 0 <= sel < len(ctx.node_ids):
                    assignments[int(u)] = int(ctx.node_ids[sel])
                else:
                    assignments[int(u)] = -1
        else:
            # Per-controller dispatch
            for c in plane.controllers:
                if not c.owns("handover"): continue
                strat = c.strategy("handover")
                if strat is None: continue
                # Slice links to nodes whose (operator_id, node_type) is
                # covered by this controller's scope.
                # Vectorized scope match: dedupe (op_id, node_type) pairs,
                # call matches() once per pair, broadcast back to row mask.
                op_arr = links["operator_id"].astype(int).to_numpy()
                nt_arr = links["node_type"].astype(str).to_numpy()
                pairs = pd.DataFrame({"op": op_arr, "nt": nt_arr}) \
                          .drop_duplicates()
                allowed = {(int(r.op), str(r.nt))
                           for r in pairs.itertuples(index=False)
                           if c.scope.matches(int(r.op), str(r.nt))}
                mask = np.array([(int(o), str(t)) in allowed
                                 for o, t in zip(op_arr, nt_arr)])
                if not mask.any(): continue
                sub_ctx = _build_ctx(self.cfg, links, self._prev, mask_rows=mask)
                out = strat.assign(sub_ctx)
                # Cluster-aware strategies in 'beam_tagged' mode populate
                # sub_ctx.out_beam_ids during assign(); merge into the global
                # beam map keyed by user_id.
                if getattr(sub_ctx, "out_beam_ids", None):
                    beam_assignments.update(sub_ctx.out_beam_ids)
                if getattr(sub_ctx, "out_assoc_phase", None):
                    assoc_phase.update(sub_ctx.out_assoc_phase)
                # Map per-slice node_idx back to a stable token. To keep this
                # cross-controller-merge-safe we record (controller_id, sat_idx)
                # tuples; downstream consumers resolve via node_ids in
                # sub_ctx if needed. For now we just keep the global node_idx
                # by looking it up in the unsliced links table.
                node_id_for_slice_idx = sub_ctx.node_ids
                for i, u in enumerate(sub_ctx.users):
                    sel = int(out[i])
                    if sel < 0:
                        assignments.setdefault(int(u), -1)
                        continue
                    chosen_node_id = int(node_id_for_slice_idx[sel])
                    # Resolve conflicts via arbiter: if user already has an
                    # assignment from a different controller, ask the arbiter.
                    if u in assignments and assignments[int(u)] != chosen_node_id \
                       and assignments[int(u)] >= 0:
                        # Minimal arbiter: accept this controller's decision
                        # only if no higher-priority handover controller
                        # exists. (Full arbiter call deferred until per-user
                        # controller-of-record is tracked.)
                        if any(prev_c.priority > c.priority for prev_c
                               in plane.controllers
                               if prev_c is not c and prev_c.owns("handover")):
                            continue
                    assignments[int(u)] = chosen_node_id

        state.data["assignments"] = assignments
        state.data["prev_assign_snapshot"] = prev_snapshot
        state.data["beam_assignments"] = beam_assignments
        state.data["assignment_phase"] = assoc_phase
        # GENIE bound strategies (B-*): flag min_power_miso to size power on the real
        # per-sub-slot channel (perfect CSI, no outage reserve) -> 0 outage, min power.
        state.data["genie_power"] = bool(getattr(self._fallback, "genie_power", False))
        self._prev_phase = dict(assoc_phase)      # persist for the held sub-slots
        # per-sat DT cold-start on handover. When user u moves
        # from sat A to sat B, release u from A so A's predictor stops
        # tracking it; B will cold-start on the next tick automatically.
        eng_sats = getattr(getattr(self, "engine", None), "satellites", None)
        if eng_sats:
            for u, prev_sat in prev_snapshot.items():
                new_sat = assignments.get(int(u), -1)
                if int(prev_sat) >= 0 and int(prev_sat) != int(new_sat):
                    sat_obj = eng_sats.get(int(prev_sat))
                    if sat_obj is not None:
                        sat_obj.release_user(int(u))
        self._prev = {u: s for u, s in assignments.items() if s >= 0}
