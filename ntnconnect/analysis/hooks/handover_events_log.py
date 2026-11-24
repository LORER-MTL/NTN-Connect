# Hook: handover_events_log
#
# Detects handover events by diffing state.data["assignments"] vs
# state.data["prev_assign_snapshot"] every tick and records:
#
#  handover_events.parquet:
#  t_idx, user_id, from_sat_id, to_sat_id,
#  from_elev_deg, to_elev_deg, trigger_type,
#  ho_success (bool), service_gap_s,
#  user_lat, user_lon -> for geographic plots
#
# trigger_type heuristic (from cfg.handover.handover_strategy when available,
# else inferred from before/after metrics):
#  * "elev_drop" - from-sat's elev_deg dropped below cfg.handover.elev_min_deg
#  * "sinr_gain" - to-sat had strictly higher predicted SINR than from-sat
#  * "load" - strategy == "load_aware" produced the change
#  * "init" - first assignment for this user (from_sat = -1)
#  * "leave" - assignment lost (to_sat = -1)
#  * "strategy" - default fallback when the cause cannot be inferred
#
# service_gap_s = ticks between losing service and regaining it (only set
# when a user briefly had to_sat = -1 and then re-acquired one).
from __future__ import annotations
import logging
import pandas as pd

from ntnconnect.orchestration.orchestrator import SimulationHook
from ntnconnect.analysis.hooks.registry import register_hook

_log = logging.getLogger(__name__)


@register_hook("handover_events_log")
def _build(engine):
    return HandoverEventsLog(engine)


class HandoverEventsLog(SimulationHook):
    """Per-tick handover-event recorder."""

    def __init__(self, engine):
        self.engine = engine
        self._rows: list[dict] = []
        # Per-user "service-loss-open" event tracker:
        #  user_id -> t_idx when service was lost (to_sat == -1)
        self._loss_open: dict[int, int] = {}
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
        try:
            self._strategy = str(
                getattr(self.engine.cfg.handover, "handover_strategy", "")
                or getattr(self.engine.cfg.handover, "mode", "") or "")
        except Exception:
            self._strategy = ""

    def on_step_end(self, state):
        assigns = state.data.get("assignments", {}) or {}
        prev    = state.data.get("prev_assign_snapshot", {}) or {}
        if not assigns and not prev:
            return
        t_idx = int(state.t_idx)
        links = state.data.get("links")

        # Per-(user_id, sat_id) -> elev_deg lookup; vectorised via dict(zip(...))
        elev_lookup: dict[tuple[int, int], float] = {}
        uid_to_pos: dict[int, tuple[float, float]] = {}
        if isinstance(links, pd.DataFrame) and not links.empty \
           and {"user_id", "node_id", "elev_deg"}.issubset(links.columns):
            u = links["user_id"].astype(int).to_numpy()
            n = links["node_id"].astype(int).to_numpy()
            e = links["elev_deg"].astype(float).to_numpy()
            elev_lookup = dict(zip(zip(u.tolist(), n.tolist()), e.tolist()))
            if {"user_lat", "user_lon"}.issubset(links.columns):
                pos = (links[["user_id", "user_lat", "user_lon"]]
                       .drop_duplicates("user_id"))
                uid_to_pos = dict(zip(
                    pos["user_id"].astype(int).to_numpy().tolist(),
                    zip(pos["user_lat"].astype(float).to_numpy().tolist(),
                        pos["user_lon"].astype(float).to_numpy().tolist())))

        # Build the union of users tracked across prev/cur
        users = set(int(u) for u in assigns) | set(int(u) for u in prev)
        for u in users:
            old = int(prev.get(u, -1))
            new = int(assigns.get(u, -1))
            if old == new:
                # No HO event - but track service-loss/regain edges.
                if new == -1 and u not in self._loss_open:
                    self._loss_open[u] = t_idx
                continue
            # Service loss / regain handling for service_gap_s
            service_gap_s = float("nan")
            if old != -1 and new == -1:
                # Service lost this tick
                self._loss_open[u] = t_idx
            if old == -1 and new != -1 and u in self._loss_open:
                t_lost = self._loss_open.pop(u)
                service_gap_s = float((t_idx - t_lost) * self._dt_s)

            # Trigger inference
            if old == -1 and new != -1:
                trigger = "init"
            elif new == -1:
                trigger = "leave"
            else:
                from_elev = elev_lookup.get((u, old), float("nan"))
                if from_elev == from_elev and from_elev < self._elev_min:
                    trigger = "elev_drop"
                elif "load" in self._strategy.lower():
                    trigger = "load"
                elif self._strategy.lower() in ("max_sinr", "max_rx_power"):
                    trigger = "sinr_gain"
                elif self._strategy:
                    trigger = "strategy"
                else:
                    trigger = "strategy"

            lat, lon = uid_to_pos.get(u, (float("nan"), float("nan")))
            # `is_handover` distinguishes a true HO (sat → sat swap) from
            # bookkeeping events (init / leave). HO-rate analytics should
            # filter on this flag and never recount initial attachments.
            is_handover = bool(old >= 0 and new >= 0 and old != new)
            self._rows.append({
                "t_idx":         t_idx,
                "user_id":       u,
                "from_sat_id":   old,
                "to_sat_id":     new,
                "from_elev_deg": elev_lookup.get((u, old), float("nan")),
                "to_elev_deg":   elev_lookup.get((u, new), float("nan")),
                "trigger_type":  trigger,
                "is_handover":   is_handover,
                "ho_success":    bool(new != -1),
                "service_gap_s": service_gap_s,
                "user_lat":      lat,
                "user_lon":      lon,
            })

    def on_simulation_end(self, state):
        out_dir = self.engine.sim_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        df.to_parquet(out_dir / "handover_events.parquet", index=False)
        _log.info("handover_events_log: %d events -> %s",
                  len(df), out_dir / "handover_events.parquet")
