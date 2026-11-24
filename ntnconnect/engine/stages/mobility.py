# Per-node mobility. Every node type carries a NodeMobility.model:
#   dataset / static / hover -> keep the input position (offline; no-op)
#   gauss_markov / random_waypoint / waypoint / drift -> live terrestrial motion
#   sgp4 / kepler -> satellites propagated live (online, no dataset)
# Position evolves on the stage: state is rebuilt each tick and the input layer
# re-seeds, so we seed a node once then carry its own position forward. This
# keeps mobility node-dependent and works both offline (dataset) and online.
from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_NOOP = {"", "dataset", "static", "hover", "none"}
_TABLE_ATTR = {
    "haps_t": "haps", "uavs_t": "uavs",
    "ground_stations_t": "ground_stations", "base_stations_t": "base_stations",
}


@register_stage("mobility")
class Mobility(Stage):
    """Applies each node type's NodeMobility.model. Missing tables are skipped;
    'dataset'/'static'/'hover' and phase==1 nodes are no-ops (position from the
    input/dataset layer); satellites with 'sgp4'/'kepler' are propagated live."""
    reads  = ["users_t", "satellites_t", "haps_t", "uavs_t",
              "ground_stations_t", "base_stations_t"]
    writes = ["users_t", "satellites_t", "haps_t", "uavs_t",
              "ground_stations_t", "base_stations_t"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pos = {}        # table -> {node_id: (lat, lon)}
        self._wp  = {}        # table -> {node_id: waypoint}
        self._vel = {}        # table -> {node_id: velocity}
        self._satgen = {}     # operator_id -> SatelliteGenerator
        self._t0 = None       # epoch for online propagation (first tick seen)

    def _op_spec(self, key, oid):
        op = self.cfg.operator_by_id(oid)
        return None if op is None else getattr(op, _TABLE_ATTR[key]).mobility

    def _step_rows(self, key, spec, ids, seed_lat, seed_lon, dt, lat_b, lon_b):
        from ntnconnect.scenario.mobility_model import step_users, user_params
        pos = self._pos.setdefault(key, {})
        lat = np.array([pos.get(int(u), (sl, so))[0]
                        for u, sl, so in zip(ids, seed_lat, seed_lon)], dtype=float)
        lon = np.array([pos.get(int(u), (sl, so))[1]
                        for u, sl, so in zip(ids, seed_lat, seed_lon)], dtype=float)
        if spec.model == "drift":
            s = float(spec.drift_deg)
            new_lat = np.clip(lat + np.random.normal(0, s, len(lat)), lat_b[0], lat_b[1])
            new_lon = np.clip(lon + np.random.normal(0, s, len(lon)), lon_b[0], lon_b[1])
        else:
            model = "random_waypoint" if spec.model == "waypoint" else spec.model
            wp = self._wp.setdefault(key, {}); vel = self._vel.setdefault(key, {})
            new_lat, new_lon = step_users(lat, lon, ids, model, user_params(spec),
                                          dt, lat_b, lon_b, wp, vel, np.random)
        for u, a, b in zip(ids, new_lat, new_lon):
            pos[int(u)] = (float(a), float(b))
        return new_lat, new_lon

    def _apply(self, state, key, idcol, spec_fn):
        df = state.data.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty: return
        dt = float(self.cfg.env.step_size_s)
        lat_b = list(self.cfg.env.lat_bounds); lon_b = list(self.cfg.env.lon_bounds)
        out = df.copy(); changed = False
        groups = list(out.groupby("operator_id")) if "operator_id" in out.columns \
            else [(None, out)]
        for oid, g in groups:
            spec = spec_fn(None if oid is None else int(oid))
            if spec is None or spec.model in _NOOP or int(spec.phase) == 1:
                continue
            ids = g[idcol].astype(int).to_numpy()
            nl, no = self._step_rows(key, spec, ids,
                                     g["lat"].astype(float).to_numpy(),
                                     g["lon"].astype(float).to_numpy(),
                                     dt, lat_b, lon_b)
            out.loc[g.index, "lat"] = nl
            out.loc[g.index, "lon"] = no
            changed = True
        if changed:
            state.data[key] = out

    def _apply_satellites(self, state):
        ts = state.timestamp
        if ts is None: return
        online = [op for op in self.cfg.operators
                  if op.satellites.mobility.model not in _NOOP]
        if not online: return
        existing = state.data.get("satellites_t")
        frames = []; have = set()
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            frames.append(existing)
            if "operator_id" in existing.columns:
                have = set(existing["operator_id"].astype(int).unique())
        from ntnconnect.scenario.sat_generator import SatelliteGenerator
        if self._t0 is None:
            self._t0 = ts
        # Anchor propagation to a fixed epoch so Kepler (relative-dt) advances;
        # SGP4 is absolute either way. Keep only this tick's rows.
        tl = [self._t0] if ts == self._t0 else [self._t0, ts]
        added = False
        for op in online:
            if int(op.id) in have:               # dataset already provided
                continue
            gen = self._satgen.get(int(op.id))
            if gen is None:
                gen = SatelliteGenerator(self.cfg, op)
                self._satgen[int(op.id)] = gen
            g = gen.generate(tl)
            if isinstance(g, pd.DataFrame) and not g.empty:
                if "timestamp" in g.columns:
                    g = g[g["timestamp"] == ts].copy()
                if "t_idx" in g.columns:
                    g["t_idx"] = state.t_idx
                if not g.empty:
                    frames.append(g); added = True
        if added:
            state.data["satellites_t"] = (pd.concat(frames, ignore_index=True)
                                          if len(frames) > 1 else frames[0])

    def run(self, state):
        self._apply(state, "users_t", "user_id", lambda oid: self.cfg.users.mobility)
        for key in ("haps_t", "uavs_t", "ground_stations_t", "base_stations_t"):
            self._apply(state, key, "node_id", lambda oid, k=key: self._op_spec(k, oid))
        self._apply_satellites(state)
