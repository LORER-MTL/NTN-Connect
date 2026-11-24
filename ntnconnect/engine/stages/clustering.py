# Stage: clustering - recompute each user's Earth-cell id vs the fixed grid;
# flag cluster_changed. Pure geographic lookup.
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry   import register_stage
from ntnconnect.scenario.grid      import lat_lon_to_cluster_id


@register_stage("clustering")
class Clustering(Stage):
    reads       = ["users_t"]
    writes      = ["users_t"]
    thread_safe = False     # mutates the shared users_t frame

    def __init__(self, cfg):
        super().__init__(cfg)
        self._prev_cid: dict[int, int] = {}

    def run(self, state):
        users = state.data.get("users_t")
        if not isinstance(users, pd.DataFrame) or users.empty:
            return
        if not ({"lat", "lon", "user_id"} <= set(users.columns)):
            return
        out = users.copy()
        uids = out["user_id"].astype(int).to_numpy()
        # Association granularity, from cfg.users.cluster_granularity:
        #  "user" -> each user is its own cluster (cluster_id = user_id): the
        #           pipeline is user-level (N=1 per cluster, no FDMA split).
        #  "cell" (default) -> fixed geographic Earth-cell grid = cluster level.
        # WISEE_USER_LEVEL env var overrides the config for quick research runs.
        gran = str(getattr(self.cfg.users, "cluster_granularity", "cell")).lower()
        env = os.environ.get("WISEE_USER_LEVEL")
        if env is not None:
            gran = "user" if env.lower() in ("1", "true", "yes") else "cell"
        if gran == "user":
            cid = uids.astype(np.int32)
        else:
            cid = lat_lon_to_cluster_id(out["lat"].to_numpy(),
                                        out["lon"].to_numpy(), self.cfg)
        prev = np.array([int(self._prev_cid.get(int(u), -1)) for u in uids],
                        dtype=np.int32)
        out["cluster_id"]      = cid.astype(np.int32)
        out["prev_cluster_id"] = prev
        out["cluster_changed"] = (cid != prev) & (prev >= 0)
        # Update history for next tick
        for u, c in zip(uids.tolist(), cid.tolist()):
            self._prev_cid[int(u)] = int(c)
        state.data["users_t"] = out
