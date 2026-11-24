# Online CSI-update timing consumption: a per-(node,user) history of the
# LARGE-SCALE CSI, so a decision (power sizing / association) reads the channel
# AGED by the availability delay while the current channel still grades the
# outcome. The delay is physically grounded - feedback RTT = distance/c (per link)
# + processing - per cfg.csi_timing. At coarse slots where the delay rounds to 0
# slots the aged value == current, so the path is byte-identical to no aging (the
# handover study at 1 s slots is unchanged); aging only bites at fine slots or
# large delays (GEO). Aligned with the handover study: the delay applies to the SLOW
# (large-scale, path_loss_mean_db) CSI; fast fading stays statistical (reserve).

from __future__ import annotations
import math
import numpy as np
import pandas as pd

from ntnconnect.common.physics import propagation_delay_ns


class CSIBuffer:
    def __init__(self, cfg, max_slots: int = 64):
        self.cfg = cfg
        self._hist: dict = {}          # (node,user) -> {t_idx: path_loss_mean_db}
        self._max = int(max_slots)

    # sim slot in ns (study step when datasets are built, else env step)
    def _step_ns(self) -> int:
        exp = self.cfg.experiment
        step_s = (float(self.cfg.study_timing.step_s)
                  if bool(getattr(exp, "generate_coverage", True))
                  else float(self.cfg.env.step_size_s))
        return max(1, int(round(step_s * 1e9)))

    def _avail_delay_ns(self, prop_rtt_ns: int, slot_ns: int) -> int:
        ct = self.cfg.csi_timing
        proc = int(round(float(ct.processing_delay_s) * 1e9))
        if ct.availability == "immediate":
            return proc
        base = int(prop_rtt_ns) + proc
        if ct.availability == "propagation" or slot_ns <= 0 or slot_ns > base:
            return base                              # propagation (physical) / collapse
        return int(math.ceil(base / slot_ns)) * slot_ns   # next_slot quantised

    def aged(self, links: pd.DataFrame, t_idx: int,
             column: str = "path_loss_mean_db") -> np.ndarray:
        """Push the current value of `column` into the history and return the AGED
        value per link for the decision. slots_back = round(delay / slot);
        slots_back == 0 (coarse slots) returns the current value (byte-identical).
        Radio ages the large-scale 'path_loss_mean_db'; handover (central agent)
        ages the realized 'path_loss_db' it associates on."""
        n = len(links)
        plm = links.get(column,
                        links.get("path_loss_db", pd.Series(np.zeros(n)))
                        ).astype(float).to_numpy()
        node = links.get("node_id", pd.Series(np.arange(n))).astype(int).to_numpy()
        user = links.get("user_id", pd.Series(np.arange(n))).astype(int).to_numpy()
        # slant distance for the propagation delay (fallbacks kept broad)
        dist = links.get("slant_range_m",
                         links.get("dist_m",
                                   links.get("ground_dist_m", pd.Series(np.zeros(n)))
                                   )).astype(float).to_numpy()
        step_ns = self._step_ns()
        ct = self.cfg.csi_timing
        instant = (ct.availability == "immediate")
        aged = plm.copy()
        for k in range(n):
            key = (int(node[k]), int(user[k]))
            h = self._hist.setdefault(key, {})
            h[t_idx] = float(plm[k])                  # record current
            if len(h) > self._max:
                del h[min(h)]
            if instant:
                continue
            prop_rtt = int(propagation_delay_ns(dist[k], round_trip=True)) if dist[k] > 0 else 0
            slots_back = int(round(self._avail_delay_ns(prop_rtt, step_ns) / step_ns)) if step_ns else 0
            if slots_back > 0:
                aged[k] = h.get(t_idx - slots_back, float(plm[k]))   # aged; else current
        return aged
