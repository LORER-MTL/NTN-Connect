# Stage: measurement - coarse link quality for ALL visible links (pre-handover).
# Large-scale RSRP (FSPL + fixed gains, no fading/beamforming), a coarse
# co-channel interference ESTIMATE, and a per-node load proxy. Feeds handover
# (ranks on RSRP) and the coarse load view. The fine, served-only channel
# (fading, beamforming) is the separate `channel` stage; the precise interference
# is the separate `interference` stage - this is the cheap estimate for the
# association decision.
from __future__ import annotations
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_KB = 1.380649e-23        # Boltzmann [J/K]
_C  = 299_792_458.0       # speed of light [m/s]


@register_stage("measurement")
class Measurement(Stage):
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.defaults_radio
        self._def = dict(ptx_w=float(d.power_max_w), gtx_db=float(d.antenna_gain_tx_db),
                         grx_db=float(d.antenna_gain_rx_db), f_hz=float(d.carrier_freq_hz),
                         bw_hz=float(d.bandwidth_hz), t_sys=float(d.system_temperature_k),
                         nf_db=float(d.noise_figure_db))
        self._op = {}
        for op in cfg.operators:
            r = getattr(op, "radio", None) or d
            self._op[int(op.id)] = dict(
                ptx_w=float(getattr(r, "power_max_w", self._def["ptx_w"])),
                gtx_db=float(getattr(r, "antenna_gain_tx_db", self._def["gtx_db"])),
                grx_db=float(getattr(r, "antenna_gain_rx_db", self._def["grx_db"])),
                f_hz=float(getattr(r, "carrier_freq_hz", self._def["f_hz"])),
                bw_hz=float(getattr(r, "bandwidth_hz", self._def["bw_hz"])),
                t_sys=float(getattr(r, "system_temperature_k", self._def["t_sys"])),
                nf_db=float(getattr(r, "noise_figure_db", self._def["nf_db"])))

    def _col(self, opid, key):
        return np.array([self._op.get(int(o), self._def)[key] for o in opid], float)

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty \
                or "dist_m" not in links.columns:
            return
        out = links
        opid = out["operator_id"].astype(int).to_numpy() if "operator_id" in out.columns \
            else np.zeros(len(out), dtype=int)
        d_m = out["dist_m"].astype(float).to_numpy()
        ptx = self._col(opid, "ptx_w"); gtx = self._col(opid, "gtx_db")
        grx = self._col(opid, "grx_db"); f = self._col(opid, "f_hz")
        bw = self._col(opid, "bw_hz"); tsys = self._col(opid, "t_sys")
        nf = self._col(opid, "nf_db")

        # Coarse RSRP: large-scale only (FSPL + fixed antenna gains, no fading).
        fspl = (4.0 * np.pi * np.maximum(d_m, 1.0) * f / _C) ** 2
        eirp_w = ptx * 10.0 ** (gtx / 10.0)
        rx_w = eirp_w * 10.0 ** (grx / 10.0) / np.maximum(fspl, 1e-30)
        # Noise with figure applied to T_sys (matches link_budget / the handover study).
        t_eff = tsys * 10.0 ** (nf / 10.0)
        noise_w = _KB * t_eff * np.maximum(bw, 1.0)
        out["rx_power_w_coarse"] = rx_w
        out["rsrp_dbm"] = 10.0 * np.log10(np.maximum(rx_w, 1e-30)) + 30.0
        out["noise_w"] = noise_w

        # Coarse co-channel interference ESTIMATE: worst-case aggregate of every
        # OTHER visible link's received power at the same user. Cheap upper bound
        # for the association decision; the precise I is the interference stage.
        uid = out["user_id"].astype(int).to_numpy() if "user_id" in out.columns \
            else np.zeros(len(out), dtype=int)
        tot = pd.Series(rx_w).groupby(uid).transform("sum").to_numpy()
        i_est = np.maximum(tot - rx_w, 0.0)
        out["i_est_w"] = i_est
        out["sinr_est_db"] = 10.0 * np.log10(
            np.maximum(rx_w, 1e-30) / np.maximum(noise_w + i_est, 1e-30))

        # Per-node load proxy: candidate links on each node this tick.
        # TODO: replace with real occupancy (admitted users / RB / power usage).
        nid = out["node_id"].astype(int).to_numpy() if "node_id" in out.columns \
            else np.zeros(len(out), dtype=int)
        out["node_load"] = pd.Series(np.ones(len(out))).groupby(nid).transform("sum").to_numpy()

        state.data["links"] = out
