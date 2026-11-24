# Stage: signal/noise/interference/SINR. interference.model: off | fixed (floor I0) |
# physical (co-band aggregation; intra/inter toggles, isolation dB, frequency_reuse_factor).
# P_rx = P_tx*10^(G/10)*|h|^2*10^(-PL/10); N = kB*T_sys*10^(NF/10)*BW.

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_log = logging.getLogger(__name__)
kB = 1.380649e-23


@register_stage("interference", internal=True)
class InterferenceStage(Stage):
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return

        # System-wide interference settings live at cfg.defaults_interference.
        # Per-operator overrides at cfg.operators[*].interference are possible
        # but here we use the global one (a future per-op variant can replace
        # this lookup with a dict keyed on operator_id).
        ic = self.cfg.defaults_interference

        # Signal power
        op_id_to_pt_w = {op.id: op.radio.power_max_w for op in self.cfg.operators}
        Pt_w = np.array([op_id_to_pt_w.get(int(o), 1.0)
                         for o in links["operator_id"].values], dtype=float)
        gain_lin = 10.0 ** (links["antenna_gain_db"].values / 10.0)
        pl_lin   = 10.0 ** (-links["path_loss_db"].values / 10.0)
        chan_g   = np.asarray(links.get("chan_gain_lin", np.ones(len(links))),
                              dtype=float)
        signal_w = Pt_w * gain_lin * np.maximum(chan_g, 1e-30) * pl_lin

        # Thermal noise
        T_sys = (float(self.cfg.defaults_radio.system_temperature_k)
                 * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        bw = links["bandwidth_hz"].values
        noise_w = kB * T_sys * np.maximum(bw, 1.0)

        # Interference per cfg.interference.model
        model = (ic.model or "off").lower()
        if model == "off":
            interf_w = np.zeros_like(signal_w)
        elif model == "fixed":
            interf_w = np.full_like(signal_w,
                                    10.0 ** (float(ic.fixed_interference_dbw) / 10.0))
        elif model == "physical":
            interf_w = self._physical_interference(links, signal_w, ic)
        else:
            _log.warning("Unknown interference.model='%s' - assuming 'off'.", model)
            interf_w = np.zeros_like(signal_w)

        snr  = signal_w / np.maximum(noise_w, 1e-30)
        sinr = signal_w / np.maximum(interf_w + noise_w, 1e-30)
        se   = np.log2(1.0 + sinr)

        out = links.copy()
        out["signal_w"]                = signal_w
        out["noise_w"]                 = noise_w
        out["interference_w"]          = interf_w
        out["snr_db"]                  = 10.0 * np.log10(np.maximum(snr,  1e-30))
        out["sinr_db"]                 = 10.0 * np.log10(np.maximum(sinr, 1e-30))
        out["spectral_efficiency_bps_hz"] = se
        out["link_rate_bps"]           = se * bw
        out["interference_model"]      = model
        state.data["links"] = out

    @staticmethod
    def _physical_interference(links: pd.DataFrame,
                               signal_w: np.ndarray,
                               ic) -> np.ndarray:
        """Co-band aggregation honoring intra/inter toggles, isolation, reuse."""
        n = len(links)
        interf_w = np.zeros(n, dtype=float)

        intra_atten = 10.0 ** (-float(ic.intra_beam_isolation_db) / 10.0)
        inter_atten = 10.0 ** (-float(ic.co_channel_isolation_db) / 10.0)
        reuse = float(ic.frequency_reuse_factor)
        # reuse_factor in (0, 1]: fraction of aggressors actually re-used.
        # We model this as a scalar gain on the aggregate (mean-field).
        reuse_gain = max(0.0, min(1.0, reuse))

        node_id = links["node_id"].astype(int).to_numpy()
        car     = links["carrier_hz"].astype(float).to_numpy()
        user    = links["user_id"].astype(int).to_numpy()
        intra_on = bool(ic.enable_intra_satellite_interference)
        inter_on = bool(ic.enable_inter_satellite_interference)

        # Vectorized: per-group "sum and per-node sum" lookups, then derive
        # the victim's aggregate as (group total - per-node total) for inter
        # and (per-node total - self) for intra. No Python loops over rows.
        df = pd.DataFrame({
            "_g":   list(zip(user.tolist(), car.tolist()))
                   if bool(ic.same_user_only) else car.tolist(),
            "_n":   node_id,
            "_sig": signal_w,
        })
        g_total  = df.groupby("_g")["_sig"].transform("sum").to_numpy()
        gn_total = df.groupby(["_g", "_n"])["_sig"].transform("sum").to_numpy()

        intra_contrib = (gn_total - signal_w) if intra_on else np.zeros(n)
        inter_contrib = (g_total  - gn_total) if inter_on else np.zeros(n)
        interf_w = (intra_contrib * intra_atten
                    + inter_contrib * inter_atten) * reuse_gain
        return interf_w
