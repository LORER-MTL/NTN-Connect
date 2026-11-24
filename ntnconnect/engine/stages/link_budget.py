# Stage: sinr_calc = link budget / SINR. Signal (P_tx*G*PL), ext interference,
# SINR/SNR/SE/rate, KPIs. Emits sinr_db_paper/sinr_db_socp. ext_mode swappable.
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_log = logging.getLogger(__name__)
_kB = 1.380649e-23


# Single-user-per-cell SINR stage (the single, canonical budget stage)
#
# note: the former `radio_allocation` + `link_budget` two-stage pair was
# removed - it was an UNUSED alternative decomposition (no scenario composed
# it) that duplicated exactly what `sinr_calc` does in one pass. `sinr_calc`
# is now the sole link-budget/SINR stage; the CVX optimisation path uses
# `scn_radio_link` (allocation+beamforming) and `paper_sinr` (figure SINR).
#
# When there is exactly one user per cell the intra-cell MU-MIMO term is
# zero, there is no power allocation, no scheduling and no beamforming
# optimisation - the radio layer is purely
#  SINR_u = P_rx,u / (I_ext + N0 B)
# with I_ext the (geometric) inter-cell interference. This stage does that
# whole calculation in one pass: external interference (constant) + signal
# + SINR + spectral efficiency + rate. It replaces the
# radio_allocation + link_budget pair for the single-user scenario.
@register_stage("sinr_calc", internal=True)
class SinrCalc(Stage):
    reads  = ["links", "assignments"]
    writes = ["links"]
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        from ntnconnect.engine.stages.external_interference import ExternalInterferenceConst
        self._iext = ExternalInterferenceConst(cfg)

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        assigns = state.data.get("assignments", {}) or {}

        # 1) Inter-cell interference (geometric; 7 modes) - computed ONCE.
        #  If an upstream `external_interference_const` stage already wrote
        #  i_ext_w, CONSUME it (no recompute). Otherwise run the same canonical
        #  ExternalInterferenceConst model inline as a fallback, so sinr_calc
        #  is self-sufficient without duplicating the interference logic.
        if "i_ext_w" not in links.columns:
            self._iext.engine = getattr(self, "engine", None)
            self._iext.run(state)
            links = state.data["links"]
        else:
            _log.debug("sinr_calc: consuming upstream i_ext_w (no recompute)")
        n = len(links)

        # 2) Noise floor
        T_sys = (float(self.cfg.defaults_radio.system_temperature_k)
                 * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        # Bandwidth: consume the per-link value written upstream (channel stage,
        # from op.radio.bandwidth_hz); fall back to the CONFIGURED default (not a
        # hardcoded constant) so an allocation sized for B and the SINR/rate
        # computed here always use the same bandwidth, for any pipeline.
        _bw_cfg = float(getattr(self.cfg.defaults_radio, "bandwidth_hz", 20e6))
        bw = links.get("bandwidth_hz",
                       pd.Series(np.full(n, _bw_cfg))).astype(float).to_numpy()
        noise_w = _kB * T_sys * np.maximum(bw, 1.0)

        # 3) Transmit power.
        #  If an upstream RRM stage (scn_radio_link / rrm_power_*) has already
        #  ALLOCATED power, CONSUME it - sinr_calc is the single authoritative
        #  SINR/rate/EE writer fed by the optimiser. Otherwise fall back to
        #  full power P_max (the coverage-study regime: every served beam at
        #  P_max). This is what makes RRM→SINR→rate/EE one unified flow.
        op_p_w = {int(op.id): float(getattr(op.radio, "power_max_w", 1.0))
                  for op in self.cfg.operators}
        op_ids = links.get("operator_id",
                            pd.Series(np.ones(n, dtype=int))
                          ).astype(int).to_numpy()
        # Reliability guard: a link whose operator_id is not in the power map
        # silently falls back to 1 W (the operator-id-mismatch bug). Fail loud.
        _uncovered = set(int(o) for o in np.unique(op_ids)) - set(op_p_w)
        if _uncovered:
            _log.error("sinr_calc: operator_id %s missing from power map %s — "
                       "p_tx defaults to 1 W for those links (likely config bug).",
                       sorted(_uncovered), sorted(op_p_w))
        alloc = links.get("tx_power_w")
        if alloc is not None:
            p_alloc = alloc.astype(float).to_numpy()
            if np.isfinite(p_alloc).all() and (p_alloc > 0).any():
                p_tx = p_alloc                      # honour RRM allocation
                _log.debug("sinr_calc: consuming upstream tx_power_w allocation")
            else:
                p_tx = np.array([op_p_w.get(int(o), 1.0) for o in op_ids], dtype=float)
        else:
            p_tx = np.array([op_p_w.get(int(o), 1.0) for o in op_ids], dtype=float)
        # Received signal power = P_tx · ||h||^2, where ||h||^2 (antenna gain ·
        # path gain · normalised small-scale fading) comes from the SHARED
        # physics.channel_power_gain - the same effective channel the allocator
        # used, general for SISO/MISO/MU-MIMO. sinr_calc only CONSUMES it here.
        from ntnconnect.common.physics import channel_power_gain
        g_db = links.get("antenna_gain_db",
                          pd.Series(np.zeros(n))).astype(float).to_numpy()
        pl_db = links.get("path_loss_db",
                           pd.Series(np.zeros(n))).astype(float).to_numpy()
        signal_w = p_tx * channel_power_gain(links, self.cfg.operators)
        i_ext_w  = links.get("i_ext_w",
                              pd.Series(np.zeros(n))).astype(float).to_numpy()

        # Cluster resource sharing (FDMA + power allocation)
        # Users served in the same cluster share one satellite beam, so they
        # split the beam's bandwidth AND power: each of the N co-cell served
        # users gets B/N and P_max/N. Signal (∝P/N) and noise+interference
        # (∝B/N) both scale by 1/N, so the per-user SINR is UNCHANGED while the
        # per-user rate becomes rate_full/N. This bounds a cluster's aggregate
        # throughput to a single beam's resource, instead of giving every
        # co-cell user a full-power full-band link. Splitting bandwidth without
        # power (bare FDMA) is not physically justified, so both are divided.
        if "cluster_id" in links.columns:
            from ntnconnect.common.physics import served_mask_from_assignments as _smask
            _sm = _smask(links, assigns)
            if _sm is not None and _sm.any():
                cid = links["cluster_id"].astype(int).to_numpy()
                uniq, cnt = np.unique(cid[_sm], return_counts=True)
                occ = dict(zip(uniq.tolist(), cnt.tolist()))
                N = np.array([occ.get(int(c), 1) for c in cid], dtype=float)
                N = np.where(_sm, np.maximum(N, 1.0), 1.0)   # divide served only
                bw       = bw / N
                p_tx     = p_tx / N
                i_ext_w  = i_ext_w / N
                noise_w  = _kB * T_sys * np.maximum(bw, 1.0)
                signal_w = signal_w / N

        out = links.copy()
        out["tx_power_w"] = p_tx
        out["signal_w"]   = signal_w
        out["noise_w"]    = noise_w
        out["radio_status"] = "sinr_calc"

        # 4) SINR / SE / rate / SNR + generalised KPIs - via the single shared
        #  core (ntnconnect.common.physics.link_metrics) so the formula is not
        #  duplicated across radio stages.
        from ntnconnect.common.physics import (served_mask_from_assignments,
                                served_link_metrics, energy_kpis,
                                sinr_db_from_powers, spatial_streams)
        # Preserve any upstream optimiser SINR (e.g. scn_radio_link SOCP) for
        # audit before we overwrite sinr_db with the achieved value. (Merged
        # from the former `paper_sinr` stage.)
        if "sinr_db" in links.columns:
            out["sinr_db_socp"] = links["sinr_db"].astype(float).to_numpy()

        served_mask = served_mask_from_assignments(out, assigns)
        if served_mask is not None:
            # MIMO spatial-multiplexing order per link (SISO/MISO/SIMO -> 1, so
            # SE is unchanged unless an operator sets channel_model.spatial_streams).
            n_streams = spatial_streams(out, self.cfg.operators)
            # TODO(beam-hopping): when beam_id / beam hopping lands, gate signal
            # and interference by the beams LIT this slot (an unlit beam neither
            # transmits nor interferes) before computing these metrics.
            m = served_link_metrics(signal_w, noise_w, i_ext_w, bw, served_mask,
                                    n_streams=n_streams)
            for k, v in m.items():
                if k != "sinr_db_full":
                    out[k] = v
            eta_pa = float(getattr(self.cfg.defaults_radio, "pa_efficiency", 0.35))
            p_circ = float(getattr(self.cfg.defaults_radio, "circuit_power_w", 10.0))
            for k, v in energy_kpis(p_tx, m["link_rate_bps"], signal_w,
                                    m["spectral_efficiency_bps_hz"], served_mask,
                                    eta_pa=eta_pa, p_circ_w=p_circ).items():
                out[k] = v

            # Full-power link-budget REFERENCE SINR (the former `paper_sinr`):
            # always reported alongside the achieved SINR. Equals sinr_db when
            # no RRM allocation ran (coverage case); differs when an optimiser
            # reduced power below P_max.
            p_max = np.array([op_p_w.get(int(o), 1.0) for o in op_ids], dtype=float)
            rx_full = p_max * 10.0 ** (g_db / 10.0) * 10.0 ** (-pl_db / 10.0)
            import os as _osrb
            if _osrb.environ.get("WISEE_TT_CHANNEL"):
                # two-TIMESCALE realized SINR: the power was sized on the LARGE-SCALE
                # channel (min_power_miso divides out chan_gain_lin), so here the full
                # channel including the per-sub-slot geometric-Rician fast fading
                # (already in signal_w = p_tx * channel_power_gain) gives the SINR the
                # served user actually experiences this sub-slot. Fading is not
                # compensated -> this is the true outage metric.
                paper_full = sinr_db_from_powers(signal_w, noise_w, i_ext_w)
            else:
                paper_full = sinr_db_from_powers(rx_full, noise_w, i_ext_w)  # p_max ref (legacy)
            paper_col = np.full(n, -300.0)
            paper_col[served_mask] = paper_full[served_mask]
            out["sinr_db_paper"] = paper_col
            out["rx_power_w_paper"] = rx_full
        state.data["links"] = out


