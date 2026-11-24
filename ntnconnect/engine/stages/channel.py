# Stage: one ChannelPipeline per operator (base model + components from cfg).

from __future__ import annotations
import logging
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage
from ntnconnect.registry.channels.pipeline import ChannelModelFactory

_log = logging.getLogger(__name__)

# TR 38.811 elevation-dependent shadowing sigma(el) - the same table as the
# per-tick Shadowing38811Component, used here for the two-timescale (WISEE_TT_
# CHANNEL) per-SLOT held draw and to expose sigma_sh_db to the reserve power sizing.
_EL_REF_TT = np.array([0.0, 10.0, 30.0, 60.0, 90.0])
_SIG_SH_TT = {
    "dense_urban": [6.0, 4.5, 3.0, 1.8, 1.0],
    "urban":       [5.5, 4.0, 2.5, 1.5, 0.8],
    "suburban":    [4.0, 3.0, 2.0, 1.2, 0.6],
    "rural":       [3.0, 2.5, 1.5, 0.8, 0.4],
}


def _flatten_channel_cfg(op):
    """Combine operator.radio + operator.channel_model into one namespace
    that channel models/components read with strict cfg.<field> access."""
    out = SimpleNamespace()
    for src in (op.radio, op.channel_model):
        for k, v in vars(src).items():
            setattr(out, k, v)
    return out


@register_stage("channel")
class ChannelStage(Stage):
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False    # mutates the shared links frame

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pipelines = {}
        for op in cfg.operators:
            try:
                self._pipelines[op.id] = ChannelModelFactory.build(_flatten_channel_cfg(op))
                _log.info("Channel pipeline built for op=%s (model=%s, components=%s)",
                          op.name, op.channel_model.model_type,
                          [c.name for c in self._pipelines[op.id].components])
            except Exception as e:
                _log.warning("op=%s channel pipeline build failed: %s", op.name, e)

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return

        # Reliability guard: every link's operator must have a matching
        # config operator, else the per-operator loop below silently skips it
        # (→ 0 dB path-loss/gain, garbage SINR). Fail LOUD instead of silent.
        if "operator_id" in links.columns:
            cfg_ops = {int(op.id) for op in self.cfg.operators}
            link_ops = set(int(o) for o in links["operator_id"].dropna().unique())
            uncovered = link_ops - cfg_ops
            if uncovered:
                _log.error(
                    "channel: links carry operator_id %s with NO matching cfg "
                    "operator %s — those links would get NO channel (FSPL only) "
                    "and produce invalid SINR. Likely an operator-id config bug.",
                    sorted(uncovered), sorted(cfg_ops))

        n = len(links)
        pl_db    = np.zeros(n)
        gain_db  = np.zeros(n)
        carrier  = np.zeros(n)
        bw       = np.zeros(n)
        dop      = np.zeros(n)
        chan_g   = np.ones(n)
        k_db     = np.full(n, np.nan)         # Rician K-factor per link (if model emits it)
        shadow_db = np.zeros(n)               # pipeline per-tick 38.811 shadowing draw
        h_per_ant_buf = []                    # list[(mask, ndarray)] for state.data["h_per_antenna"]

        # Served-only (default): channel runs AFTER handover, so the EXPENSIVE
        # per-antenna fading pipeline is evaluated only on served (user, sat)
        # links. Non-served links still get a cheap, geometrically-correct FSPL
        # path-loss (consumed by external-interference / access-delay) but no
        # per-antenna h. Collapses O(users x visible_sats) -> O(users).
        # Defensive fallback to full only if no assignments are present (e.g. the
        # channel is placed before handover in a custom pipeline).
        served_only = True
        served_mask = None
        assigns = state.data.get("assignments", {}) or {}
        if assigns and {"user_id", "node_id"}.issubset(links.columns):
            uid = links["user_id"].astype(int).to_numpy()
            nid = links["node_id"].astype(int).to_numpy()
            served_for_uid = np.array(
                [assigns.get(int(u), -1) for u in uid], dtype=int)
            served_mask = (served_for_uid == nid)
        else:
            served_only = False

        # node_type per row - the satellite atmospheric/MIMO pipeline runs only
        # on satellite links; terrestrial/aerial nodes get their own large-scale
        # model added below (see access_models). Defaults to "satellite" so a
        # sat-only scenario (e.g. WISEE) is unchanged.
        if "node_type" in links.columns:
            nt_all = links["node_type"].astype(str).to_numpy()
        else:
            nt_all = np.full(len(links), "satellite", dtype=object)
        is_sat = (nt_all == "satellite")

        for op in self.cfg.operators:
            mask = (links["operator_id"] == op.id).values
            if not mask.any(): continue
            # When served-only, the heavy pipeline runs on the served subset
            # of this operator's rows; FSPL is filled for the rest below.
            # Restrict to satellite rows - non-sat access nodes use access_models.
            pipe_mask = mask & is_sat
            if served_only and served_mask is not None:
                pipe_mask = pipe_mask & served_mask
            sub = links.loc[pipe_mask]
            f      = float(op.radio.carrier_freq_hz)
            band   = float(op.radio.bandwidth_hz)
            gain   = float(op.radio.antenna_gain_tx_db + op.radio.antenna_gain_rx_db)
            carrier[mask] = f
            bw[mask]      = band
            gain_db[mask] = gain

            # Cheap geometric FSPL for every link of this operator
            # (served-only mode keeps this for the non-served rows so the
            #  interference + access-delay terms remain geometrically valid;
            #  full mode overwrites these rows with the pipeline loss below.)
            c0 = 299_792_458.0
            dist_all = np.maximum(links.loc[mask, "dist_m"].values, 1.0)
            fspl_all = (20*np.log10(dist_all) + 20*np.log10(f)
                         + 20*np.log10(4*np.pi/c0))
            pl_db[mask]  = fspl_all
            chan_g[mask] = 1.0

            if len(sub) == 0:
                continue

            dist_m   = np.maximum(sub["dist_m"].values, 1.0)
            elev_deg = sub.get("elev_deg", pd.Series(np.zeros(len(sub)))).values
            # Accept either "azimuth_deg" (new) or "azimuth" (legacy alias).
            if "azimuth_deg" in sub.columns:
                az_deg = sub["azimuth_deg"].values
            elif "azimuth" in sub.columns:
                az_deg = sub["azimuth"].values
            else:
                az_deg = np.zeros(len(sub))
            rate_mps = (sub.get("range_rate_km_s",
                                pd.Series(np.zeros(len(sub)))).values * 1000.0)

            pipe = self._pipelines.get(op.id)
            if pipe is None:
                # FSPL already filled above for all rows; nothing more to do.
                continue
            try:
                loss, h, doppler, extras = pipe.execute(
                    dist_m=dist_m, elev_deg=elev_deg,
                    azimuth_deg=az_deg,
                    range_rate_m_s=rate_mps,
                    dt_sec=float(self.cfg.env.step_size_s),
                    pos_m=None,
                )
                nsub = int(pipe_mask.sum())
                pl_db[pipe_mask] = np.asarray(loss).reshape(-1)[:nsub]
                dop[pipe_mask]   = np.asarray(doppler).reshape(-1)[:nsub]
                h_arr = np.asarray(h)
                if h_arr.ndim > 1:
                    chan_g[pipe_mask] = np.sum(np.abs(h_arr)**2,
                                          axis=tuple(range(1, h_arr.ndim)))
                    h_per_ant_buf.append((pipe_mask, h_arr))
                else:
                    chan_g[pipe_mask] = np.abs(h_arr)**2
                if isinstance(extras, dict) and "k_factor_db" in extras:
                    kvals = np.asarray(extras["k_factor_db"]).reshape(-1)
                    k_db[pipe_mask] = kvals[:nsub]
                if isinstance(extras, dict) and "shadow_db" in extras:
                    sv = np.asarray(extras["shadow_db"]).reshape(-1)
                    shadow_db[pipe_mask] = sv[:nsub]
            except Exception as e:
                _log.warning("op=%s channel pipeline run failed at t=%s: %s",
                             op.name, state.timestamp, e)

        # Node-type-aware large-scale loss for non-satellite access nodes
        # Terrestrial base stations (3GPP TR 38.901) and aerial HAPS/UAV
        # (TR 38.811) add excess loss over the FSPL already in pl_db. Satellite
        # rows get 0 excess (their loss came from the pipeline above). The
        # per-node-type model map is overridable via cfg.defaults_channel.
        if not bool(is_sat.all()):
            try:
                from ntnconnect.registry.channels.access_models import excess_loss_db
                mapping = getattr(self.cfg.defaults_channel, "access_models", None)
                elev_all = (links["elev_deg"].to_numpy()
                            if "elev_deg" in links.columns else None)
                exc = excess_loss_db(nt_all,
                                     links["dist_m"].to_numpy(),
                                     elev_all, carrier, mapping)
                pl_db = pl_db + exc
            except Exception as e:
                _log.warning("access_models excess-loss failed: %s", e)

        # WISEE stochastic weather (all strategies): per-link Gaussian rain+cloud
        #  deviation A_wx ~ N(0, sigma_wx^2) added to the satellite path loss, so
        #  the realized channel carries UNKNOWN weather - matching the standalone.
        #  The per-tick draw has the same marginal blocking as a per-slot draw
        #  (only burstiness differs). the allocator separately reserves sigma_wx in its
        #  risk margin (_min_power). Gated so default runs are unchanged.
        import os as _os
        _wxenv = _os.environ.get("WISEE_STOCH_WEATHER")
        _wx_on = ((_wxenv not in ("0", "", "false", "False")) if _wxenv is not None
                  else bool(getattr(self.cfg.defaults_channel, "wisee_stoch_weather", False)))
        _tt_on = bool(_os.environ.get("WISEE_TT_CHANNEL"))

        # pl_mean_db = the MEAN large-scale loss (FSPL+gas+rain+cloud, no random
        # shadowing/weather) - what the reserve power sizing uses. sigma_*_db are the
        # per-link log-normal std of the UNKNOWN shadowing/weather the outage reserves.
        pl_mean_db  = np.asarray(pl_db, float).copy()
        sigma_sh_db = np.zeros(n)
        sigma_wx_db = np.zeros(n)

        if _tt_on:
            # Paper-consistent two-timescale large-scale model
            # The reserve does not know the shadowing/weather realisation; it knows
            # only the mean (pl_mean_db) and the std (sigma_*_db). The realised
            # channel carries a per-SLOT (slow) shadowing+weather draw, HELD across
            # the sub-slots of a slot, that the outage margin must absorb.
            #
            # (1) MEAN: strip the pipeline's per-tick zero-mean 38.811 shadowing.
            pl_mean_db = np.asarray(pl_db, float) - np.asarray(shadow_db, float)
            elev_all = (links["elev_deg"].to_numpy(float)
                        if "elev_deg" in links.columns else np.zeros(n))
            # (2) shadowing std sigma(el) (38.811 elevation table).
            _scn = str(getattr(self.cfg.defaults_channel,
                               "shadowing_38811_scenario", "suburban")).lower()
            if _scn not in _SIG_SH_TT:
                _scn = "suburban"
            sigma_sh_db = np.where(is_sat,
                                   np.interp(elev_all, _EL_REF_TT, _SIG_SH_TT[_scn]),
                                   0.0)
            # (3) weather std sigma_wx (P.838 rain + P.840 cloud), if enabled.
            if _wx_on:
                try:
                    from ntnconnect.registry.channels.components.wisee_weather import weather_sigma_db
                    _fghz = float(np.asarray(carrier).ravel()[0]) / 1e9
                    sigma_wx_db = np.where(is_sat,
                                           weather_sigma_db(elev_all, _fghz), 0.0)
                except Exception as e:
                    _log.warning("wisee weather sigma failed: %s", e)
            # (4) per-SLOT realisation of shadowing (+weather), HELD across the
            #  sub-slots of a slot (slow). Keyed by (user,sat) so a link keeps
            #  its draw within the slot and redraws only at the slot boundary.
            _sub = max(1, int(getattr(self.cfg.handover, "run_every_n_ticks", 1)))
            uid = (links["user_id"].to_numpy(np.int64)
                   if "user_id" in links.columns else np.arange(n, dtype=np.int64))
            nid = (links["node_id"].to_numpy(np.int64)
                   if "node_id" in links.columns else np.arange(n, dtype=np.int64))
            keyi = uid * np.int64(100003) + nid
            tick = int(getattr(self, "_tt_tick", 0)); slot = tick // _sub
            if getattr(self, "_tt_slot", None) != slot:          # slot boundary: redraw
                self._tt_slot = slot
                dev = (sigma_sh_db * np.random.randn(n)
                       + sigma_wx_db * np.random.randn(n))
                self._tt_cache = dict(zip(keyi.tolist(), dev.tolist()))
            else:                                                # within slot: reuse
                cache = getattr(self, "_tt_cache", {})
                cser = pd.Series(cache) if cache else pd.Series(dtype=float)
                dev = cser.reindex(keyi).to_numpy()
                miss = ~np.isfinite(dev)
                if miss.any():                                   # new visibility: draw
                    fresh = (sigma_sh_db * np.random.randn(n)
                             + sigma_wx_db * np.random.randn(n))
                    dev = np.where(miss, fresh, dev)
                    self._tt_cache.update(dict(zip(keyi[miss].tolist(),
                                                   dev[miss].tolist())))
            self._tt_tick = tick + 1
            pl_db = pl_mean_db + np.where(is_sat, dev, 0.0)      # realised large-scale
        else:
            # Legacy stochastic-weather + per-slot hold (non-TT runs)
            if _wx_on:
                try:
                    from ntnconnect.registry.channels.components.wisee_weather import weather_sigma_db
                    elev_all = (links["elev_deg"].to_numpy(float)
                                if "elev_deg" in links.columns else np.zeros(len(links)))
                    f_ghz = float(np.asarray(carrier).ravel()[0]) / 1e9   # scalar carrier
                    sw = weather_sigma_db(elev_all, f_ghz)
                    pl_db = pl_db + np.where(is_sat, sw * np.random.randn(len(links)), 0.0)
                except Exception as e:
                    _log.warning("wisee stochastic weather failed: %s", e)
            _sub = max(1, int(getattr(self.cfg.handover, "run_every_n_ticks", 1)))
            if _wx_on and _sub > 1 and {"user_id", "node_id"} <= set(links.columns):
                try:
                    uid = links["user_id"].to_numpy(np.int64)
                    nid = links["node_id"].to_numpy(np.int64)
                    keyi = uid * np.int64(100003) + nid
                    tick = int(getattr(self, "_ls_tick", 0)); slot = tick // _sub
                    if getattr(self, "_ls_slot", None) != slot:      # slot boundary: cache
                        self._ls_slot = slot
                        self._ls_cache = dict(zip(keyi[is_sat].tolist(),
                                                 np.asarray(pl_db)[is_sat].tolist()))
                    else:                                            # within slot: reuse
                        cache = getattr(self, "_ls_cache", {})
                        cser = pd.Series(cache) if cache else pd.Series(dtype=float)
                        cached = cser.reindex(keyi).to_numpy()       # NaN where unseen
                        m = ~np.isnan(cached) & is_sat
                        pl_db = np.asarray(pl_db, float).copy(); pl_db[m] = cached[m]
                    self._ls_tick = tick + 1
                except Exception as e:
                    _log.warning("wisee per-slot large-scale hold failed: %s", e)

        out = links.copy()
        out["path_loss_db"]      = pl_db          # realised (mean + slot shadowing/weather)
        out["path_loss_mean_db"] = pl_mean_db     # mean only - reserve power sizing
        out["sigma_sh_db"]       = sigma_sh_db    # shadowing std (outage reserve)
        out["sigma_wx_db"]       = sigma_wx_db    # weather std (outage reserve)
        out["antenna_gain_db"] = gain_db
        out["carrier_hz"]      = carrier
        out["bandwidth_hz"]    = bw
        out["doppler_hz"]      = dop
        out["chan_gain_lin"]   = chan_g
        out["k_factor_db"]     = k_db
        state.data["links"]    = out
        # Expose the per-antenna h produced by SteeringUPAComponent (if
        # enabled) so the DT pipeline can read it without re-running
        # steering construction. Keyed by row index in the links frame.
        if h_per_ant_buf:
            h_per_link: dict[int, np.ndarray] = {}
            for mask, h_arr in h_per_ant_buf:
                row_idx = np.where(mask)[0]
                for j, ri in enumerate(row_idx):
                    h_per_link[int(ri)] = h_arr[j]
            state.data["h_per_antenna"] = h_per_link
