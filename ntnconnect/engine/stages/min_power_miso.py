# MISO min-power (MRT): P_min = gamma*(beta+N0)/||h||^2, gamma=2^(R0/B)-1, cap P_max.
# Runs after channel+interference, before sinr_calc. Per-sat P_sat budget: admit
# cheapest-first until P_sat, rest blocked (min_power_capped, tx_power=0).

from __future__ import annotations
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_stage

_KB = 1.380649e-23

# Cache: eps -> (K_dB grid, fast-fading outage margin grid). Exact Rician margin =
# 1/q_eps, q_eps the eps-quantile of unit-mean |h|^2 (2(K+1)|h|^2 ~ ncx2(2, 2K)).
# ncx2.ppf is expensive, so evaluate once over a K grid and interpolate per link.
_FF_MARGIN_CACHE: dict = {}


def _ff_margin_grid(eps: float):
    key = round(float(eps), 5)
    if key not in _FF_MARGIN_CACHE:
        from scipy.stats import ncx2
        kdb = np.arange(-10.0, 30.01, 0.25)
        K = 10.0 ** (kdb / 10.0)
        q = ncx2.ppf(eps, 2.0, 2.0 * K) / (2.0 * (K + 1.0))
        _FF_MARGIN_CACHE[key] = (kdb, 1.0 / np.maximum(q, 1e-9))
    return _FF_MARGIN_CACHE[key]


@register_stage("min_power_miso", internal=True)
class MinPowerMISO(Stage):
    reads  = ["links"]
    writes = ["links"]
    thread_safe = False

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return
        n = len(links)
        ho = self.cfg.handover

        # QoS rate R0 -> SINR target gamma = 2^(R0/B) - 1
        R0 = float(getattr(ho, "qos_rate_bps", 5.0e6))
        bw = links.get("bandwidth_hz",
                       pd.Series(np.full(n, float(getattr(self.cfg.defaults_radio,
                                                          "bandwidth_hz", 5e6))))
                      ).astype(float).to_numpy()
        gamma = 2.0 ** (R0 / np.maximum(bw, 1.0)) - 1.0

        # Channel gain ||h||^2 per unit tx = G * PL^-1 * fading, via the shared
        # physics.channel_power_gain (same number sinr_calc uses -> SINR == gamma).
        from ntnconnect.common.physics import channel_power_gain
        op_ids = links.get("operator_id",
                           pd.Series(np.ones(n, dtype=int))).astype(int).to_numpy()
        chan = channel_power_gain(links, self.cfg.operators)
        # two-timescale (WISEE_TT_CHANNEL): size power on the large-scale channel only
        # (fading divided out), so uncompensated fast fading surfaces as outage in
        # sinr_calc. genie (B-* strategies): size on the full channel, no reserve.
        import os as _osmp
        _genie = bool(state.data.get("genie_power"))
        if _osmp.environ.get("WISEE_TT_CHANNEL") and not _genie:
            # mean gain = antenna gain x mean path loss; path_loss_mean_db keeps
            # shadowing as an unknown the outage margin must absorb (path_loss_db would
            # cancel it). Falls back to path_loss_db when the mean column is absent.
            _gd = links.get("antenna_gain_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
            _plm = links.get("path_loss_mean_db",
                             links.get("path_loss_db", pd.Series(np.zeros(n)))
                             ).astype(float).to_numpy()
            chan = 10.0 ** (_gd / 10.0) * 10.0 ** (-_plm / 10.0)

        # noise floor + external interference
        T_sys = (float(self.cfg.defaults_radio.system_temperature_k)
                 * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        noise_w = _KB * T_sys * np.maximum(bw, 1.0)
        # beta_{u,s}: aggregate co-channel interference [W], persisted as 'i_ext_w'
        beta_w = links.get("i_ext_w", pd.Series(np.zeros(n))).astype(float).to_numpy()

        # closed-form min power to meet R0 (uncapped nominal)
        p_req = gamma * (noise_w + beta_w) / np.maximum(chan, 1e-30)
        # outage reserve for the composite lognormal (shadow+weather) x Rician fade,
        # sized on the mean gain: product of marginal (1-eps) quantiles (conservative).
        _eps_ff = float(_osmp.environ.get("WISEE_OUTAGE_EPS", "0.05"))
        if _osmp.environ.get("WISEE_TT_CHANNEL") and not _genie:   # genie: no reserve
            _ssh = links.get("sigma_sh_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
            _swx = links.get("sigma_wx_db", pd.Series(np.zeros(n))).astype(float).to_numpy()
            _sig_tot = np.sqrt(_ssh ** 2 + _swx ** 2)
            _kdb = links.get("k_factor_db", pd.Series(np.full(n, 10.0))).astype(float).to_numpy()
            _kdb = np.where(np.isfinite(_kdb), _kdb, 10.0)
            from scipy.stats import norm as _norm_ls
            _z_ls = float(_norm_ls.ppf(1.0 - _eps_ff))
            p_req = p_req * 10.0 ** (_sig_tot * _z_ls / 10.0)  # (a) shadowing+weather
            if _osmp.environ.get("WISEE_FF_GAUSSIAN"):
                from scipy.stats import norm as _norm
                _z = float(_norm.ppf(1.0 - _eps_ff)); _K = 10.0 ** (_kdb / 10.0)
                _sig_ff = (10.0 / np.log(10.0)) * np.sqrt(2.0 * _K + 1.0) / (_K + 1.0)
                _margin = 10.0 ** (_sig_ff * _z / 10.0)       # Gaussian-approx margin
            else:
                _grid, _mgrid = _ff_margin_grid(_eps_ff)       # exact Rician margin
                _margin = np.interp(_kdb, _grid, _mgrid)
            p_req = p_req * _margin                            # (b) fast fading

        # Per-satellite budget P_sat (total RF power, not per-beam), enforced by a
        # common cheapest-first admission for every strategy: admit until cumulative
        # p_req would exceed P_sat, rest blocked. Restricted to served links (the
        # association) via state.data["assignments"], else a sat accrues phantom
        # demand from every user that can merely see it.
        import os as _os
        _op_psat = {int(op.id): float(getattr(op.radio, "power_max_w", 12.6))
                    for op in self.cfg.operators}
        _env_psat = _os.environ.get("WISEE_PMAX_W")
        node = links.get("node_id", pd.Series(np.arange(n))).astype(int).to_numpy()
        user = links.get("user_id", pd.Series(np.arange(n))).astype(int).to_numpy()
        assigns = state.data.get("assignments", {}) or {}
        served = np.array([assigns.get(int(user[k]), -2) == int(node[k])
                           for k in range(n)], dtype=bool)

        admitted = np.zeros(n, dtype=bool)
        cum = {}                                  # sat -> admitted power (this tick)
        srv_idx = np.where(served)[0]
        for k in srv_idx[np.argsort(p_req[srv_idx], kind="stable")]:  # cheapest first
            s = int(node[k])
            psat = float(_env_psat) if _env_psat else _op_psat.get(int(op_ids[k]), 12.6)
            c = cum.get(s, 0.0)
            if c + p_req[k] <= psat + 1e-12:
                admitted[k] = True; cum[s] = c + p_req[k]

        tx = np.where(served & admitted, p_req, 0.0)
        # WISEE_BESTEFFORT_POWER: guaranteed set unchanged; share the leftover
        # P_sat - sum(admitted) among blocked users proportional to p_req (max-min
        # fair, equal sub-QoS SINR). slot_admitted / min_power_capped unchanged.
        if _os.environ.get("WISEE_BESTEFFORT_POWER"):
            cap_idx = np.where(served & ~admitted)[0]
            by_sat = {}
            for k in cap_idx:
                by_sat.setdefault(int(node[k]), []).append(k)
            for s, ks in by_sat.items():
                psat = float(_env_psat) if _env_psat else _op_psat.get(int(op_ids[ks[0]]), 12.6)
                residual = max(psat - cum.get(s, 0.0), 0.0)
                need = float(np.sum(p_req[ks]))
                if residual > 0.0 and need > 0.0:
                    scale = residual / need                     # < 1 (best-effort)
                    for k in ks:
                        tx[k] = p_req[k] * scale                # equal-SINR share
        out = links.copy()
        out["tx_power_w"]       = tx              # consumed by sinr_calc
        out["min_power_w"]      = p_req           # required power (energy metric)
        out["slot_admitted"]    = served & admitted
        out["min_power_capped"] = served & ~admitted
        # assoc phase: 1 = guaranteed-QoS set, 0 = best-effort
        _phase_map = state.data.get("assignment_phase", {}) or {}
        out["assoc_guaranteed"] = np.array(
            [int(_phase_map.get(int(user[k]), 0)) for k in range(n)], dtype=np.int8)
        state.data["links"] = out
