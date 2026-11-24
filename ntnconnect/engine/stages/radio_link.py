# Stage: scn_radio_link = BF + RRM power + interference/SINR in one pass.
# rrm_power_algorithm: equal|waterfilling|utility. interference.model: off|fixed|physical.
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

# CVXPY is a soft dependency: only required when the user picks
# rrm_power_algorithm = "cvx_mimo_min_power".
try:
    import cvxpy as cp                  # type: ignore
    _HAS_CVXPY = True
except Exception:                       # pragma: no cover
    cp = None
    _HAS_CVXPY = False

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry   import register_stage
from ntnconnect.registry.agents.radio_algorithm import get_radio_algorithm

_log = logging.getLogger(__name__)

# Earth radius (m) for off-nadir angle computation
_EARTH_R_M = 6_371_000.0
kB    = 1.380649e-23


@register_stage("scn_radio_link", internal=True)
class ScnRadioLink(Stage):
    """One-shot radio: BF gain, RRM power, signal/noise/interference, SINR."""
    reads       = ["links", "assignments"]
    writes      = ["links"]
    thread_safe = False

    # BF (matches the formula in stages/rrm.py::ChanBeamforming)
    @staticmethod
    def _bf_gain_db(cfg, op, n_users_on_sat: int) -> float:
        cm   = op.channel_model
        n_tx = max(1, int(cm.n_tx_x) * int(cm.n_tx_y))
        n_rx = max(1, int(cm.n_rx))
        base = n_tx * n_rx
        mode = (op.radio.beamforming_mode or "none").lower()
        n_u  = max(1, int(n_users_on_sat))
        if mode == "digital":
            eff, mux = float(cm.bf_eff_digital), 1.0
        elif mode == "analog":
            eff, mux = float(cm.bf_eff_analog), 1.0 / n_u
        elif mode == "hybrid":
            eff, mux = float(cm.bf_eff_hybrid), min(1.0, float(cm.num_rf_chains) / n_u)
        else:
            return 0.0
        return 10.0 * np.log10(max(base * eff * mux, 1e-12)) if eff > 0 else 0.0

    # RRM power algorithms (per-sat)
    def _allocate_power(self, links: pd.DataFrame, assigns: dict) -> np.ndarray:
        algo = (getattr(self.cfg.handover, "rrm_power_algorithm", "equal") or "equal").lower()
        pmax = float(self.cfg.operators[0].radio.power_max_w)
        n_arr = links["node_id"].astype(int).to_numpy()
        u_arr = links["user_id"].astype(int).to_numpy()
        g     = links.get("chan_gain_lin", pd.Series(np.ones(len(links)))).astype(float).to_numpy()
        pl    = links.get("path_loss_db",  pd.Series(np.zeros(len(links)))).astype(float).to_numpy()
        g_eff = np.maximum(g * 10**(-pl / 10.0), 1e-30)
        tx    = np.zeros(len(links))

        # Group served users by their assigned sat
        served = {}
        for u, s in assigns.items():
            if int(s) >= 0:
                served.setdefault(int(s), []).append(int(u))
        for sat, uids in served.items():
            mask = (n_arr == sat) & np.isin(u_arr, uids)
            if not mask.any(): continue
            idx = np.where(mask)[0]
            gg  = g_eff[idx]
            if algo == "equal":
                tx[idx] = pmax / len(idx)
            elif algo == "waterfilling":
                n0  = float(getattr(self.cfg.handover, "waterfilling_noise_floor_w", 1e-13))
                n_g = n0 / gg
                lo, hi = n_g.min(), n_g.min() + pmax + n_g.max()
                for _it in range(50):
                    mu = 0.5 * (lo + hi)
                    if np.maximum(mu - n_g, 0.0).sum() > pmax: hi = mu
                    else: lo = mu
                p = np.maximum(mu - n_g, 0.0)
                if p.sum() > 0: p *= pmax / p.sum()
                tx[idx] = p
            elif algo == "utility":
                floor = float(getattr(self.cfg.handover, "utility_min_share", 0.05))
                w = gg / gg.sum()
                w = np.maximum(w, floor)
                tx[idx] = (w / w.sum()) * pmax
            else:
                raise ValueError(f"Unknown rrm_power_algorithm: {algo!r}")
        return tx

    # Interference / SINR (matches the existing InterferenceStage)
    @staticmethod
    def _co_band_interference(links: pd.DataFrame, signal_w: np.ndarray, ic) -> np.ndarray:
        node_id  = links["node_id"].astype(int).to_numpy()
        car      = links["carrier_hz"].astype(float).to_numpy()
        user     = links["user_id"].astype(int).to_numpy()
        intra_on = bool(ic.enable_intra_satellite_interference)
        inter_on = bool(ic.enable_inter_satellite_interference)
        intra_atten = 10.0 ** (-float(ic.intra_beam_isolation_db) / 10.0)
        inter_atten = 10.0 ** (-float(ic.co_channel_isolation_db) / 10.0)
        reuse_gain  = max(0.0, min(1.0, float(ic.frequency_reuse_factor)))
        df = pd.DataFrame({
            "_g":   list(zip(user.tolist(), car.tolist()))
                   if bool(ic.same_user_only) else car.tolist(),
            "_n":   node_id,
            "_sig": signal_w,
        })
        g_total  = df.groupby("_g")["_sig"].transform("sum").to_numpy()
        gn_total = df.groupby(["_g", "_n"])["_sig"].transform("sum").to_numpy()
        intra_contrib = (gn_total - signal_w) if intra_on else np.zeros(len(links))
        inter_contrib = (g_total  - gn_total) if inter_on else np.zeros(len(links))
        return (intra_contrib * intra_atten + inter_contrib * inter_atten) * reuse_gain

    # MIMO geometry helpers (used by the CVX SOCP branch)
    @staticmethod
    def _off_nadir_rad(elev_deg: float, sat_alt_m: float) -> float:
        """Off-nadir angle (rad) at the satellite, given user elevation."""
        return float(np.arcsin(
            _EARTH_R_M * np.cos(np.radians(np.clip(elev_deg, 0.0, 90.0)))
            / (_EARTH_R_M + max(sat_alt_m, 1.0))))

    @staticmethod
    def _steering_vector(theta_rad: float, phi_rad: float,
                          n_x: int, n_y: int,
                          d_over_lambda: float = 0.5) -> np.ndarray:
        """UPA steering vector. Returns a (n_x*n_y,) complex ndarray with
        ||a||^2 = n_x*n_y."""
        kx = np.arange(n_x)
        ky = np.arange(n_y)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        phase = 2.0 * np.pi * d_over_lambda * np.sin(theta_rad) * (
            KX * np.cos(phi_rad) + KY * np.sin(phi_rad))
        return np.exp(1j * phase).ravel()

    # CVXPY MU-MIMO SOCP: minimum-power downlink with QoS rate
    def _solve_cvx_mimo_min_power(self, out: pd.DataFrame,
                                  assigns: dict) -> pd.DataFrame:
        """Per-sat MU-MIMO SOCP. Mutates and returns `out` with tx_power_w,
        signal_w, noise_w, interference_w, sinr_db, link_rate_bps populated.

        Model (per sat s, independent):
            min sum_u ||w_u||^2
            s.t. G_rx*|h^H w_u|^2  >=  gamma_min * (sigma^2 + I_bar + sum_{j!=u} G_rx*|h^H w_j|^2)
                 sum_u ||w_u||^2 <= P_max
        SOC-reformulated via Bengtsson-Ottersten phase rotation.
        """
        if not _HAS_CVXPY:
            _log.error("cvx_mimo_min_power requested but cvxpy is not installed. "
                       "Falling back to equal-power.")
            return self._fallback_equal_power(out, assigns)

        cfg_h  = self.cfg.handover
        cfg_op = self.cfg.operators[0]            # single LEO operator in this scenario

        # Constants (all linear units)
        fixed_ici_w = 10.0 ** (float(cfg_h.cvx_fixed_ici_dbw) / 10.0)   # I-bar (W)
        qos_rate    = float(cfg_h.qos_rate_bps)
        d_lambda    = float(cfg_h.cvx_antenna_spacing_d_over_lambda)
        n_x         = int(cfg_op.channel_model.n_tx_x)
        n_y         = int(cfg_op.channel_model.n_tx_y)
        n_tx        = n_x * n_y
        pmax        = float(cfg_op.radio.power_max_w)
        rx_gain_lin = 10.0 ** (float(cfg_op.radio.antenna_gain_rx_db) / 10.0)
        bw          = float(cfg_op.radio.bandwidth_hz)
        T_sys       = (float(self.cfg.defaults_radio.system_temperature_k)
                       * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        noise_w     = kB * T_sys * bw
        N_bar       = noise_w + fixed_ici_w
        gamma_min   = 2.0 ** (qos_rate / bw) - 1.0
        sat_alt_m   = (float(cfg_op.satellites.shells[0].altitude_km) * 1000.0
                       if cfg_op.satellites.shells else 600_000.0)
        solver_name = str(cfg_h.cvx_solver).upper()
        infeas_pol  = str(cfg_h.cvx_infeasibility_policy).lower()
        slack_lambda = float(cfg_h.cvx_infeas_slack_lambda)

        # Per-link arrays
        n_arr  = out["node_id"].astype(int).to_numpy()
        u_arr  = out["user_id"].astype(int).to_numpy()
        elev   = out.get("elev_deg", pd.Series(np.zeros(len(out)))).astype(float).to_numpy()
        az     = out.get("azimuth",  pd.Series(np.zeros(len(out)))).astype(float).to_numpy()
        gain   = out.get("chan_gain_lin", pd.Series(np.ones(len(out)))).astype(float).to_numpy()
        pl_db  = out.get("path_loss_db",  pd.Series(np.zeros(len(out)))).astype(float).to_numpy()

        tx_pow   = np.zeros(len(out))
        sig_w    = np.zeros(len(out))
        intra_w  = np.zeros(len(out))
        status   = np.full(len(out), "pending", dtype=object)

        # Build sat -> list of served user_ids
        served_by_sat: dict[int, list[int]] = {}
        for u, s in assigns.items():
            if int(s) >= 0:
                served_by_sat.setdefault(int(s), []).append(int(u))

        # Per-sat independent SOCP
        for sat_id, uids in served_by_sat.items():
            # Resolve row indices for users served by this sat
            rows = {}
            for u in uids:
                m = (n_arr == sat_id) & (u_arr == u)
                idx = np.where(m)[0]
                if len(idx): rows[u] = int(idx[0])
            if not rows: continue
            user_list = list(rows.keys())
            k = len(user_list)

            # Build channel matrix H ∈ C^{n_tx x k}: columns are h_{s,u}
            H = np.zeros((n_tx, k), dtype=complex)
            for col, u in enumerate(user_list):
                r = rows[u]
                theta_rad = self._off_nadir_rad(elev[r], sat_alt_m)
                phi_rad   = float(np.radians(az[r]))
                a_vec     = self._steering_vector(theta_rad, phi_rad, n_x, n_y, d_lambda)
                amp       = float(np.sqrt(max(gain[r] * 10.0 ** (-pl_db[r] / 10.0), 1e-30)))
                H[:, col] = amp * a_vec

            # CVXPY problem (real / imag split)
            W_re = cp.Variable((n_tx, k))
            W_im = cp.Variable((n_tx, k))
            H_re, H_im = H.real, H.imag
            sqrt_rx     = float(np.sqrt(rx_gain_lin))
            sqrt_N_bar  = float(np.sqrt(N_bar))
            soc_factor  = float(np.sqrt(1.0 + 1.0 / max(gamma_min, 1e-30)))
            cons = []

            # Build QoS SOC + phase-rotation constraints
            for ui in range(k):
                h_re = H_re[:, ui]; h_im = H_im[:, ui]
                # h^H w_u = (h_re·w_re + h_im·w_im) + j(h_re·w_im - h_im·w_re)
                re_u =  h_re @ W_re[:, ui] + h_im @ W_im[:, ui]
                im_u =  h_re @ W_im[:, ui] - h_im @ W_re[:, ui]
                cons.append(im_u == 0)        # phase fix → Im(h_u^H w_u) = 0
                rhs_stack = []
                for j in range(k):
                    if j == ui: continue
                    re_j = h_re @ W_re[:, j] + h_im @ W_im[:, j]
                    im_j = h_re @ W_im[:, j] - h_im @ W_re[:, j]
                    rhs_stack += [sqrt_rx * re_j, sqrt_rx * im_j]
                rhs_stack.append(sqrt_N_bar)
                rhs = cp.hstack(rhs_stack)
                lhs = soc_factor * sqrt_rx * re_u
                cons.append(cp.SOC(lhs, rhs))

            # Per-sat power budget (convex L2 ball)
            cons.append(cp.sum_squares(W_re) + cp.sum_squares(W_im) <= pmax)

            obj = cp.Minimize(cp.sum_squares(W_re) + cp.sum_squares(W_im))

            # Solve
            problem = cp.Problem(obj, cons)
            solved_status = None
            try:
                solver = getattr(cp, solver_name, cp.CLARABEL)
                problem.solve(solver=solver, verbose=False)
                solved_status = problem.status
            except Exception as e:
                _log.warning("CVXPY solve failed for sat %s: %s", sat_id, e)

            # Infeasibility fallback
            if solved_status not in ("optimal", "optimal_inaccurate"):
                if infeas_pol == "soft_slack":
                    # Soft-slack: relax QoS, penalise. Re-build + re-solve.
                    xi = cp.Variable(k, nonneg=True)
                    cons2 = [cp.sum_squares(W_re) + cp.sum_squares(W_im) <= pmax]
                    for ui in range(k):
                        h_re = H_re[:, ui]; h_im = H_im[:, ui]
                        re_u =  h_re @ W_re[:, ui] + h_im @ W_im[:, ui]
                        im_u =  h_re @ W_im[:, ui] - h_im @ W_re[:, ui]
                        cons2.append(im_u == 0)
                        rhs_stack = []
                        for j in range(k):
                            if j == ui: continue
                            re_j = h_re @ W_re[:, j] + h_im @ W_im[:, j]
                            im_j = h_re @ W_im[:, j] - h_im @ W_re[:, j]
                            rhs_stack += [sqrt_rx * re_j, sqrt_rx * im_j]
                        rhs_stack.append(sqrt_N_bar)
                        # Relaxed factor sqrt(1 + 1/(gamma_min/(1+xi))) ≈ sqrt(1 + (1+xi)/gamma_min)
                        soc_factor_relaxed = cp.sqrt(1.0 + (1.0 + xi[ui]) / max(gamma_min, 1e-30))
                        cons2.append(cp.SOC(soc_factor_relaxed * sqrt_rx * re_u, cp.hstack(rhs_stack)))
                    obj2 = cp.Minimize(cp.sum_squares(W_re) + cp.sum_squares(W_im)
                                       + slack_lambda * cp.sum(xi))
                    try:
                        cp.Problem(obj2, cons2).solve(solver=getattr(cp, solver_name, cp.CLARABEL),
                                                       verbose=False)
                        solved_status = "soft_slack"
                    except Exception as e:
                        _log.warning("Soft-slack solve also failed for sat %s: %s", sat_id, e)
                else:
                    # drop_worst: equal-power among visible users (graceful fallback)
                    for col, u in enumerate(user_list):
                        r = rows[u]
                        p_u = pmax / k
                        tx_pow[r] = p_u
                        sig_w[r]  = float(p_u * rx_gain_lin
                                          * np.abs(H[:, col]).sum() ** 2 / n_tx)
                        status[r] = solved_status or "infeasible_fallback_equal"
                    continue

            # Extract optimal precoders W ∈ C^{n_tx x k}
            if W_re.value is None or W_im.value is None:
                # numerical failure
                for col, u in enumerate(user_list):
                    r = rows[u]
                    tx_pow[r] = pmax / k
                    status[r] = "solver_no_value"
                continue
            W = W_re.value + 1j * W_im.value

            # Per-user outputs from the precoders
            for col, u in enumerate(user_list):
                r = rows[u]
                w_u   = W[:, col]
                h_u   = H[:, col]
                p_u   = float(np.linalg.norm(w_u) ** 2)
                s_u   = float(rx_gain_lin * np.abs(np.vdot(h_u, w_u)) ** 2)
                i_u   = 0.0
                for j in range(k):
                    if j == col: continue
                    i_u += np.abs(np.vdot(h_u, W[:, j])) ** 2
                tx_pow[r]  = p_u
                sig_w[r]   = s_u
                intra_w[r] = float(rx_gain_lin * i_u)
                status[r]  = "optimal" if solved_status == "optimal" else solved_status

        # Aggregate per-link metrics (via the single shared core)
        # The interference fed to the shared SINR is the intra-cell term on top
        # of the effective noise N_bar (which already folds in I_ext).
        from ntnconnect.common.physics import (sinr_db_from_powers, snr_db_from_powers,
                               spectral_efficiency_bps_hz)
        sinr_db = sinr_db_from_powers(sig_w, N_bar, intra_w)
        se      = spectral_efficiency_bps_hz(sinr_db)
        out["tx_power_w"]                 = tx_pow
        out["signal_w"]                   = sig_w
        out["noise_w"]                    = noise_w
        out["interference_w"]             = intra_w + fixed_ici_w
        out["interference_intra_w"]       = intra_w
        out["interference_ext_w"]         = fixed_ici_w
        out["snr_db"]                     = snr_db_from_powers(sig_w, noise_w)
        out["sinr_db"]                    = sinr_db
        out["spectral_efficiency_bps_hz"] = se
        out["link_rate_bps"]              = se * bw
        out["interference_model"]         = "fixed_ici_plus_intra"
        out["radio_status"]               = status
        # BF gain is implicit in the precoder; report effective per-link gain
        out["bf_gain_db"] = 10.0 * np.log10(np.maximum(
            sig_w / np.maximum(tx_pow * 10.0 ** (-pl_db / 10.0)
                                * np.maximum(gain, 1e-30) * rx_gain_lin, 1e-30),
            1e-30))
        return out

    def _fallback_equal_power(self, out: pd.DataFrame, assigns: dict) -> pd.DataFrame:
        """Used when cvxpy is unavailable. Equivalent to rrm_power_algorithm='equal'."""
        out["tx_power_w"] = self._allocate_power(out, assigns)   # algo='equal' path
        return out

    # orchestrator entry
    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty: return
        assigns = state.data.get("assignments", {}) or {}
        if not assigns:
            # No HO yet (first tick before HO ran) - leave the frame untouched.
            return

        out = links.copy()

        # Registry dispatch (Path B)
        # Any algorithm registered with @register_radio_algorithm("...")
        # in v1/ntnconnect/agents/radio/*.py is selectable by its name in
        # cfg.handover.rrm_power_algorithm. The algorithm writes every
        # radio column on `links` itself and we return.
        #
        # The closed-form algorithms (equal | waterfilling | utility) are
        # not in the registry -- they continue to use the
        # inline BF + RRM + interference path below.
        algo_obj = get_radio_algorithm(self.cfg)
        if algo_obj is not None:
            state.data["links"] = algo_obj.compute(out, assigns, state=state)
            return

        # 1) Beamforming gain per operator and per sat
        n_arr = out["node_id"].astype(int).to_numpy()
        # users-per-sat from current assignments
        n_on_sat = {}
        for u, s in assigns.items():
            if int(s) >= 0:
                n_on_sat[int(s)] = n_on_sat.get(int(s), 0) + 1
        op_by_id = {op.id: op for op in self.cfg.operators}
        bf_db = np.zeros(len(out))
        for i, s in enumerate(n_arr):
            oid = int(out["operator_id"].iloc[i])
            op  = op_by_id.get(oid)
            if op is None: continue
            bf_db[i] = self._bf_gain_db(self.cfg, op, n_on_sat.get(int(s), 1))
        out["bf_gain_db"] = bf_db
        out["antenna_gain_db"] = out.get(
            "antenna_gain_db", pd.Series(np.zeros(len(out)))
        ).astype(float).to_numpy() + bf_db

        # 2) RRM power per link
        out["tx_power_w"] = self._allocate_power(out, assigns)

        # 3) Signal, noise, interference, SINR
        ic       = self.cfg.defaults_interference
        gain_lin = 10.0 ** (out["antenna_gain_db"].astype(float).to_numpy() / 10.0)
        pl_lin   = 10.0 ** (-out["path_loss_db"].astype(float).to_numpy() / 10.0)
        chan_g   = np.asarray(out.get("chan_gain_lin", np.ones(len(out))), dtype=float)
        signal_w = out["tx_power_w"].astype(float).to_numpy() * \
                   gain_lin * np.maximum(chan_g, 1e-30) * pl_lin
        T_sys    = (float(self.cfg.defaults_radio.system_temperature_k)
                    * 10.0 ** (self.cfg.defaults_radio.noise_figure_db / 10.0))
        bw       = out["bandwidth_hz"].astype(float).to_numpy()
        noise_w  = kB * T_sys * np.maximum(bw, 1.0)

        model = (ic.model or "off").lower()
        if   model == "off":      interf_w = np.zeros_like(signal_w)
        elif model == "fixed":    interf_w = np.full_like(signal_w,
                                                          10.0 ** (float(ic.fixed_interference_dbw) / 10.0))
        elif model == "physical": interf_w = self._co_band_interference(out, signal_w, ic)
        else:
            _log.warning("Unknown interference.model='%s' - assuming off.", model)
            interf_w = np.zeros_like(signal_w)

        sinr = signal_w / np.maximum(interf_w + noise_w, 1e-30)
        snr  = signal_w / np.maximum(noise_w,            1e-30)
        out["signal_w"]                   = signal_w
        out["noise_w"]                    = noise_w
        out["interference_w"]             = interf_w
        out["snr_db"]                     = 10.0 * np.log10(np.maximum(snr,  1e-30))
        out["sinr_db"]                    = 10.0 * np.log10(np.maximum(sinr, 1e-30))
        out["spectral_efficiency_bps_hz"] = np.log2(1.0 + sinr)
        out["link_rate_bps"]              = out["spectral_efficiency_bps_hz"] * bw
        out["interference_model"]         = model

        state.data["links"] = out
