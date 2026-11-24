# Radio algorithm: cvx_mimo_min_power - per-sat MU-MIMO downlink beamforming SOCP.
# Min total power s.t. per-user QoS rate + per-sat budget; inter-sat interference fixed.
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

try:
    import cvxpy as cp                                # type: ignore
    _HAS_CVXPY = True
except Exception:                                     # pragma: no cover
    cp = None
    _HAS_CVXPY = False

from ntnconnect.registry.agents.radio_algorithm import BaseRadioAlgorithm, register_radio_algorithm
from ntnconnect.registry.agents.radio._utils import (
    build_channel_matrix, per_sat_groups, finalize_radio_columns,
)

_log = logging.getLogger(__name__)


@register_radio_algorithm("cvx_mimo_min_power")
class CvxMimoMinPower(BaseRadioAlgorithm):
    """Per-sat SOCP. Minimum total tx-power with QoS rate per user."""

    def compute(self, links: pd.DataFrame, assigns: dict,
                state=None) -> pd.DataFrame:
        out = links     # caller already handed us a writable copy
        if not _HAS_CVXPY:
            _log.error("cvx_mimo_min_power requires cvxpy; falling back to "
                       "equal-power split.")
            return self._fallback_equal_power(out, assigns)

        cfg_h  = self.cfg.handover
        cfg_op = self.cfg.operators[0]

        # Constants
        fixed_ici_w = 10.0 ** (float(cfg_h.cvx_fixed_ici_dbw) / 10.0)
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
        noise_w     = 1.380649e-23 * T_sys * bw
        N_bar       = noise_w + fixed_ici_w
        gamma_min   = 2.0 ** (qos_rate / bw) - 1.0
        sat_alt_m   = (float(cfg_op.satellites.shells[0].altitude_km) * 1000.0
                       if cfg_op.satellites.shells else 600_000.0)
        solver_name  = str(cfg_h.cvx_solver).upper()
        infeas_pol   = str(cfg_h.cvx_infeasibility_policy).lower()
        slack_lambda = float(cfg_h.cvx_infeas_slack_lambda)

        # Per-link output arrays
        tx_pow  = np.zeros(len(out))
        sig_w   = np.zeros(len(out))
        intra_w = np.zeros(len(out))
        status  = np.full(len(out), "pending", dtype=object)

        # One SOCP per sat
        for sat_id, rows, user_list in per_sat_groups(out, assigns):
            k = len(user_list)
            H = build_channel_matrix(out, rows, user_list,
                                     n_x, n_y, d_lambda, sat_alt_m)
            # CVXPY (real/imag split)
            W_re = cp.Variable((n_tx, k))
            W_im = cp.Variable((n_tx, k))
            H_re, H_im = H.real, H.imag
            sqrt_rx     = float(np.sqrt(rx_gain_lin))
            sqrt_N_bar  = float(np.sqrt(N_bar))
            soc_factor  = float(np.sqrt(1.0 + 1.0 / max(gamma_min, 1e-30)))
            cons = []
            for ui in range(k):
                h_re = H_re[:, ui]; h_im = H_im[:, ui]
                re_u =  h_re @ W_re[:, ui] + h_im @ W_im[:, ui]
                im_u =  h_re @ W_im[:, ui] - h_im @ W_re[:, ui]
                cons.append(im_u == 0)                # phase fix
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
            cons.append(cp.sum_squares(W_re) + cp.sum_squares(W_im) <= pmax)
            obj = cp.Minimize(cp.sum_squares(W_re) + cp.sum_squares(W_im))

            solved_status = None
            try:
                solver = getattr(cp, solver_name, cp.CLARABEL)
                cp.Problem(obj, cons).solve(solver=solver, verbose=False)
                solved_status = "optimal"            # CVXPY sets prob.status
            except Exception as e:
                _log.warning("cvx_mimo_min_power: sat %s failed: %s", sat_id, e)

            # Infeasibility fallback
            if solved_status not in ("optimal", "optimal_inaccurate") or W_re.value is None:
                if infeas_pol == "soft_slack":
                    self._soft_slack_solve(H, n_tx, k, sqrt_rx, sqrt_N_bar,
                                           soc_factor, gamma_min, pmax,
                                           solver_name, slack_lambda,
                                           W_re, W_im, cons)
                else:
                    for col, u in enumerate(user_list):
                        r = rows[u]
                        p_u = pmax / k
                        tx_pow[r] = p_u
                        sig_w[r]  = float(p_u * rx_gain_lin
                                           * np.abs(H[:, col]).sum() ** 2 / n_tx)
                        status[r] = solved_status or "infeasible_fallback_equal"
                    continue

            if W_re.value is None or W_im.value is None:
                for col, u in enumerate(user_list):
                    r = rows[u]; tx_pow[r] = pmax / k; status[r] = "solver_no_value"
                continue
            W = W_re.value + 1j * W_im.value

            for col, u in enumerate(user_list):
                r = rows[u]
                w_u = W[:, col]; h_u = H[:, col]
                tx_pow[r] = float(np.linalg.norm(w_u) ** 2)
                sig_w[r]  = float(rx_gain_lin * np.abs(np.vdot(h_u, w_u)) ** 2)
                acc = 0.0
                for j in range(k):
                    if j == col: continue
                    acc += np.abs(np.vdot(h_u, W[:, j])) ** 2
                intra_w[r] = float(rx_gain_lin * acc)
                status[r]  = "optimal"

        return finalize_radio_columns(out, tx_pow, sig_w, intra_w, status,
                                      self.cfg,
                                      fixed_ici_w=fixed_ici_w,
                                      interference_model_name="fixed_ici_plus_intra")

    # Fallback when cvxpy is missing
    def _fallback_equal_power(self, out: pd.DataFrame, assigns: dict) -> pd.DataFrame:
        cfg_op = self.cfg.operators[0]
        pmax   = float(cfg_op.radio.power_max_w)
        tx     = np.zeros(len(out))
        n_arr  = out["node_id"].astype(int).to_numpy()
        u_arr  = out["user_id"].astype(int).to_numpy()
        served = {}
        for u, s in assigns.items():
            if int(s) >= 0:
                served.setdefault(int(s), []).append(int(u))
        for s, uids in served.items():
            m = (n_arr == s) & np.isin(u_arr, uids)
            idx = np.where(m)[0]
            if len(idx): tx[idx] = pmax / len(idx)
        out["tx_power_w"] = tx
        return out

    # Optional: soft-slack solve for infeasibility (kept inline)
    @staticmethod
    def _soft_slack_solve(H, n_tx, k, sqrt_rx, sqrt_N_bar, soc_factor,
                          gamma_min, pmax, solver_name, slack_lambda,
                          W_re, W_im, cons):
        """Soft-slack reformulation. Tries to relax QoS targets with a
        nonneg slack xi per user and minimise power + lambda * sum(xi)."""
        xi = cp.Variable(k, nonneg=True)
        cons2 = [cp.sum_squares(W_re) + cp.sum_squares(W_im) <= pmax]
        H_re, H_im = H.real, H.imag
        for ui in range(k):
            h_re = H_re[:, ui]; h_im = H_im[:, ui]
            re_u = h_re @ W_re[:, ui] + h_im @ W_im[:, ui]
            im_u = h_re @ W_im[:, ui] - h_im @ W_re[:, ui]
            cons2.append(im_u == 0)
            rhs_stack = []
            for j in range(k):
                if j == ui: continue
                re_j = h_re @ W_re[:, j] + h_im @ W_im[:, j]
                im_j = h_re @ W_im[:, j] - h_im @ W_re[:, j]
                rhs_stack += [sqrt_rx * re_j, sqrt_rx * im_j]
            rhs_stack.append(sqrt_N_bar)
            f_relaxed = cp.sqrt(1.0 + (1.0 + xi[ui]) / max(gamma_min, 1e-30))
            cons2.append(cp.SOC(f_relaxed * sqrt_rx * re_u, cp.hstack(rhs_stack)))
        obj2 = cp.Minimize(cp.sum_squares(W_re) + cp.sum_squares(W_im)
                           + slack_lambda * cp.sum(xi))
        try:
            cp.Problem(obj2, cons2).solve(solver=getattr(cp, solver_name, cp.CLARABEL),
                                          verbose=False)
        except Exception:
            pass
