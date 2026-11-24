# Stage: DT glue (after channel, before radio). Build h_{s,u}, push to channel_twin,
# run cfg.handover.channel_predictor -> state["channel_predictions"]. No-op if unset.
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry   import register_dt_stage
from ntnconnect.registry.channels.predictor import get_channel_predictor
from ntnconnect.registry.channels._history  import ChannelHistoryStore
from ntnconnect.registry.channels.dt_channel import DTChannel
from ntnconnect.registry.agents.radio._utils import steering_vector, off_nadir_rad

# Aged-geometry noise (matches Colab DigitalTwins.generate_matrix).
_NOMINAL_AGE_ELEV_STD_DEG = 0.2
_NOMINAL_AGE_AZ_STD_DEG   = 0.2
# Fixed "other loss" on top of FSPL for the nominal channel - matches the
# Colab `gain = 10 ** -(fspl + 0.5)/20`.
_NOMINAL_OTHER_LOSS_DB    = 0.5

_log = logging.getLogger(__name__)


@register_dt_stage("scn_channel_predict")
class ScnChannelPredict(Stage):
    reads       = ["links"]
    writes      = ["twin_channel"]      # + state.data["channel_predictions"] (side)
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._predictor = None
        self._ws = int(getattr(cfg.handover, "channel_predictor_window", 20))

    def _ensure_store(self):
        # per-sat DTs live on the Satellite objects. The engine
        # still keeps a `channel_twin` attribute, but it's now a thin
        # dict[(sat_id, user_id) -> deque] aggregated VIEW for back-compat
        # rather than a global store. Each Satellite owns the canonical
        # history.
        if getattr(self.engine, "channel_twin", None) is None:
            n_x = int(self.cfg.defaults_channel.n_tx_x)
            n_y = int(self.cfg.defaults_channel.n_tx_y)
            # Aggregated view; populated lazily as sats are touched.
            self.engine.channel_twin = ChannelHistoryStore(
                window_size=self._ws, n_tx=n_x * n_y)
        if self._predictor is None:
            self._predictor = get_channel_predictor(self.cfg)

    def _mirror_store_into_state(self, state):
        # expose engine.channel_twin via state.data so consumers
        # (radio algorithms, hooks) can read it without reaching back into
        # the engine. Same object - no copy - so back-compat is total.
        if state.data.get("channel_twin") is not self.engine.channel_twin:
            state.data["channel_twin"] = self.engine.channel_twin
        # expose per-sat dict {sat_id -> DTChannel} so radio
        # algorithms can pick the right DT for the (sat, user) link they
        # are designing for. Legacy `state.data["dt_channel"]` is kept
        # as the first sat's DT (a coarse aggregator) for back-compat.
        sats = getattr(self.engine, "satellites", {}) or {}
        state.data["per_sat_dt_channels"] = {
            sid: s.dt_channel for sid, s in sats.items() if s.dt_channel is not None
        }
        # Legacy single DTChannel - pick first sat's so legacy radio
        # algorithms that still read state.data["dt_channel"] get
        # something usable. The recommended path is per_sat_dt_channels.
        if state.data["per_sat_dt_channels"]:
            state.data["dt_channel"] = next(iter(state.data["per_sat_dt_channels"].values()))
        else:
            state.data["dt_channel"] = None

    def run(self, state):
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            state.data["channel_predictions"] = {}
            return
        self._ensure_store()
        self._mirror_store_into_state(state)
        if self._predictor is None:
            state.data["channel_predictions"] = {}
            return
        self._predictor.on_tick_start()

        cfg_op = self.cfg.operators[0]
        n_x = int(cfg_op.channel_model.n_tx_x)
        n_y = int(cfg_op.channel_model.n_tx_y)
        d_lambda = float(getattr(self.cfg.handover,
                                 "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = (float(cfg_op.satellites.shells[0].altitude_km) * 1000.0
                     if cfg_op.satellites.shells else 600_000.0)

        u_arr = links["user_id"].astype(int).to_numpy()
        n_arr = links["node_id"].astype(int).to_numpy()
        e_arr = links.get("elev_deg", pd.Series(np.zeros(len(links)))).astype(float).to_numpy()
        # BUG #6 FIX: geometry writes "azimuth_deg"; accept either name.
        if "azimuth_deg" in links.columns:
            a_arr = links["azimuth_deg"].astype(float).to_numpy()
        elif "azimuth" in links.columns:
            a_arr = links["azimuth"].astype(float).to_numpy()
        else:
            a_arr = np.zeros(len(links), dtype=float)
        g_arr = links.get("chan_gain_lin", pd.Series(np.ones(len(links)))).astype(float).to_numpy()
        pl_arr= links.get("path_loss_db",  pd.Series(np.zeros(len(links)))).astype(float).to_numpy()
        # BUG #3 + #4 + #5 FIX: pull per-link Rician K-factor from the
        # channel model so h_meas has real per-antenna NLoS variation
        # instead of being a deterministic function of geometry.
        if "k_factor_db" in links.columns:
            k_arr = links["k_factor_db"].astype(float).to_numpy()
        else:
            k_arr = np.full(len(links),
                            float(getattr(self.cfg.handover, "mc_rician_k_db", 8.0)))
        # Seeded RNG for the per-antenna NLoS injection - deterministic
        # given the engine master seed.
        rng = getattr(self.engine, "rng_channel", np.random.default_rng())

        predictions: dict = {}
        ground_truth: dict = {}

        # PREDICT first using only the strict-past history, THEN
        # push the current tick's measurement so the next tick has it.
        # Push-before-predict allows the predictor to read its own
        # ground-truth back out of the deque on the same tick - trivial
        # zero-error prediction. The fix below is the standard
        # online-learning protocol (data hygiene).
        #
        # two-PIPELINE DT ARCHITECTURE:
        #  h_meas (real channel) = amp_real · (√(K/(K+1)) · a_real
        #  + √(1/(K+1)) · CN(0,I/N_t))
        #  uses path_loss_db (incl. rain/gas/shadow),
        #  k_factor_db, and steering from true geometry.
        #  h_nom (NOMINAL channel) = amp_nom · a_aged
        #  uses only FSPL + a small fixed loss and
        #  steering from AGED (noisy) geometry.
        #  Equivalent to DigitalTwins.generate_matrix
        #  in the Colab.
        # Both are pushed to ChannelHistoryStore so the scalar-correction
        # predictor can fit `h_meas ≈ alpha · h_nom`.
        n_tx = n_x * n_y
        f_hz       = float(cfg_op.radio.carrier_freq_hz)
        f_ghz_log  = 20.0 * np.log10(f_hz / 1.0e9)
        fspl_intercept = float(getattr(self.cfg.defaults_channel,
                                       "threegpp_fspl_intercept", 92.45))
        # Per-antenna h from the channel pipeline (SteeringUPAComponent).
        # When present, we read h_meas directly from here instead of
        # re-building it from the scalar Rician + per-antenna NLoS.
        h_per_ant = (state.data.get("h_per_antenna", None)
                     if state is not None else None)
        # Per-antenna h_nom from the DT channel pipeline (dt_channel stage).
        # When present, we prefer it over the inline geometric-nominal
        # construction further down. When absent (DT stage not enabled
        # or per-op DT cfg None), the inline fallback runs.
        h_nom_per_ant = (state.data.get("h_nom_per_antenna", None)
                          if state is not None else None)
        for r in range(len(links)):
            sat_id  = int(n_arr[r]); user_id = int(u_arr[r])
            # real channel h_meas
            # PREFER the per-antenna h produced by the channel pipeline's
            # SteeringUPAComponent (if enabled) so the BF and the DT
            # consume the same channel realisation. Fallback to the
            # inline construction when the component is disabled.
            h_pipeline = h_per_ant.get(int(r)) if h_per_ant is not None else None
            theta   = off_nadir_rad(e_arr[r], sat_alt_m)
            phi     = float(np.radians(a_arr[r]))
            a_vec   = steering_vector(theta, phi, n_x, n_y, d_lambda)
            if h_pipeline is not None and h_pipeline.size == n_tx:
                # Channel pipeline already lifted h to per-antenna. Apply
                # large-scale amplitude here (the pipeline returns the
                # small-scale fading only).
                amp_ls = float(np.sqrt(max(10.0 ** (-pl_arr[r] / 10.0), 1e-30)))
                h_meas = amp_ls * h_pipeline
            else:
                amp     = float(np.sqrt(max(g_arr[r] * 10.0 ** (-pl_arr[r] / 10.0), 1e-30)))
                K_lin   = 10.0 ** (k_arr[r] / 10.0)
                los_w   = float(np.sqrt(K_lin / (K_lin + 1.0)))
                nlos_w  = float(np.sqrt(1.0   / (K_lin + 1.0)))
                nlos    = (rng.standard_normal(n_tx)
                           + 1j * rng.standard_normal(n_tx)) / np.sqrt(2.0 * n_tx)
                h_meas  = amp * (los_w * a_vec + nlos_w * nlos)
            ground_truth[(sat_id, user_id)] = h_meas

            # NOMINAL channel h_nom (Model #2)
            # PREFERRED: read from the dt_channel stage's pipeline.
            h_nom_pipeline = (h_nom_per_ant.get(int(r))
                               if h_nom_per_ant is not None else None)
            if h_nom_pipeline is not None and h_nom_pipeline.size == n_tx:
                h_nom = h_nom_pipeline
            else:
                # Fallback: inline construction from aged geometry +
                # FSPL-only path loss. Kept for back-compat when the
                # dt_channel stage isn't configured.
                elev_nom  = float(np.clip(
                    e_arr[r] + rng.normal(0.0, _NOMINAL_AGE_ELEV_STD_DEG),
                    0.0, 90.0))
                az_nom    = float(a_arr[r] + rng.normal(0.0, _NOMINAL_AGE_AZ_STD_DEG))
                theta_nom = off_nadir_rad(elev_nom, sat_alt_m)
                phi_nom   = float(np.radians(az_nom))
                a_vec_nom = steering_vector(theta_nom, phi_nom, n_x, n_y, d_lambda)
                dist_m_link = float(links.iloc[r].get(
                    "dist_m", links.iloc[r].get("ground_dist_m", 1.0)))
                fspl_db_link = (fspl_intercept + f_ghz_log
                                + 20.0 * np.log10(max(dist_m_link / 1.0e3, 1e-6)))
                pl_nom_db = fspl_db_link + _NOMINAL_OTHER_LOSS_DB
                amp_nom   = float(np.sqrt(max(10.0 ** (-pl_nom_db / 10.0), 1e-30)))
                h_nom     = amp_nom * a_vec_nom

            # 1) Predict on history-before-this-tick. view() may be None
            #  if this is the first time this (sat,user) pair is seen.
            # ask this satellite's own DT.
            sat_obj = self.engine.get_or_create_satellite(sat_id)
            view = (sat_obj.channel_twin.view(sat_id, user_id)
                    if sat_obj is not None else None)
            sat_pred = (sat_obj.predictor if sat_obj is not None
                        else self._predictor)
            if view is not None and sat_pred is not None:
                h_pred, Sigma, status = sat_pred.predict(view, lag_steps=1)
                # defensively replace any NaN/Inf the predictor produces.
                if (not np.all(np.isfinite(h_pred))) or (not np.all(np.isfinite(Sigma))):
                    _log.warning("Predictor %s produced non-finite output for "
                                 "(sat=%s,user=%s); using cold-start surrogate",
                                 type(self._predictor).__name__, sat_id, user_id)
                    pwr   = float(np.mean(np.abs(h_meas)**2)); pwr = max(pwr, 1e-30)
                    h_pred = h_meas
                    Sigma  = (0.1 * pwr + 1e-12) * np.eye(h_meas.size, dtype=complex)
                    status = "nan_recover"
                predictions[(sat_id, user_id)] = {
                    "h_pred":  h_pred,
                    "Sigma":   Sigma,
                    "status":  status,
                }
                # Mirror into this sat's DTChannel posterior cache so
                # sat.dt_channel.mean()/.cov()/.sample() all work.
                if sat_obj is not None and sat_obj.dt_channel is not None:
                    sat_obj.dt_channel._posterior[(int(sat_id), int(user_id))] = {
                        "h_pred": h_pred, "Sigma": Sigma,
                        "status": status, "L": None,
                    }

            # 2) THEN push the realised channel so the next tick has it.
            # Push both the realised channel (h_meas) AND the geometric
            # nominal (h_nom) into this satellite's history store (per-sat
            # learning). Also mirror into engine.channel_twin for legacy
            # callers that still read the global store.
            if sat_obj is not None:
                sat_obj.channel_twin.push(sat_id, user_id, h_meas, h_nominal=h_nom)
            self.engine.channel_twin.push(sat_id, user_id, h_meas, h_nominal=h_nom)

        # Hand the realised channels to the predictor for online learning
        self._predictor.on_tick_end(ground_truth)
        state.data["channel_predictions"] = predictions

        # Publish a links-aligned twin channel to SHARED STATE so the radio
        # context (dt branch) can size power on the PREDICTED channel exactly as
        # it sizes on the true channel. Units match link_metrics.channel_power_gain:
        #   twin_gain = 10^(G/10) * ||h_pred||^2 / N_t   (== true_gain form).
        # Rows with no prediction yet (cold start / first sighting) keep the known
        # large-scale gain (antenna x path loss), so they are sized on geometry
        # rather than spuriously blocked. sigma = 10^(G/10)*trace(Sigma)/N_t is a
        # per-link prediction-uncertainty proxy (informational).
        g_db = links.get("antenna_gain_db",
                         pd.Series(np.zeros(len(links)))).astype(float).to_numpy()
        g_lin = 10.0 ** (g_db / 10.0)
        tw_gain  = g_lin * 10.0 ** (-pl_arr / 10.0)      # default: known large-scale
        tw_sigma = np.zeros(len(links), dtype=float)
        for r in range(len(links)):
            pr = predictions.get((int(n_arr[r]), int(u_arr[r])))
            if pr is None:
                continue
            hp = np.asarray(pr["h_pred"]); sg = np.asarray(pr["Sigma"])
            gfac = g_lin[r] / max(n_tx, 1)
            tw_gain[r]  = gfac * float(np.sum(np.abs(hp) ** 2))
            tw_sigma[r] = gfac * float(np.trace(sg).real)
        state.data["twin_channel"] = {"gain": tw_gain, "sigma": tw_sigma}
