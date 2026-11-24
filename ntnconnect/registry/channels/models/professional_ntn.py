# Professional NTN: per-antenna Rician (TR 38.811). FSPL/rain/gas/shadow/Doppler
# are pipeline components; model produces only h. model_type="Professional_NTN".
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


# Smooth-interpolated 3GPP-NTN K-factor table (TR 38.811-style)
_EL_REF  = np.array([0.0, 10.0, 30.0, 60.0, 90.0])
_MU_REF  = np.array([4.0,  6.0,  9.0, 14.0, 16.0])   # dB
_SIG_REF = np.array([4.0,  3.5,  2.5,  1.5,  1.0])   # dB


@register_model("Professional_NTN")
class ProfessionalNTN(AbstractChannelModel):
    """3GPP TR 38.811-style NTN channel.

    UNIFORM contract with the other 7 models: this base model produces
    only the physical effects that make this model **unique**:
        - FSPL with configurable intercept
        - Rain attenuation (AR-1 Gilbert-Elliott-like)
        - Atmospheric gas absorption (elevation-dependent)
        - Rician scalar with elevation-dependent K-factor (AR-1, TR 38.811)

    Shadowing, Doppler shift, and propagation-delay phase are LEFT to
    the pipeline components (ShadowingComponent, DopplerComponent,
    PropagationDelayComponent) so the model contract matches the other
    7 channel models. The DT predictor in scn_channel_predict consumes
    the fully-assembled channel after the pipeline has run.
    """

    # Contract marker: empty set means "no overlap with components"
    # i.e. pipeline runs every configured component on top of this model.
    handles_internally = frozenset()
    # Owns its own per-antenna construction - SteeringUPAComponent becomes
    # a no-op for this model. Avoids the wasted "scalar Rician + steering"
    # double-draw.
    produces_per_antenna_h = True

    def __init__(self, cfg):
        super().__init__(cfg)
        # Knobs still readable here for back-compat introspection - but
        # the actual physics moved into the corresponding components.

    # main compute_channel call
    def compute_channel(self, dist_m, elev_deg, dt_sec,
                        state=None, pos_m=None,
                        range_rate_m_s=None, azimuth_deg=None):
        """Per-antenna Rician channel - built directly, no double-draw.

        FSPL is provided by FSPLComponent (pre-model). K-factor comes from
        whichever K-factor component ran in the pre-model phase, via
        state.data["k_factor_db_pending"]. We then build h_per_antenna
        = sqrt(K/(K+1))·a + sqrt(1/(K+1))·CN(0,I/N_t) in one pass.
        """
        from ntnconnect.registry.channels._steering import per_antenna_rician_batch, off_nadir_rad
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        n = len(d)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))

        # K-factor - preferred path: pre-model K component set it.
        k_db_default = float(getattr(self.cfg, "threegpp_k_factor_los_db", 10.0))
        k_db = (state.data.get("k_factor_db_pending") if state is not None else None)
        if k_db is None or np.asarray(k_db).size != n:
            k_db = np.full(n, k_db_default)
        else:
            k_db = np.asarray(k_db, dtype=float)

        # Per-antenna build via the shared helper.
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y
        d_lambda  = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        strategy   = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()

        # Vectorised per-antenna build for all links at once. Per-element
        # unit-variance diffuse (E[||g||^2]=n_tx) so the realized per-antenna K
        # equals the configured K; rician_unit_mean_fading=False restores legacy.
        unit_mean = bool(getattr(self.cfg, "rician_unit_mean_fading", True))
        rng = getattr(state, "rng", None) or np.random
        theta = off_nadir_rad(elev, sat_alt_m)
        phi   = np.radians(az)
        h = per_antenna_rician_batch(
            k_db, theta, phi, n_x, n_y, strategy=strategy,
            dist_m=d, carrier_hz=carrier_hz, d_over_lambda=d_lambda,
            unit_mean=unit_mean, rng=rng)

        # loss_db = 0 because FSPL is owned by FSPLComponent. Post-model
        # rain/gas/shadow then add to loss_db.
        return ChannelData(
            loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
            extras={"k_factor_db": k_db},
        )
