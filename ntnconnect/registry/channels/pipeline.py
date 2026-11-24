# Channel model registry + pipeline: base model -> ordered components (from cfg).
from __future__ import annotations
import logging
from typing import Any, Dict, List, Type, Optional
import numpy as np

from ntnconnect.registry.channels.base import (
    AbstractChannelModel, AbstractChannelComponent,
    ChannelData, ChannelState,
)
from ntnconnect.registry.channels.components.shadowing import ShadowingComponent
from ntnconnect.registry.channels.components.doppler import DopplerComponent
from ntnconnect.registry.channels.components.propagation_delay import PropagationDelayComponent
from ntnconnect.registry.channels.components.polarization import PolarizationComponent
from ntnconnect.registry.channels.components.spatial_correlation import SpatialCorrelationComponent
from ntnconnect.registry.channels.components.jakes import JakesCorrelationComponent
from ntnconnect.registry.channels.components.rain_ar1 import RainAR1Component
from ntnconnect.registry.channels.components.rain_p618 import RainP618Component
from ntnconnect.registry.channels.components.gas_absorption import GasCosecComponent
from ntnconnect.registry.channels.components.k_factor_38811 import KFactor38811Component
from ntnconnect.registry.channels.components.fspl import FSPLComponent
from ntnconnect.registry.channels.components.k_factor_constant import KFactorConstantComponent
from ntnconnect.registry.channels.components.k_factor_sigmoid  import KFactorSigmoidComponent
from ntnconnect.registry.channels.components.steering_upa      import SteeringUPAComponent
from ntnconnect.registry.channels.components.dt_observation_noise import DTObservationNoiseComponent
from ntnconnect.registry.channels.components.k_factor_38811_scenario import KFactor38811ScenarioComponent
from ntnconnect.registry.channels.components.k_factor_p681           import KFactorP681Component
from ntnconnect.registry.channels.components.k_factor_distance       import KFactorDistanceComponent
from ntnconnect.registry.channels.components.shadowing_38811         import Shadowing38811Component
from ntnconnect.registry.channels.components.shadowing_gudmundson    import ShadowingGudmundsonComponent
from ntnconnect.registry.channels.components.rain_p838_hv            import RainP838HVComponent
from ntnconnect.registry.channels.components.gas_p676                import GasP676Component
from ntnconnect.registry.channels.components.cloud_p840              import CloudP840Component
from ntnconnect.registry.channels.components.refraction_p834         import RefractionP834Component
from ntnconnect.registry.channels.components.beam_pattern            import BeamPatternComponent

_log = logging.getLogger(__name__)
_REGISTRY: Dict[str, Type[AbstractChannelModel]] = {}

# Single-choice category selectors (see ChannelConfig.rain_model, etc.)
# Each maps a category enum value -> the underlying enable_* boolean. Selecting a
# category resolves to exactly one model (all others cleared), so mutually
# exclusive models (two rain models, two K-factor sources) cannot coexist.
_CATEGORY_MAPS = {
    "rain_model": {"p838": "enable_rain_p838_hv", "p618": "enable_rain_p618",
                   "ar1": "enable_rain_ar1"},
    "gas_model":  {"p676": "enable_gas_p676", "cosec": "enable_gas_absorption"},
    "shadowing_model": {"p38811": "enable_shadowing_38811",
                        "gudmundson": "enable_shadowing_gudmundson",
                        "lognormal": "enable_shadowing"},
    "kfactor_model": {"p38811": "enable_k_factor_38811",
                      "p38811_scenario": "enable_k_factor_38811_scenario",
                      "constant": "enable_k_factor_constant",
                      "sigmoid": "enable_k_factor_sigmoid",
                      "p681": "enable_k_factor_p681",
                      "distance": "enable_k_factor_distance"}}


def resolve_category_selectors(cfg) -> None:
    """Map single-choice category enums onto the underlying enable_* booleans,
    enforcing exactly one model per category. A category whose selector is None
    is left to its legacy enable_* flags; "off" clears the whole category."""
    for sel_attr, mapping in _CATEGORY_MAPS.items():
        sel = getattr(cfg, sel_attr, None)
        if sel is None:
            continue                       # legacy: leave enable_* flags as set
        sel = str(sel).lower()
        for key, flag in mapping.items():
            try:
                setattr(cfg, flag, (key == sel))   # only the chosen model = True
            except Exception:
                pass


def register_model(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def list_models(): return dict(_REGISTRY)


def get_model_class(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown channel model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


# ChannelPipeline
class ChannelPipeline:
    """Chained execution: base model -> [components].

    Components are auto-selected from the cfg flags. Users can pass an
    explicit list to override.
    """
    def __init__(self, model: AbstractChannelModel,
                 components: Optional[List[AbstractChannelComponent]] = None,
                 cfg: Any = None,
                 role: str = "physical",
                 rng=None):
        # role="physical" -> the true channel pipeline (produces h_meas).
        # role="dt" -> the digital-twin pipeline; DT-only components (e.g.
        #  observation noise) are added only here, keeping the
        #  physical pipeline free of twin concerns.
        self.model = model
        self.cfg = cfg if cfg is not None else getattr(model, "cfg", None)
        self.role = str(role).lower()
        # Each pipeline owns an INDEPENDENT RNG so the physical and DT pipelines
        # never share a random stream (activating one must not perturb the other).
        # Without this, state.rng=None makes every model fall back to the global
        # np.random, coupling the two pipelines. Seeded from cfg.channel_rng_seed
        # when present (reproducible), with a per-role offset so physical and DT
        # get DISTINCT independent streams; otherwise a fresh Generator.
        if rng is None:
            seed = getattr(self.cfg, "channel_rng_seed", None) if self.cfg is not None else None
            role_code = 1 if self.role == "dt" else 0
            rng = (np.random.default_rng([int(seed), role_code]) if seed is not None
                   else np.random.default_rng())
        self.state = ChannelState(rng=rng)
        self.components = components if components is not None \
                          else self._auto_components()
        self._check_compatibility()

    def _check_compatibility(self):
        """Warn on combinations that silently break physics.

        Catches the three failure modes documented in base.py:
          1. multi-tap / spatially-structured model + steering component
             without `produces_per_antenna_h = True` → steering destroys
             the model's spatial structure.
          2. K-factor component enabled + model that doesn't read
             state["k_factor_db_pending"] → K-factor reported in extras
             but the actual fading uses the model's own K.
          3. Scalar-h model + steering component disabled → downstream
             code receives a scalar h instead of per-antenna.
        """
        comp_names = {c.name for c in self.components}
        model_name = type(self.model).__name__

        has_steering = "steering_upa" in comp_names
        has_k_38811   = "k_factor_38811"   in comp_names
        has_k_sigmoid = "k_factor_sigmoid" in comp_names
        has_k_constant= "k_factor_constant" in comp_names
        any_k_component = has_k_38811 or has_k_sigmoid or has_k_constant

        per_ant   = bool(getattr(self.model, "produces_per_antenna_h", False))
        intrinsic = bool(getattr(self.model, "has_intrinsic_spatial_structure", False))
        consumes_k = bool(getattr(self.model, "consumes_external_k_factor", True))

        # Failure 1
        if has_steering and intrinsic and not per_ant:
            _log.warning(
                "ChannelPipeline: %s has intrinsic spatial structure but does "
                "NOT declare produces_per_antenna_h=True. The SteeringUPAComponent "
                "will overwrite the model's spatial structure with a Rician lift. "
                "Either set produces_per_antenna_h=True on %s OR disable steering "
                "(enable_steering_upa=False).",
                model_name, model_name)

        # Failure 2
        if any_k_component and not consumes_k:
            _log.warning(
                "ChannelPipeline: %s does NOT consume external K-factor "
                "(consumes_external_k_factor=False), but a K-factor component is "
                "enabled. The K-factor written to extras['k_factor_db'] is NOT "
                "the K-factor used by %s. KPIs and predictors will see "
                "mismatched values. Disable the K-factor component OR replace %s.",
                model_name, model_name, model_name)

        # Failure 3 - scalar-h model + no steering means downstream code
        # has to lift it itself, with no guarantee it honours steering_strategy.
        if (not has_steering) and (not per_ant) and (not intrinsic):
            _log.info(
                "ChannelPipeline: model=%s returns scalar h and SteeringUPAComponent "
                "is disabled. Downstream consumers (scn_channel_predict.py) will "
                "build per-antenna h with their own steering logic, which DOES honour "
                "cfg.steering_strategy via steering_vector_for().",
                model_name)

        # Failure 4 - enforce the fast_fading vs complete_lms contract.
        # A complete_lms model owns shadowing intrinsically, so it must claim the
        # shadowing components in handles_internally, else shadowing double-counts.
        # A fast_fading model must not claim shadowing (it does not provide it), else
        # enabling the external component is suppressed and shadowing is lost.
        SHADOW_NAMES = {"shadowing", "shadowing_38811", "shadowing_gudmundson"}
        chan_class = str(getattr(self.model, "channel_class", "fast_fading"))
        handles    = set(getattr(self.model, "handles_internally", set()))
        if chan_class == "complete_lms" and not (SHADOW_NAMES & handles):
            _log.warning(
                "ChannelPipeline: %s is channel_class='complete_lms' (models shadowing "
                "intrinsically) but does NOT declare a shadowing name in "
                "handles_internally. An enabled shadowing component would DOUBLE-COUNT "
                "shadowing. Add the shadowing names to handles_internally.", model_name)
        if chan_class != "complete_lms" and (SHADOW_NAMES & handles):
            _log.warning(
                "ChannelPipeline: %s is channel_class='%s' but claims shadowing in "
                "handles_internally; it will SUPPRESS the external shadowing component "
                "without providing shadowing itself (the link loses shadow fading).",
                model_name, chan_class)

    def _auto_components(self) -> List[AbstractChannelComponent]:
        """Auto-select reusable components based on cfg flags.

        Components carry a `phase` attribute (default 'post_model') so
        pre-model effects (FSPL, K-factor selector) run before the
        base channel model, while shadow/doppler/etc. run after.
        """
        cfg = self.cfg
        # Resolve single-choice category enums (rain/gas/shadowing/kfactor) onto
        # the enable_* booleans before assembly, so each category yields exactly
        # one model and mutually-exclusive models cannot both be added.
        resolve_category_selectors(cfg)
        comps: List[AbstractChannelComponent] = []
        handles = set(getattr(self.model, "handles_internally", set()))

        def _skip(name: str) -> bool:
            if name in handles:
                _log.info("ChannelPipeline: skipping '%s' component "
                          "(handled by base model %s)",
                          name, type(self.model).__name__)
                return True
            return False

        # PRE-MODEL: things the channel model itself relies on
        # Refraction runs first among pre-model components so the corrected
        # elevation is seen by every downstream consumer - FSPL (in case a future
        # elevation-dependent term is added), the K-factor selector, and the base
        # model (which the pipeline now feeds data.extras["elev_deg"]).
        if bool(getattr(cfg, "enable_refraction_p834", False)) and not _skip("refraction_p834"):
            comps.append(RefractionP834Component())
        # FSPL must run before the model so the model never re-computes path loss.
        if bool(getattr(cfg, "enable_fspl", True)) and not _skip("fspl"):
            comps.append(FSPLComponent())
        # K-factor strategy - exactly one source should be active.
        # Selection precedence (highest wins): TR 38.811 → sigmoid → constant.
        # If none is explicitly enabled, the constant source is forced on
        # so Rician models always have a K to read.
        # K-factor - pick at most one (precedence 38811 > scenario > p681 >
        # distance > sigmoid > constant). Single-selection so two sources can't
        # both apply (the old code appended every enabled flag → "last wins").
        k_factor_components = []
        if bool(getattr(cfg, "enable_k_factor_38811", False)) and not _skip("k_factor_38811"):
            k_factor_components.append(KFactor38811Component())
        elif bool(getattr(cfg, "enable_k_factor_38811_scenario", False)) and not _skip("k_factor_38811_scenario"):
            k_factor_components.append(KFactor38811ScenarioComponent())
        elif bool(getattr(cfg, "enable_k_factor_p681", False)) and not _skip("k_factor_p681"):
            k_factor_components.append(KFactorP681Component())
        elif bool(getattr(cfg, "enable_k_factor_distance", False)) and not _skip("k_factor_distance"):
            k_factor_components.append(KFactorDistanceComponent())
        elif bool(getattr(cfg, "enable_k_factor_sigmoid", False)) and not _skip("k_factor_sigmoid"):
            k_factor_components.append(KFactorSigmoidComponent())
        elif bool(getattr(cfg, "enable_k_factor_constant", False)) and not _skip("k_factor_constant"):
            k_factor_components.append(KFactorConstantComponent())
        if not k_factor_components and not _skip("k_factor_constant"):
            # default - silent fallback so the system always has a K source
            _log.info("No K-factor component enabled; defaulting to constant "
                      "(threegpp_k_factor_los_db). Set enable_k_factor_38811 "
                      "or enable_k_factor_sigmoid to override.")
            # Force the constant component on for this run
            class _Forced(KFactorConstantComponent):
                def apply(self, data, cfg, dt_sec, state=None):
                    # bypass the flag check
                    import numpy as np
                    n = data.n_links() or len(data.extras.get("elev_deg", []))
                    if n == 0: return
                    k_db = np.full(n, float(getattr(cfg, "threegpp_k_factor_los_db", 10.0)))
                    if state is not None: state.data["k_factor_db_pending"] = k_db
                    data.extras["k_factor_db"] = k_db
            k_factor_components.append(_Forced())
        if len(k_factor_components) > 1:
            _log.warning("Multiple K-factor strategies enabled (%s); precedence "
                         "is left-to-right and the last one wins per-link.",
                         [c.name for c in k_factor_components])
        comps.extend(k_factor_components)

        # POST-MODEL: layered on top of the fading the model produced
        # Temporal (Jakes) correlation runs first among post-model steps so it
        # filters the raw small-scale fading straight from the model, before any
        # h-modifying component. Ordering constraints it satisfies here:
        #  * before PropagationDelayComponent - that applies a per-tick, geometry-
        #  dependent phase; correlating across ticks after it would mix different
        #  delay phases and corrupt the fading.
        #  * before SpatialCorrelation / Polarization - those are time-invariant
        #  linear maps on h, so filtering the raw fading first is unambiguous.
        #  * before DTObservationNoise - estimation noise must stay iid, not be
        #  fed into the temporal filter.
        # It self-skips when the model already produced temporal fading
        # (state["model_did_temporal_fading"]), so ar1/sos model modes are not
        # double-correlated.
        if bool(cfg.enable_jakes_corr) and not _skip("jakes"):
            comps.append(JakesCorrelationComponent())
        # Rain - pick at most one (precedence: p838_hv > p618 > ar1).
        if bool(getattr(cfg, "enable_rain_p838_hv", False)) and not _skip("rain_p838_hv"):
            comps.append(RainP838HVComponent())
        elif bool(getattr(cfg, "enable_rain_p618", False)) and not _skip("rain_p618"):
            comps.append(RainP618Component())
        elif bool(getattr(cfg, "enable_rain_ar1", False)) and not _skip("rain_ar1"):
            comps.append(RainAR1Component())
        # Gas - pick at most one (precedence: p676 > cosec).
        if bool(getattr(cfg, "enable_gas_p676", False)) and not _skip("gas_p676"):
            comps.append(GasP676Component())
        elif bool(getattr(cfg, "enable_gas_absorption", False)) and not _skip("gas_absorption"):
            comps.append(GasCosecComponent())
        # Cloud - additive on top of gas.
        if bool(getattr(cfg, "enable_cloud_p840", False)) and not _skip("cloud_p840"):
            comps.append(CloudP840Component())
        # Shadowing - pick at most one (precedence: 38811 > gudmundson > legacy).
        if bool(getattr(cfg, "enable_shadowing_38811", False)) and not _skip("shadowing_38811"):
            comps.append(Shadowing38811Component())
        elif bool(getattr(cfg, "enable_shadowing_gudmundson", False)) and not _skip("shadowing_gudmundson"):
            comps.append(ShadowingGudmundsonComponent())
        elif bool(cfg.enable_shadowing) and not _skip("shadowing"):
            comps.append(ShadowingComponent())
        # Antenna beam pattern (off-boresight loss).
        if bool(getattr(cfg, "enable_beam_pattern", False)) and not _skip("beam_pattern"):
            comps.append(BeamPatternComponent())
        if bool(cfg.enable_polarization) and not _skip("polarization"):
            comps.append(PolarizationComponent())
        if (cfg.spatial_corr_model or "iid").lower() != "iid" \
                and not _skip("spatial_correlation"):
            comps.append(SpatialCorrelationComponent())
        if bool(cfg.enable_doppler) and not _skip("doppler"):
            comps.append(DopplerComponent())
        if bool(cfg.enable_propagation_delay) and not _skip("propagation_delay"):
            comps.append(PropagationDelayComponent())
        # Steering / per-antenna lift - DEPRECATED. Every registered
        # channel model now produces per-antenna h directly via shared
        # helpers in ntnconnect.registry.channels._steering. The SteeringUPA
        # component remains in the registry as a no-op safety net (it
        # detects h.ndim > 1 or produces_per_antenna_h=True and skips).
        # Kept here only for back-compat with hand-rolled models that
        # still return scalar h.
        if bool(getattr(cfg, "enable_steering_upa", False)) and not _skip("steering_upa"):
            _log.info("ChannelPipeline: enable_steering_upa=True is no longer "
                      "required — all registered models now produce per-antenna h.")
            comps.append(SteeringUPAComponent())
        # DT observation noise is a DIGITAL-TWIN concern, not a physical-channel
        # effect. It is added only on the DT pipeline (role="dt"); the physical
        # pipeline never carries it, so the two stay cleanly separated.
        if bool(getattr(cfg, "enable_dt_obs_noise", False)) and not _skip("dt_obs_noise"):
            if self.role == "dt":
                comps.append(DTObservationNoiseComponent())
            else:
                _log.info("ChannelPipeline: enable_dt_obs_noise=True is ignored on "
                          "the physical pipeline (role=%r). It applies only on the "
                          "DT pipeline — build it with ChannelModelFactory.build_dt() "
                          "or role='dt'.", self.role)
        return comps

    def execute(self,
                dist_m: np.ndarray,
                elev_deg: np.ndarray,
                range_rate_m_s: np.ndarray,
                dt_sec: float,
                azimuth_deg=None,
                pos_m=None):
        # Two-phase pipeline:
        #  1) Initialise a blank ChannelData, stash geometry in extras
        #  so pre-model components (FSPL, K-factor) can read it.
        #  2) Run pre-model components (loss_db = FSPL, extras['k_factor_db']).
        #  3) Run the base channel model - produces h (small-scale).
        #  4) Run post-model components (rain, gas, shadow, doppler, delay,
        #  and SteeringUPAComponent - which lifts scalar h to
        #  per-antenna using el + az + K).
        data = ChannelData()
        n_links = int(np.asarray(dist_m).size)
        data.extras["dist_m"]         = np.asarray(dist_m, dtype=float)
        data.extras["range_rate_m_s"] = np.asarray(range_rate_m_s, dtype=float)
        data.extras["elev_deg"]       = np.asarray(elev_deg, dtype=float)
        data.extras["azimuth_deg"]    = (np.asarray(azimuth_deg, dtype=float)
                                          if azimuth_deg is not None
                                          else np.zeros(n_links, dtype=float))

        pre  = [c for c in self.components if getattr(c, "phase", "post_model") == "pre_model"]
        post = [c for c in self.components if getattr(c, "phase", "post_model") != "pre_model"]

        # 2) pre-model phase
        for comp in pre:
            try:
                comp.apply(data, self.cfg, dt_sec, self.state)
            except Exception as e:
                _log.warning("Channel pre-model component %s failed: %s", comp.name, e)

        # 3) base model - produces the small-scale fading h.
        # Models may still write loss_db (legacy path); we ADD to whatever
        # FSPL already deposited so we don't lose information.
        # Expose the model's per-antenna-h contract to downstream components.
        if self.state is not None:
            self.state.data["model_produces_per_antenna_h"] = bool(
                getattr(self.model, "produces_per_antenna_h", False))
        # Elevation may have been refraction-corrected by a pre-model component
        # (RefractionP834Component writes data.extras["elev_deg"]). Feed the model
        # the corrected value so its steering / off-nadir / in-model K use the same
        # elevation as the post-model atmospheric components. Falls back to the raw
        # argument when no correction ran.
        elev_for_model = data.extras.get("elev_deg", elev_deg)
        out = self.model.compute_channel(
            dist_m, elev_for_model, dt_sec, self.state, pos_m,
            azimuth_deg=data.extras.get("azimuth_deg"))
        if out is None:
            return None, None, None, dict(data.extras)
        if isinstance(out, tuple):
            loss_db, h, doppler_hz = out
            out = ChannelData(loss_db=loss_db, h=h, doppler_hz=doppler_hz)
        # h is always taken from the model. loss_db is added.
        data.h          = out.h
        data.doppler_hz = out.doppler_hz if out.doppler_hz is not None \
                          else np.zeros(data.n_links())
        if out.loss_db is not None:
            if data.loss_db is None:
                data.loss_db = np.asarray(out.loss_db, dtype=float)
            else:
                data.loss_db = np.asarray(data.loss_db, dtype=float) + np.asarray(out.loss_db, dtype=float)
        # Forward any extras the model produced without overwriting what
        # the pre-model components already set (e.g. k_factor_db).
        for k, v in out.extras.items():
            data.extras.setdefault(k, v)

        # 4) post-model phase
        for comp in post:
            try:
                comp.apply(data, self.cfg, dt_sec, self.state)
            except Exception as e:
                _log.warning("Channel post-model component %s failed: %s", comp.name, e)

        return data.loss_db, data.h, data.doppler_hz, dict(data.extras)


# Public facade
class ChannelModelFactory:
    @classmethod
    def build(cls, channel_cfg, role: str = "physical") -> ChannelPipeline:
        # Trigger model registration
        from ntnconnect.registry.channels.models import (
            three_gpp, loo, lutz, perez_fontan, ntn_tdl,
            saleh_valenzuela, spatial_geometric_rician,
            professional_ntn, geometric_nominal,
        )
        name = channel_cfg.model_type
        Cls = get_model_class(name)
        model = Cls(channel_cfg)
        return ChannelPipeline(model, cfg=channel_cfg, role=role)

    @classmethod
    def build_dt(cls, channel_cfg) -> ChannelPipeline:
        """Build the DIGITAL-TWIN channel pipeline (role='dt').

        Identical to build() but adds DT-only components (e.g. observation
        noise). Use this for the twin's channel realization; use build() for
        the true physical channel so the two remain separated.
        """
        return cls.build(channel_cfg, role="dt")

    # Legacy-compat alias used by older code paths
    @classmethod
    def get_pipeline(cls, channel_cfg):
        return cls.build(channel_cfg)
