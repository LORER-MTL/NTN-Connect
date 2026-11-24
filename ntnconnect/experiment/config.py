# SkyNTNPro v1 - typed configuration model; parses configs/master.yaml into one SimulationConfig.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import yaml


# Leaf configs
@dataclass
class ExperimentConfig:
    name: str = "baseline"
    random_seed: int = 42                 # legacy seed - still honoured
    global_seed: Optional[int] = None     # when set, seeds every stochastic subsystem deterministically
    notes: str = ""
    strict_config: bool = False    # when True, setup() raises on any missing required field
    # Phase-1 dataset generation; idempotent (matching manifest is reused). False = use env window.
    generate_coverage: bool = True


@dataclass
class EnvConfig:
    start_utc: str = "2026-05-21T12:00:00Z"
    duration_s: int = 3600
    step_size_s: int = 60
    lat_bounds: List[float] = field(default_factory=lambda: [-90.0, 90.0])
    lon_bounds: List[float] = field(default_factory=lambda: [-180.0, 180.0])
    region_name: str = "Global"


@dataclass
class DatasetTimingConfig:
    """Unified Phase-1 dataset timing consumed by every study; coarse_step_s=0 disables the visibility pre-pass."""
    utc: str = "2026-05-21T12:00:00Z"
    duration_s: float = 86400.0
    step_s: float = 30.0
    coarse_step_s: float = 0.0
    batch_size_s: float = 900.0

    def __post_init__(self):
        import warnings
        if not (self.duration_s > 0 and self.step_s > 0 and self.batch_size_s > 0):
            raise ValueError("dataset timing: duration_s/step_s/batch_size_s must be > 0")
        if abs(self.batch_size_s / self.step_s
               - round(self.batch_size_s / self.step_s)) > 1e-9:
            warnings.warn(f"dataset timing: batch_size_s ({self.batch_size_s}) is not a "
                          f"whole number of step_s ({self.step_s}).")
        if abs(self.duration_s / self.batch_size_s
               - round(self.duration_s / self.batch_size_s)) > 1e-9:
            warnings.warn(f"dataset timing: duration_s ({self.duration_s}) is not a "
                          f"whole number of batch_size_s ({self.batch_size_s}).")

    @property
    def n_steps(self) -> int:
        return max(1, int(self.duration_s // self.step_s))

    @property
    def steps_per_batch(self) -> int:
        return max(1, int(self.batch_size_s // self.step_s))

    @property
    def n_batches(self) -> int:
        return max(1, int(self.duration_s // self.batch_size_s))


# ns-3-style integer time base: 1 s = 1e9 ns (exact int64 time, no float drift).
NS_PER_S = 1_000_000_000


@dataclass
class ResolvedTiming:
    """Concrete integer-ns timing for one run (from SimulationTimingConfig.resolved)."""
    resolution_ns: int          # smallest quantum (ns-3 floor, default 1 ns)
    start_offset_ns: int        # window start, from dataset.utc
    duration_ns: int            # window length
    step_ns: int                # slot / decision step
    fast_fading_ns: int         # sub-slot (fast-fading) step
    sub_slots: int              # = step_ns // fast_fading_ns
    n_ticks: int                # = duration_ns // step_ns


@dataclass
class SimulationTimingConfig:
    """Phase-2 sim timing (a window into the dataset trace); seconds in, integer ns out via resolved(). 0 = inherit/default."""
    start_offset_s: float = 0.0
    duration_s: float = 0.0
    step_s: float = 0.0
    resolution_s: float = 1e-9
    fast_fading_s: float = 0.0

    def resolved(self, dataset: "DatasetTimingConfig") -> ResolvedTiming:
        res_ns = max(1, int(round(self.resolution_s * NS_PER_S)))

        def _ns(x: float) -> int:
            # snap seconds to an integer number of resolution quanta
            return int(round(x * NS_PER_S / res_ns)) * res_ns

        step_ns = _ns(self.step_s) or _ns(dataset.step_s)
        off_ns  = _ns(self.start_offset_s)
        dur_ns  = _ns(self.duration_s) or (_ns(dataset.duration_s) - off_ns)
        ff_ns   = _ns(self.fast_fading_s) or step_ns     # default: 1 sub-slot/slot
        # exact integer-multiple hierarchy: step >= fast_fading >= resolution
        if step_ns % res_ns or ff_ns % res_ns:
            raise ValueError("sim timing: step/fast_fading must be multiples of resolution")
        if step_ns % ff_ns:
            raise ValueError("sim timing: step_s must be a whole number of fast_fading_s")
        if dur_ns <= 0 or step_ns <= 0:
            raise ValueError("sim timing: duration_s/step_s must be > 0")
        return ResolvedTiming(
            resolution_ns=res_ns, start_offset_ns=off_ns, duration_ns=dur_ns,
            step_ns=step_ns, fast_fading_ns=ff_ns,
            sub_slots=step_ns // ff_ns, n_ticks=max(1, dur_ns // step_ns))


@dataclass
class ResolvedCSITiming:
    generation_period_ns: int       # how often new CSI is produced
    availability: str               # propagation | next_slot | immediate
    processing_delay_ns: int        # extra processing on top of availability
    handover_central: bool          # association decided at a central agent
    handover_assume_instant: bool   # central agent has all CSI with no delay

    def availability_delay_ns(self, prop_rtt_ns: int, slot_ns: int) -> int:
        """CSI availability delay (ns): propagation RTT + processing; next_slot quantises up but collapses to propagation when slot > delay; immediate = processing only."""
        base = int(prop_rtt_ns) + self.processing_delay_ns     # physical propagation
        if self.availability == "immediate":
            return self.processing_delay_ns
        if self.availability == "propagation" or slot_ns <= 0:
            return base
        # next_slot: quantise to next slot, but use propagation when slot > delay
        if slot_ns > base:
            return base
        import math
        return int(math.ceil(base / slot_ns)) * slot_ns


@dataclass
class CSITimingConfig:
    """Online CSI update timing: when channel state is regenerated and when it becomes usable. availability = propagation (physical) | next_slot | immediate (genie)."""
    generation_period_s: float = 0.0     # 0 = every fast-fading sub-slot; must be a whole number of sub-slots
    availability: str = "propagation"    # propagation | next_slot | immediate
    processing_delay_s: float = 0.0
    handover_central: bool = True
    handover_assume_instant: bool = False

    def resolved(self, sim: "ResolvedTiming") -> ResolvedCSITiming:
        res = sim.resolution_ns

        def _ns(x: float) -> int:
            return int(round(x * NS_PER_S / res)) * res

        gen = _ns(self.generation_period_s) or sim.fast_fading_ns
        if gen % sim.fast_fading_ns:
            raise ValueError("csi generation_period_s must be a whole number of "
                             "fast-fading sub-slots")
        if self.availability not in ("propagation", "next_slot", "immediate"):
            raise ValueError("csi availability must be propagation|next_slot|immediate")
        return ResolvedCSITiming(
            generation_period_ns=gen, availability=self.availability,
            processing_delay_ns=_ns(self.processing_delay_s),
            handover_central=bool(self.handover_central),
            handover_assume_instant=bool(self.handover_assume_instant))


@dataclass
class StorageConfig:
    data_dir: str = "outputs"
    run_id: int = 1
    partition_by: Dict[str, List[str]] = field(default_factory=lambda: {
        "nodes":      ["operator_id"],
        "links":      ["link_type", "node_type", "operator_id"],
        "node_state": ["node_type", "operator_id"],
        "tasks":      ["tick_bucket"],
    })
    tick_bucket_size: int = 1000
    parquet_compression: str = "zstd"


@dataclass
class RadioConfig:
    carrier_freq_hz: float = 2.0e9
    bandwidth_hz:    float = 20.0e6
    power_max_w:     float = 10.0
    noise_figure_db: float = 3.0
    system_temperature_k: float = 290.0
    antenna_gain_tx_db: float = 30.0
    antenna_gain_rx_db: float = 5.0
    beamforming_mode: str = "hybrid"


@dataclass
class ChannelConfig:
    # Selection
    model_type: str = "3GPP"               # 3GPP | Loo | Lutz | Perez_Fontan | NTN_TDL | SV | Spatial_Geometric_Rician
    scenario: str = "Urban"
    # node_type -> access-model name for non-satellite nodes; empty = built-in defaults.
    access_models: Dict[str, str] = field(default_factory=dict)

    # Component flags
    enable_doppler: bool = True
    enable_multipath: bool = True
    enable_shadowing: bool = True
    enable_polarization: bool = True
    enable_propagation_delay: bool = True
    enable_jakes_corr: bool = True
    enable_atmospheric: bool = False
    # Single-choice category selectors (preferred over the flat enable_* flags); None = fall back to enable_*.
    rain_model:      Optional[str] = None   # off | p838 | p618 | ar1
    gas_model:       Optional[str] = None   # off | p676 | cosec
    shadowing_model: Optional[str] = None   # off | p38811 | gudmundson | lognormal
    kfactor_model:   Optional[str] = None   # off | p38811 | p38811_scenario | constant | sigmoid | p681 | distance
    # Reusable components factored out of Professional_NTN; default OFF.
    enable_rain_ar1: bool          = False    # → RainAR1Component (Colab AR-1)
    enable_rain_p618: bool         = False    # → RainP618Component (ITU-R P.618-style)
    enable_gas_absorption: bool    = False    # → GasCosecComponent
    enable_fspl: bool              = True     # → FSPLComponent (pre-model)
    # K-factor strategies (pick at most one; precedence 38811 > sigmoid > constant).
    enable_k_factor_38811: bool    = False    # → KFactor38811Component (pre-model)
    enable_k_factor_sigmoid: bool  = False    # → KFactorSigmoidComponent (pre-model)
    enable_k_factor_constant: bool = False    # → KFactorConstantComponent (pre-model)
    enable_steering_upa: bool      = False    # → SteeringUPAComponent (post-model, runs last)
    steering_strategy: str         = "upa_kron"   # upa_kron | ula | upa_geometric_los

    # DT-pipeline knobs (dt_channel stage / Geometric_Nominal model).
    # Aged geometry: std-dev of N(0,σ²) noise the DT adds to model position-knowledge mismatch.
    geo_nom_aged_elev_std_deg: float = 0.2   # deg
    geo_nom_aged_az_std_deg:   float = 0.2   # deg
    geo_nom_other_loss_db:     float = 0.5   # constant other-loss added on top of FSPL

    # Optional DT observation-noise component: adds CN(0,σ²·I) to the DT's per-antenna h.
    enable_dt_obs_noise: bool      = False
    dt_obs_noise_sigma:  float     = 0.0     # std of complex Gaussian per antenna

    # Extended K-factor strategies (new, all default OFF)
    enable_k_factor_38811_scenario: bool = False
    k_factor_scenario_name:         str  = "rural"   # dense_urban|urban|suburban|rural
    enable_k_factor_p681:           bool = False
    p681_environment:               str  = "rural_wooded"  # open|rural_wooded|suburban|urban
    enable_k_factor_distance:       bool = False
    k_distance_k0_db:               float = 12.0
    k_distance_ref_m:               float = 6.0e5
    k_distance_gamma:               float = 0.5

    # Extended shadowing strategies (new, default OFF)
    enable_shadowing_38811:         bool  = False
    shadowing_38811_scenario:       str   = "rural"  # dense_urban|urban|suburban|rural
    enable_shadowing_gudmundson:    bool  = False
    shadowing_gudmundson_corr_m:    float = 50.0
    shadowing_gudmundson_user_speed_m_s: float = 0.0

    # Extended rain strategy (new, default OFF)
    enable_rain_p838_hv:            bool  = False
    rain_polarization:              str   = "circular"  # H | V | circular

    # Extended atmosphere components (new, default OFF)
    enable_gas_p676:                bool  = False
    p676_water_vapour_g_m3:         float = 7.5
    p676_effective_height_km:       float = 6.0
    enable_cloud_p840:              bool  = False
    p840_cloud_liquid_kg_m2:        float = 0.5
    enable_refraction_p834:         bool  = False

    # Beam pattern (new, default OFF)
    enable_beam_pattern:            bool  = False
    beam_3db_deg:                   float = 4.0
    beam_max_loss_db:               float = 30.0

    # Antenna geometry
    n_rx: int = 4
    n_tx_x: int = 8
    n_tx_y: int = 8
    # Hybrid-BF RF chain count
    num_rf_chains: int = 4
    antenna_mode: str = "auto"        # auto | siso | miso | simo | mimo ("auto" derived from N_t,N_r)
    spatial_streams: int = 1          # MIMO multiplexing order N_s, capped by min(N_t,N_r); 1 = single-stream
    multiple_access: str = "fdma"     # fdma | tdma | ofdma | sdma | noma
    beam_hopping: bool = False        # when on, beam_id is a per-slot decision
    # Analog/hybrid efficiency factors on the geometric array gain
    bf_eff_digital: float = 1.0
    bf_eff_analog:  float = 0.8
    bf_eff_hybrid:  float = 0.9

    # Shadowing
    shadowing_std_db: float = 2.0
    shadowing_correlation_distance_m: float = 50.0

    # Polarization
    xpr_db: float = 7.0

    # Spatial correlation
    spatial_corr_model: str = "iid"        # iid | exp
    tx_corr_rho: float = 0.0
    rx_corr_rho: float = 0.0

    # 3GPP TR 38.811 params
    threegpp_fspl_intercept: float = 32.45
    threegpp_k_factor_los_db: float = 12.0
    threegpp_k_factor_nlos_db: float = -30.0
    threegpp_los_prob_offset: float = 0.5
    threegpp_los_prob_scale: float = 12.0

    # 3GPP TR 38.811 fast-fading temporal model (see channels/models/three_gpp.py).
    threegpp_fading_mode: str = "iid"       # iid (block) | ar1 (Gauss-Markov) | sos (Clarke/Jakes)
    threegpp_k_mode:      str = "constant"  # constant | elevation (k_factor_38811 overrides both)
    threegpp_sos_M:       int = 16          # sinusoids for the "sos" Doppler spectrum
    ue_speed_mps:         float = 0.0       # terminal speed -> Doppler f_D = v f_c / c; 0 = quasi-static

    # Professional NTN model params (rain / gas / AR-1 K-factor)
    pro_ntn_rain_corr:     float = 2e-4         # AR-1 rate for rain state
    pro_ntn_gas_factor_db: float = 0.5          # gas-attenuation baseline (dB)
    pro_ntn_kappa_rho:     float = 0.95         # AR-1 rate for Rician K (per link)

    # Loo (1985) shadowed-Rician LMS channel (see channels/models/loo.py).
    loo_mode: str = "iid"                       # iid | correlated (AR-1 LoS shadow + Doppler multipath)
    loo_los_rho: float = 0.995                  # slow-shadow AR-1 coefficient
    loo_mu: float = -0.5
    loo_sigma: float = 1.0
    loo_multipath_var: float = 0.1

    sgr_fading_mode: str = "iid"   # Spatial_Geometric_Rician fading: iid | ar1 | sos

    # Lutz - two-state Markov LMS (Lutz et al. 1991).
    lutz_mode: str = "iid"         # iid | correlated (Doppler good/bad states + AR-1 shadow)
    lutz_shadow_rho: float = 0.99
    lutz_P: List[List[float]] = field(default_factory=lambda: [[0.95,0.05],[0.10,0.90]])
    lutz_k_factor_db: float = 10.0
    lutz_mu_bad: float = -2.0
    lutz_sigma_bad: float = 2.0

    # Perez-Fontan - three-state Markov LMS (Perez-Fontan et al.).
    pf_mode: str = "iid"           # iid | correlated (AR-1 LoS + Doppler multipath per state)
    pf_shadow_rho: float = 0.99
    pf_P: List[List[float]] = field(default_factory=lambda:
        [[0.98,0.01,0.01],[0.05,0.90,0.05],[0.01,0.05,0.94]])
    pf_mu: List[float] = field(default_factory=lambda: [0.0, -3.0, -10.0])
    pf_sigma: List[float] = field(default_factory=lambda: [0.5, 2.0, 4.0])
    pf_multipath_var: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])

    # NTN-TDL (3GPP TR 38.811 §6.9.2, Tables 6.9.2-1..4).
    tdl_mode: str = "simplified"                 # simplified (narrowband collapse) | full (tabulated profile)
    tdl_profile: str = "A"                       # A | B | C | D
    tdl_fading: str = "iid"                      # iid | ar1 | sos
    tdl_num_taps: int = 5
    tdl_rms_delay_spread_ns: float = 50.0

    # Saleh-Valenzuela (Saleh & Valenzuela 1987).
    sv_mode: str = "simplified"                  # simplified (narrowband collapse) | full (cluster/ray)
    sv_fading: str = "iid"                       # iid | ar1 | sos
    sv_num_clusters: int = 3
    sv_rays_per_cluster: int = 4
    sv_lambda_cluster_ns: float = 200.0
    sv_lambda_ray_ns: float = 5.0
    sv_gamma_cluster_ns: float = 500.0
    sv_gamma_ray_ns: float = 30.0


@dataclass
class InterferenceConfig:
    # legacy in-pipeline interference (consumed by stage "interference")
    model: str = "physical"
    formulation: str = "explicit"
    enable_intra_satellite_interference: bool = True
    enable_inter_satellite_interference: bool = True
    same_user_only: bool = True
    fixed_interference_dbw: float = -110.0
    co_channel_isolation_db: float = 25.0
    intra_beam_isolation_db: float = 30.0
    frequency_reuse_factor: float = 1.0

    # Optimization-safe external interference (stage "external_interference_const"):
    # writes links["i_ext_w"] so I_ext enters the SINR denominator as a constant.
    # ext_mode: off | fixed_in | fixed_ci_nominal | reuse_floor_const |
    #           nvis_sidelobe_nominal | s672_geom | lognormal
    ext_mode: str = "off"
    ext_knob: float = -12.2
    ext_K_r:  int   = 1
    ext_interferer_pmax_w: float = 0.0     # interferer EIRP cap (W); 0 = use power_max_w
    ext_pol_iso_db: float = 12.0           # polarisation isolation (modes 5, 6)
    ext_offaxis_mode: str = "geom"         # geom | elev_proxy (off-axis angle source, mode 6)
    ext_sat_theta_3db_deg: float = 4.0     # sat tx 3-dB beamwidth (S.672 input)
    ext_colouring: str = "mean_field"      # mean_field | id_mod_kr | greedy_elev | dsatur
    ext_conflict_sep_deg: float = 1.5      # beam-conflict great-circle separation [deg]


@dataclass
class ISLConfig:
    enabled: bool = True
    strategy: str = "plus_grid"   # none | degree_bounded | plus_grid | range_rate_greedy | match_b4
    max_degree: int = 4
    max_range_km: float = 5500.0
    min_grazing_alt_km: float = 80.0
    require_los: bool = True
    cross_operator_isl_allowed: bool = False
    range_rate_alpha_s: float = 60.0   # range_rate_greedy separation-rate penalty [s]


@dataclass
class ShellConfig:
    name: str = "Layer"
    altitude_km: float = 1200.0
    inclination_deg: float = 87.9
    num_planes: int = 12
    num_satellites: int = 72
    phasing_factor: int = 1


@dataclass
class SatelliteFleet:
    catalog_id_start: int = 0
    generate_tle: bool = True
    tle_file_path: Optional[str] = None
    shells: List[ShellConfig] = field(default_factory=list)
    propagator: str = "auto"       # auto | skyfield (sgp4) | skyfield_auto_tle (sgp4_auto_tle) | kepler
    tle_cache_dir: str = "data/tle_cache"
    mobility: "NodeMobility" = field(default_factory=lambda: NodeMobility())
    # Optional band filters: keep only sats whose inclination/altitude fall in any [lo,hi] band; empty = no filtering.
    inc_bands: List[List[float]] = field(default_factory=list)   # [[lo,hi],...] deg
    alt_bands: List[List[float]] = field(default_factory=list)   # [[lo,hi],...] km

    def __post_init__(self):
        self.mobility = _coerce_node_mobility(self.mobility)


@dataclass
class FixedNodeEntry:
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    sector_deg: float = 360.0


@dataclass
class HAPSFleet:
    id_range: Tuple[int, int] = (0, 0)
    mobility: "NodeMobility" = field(default_factory=lambda: NodeMobility())
    positions: List[FixedNodeEntry] = field(default_factory=list)

    def __post_init__(self):
        self.mobility = _coerce_node_mobility(self.mobility)


@dataclass
class UAVFleet:
    id_range: Tuple[int, int] = (0, 0)
    mobility: "NodeMobility" = field(default_factory=lambda: NodeMobility())
    positions: List[FixedNodeEntry] = field(default_factory=list)

    def __post_init__(self):
        self.mobility = _coerce_node_mobility(self.mobility)


@dataclass
class GroundStationFleet:
    id_range: Tuple[int, int] = (0, 0)
    mobility: "NodeMobility" = field(default_factory=lambda: NodeMobility())
    positions: List[FixedNodeEntry] = field(default_factory=list)

    def __post_init__(self):
        self.mobility = _coerce_node_mobility(self.mobility)


@dataclass
class BaseStationFleet:
    id_range: Tuple[int, int] = (0, 0)
    mobility: "NodeMobility" = field(default_factory=lambda: NodeMobility())
    positions: List[FixedNodeEntry] = field(default_factory=list)

    def __post_init__(self):
        self.mobility = _coerce_node_mobility(self.mobility)


@dataclass
class ComputeCapability:
    """Static per-node compute / storage capabilities."""
    cpu_cores: int = 0
    cpu_clock_ghz: float = 0.0
    flops_peak: float = 0.0
    ram_gb: float = 0.0
    storage_gb: float = 0.0
    gpu_present: bool = False
    power_idle_w: float = 0.0
    power_per_flop_j: float = 0.0
    compute_role: str = "none"                   # none | mec | cache | both


@dataclass
class Operator:
    id: int = 0
    name: str = ""
    sat_id_range: Tuple[int, int] = (0, 0)
    radio: RadioConfig = field(default_factory=RadioConfig)
    channel_model: ChannelConfig = field(default_factory=ChannelConfig)
    # Optional second channel config for the DT pipeline (dt_channel stage); None = inline fallback.
    dt_channel_model: Optional[ChannelConfig] = None
    isl: ISLConfig = field(default_factory=ISLConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    satellites: SatelliteFleet = field(default_factory=SatelliteFleet)
    haps: HAPSFleet = field(default_factory=HAPSFleet)
    uavs: UAVFleet = field(default_factory=UAVFleet)
    ground_stations: GroundStationFleet = field(default_factory=GroundStationFleet)
    base_stations: BaseStationFleet = field(default_factory=BaseStationFleet)
    compute: Dict[str, ComputeCapability] = field(default_factory=dict)


@dataclass
class NodeMobility:
    """Per-node mobility; model selects the position source (dataset | static/hover | sgp4/kepler | gauss_markov/random_waypoint/drift/waypoint)."""
    model: str = "dataset"
    type: Optional[str] = None          # legacy alias for model
    speed_max_mps: float = 1.0
    speed_min_mps: float = 0.0
    alpha: float = 0.75
    sigma: float = 5.0
    drift_deg: float = 0.001
    phase: int = 2

    def __post_init__(self):
        if self.type is not None and self.model == "dataset":
            self.model = self.type


UserMobility = NodeMobility             # backward-compatible alias


def _coerce_node_mobility(m) -> "NodeMobility":
    """Normalize a fleet's mobility (str/dict/NodeMobility) to a NodeMobility."""
    if isinstance(m, NodeMobility):
        return m
    if isinstance(m, str):
        return NodeMobility(model=m)
    if isinstance(m, dict):
        return NodeMobility(**m)
    return NodeMobility()


@dataclass
class UserPreferences:
    preferred_operator_id: Optional[int] = None
    allow_fallback: bool = True
    fallback_cost: float = 5.0


@dataclass
class UsersConfig:
    num_seed_users: int = 10
    grid_method: str = "random"      # random | matern | fibonacci | grid | hotspot
    min_dist_km: float = 50.0
    # hotspot placement (grid_method="hotspot"): users uniform within a circle of hotspot_radius_km.
    hotspot_lat: float = 0.0
    hotspot_lon: float = 0.0
    hotspot_radius_km: float = 100.0
    land_only: bool = True
    enable_clustering: bool = False
    max_users_per_cluster: Optional[int] = None
    # Earth-cell grid (geometry/grid.py + clustering stage); fixed per MC iteration, membership recomputed each tick.
    cluster_grid_method: str = "regular_lat_lon"  # regular_lat_lon | hex | h3
    cluster_size_km: float = 200.0
    users_per_cluster: int = 5
    cluster_score_aggregation: str = "mean"   # min | mean | sum_rate
    cluster_granularity: str = "cell"   # cell (FDMA-split Earth-cell) | user (each user its own cluster)
    cluster_geometry: str = "users"     # users (aggregate per-user metrics) | cell_center (cell centroid)
    mobility: UserMobility = field(default_factory=UserMobility)
    preferences: UserPreferences = field(default_factory=UserPreferences)


@dataclass
class HandoverConfig:
    mode: str = "centralized"
    hysteresis_deg: float = 3.0                  # also used as a3-Offset in dB
    min_elevation_deg: float = 20.0
    max_users_per_sat: int = 50
    inter_operator_penalty: float = 5.0
    inter_node_type_penalty: float = 3.0
    ping_pong_window_s: float = 30.0
    objective: str = "max_elev_with_load_balance"
    satellite_admission_cap: int = 50
    top_k_local: int = 3
    # Knobs for the benchmark HO strategies + RRM power algorithms (safe defaults).
    load_penalty_db: float = 0.5
    utility_alpha: float = 0.5
    waterfilling_noise_floor_w: float = 1.0e-13
    utility_min_share: float = 0.05
    # 3GPP-A3 (gpp_a3) knobs, TS 38.331 §5.5.4.4: enter when Mn - Hys > Mp + Off; effective margin = Off + Hys.
    # Optional L3 filter (§5.5.3.2): F_n = (1-a)F_{n-1} + a M_n, a = 1/2^(k/4).
    a3_offset_db: float = 3.0
    a3_hysteresis_db: float = 0.0
    a3_ttt_ticks: int = 2
    a3_l3_filter: bool = False
    a3_l3_filter_k: int = 4
    remaining_vis_oracle: bool = False          # exact forward-scan remaining-visibility (no NaN)
    # Sat cell-capacity model (cluster-aware HO strategies):
    #  unlimited | capped (<= max_cells_per_sat) | strict_one (K=1) | beam_tagged (K beams, beam_id per (user,tick))
    sat_cell_capacity_mode: str = "unlimited"
    max_cells_per_sat: int = 8                  # max cells (= max users in WISEE) one sat serves at once
    max_users_per_cell: int = 1                # WISEE = 1; > 1 = multi-user-per-cell extension
    rrm_power_algorithm: str = "equal"         # equal | waterfilling | utility | cvx_mimo_min_power
    # CVXPY MU-MIMO SOCP knobs (used only when rrm_power_algorithm == "cvx_mimo_min_power").
    cvx_fixed_ici_dbw: float = -110.0          # I-bar, inter-sat interference floor (dBW)
    qos_rate_bps: float = 1.0e6                # R_min per user (b/s)
    cvx_infeasibility_policy: str = "drop_worst"   # drop_worst | soft_slack
    cvx_infeas_slack_lambda: float = 1.0
    cvx_antenna_spacing_d_over_lambda: float = 0.5
    cvx_solver: str = "CLARABEL"               # CLARABEL | SCS | ECOS | MOSEK
    # Optional channel-predictor knobs (scn_channel_predict); empty = no prediction.
    channel_predictor: str = ""                # "" | <registered predictor name>
    channel_predictor_window: int = 20
    # Optional robustness knobs, inert unless an external radio algorithm is selected.
    dro_epsilon: float = 0.05                  # outage level
    cvx_wasserstein_n_scenarios: int = 5
    cvx_wasserstein_radius_cap:  float = 0.15
    # Three-time-scale knobs; mc_channel_samples > 1 enables an inner MC channel loop per tick.
    run_every_n_ticks: int = 1                 # HO cadence in engine ticks
    # HO stability: ttt_ticks = must stay better before switching; mts_ticks = min-time-of-stay after a HO; 0 = off.
    ttt_ticks: int = 0
    mts_ticks: int = 0
    mc_channel_samples: int = 1                # inner-MC samples per engine tick
    mc_dt_s: float = 0.0                       # spacing between inner samples (informational)
    mc_rician_k_db: float = 8.0                # Rician K (dB) used by the inner MC channel
    mc_coherence_time_s: float = 0.0           # 0 = iid inner-MC; >0 = AR-1 NLoS, rho = exp(-mc_dt_s/this)
    # Predictor cold-start covariance scale: Σ = factor·|h|²·I for the first 3 warmup ticks.
    predictor_cold_start_scale: float = 0.1


@dataclass
class WorkloadConfig:
    enabled: bool = False
    arrival_model: str = "poisson"
    arrival_rate_per_user_per_s: float = 0.5
    cpu_required_flops_mean: float = 1.0e9
    cpu_required_flops_std: float = 3.0e8
    data_in_mb_mean: float = 1.0
    data_out_mb_mean: float = 0.1
    deadline_ms_mean: float = 300.0
    deadline_ms_std: float = 100.0
    priority_levels: List[str] = field(default_factory=lambda: ["low","normal","high"])
    priority_weights: List[float] = field(default_factory=lambda: [0.3,0.6,0.1])


@dataclass
class OffloadConfig:
    mode: str = "centralized"
    target_policy: str = "nearest_compute_capable"   # nearest_compute_capable | load_balanced | deadline_aware
    allow_cross_operator: bool = False
    feeder_fallback: bool = True
    # Offload-context knobs (network model the orchestrator sizes against).
    local_flops: float = 0.0            # device compute; 0 = no local execution
    edge_latency_ms: float = 5.0        # one-way prop to the serving access node
    cloud_latency_ms: float = 40.0      # one-way prop to a ground/cloud node
    backbone_rate_bps: float = 1.0e9    # feeder/backbone cap toward the cloud


@dataclass
class SchedulerConfig:
    mode: str = "distributed"
    policy: str = "edf"          # fifo | edf | srtf
    preempt: bool = False
    queue_capacity: int = 256    # per-node hard cap; over-arrivals -> dropped


@dataclass
class RoutingConfig:
    mode: str = "per_operator_hybrid"   # centralized | distributed | per_operator_hybrid
    algorithm: str = "dijkstra"         # dijkstra | yen_k_shortest
    k_paths: int = 1


@dataclass
class CacheConfig:
    mode: str = "per_node"
    policy: str = "lru"                       # lru | lfu | fifo
    capacity_fraction_of_storage: float = 0.5
    capacity_items: int = 128                 # fallback if storage unknown
    populate_on_task: bool = True             # touch (node, data_in_mb hash) per task


@dataclass
class EnergyConfig:
    enabled: bool = False
    include_radio: bool = True
    include_compute: bool = True
    include_idle: bool = True
    battery_aware: bool = False


@dataclass
class GifConfig:
    enabled: bool = True
    frame_duration_ms: int = 500
    keep_frames: bool = False


@dataclass
class OutputsConfig:
    # Global default format
    format: str = "parquet"                       # parquet | csv | json | jsonl | parquet+csv

    # Per-table format override (table-name -> format)
    formats: Dict[str, str] = field(default_factory=dict)

    # > 0 = flush hook buffers to *_batch{N:04d}.parquet every N ticks; 0 = single-file at sim end.
    batch_log_every_n_ticks: int = 0

    # Per-table activation; keys are TableSchema names from schemas/tables.py.
    tables: Dict[str, bool] = field(default_factory=lambda: {
        "per_user_kpis":     True,
        "node_state":        True,
        "tasks":             False,
        "ho_events":         True,
        "active_isl":        True,
        "per_operator_kpis": True,
        "routes":            False,
    })

    # Which hooks run (incl. non-table hooks like plotting/snapshot/gif); alias: `toggles`.
    hooks: Dict[str, bool] = field(default_factory=lambda: {
        "per_user_kpis":     True,
        "node_state":        True,
        "tasks":             False,
        "ho_events":         True,
        "active_isl":        True,
        "per_operator_kpis": True,
        "plotting":          True,
    })

    # Backward-compat alias: `cfg.outputs.toggles` reads/writes `hooks`.
    @property
    def toggles(self) -> Dict[str, bool]:
        return self.hooks

    @toggles.setter
    def toggles(self, value: Dict[str, bool]) -> None:
        self.hooks = value

    # Misc legacy flags kept for backwards compat
    simulation_results_csv: bool = True
    snapshot_png: bool = True
    gif: GifConfig = field(default_factory=GifConfig)


@dataclass
class VizConfig:
    enable_live_hud: bool = True
    global_view: bool = True
    margin_deg: float = 0.0
    link_alpha: float = 0.8
    max_users_to_viz: int = 10
    dpi: int = 100
    orbit_sampling_step_s: int = 60
    per_operator_colors: Dict[int, str] = field(default_factory=dict)


# KPIs + plots
@dataclass
class KPIConfig:
    computed: List[str] = field(default_factory=list)   # empty = all registered


@dataclass
class ChartSpec:
    kpi: str
    type: str = "line"                                  # line | bar | cdf | hist | heatmap | scatter | network_graph
    name: Optional[str] = None
    title: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    group_by: Optional[str] = None
    agg: Optional[str] = None
    bins: Optional[int] = None
    value: Optional[str] = None
    color: Optional[str] = None
    layout: Optional[str] = None


@dataclass
class PlotConfig:
    formats: List[str] = field(default_factory=lambda: ["png"])
    output_dir: str = "plots"
    charts: List[ChartSpec] = field(default_factory=list)


# Control plane (controllers, scopes, decisions)
# UE is intentionally not a valid scope target in v1.
VALID_NODE_TYPES_FOR_SCOPE = {
    "satellite", "haps", "uav", "ground_station", "base_station",
}

@dataclass
class ScopeSpec:
    operator_ids: List[int] = field(default_factory=list)
    node_types:   List[str] = field(default_factory=list)


@dataclass
class ControllerSpec:
    id: str
    priority: int = 100
    scope: ScopeSpec = field(default_factory=ScopeSpec)
    visibility: str = "own_scope"
    decisions: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class ControlPlaneConfig:
    arbiter: str = "priority_based"
    controllers: List[ControllerSpec] = field(default_factory=list)


@dataclass
class StageGroupSpec:
    name: str
    stages: List[str] = field(default_factory=list)
    parallel: bool = False
    max_workers: int = 4


@dataclass
class PipelineConfig:
    groups: List[StageGroupSpec] = field(default_factory=list)


@dataclass
class GeometryConstraints:
    active_engine: str = "skyfield"
    enable_culling: bool = True
    interpolation_step_s: Optional[int] = None
    satellite_mode: str = "visible_over_duration"


@dataclass
class ConstraintsConfig:
    min_elevation_deg: float = 20.0
    # Gateway-side mask [deg] for the Phase-1 visibility pre-pass union; ignored when there are no gateways.
    gateway_min_elevation_deg: float = 10.0
    rate_threshold_bps_hz: float = 1.5
    geometry: GeometryConstraints = field(default_factory=GeometryConstraints)


# Top-level config
@dataclass
class DigitalTwinConfig:
    """Digital-twin activation (off by default); tiers subset of node|operator|network."""
    enabled: bool = False
    tiers: List[str] = field(default_factory=lambda: ["node"])
    aged_elev_std_deg: float = 0.2
    aged_az_std_deg: float = 0.2
    aged_corr_time_s: float = 0.0


@dataclass
class RadioStageConfig:
    """Radio (RRM) stage: three branches keyed by the CSI assumption; only power sizing differs, grading is shared."""
    csi_mode: str = "stochastic"     # stochastic (outage reserve) | csi (genie) | dt (predicted channel)
    power: str = "min_power"         # min_power | equal | waterfilling | utility
    run_beamforming: bool = False    # compose chan_beamforming as a pre-step


@dataclass
class SimulationConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    # Coarse GEOMETRY trace: all sats (coarse_step_s=0 = no pre-pass) for whole-constellation views.
    geometry_timing: DatasetTimingConfig = field(
        default_factory=lambda: DatasetTimingConfig(
            duration_s=86400.0, step_s=30.0, batch_size_s=900.0))    # 24 h @ 30 s
    # Fine STUDY trace: per-batch coarse visibility pre-pass, only visible sats stored at 1 s.
    study_timing: DatasetTimingConfig = field(
        default_factory=lambda: DatasetTimingConfig(
            duration_s=7200.0, step_s=1.0, coarse_step_s=10.0, batch_size_s=900.0))  # 2 h @ 1 s
    sim_timing: SimulationTimingConfig = field(default_factory=SimulationTimingConfig)
    csi_timing: CSITimingConfig = field(default_factory=CSITimingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    defaults_radio: RadioConfig = field(default_factory=RadioConfig)
    defaults_channel: ChannelConfig = field(default_factory=ChannelConfig)
    defaults_interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    operators: List[Operator] = field(default_factory=list)
    users: UsersConfig = field(default_factory=UsersConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    control_plane: ControlPlaneConfig = field(default_factory=ControlPlaneConfig)
    radio: RadioStageConfig = field(default_factory=RadioStageConfig)
    dt: DigitalTwinConfig = field(default_factory=DigitalTwinConfig)
    kpis: KPIConfig = field(default_factory=KPIConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    # Helpers
    def operator_by_id(self, op_id: int) -> Optional[Operator]:
        for op in self.operators:
            if op.id == op_id:
                return op
        return None

    def all_node_types(self) -> List[str]:
        return ["satellite", "haps", "uav", "ground_station", "base_station"]

    def has_workload(self) -> bool:
        return self.workload.enabled

    def has_compute(self) -> bool:
        return any(any(c.compute_role != "none" for c in op.compute.values())
                   for op in self.operators)

    def save_used(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


# Loader: merges defaults into per-operator overrides
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_compute(d: dict) -> Dict[str, ComputeCapability]:
    return {k: ComputeCapability(**v) for k, v in (d or {}).items()}


def _parse_fixed_nodes(items: list) -> List[FixedNodeEntry]:
    return [FixedNodeEntry(**i) for i in (items or [])]


def _parse_shells(items: list) -> List[ShellConfig]:
    return [ShellConfig(**i) for i in (items or [])]


def _parse_operator(raw: dict, defaults: dict) -> Operator:
    radio_d   = _deep_merge(defaults.get("radio", {}),         raw.get("radio", {}))
    chan_d    = _deep_merge(defaults.get("channel_model", {}), raw.get("channel_model", {}))
    interf_d  = _deep_merge(defaults.get("interference", {}),  raw.get("interference", {}))

    def _mob(raw, phase=None):
        if isinstance(raw, str):
            nm = NodeMobility(model=raw)
        elif isinstance(raw, dict):
            nm = NodeMobility(**raw)
        else:
            nm = NodeMobility()
        if phase is not None:
            nm.phase = int(phase)
        return nm

    sat_raw = raw.get("satellites", {}) or {}
    sat = SatelliteFleet(
        catalog_id_start=sat_raw.get("catalog_id_start", 0),
        generate_tle=sat_raw.get("generate_tle", False),
        tle_file_path=sat_raw.get("tle_file_path", None),
        shells=_parse_shells(sat_raw.get("shells", [])),
        propagator=sat_raw.get("propagator", "auto"),
        tle_cache_dir=sat_raw.get("tle_cache_dir", "data/tle_cache"),
        mobility=_mob(sat_raw.get("mobility")),
        inc_bands=sat_raw.get("inc_bands", []) or [],   # TLE inclination-band filter
        alt_bands=sat_raw.get("alt_bands", []) or [],   # TLE altitude-band filter [km]
    )

    def _fleet(cls, raw_block):
        raw_block = raw_block or {}
        return cls(
            id_range=tuple(raw_block.get("id_range", (0, 0))),
            mobility=_mob(raw_block.get("mobility"), raw_block.get("mobility_phase")),
            positions=_parse_fixed_nodes(raw_block.get("positions", [])),
        )

    return Operator(
        id=raw.get("id", 0),
        name=raw.get("name", ""),
        sat_id_range=tuple(raw.get("sat_id_range", (0, 0))),
        radio=RadioConfig(**radio_d),
        channel_model=ChannelConfig(**chan_d),
        isl=ISLConfig(**(raw.get("isl") or {})),
        interference=InterferenceConfig(**interf_d),
        satellites=sat,
        haps=_fleet(HAPSFleet, raw.get("haps")) if isinstance(raw.get("haps"), dict) else HAPSFleet(),
        uavs=_fleet(UAVFleet, raw.get("uavs")) if isinstance(raw.get("uavs"), dict) else UAVFleet(),
        ground_stations=_fleet(GroundStationFleet, raw.get("ground_stations")[0])
                        if isinstance(raw.get("ground_stations"), list) and raw["ground_stations"]
                        else (_fleet(GroundStationFleet, raw.get("ground_stations"))
                              if isinstance(raw.get("ground_stations"), dict)
                              else GroundStationFleet()),
        base_stations=_fleet(BaseStationFleet, raw.get("base_stations"))
                      if isinstance(raw.get("base_stations"), dict)
                      else BaseStationFleet(),
        compute=_parse_compute(raw.get("compute", {})),
    )


REQUIRED_TOP_LEVEL_SECTIONS = (
    "experiment", "env", "storage", "operators", "users",
    "handover", "outputs", "pipeline",
)


def _outputs_kwargs(raw_outputs: dict) -> dict:
    """OutputsConfig kwargs from raw YAML; accepts legacy `toggles:` as alias for `hooks:`, drops `gif`."""
    d = dict(raw_outputs or {})
    if "toggles" in d:
        # legacy `toggles:` -> `hooks:` unless `hooks:` is given
        legacy = d.pop("toggles")
        d.setdefault("hooks", legacy)
    d.pop("gif", None)
    return d


def _check_required_sections(raw: dict, strict: bool) -> None:
    if not strict: return
    missing = [s for s in REQUIRED_TOP_LEVEL_SECTIONS if s not in raw]
    if missing:
        from ntnconnect.experiment.strict import MissingConfigError
        raise MissingConfigError(
            "Strict config mode: top-level sections missing from YAML: "
            f"{missing}. The simulator allows values only from this YAML "
            f"or from user_overrides() in scripts/run.py."
        )


def load_simulation_config(path: str | Path) -> SimulationConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    # master.yaml is the default base; the given file is deep-merged on top (master skips this).
    _master = Path(__file__).resolve().parents[2] / "configs" / "master.yaml"
    if _master.is_file() and Path(path).resolve() != _master.resolve():
        _base = yaml.safe_load(_master.read_text()) or {}
        raw = _deep_merge(_base, raw)
    # Detect strict mode early so we can refuse partial YAMLs.
    _strict = bool(((raw.get("experiment") or {}).get("strict_config")) is True)
    _check_required_sections(raw, _strict)
    defaults = raw.get("defaults", {}) or {}

    cfg = SimulationConfig(
        experiment=ExperimentConfig(**(raw.get("experiment") or {})),
        env=EnvConfig(**(raw.get("env") or {})),
        geometry_timing=DatasetTimingConfig(**{
            "duration_s": 86400.0, "step_s": 30.0, "batch_size_s": 900.0,
            **(raw.get("geometry_timing") or {})}),
        study_timing=DatasetTimingConfig(**{
            "duration_s": 7200.0, "step_s": 1.0, "batch_size_s": 900.0,
            **(raw.get("study_timing") or {})}),
        sim_timing=SimulationTimingConfig(**(raw.get("sim_timing") or {})),
        csi_timing=CSITimingConfig(**(raw.get("csi_timing") or {})),
        storage=StorageConfig(**(raw.get("storage") or {})),
        defaults_radio=RadioConfig(**(defaults.get("radio") or {})),
        defaults_channel=ChannelConfig(**(defaults.get("channel_model") or {})),
        defaults_interference=InterferenceConfig(**(defaults.get("interference") or {})),
        operators=[_parse_operator(o, defaults) for o in (raw.get("operators") or [])],
        users=UsersConfig(
            **{**(raw.get("users") or {}),
               "mobility":   UserMobility(**((raw.get("users") or {}).get("mobility") or {})),
               "preferences":UserPreferences(**((raw.get("users") or {}).get("preferences") or {}))},
        ),
        handover=HandoverConfig(**(raw.get("handover") or {})),
        workload=WorkloadConfig(**(raw.get("workload") or {})),
        offload=OffloadConfig(**(raw.get("offload") or {})),
        scheduler=SchedulerConfig(**(raw.get("scheduler") or {})),
        routing=RoutingConfig(**(raw.get("routing") or {})),
        cache=CacheConfig(**(raw.get("cache") or {})),
        energy=EnergyConfig(**(raw.get("energy") or {})),
        outputs=OutputsConfig(
            **{**_outputs_kwargs(raw.get("outputs") or {}),
               "gif": GifConfig(**((raw.get("outputs") or {}).get("gif") or {}))},
        ),
        viz=VizConfig(**(raw.get("viz") or {})),
        constraints=ConstraintsConfig(
            **{**(raw.get("constraints") or {}),
               "geometry": GeometryConstraints(**((raw.get("constraints") or {}).get("geometry") or {}))},
        ),
        pipeline=PipelineConfig(groups=[
            StageGroupSpec(**g) for g in ((raw.get("pipeline") or {}).get("groups") or [])
        ]),
        control_plane=_parse_control_plane(raw.get("control_plane") or {}),
        radio=RadioStageConfig(**(raw.get("radio") or {})),
        dt=DigitalTwinConfig(**(raw.get("dt") or {})),
        kpis=KPIConfig(computed=list((raw.get("kpis") or {}).get("computed") or [])),
        plots=PlotConfig(
            formats=list((raw.get("plots") or {}).get("formats") or ["png"]),
            output_dir=(raw.get("plots") or {}).get("output_dir", "plots"),
            charts=[ChartSpec(**c) for c in ((raw.get("plots") or {}).get("charts") or [])],
        ),
    )
    _validate(cfg)
    return cfg


def _parse_control_plane(raw: dict) -> ControlPlaneConfig:
    controllers = []
    for c in (raw.get("controllers") or []):
        sc = c.get("scope", {}) or {}
        controllers.append(ControllerSpec(
            id=c["id"],
            priority=int(c.get("priority", 100)),
            scope=ScopeSpec(
                operator_ids=[int(x) for x in (sc.get("operator_ids") or [])
                              if x not in ("*", None)],
                node_types=[str(x) for x in (sc.get("node_types") or [])
                            if x not in ("*", None)],
            ),
            visibility=c.get("visibility", "own_scope"),
            decisions={k: v for k, v in (c.get("decisions") or {}).items()},
        ))
    return ControlPlaneConfig(
        arbiter=raw.get("arbiter", "priority_based"),
        controllers=controllers,
    )


def _validate(cfg: SimulationConfig) -> None:
    """Enforce sat_id range non-overlap and other invariants at load time."""
    ranges: List[Tuple[int, Tuple[int, int]]] = []
    for op in cfg.operators:
        if op.sat_id_range and op.sat_id_range[1] > op.sat_id_range[0]:
            ranges.append((op.id, op.sat_id_range))
    ranges.sort(key=lambda x: x[1][0])
    for i in range(len(ranges) - 1):
        a_id, (a_lo, a_hi) = ranges[i]
        b_id, (b_lo, b_hi) = ranges[i + 1]
        if b_lo <= a_hi:
            raise ValueError(
                f"Operator {a_id} and {b_id} have overlapping sat_id ranges "
                f"[{a_lo},{a_hi}] vs [{b_lo},{b_hi}]"
            )
    op_ids = [op.id for op in cfg.operators]
    if len(set(op_ids)) != len(op_ids):
        raise ValueError(f"Duplicate operator.id in config: {op_ids}")

    # Control-plane validation
    cp = cfg.control_plane
    seen_ids = set()
    for c in cp.controllers:
        if not c.id:
            raise ValueError("Every controller must have a non-empty id.")
        if c.id in seen_ids:
            raise ValueError(f"Duplicate controller.id: {c.id}")
        seen_ids.add(c.id)
        # UE-passive enforcement (V2 future-work boundary)
        bad_types = [nt for nt in c.scope.node_types
                     if nt in ("user", "ue", "users")]
        if bad_types:
            raise ValueError(
                f"Controller '{c.id}' includes UE node types {bad_types} in scope. "
                f"v1 treats UEs as passive - UE-side control is reserved as "
                f"future work (V2_PASSIVE_TO_ACTIVE_UE)."
            )
        invalid = [nt for nt in c.scope.node_types
                   if nt and nt not in VALID_NODE_TYPES_FOR_SCOPE]
        if invalid:
            raise ValueError(
                f"Controller '{c.id}' has invalid node_types in scope: {invalid}. "
                f"Allowed: {sorted(VALID_NODE_TYPES_FOR_SCOPE)}"
            )
        if c.visibility not in {"own_node","own_scope","own_operator","peers","global"}:
            raise ValueError(
                f"Controller '{c.id}' has invalid visibility '{c.visibility}'."
            )
