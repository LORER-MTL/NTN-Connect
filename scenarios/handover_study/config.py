#!/usr/bin/env python
"""WISEE handover study - THE CONFIGURATION (this scenario only).

Every value the WISEE final campaign needs, and nothing else. It is the faithful
port of the original config: v2_public/scenarios/WISEE/handover_study/
run_final_tt.py (env vars) + config_algorithm.py (link budget).

All phase drivers read from here:
    scripts/phase1/generate_dataset.py  (geometry)
    scripts/phase2/simulate.py          (per-case cfg)
    scripts/phase3/metrics.py           (SE thresholds live in phase3)
    scripts/main_run.py                 (the sweep)

Two groups below:
  (A) SWEEP        - what the campaign iterates over (gammas, points, strategies)
  (B) PER-CASE CFG - the link budget, channel, handover and interference values
                     that build_wisee_cfg() applies to ONE (const, strat, point,
                     gamma) case. The four env vars the engine does not honour
                     (IHAAP_ALPHA2, IHAAP_TPASS, WISEE_MAX_CELLS, WISEE_QOS_RATE_BPS)
                     are mirrored onto cfg fields so behaviour is identical.
"""
from __future__ import annotations
import os, math
from pathlib import Path

from ntnconnect.experiment.config import (
    SimulationConfig, UsersConfig, Operator, SatelliteFleet, GroundStationFleet,
    FixedNodeEntry, NodeMobility,
)
from ntnconnect.scenario.presets import wisee_handover_pipeline

HERE = Path(__file__).resolve().parent
TLE_DIR = HERE / "tles"                     # the scenario's own TLEs (relative, portable)

# ===== COMMON STUDY CONSTANTS ===============================================
# The SWEEP (which gammas / operating points / strategies / powers to iterate)
# is NOT here - it lives in scripts/main_run.py. This file holds only what is
# common to every case: the admission reward, the run length, and the fixed
# per-constellation t_pass.
M    = 0.25                                 # admission reward m
MIN  = 45                                   # run length [min]
TP   = {"starlink": 180, "oneweb": 300}     # t_pass per constellation [s]
MIN_ELEV_USER_DEG = 25.0                    # user/UE elevation mask [deg] (fixed)
MIN_ELEV_GW_DEG   = 10.0                    # gateway/feeder elevation mask [deg] (fixed)

# ===== (B) PER-CASE CFG - link budget (config_algorithm.py, VERBATIM) =======
B_USER_HZ    = 62.5e6                 # SYSTEM_BW 500e6 / BEAMS 8
CARRIER_HZ   = 12e9
SAT_GAIN_DB  = {"oneweb": 38.5, "starlink": 32.0}
RX_GAIN_DB   = 20.0
NF_DB        = 1.2
TSYS_K       = 290.0
BEAMS        = 8
OP_ID        = {"oneweb": 1, "starlink": 2}
TLE_FILE     = {"oneweb": "OneWeb.txt", "starlink": "StarLink.txt"}
INC_BANDS    = [[52.6, 53.5], [87.0, 89.0]]
ALT_BANDS    = [[500.0, 600.0], [1100.0, 1300.0]]
HOTSPOT      = dict(num_seed_users=50, grid_method="hotspot",
                    hotspot_lat=52.0, hotspot_lon=-72.0, hotspot_radius_km=400.0)
GATEWAYS = {
    "oneweb": [("Cambridge_Bay_NU",69.117,-105.057,23.0),("Inuvik_NT",68.35,-133.5,34.0),
               ("Calgary_AB",51.05,-114.07,1099.0),("Allan_Park_ON",44.2,-80.9,413.0),
               ("St_Johns_NL",47.56,-52.71,52.0)],
    "starlink": [("Calgary_AB",51.05,-114.07,1099.0),("Estevan_SK",49.14,-102.99,580.0),
                 ("Saskatoon_SK",52.13,-106.67,482.0),("Plattsville_ON",43.40,-80.62,315.0),
                 ("Debert_NS",45.42,-63.45,47.0)],
}


# ===== FLAGS / FLIPS (edit these to toggle the study behaviour) =============
USER_LEVEL       = True     # user-level granularity (else cluster-level FDMA split)
TT_CHANNEL       = True     # two-timescale channel (per-slot large-scale + per-sub-slot Rician)
BESTEFFORT_POWER = True     # best-effort power block after guaranteed admission
STOCH_WEATHER    = False    # stochastic weather variation (channel weather stays ON)
OUTAGE_EPS       = 0.05     # outage epsilon for the power reserve (z = 1.645)
HO_PERIOD_S      = 2.0      # handover decision period [s]
MAX_CELLS        = BEAMS    # per-satellite beam cap


def apply_case_env(cnst: str, a1: float, a2: float, g_db: float, psat_w: float):
    """Set the per-case environment the engine honours. Only gamma/psat vary per
    case; everything else is the FLAGS above (common to the whole study). qos is
    derived per gamma as the Shannon rate at the SE threshold."""
    b = lambda x: "1" if x else "0"
    qos = 62.5e6 * math.log2(1.0 + 10 ** (g_db / 10.0))
    os.environ.update(
        WISEE_USER_LEVEL=b(USER_LEVEL), WISEE_MAX_CELLS=str(MAX_CELLS),
        WISEE_QOS_RATE_BPS=str(qos), WISEE_TT_CHANNEL=b(TT_CHANNEL),
        WISEE_BESTEFFORT_POWER=b(BESTEFFORT_POWER), WISEE_STOCH_WEATHER=b(STOCH_WEATHER),
        WISEE_OUTAGE_EPS=str(OUTAGE_EPS), WISEE_PMAX_W=str(psat_w),
        WISEE_HO_PERIOD_S=str(HO_PERIOD_S),
    )
    return qos


def build_wisee_cfg(constellation: str, strategy: str, *, a1: float, a2: float,
                    g_db: float, psat_w: float = 100.0,
                    minutes: float = MIN, step_s: float = 1.0) -> SimulationConfig:
    """Faithful Phase-2 config for one (constellation, strategy, point, gamma) case.
    Mirrors run_final_tt._run + config_algorithm.py exactly."""
    cn = constellation
    qos = apply_case_env(cn, a1, a2, g_db, psat_w)
    tp = float(TP[cn])

    cfg = SimulationConfig()
    cfg.experiment.name = f"{cn}__{strategy}"
    cfg.experiment.random_seed = 42
    cfg.experiment.generate_coverage = False
    cfg.env.start_utc = "2026-05-04T00:00:00Z"; cfg.env.region_name = "Quebec"
    cfg.env.duration_s = float(minutes) * 60.0; cfg.env.step_size_s = float(step_s)

    op = Operator(name=cn, id=OP_ID[cn])
    op.satellites = SatelliteFleet(
        tle_file_path=str(TLE_DIR / TLE_FILE[cn]), propagator="skyfield",
        inc_bands=INC_BANDS, alt_bands=ALT_BANDS, mobility=NodeMobility(model="dataset"))
    op.ground_stations = GroundStationFleet(
        id_range=(OP_ID[cn]*100000, OP_ID[cn]*100000+len(GATEWAYS[cn])),
        positions=[FixedNodeEntry(name=n, lat=la, lon=lo, alt_m=al)
                   for (n, la, lo, al) in GATEWAYS[cn]])
    op.radio.carrier_freq_hz = CARRIER_HZ
    op.radio.bandwidth_hz = B_USER_HZ
    op.radio.power_max_w = float(psat_w)
    op.radio.antenna_gain_tx_db = SAT_GAIN_DB[cn]
    op.radio.antenna_gain_rx_db = RX_GAIN_DB
    op.radio.noise_figure_db = NF_DB
    op.radio.system_temperature_k = TSYS_K
    op.channel_model.n_rx = 1; op.channel_model.antenna_mode = "miso"
    # Channel model VERBATIM from build_common_csi_cfg() (_csi_common.py): the
    # Professional_NTN base + gas absorption + rain AR-1 + shadowing + K-factor.
    # Without these the channel is FSPL-only and path loss is ~2.2 dB low, which
    # makes the required power (and thus blocking) wrong.
    cm = op.channel_model
    cm.model_type = "Professional_NTN"
    cm.enable_shadowing = True;        cm.shadowing_std_db = 2.0
    cm.enable_rain_ar1 = True;         cm.pro_ntn_rain_corr = 2e-4
    cm.enable_gas_absorption = True;   cm.pro_ntn_gas_factor_db = 0.5
    cm.enable_k_factor_38811 = True;   cm.pro_ntn_kappa_rho = 0.95
    cfg.operators = [op]
    cfg.defaults_radio.carrier_freq_hz = CARRIER_HZ
    cfg.defaults_radio.bandwidth_hz = B_USER_HZ
    cfg.defaults_radio.noise_figure_db = NF_DB
    cfg.defaults_radio.system_temperature_k = TSYS_K

    cfg.users = UsersConfig(mobility=NodeMobility(model="static"), **HOTSPOT)

    cfg.constraints.min_elevation_deg = MIN_ELEV_USER_DEG
    cfg.constraints.gateway_min_elevation_deg = MIN_ELEV_GW_DEG
    ho = cfg.handover
    ho.min_elevation_deg = MIN_ELEV_USER_DEG
    ho.mode = strategy
    ho.qos_rate_bps = qos                              # mirrors WISEE_QOS_RATE_BPS
    ho.dro_epsilon = OUTAGE_EPS
    ho.run_every_n_ticks = max(1, round(HO_PERIOD_S / step_s))  # ho_period [s]
    ho.ttt_ticks = 0; ho.mts_ticks = 0
    ho.hysteresis_deg = 0.05
    ho.max_cells_per_sat = MAX_CELLS                   # mirrors WISEE_MAX_CELLS
    ho.shadowing_38811_scenario = "suburban"
    ho.mc_channel_samples = 50                         # 50 fast Rician samples / slow tick
    ho.channel_predictor = "ar1"; ho.channel_predictor_window = 20

    cfg.radio.csi_mode = "stochastic"; cfg.radio.power = "min_power"

    ic = cfg.defaults_interference
    ic.ext_mode = "fixed_in"; ic.ext_knob = -6.0; ic.ext_K_r = 8
    ic.ext_interferer_pmax_w = 100.0; ic.ext_pol_iso_db = 12.0

    cfg.pipeline = wisee_handover_pipeline()
    return cfg
