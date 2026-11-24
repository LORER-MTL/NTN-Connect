# SimulationEngine: build orchestrator from config, then run the scheduler.

from __future__ import annotations
import importlib
import logging
import numpy as np
from pathlib import Path
from typing import List

from ntnconnect.experiment.config import SimulationConfig
from ntnconnect.orchestration.orchestrator import (
    SimulationOrchestrator, SimulationScheduler, StageGroup,
)
from ntnconnect.registry.registry import get_stage_class, list_stages, all_stages
from ntnconnect.engine.stage_base import stage_callable

_log = logging.getLogger(__name__)


# Modules that, when imported, populate the stage registry.
DEFAULT_STAGE_MODULES = [
    "ntnconnect.engine.stages.control_plane",
    "ntnconnect.engine.stages.read_geometry",
    "ntnconnect.engine.stages.dimensions",     # writes dim tables once, after first read
    "ntnconnect.engine.stages.geometry",
    "ntnconnect.engine.stages.measurement",   # coarse RSRP + interference estimate + load
    "ntnconnect.engine.stages.isl",
    "ntnconnect.engine.stages.channel",
    "ntnconnect.engine.stages.interference",
    "ntnconnect.engine.stages.external_interference",   # optimization-safe I_ext (7 modes)
    "ntnconnect.engine.stages.min_power_miso",          # closed-form MISO min-power allocation
    "ntnconnect.engine.stages.link_budget",            # sinr_calc (canonical link-budget/SINR stage)
    "ntnconnect.engine.stages.agent_step",              # User / Gateway.step()
    "ntnconnect.engine.stages.read_ephemeris",      # read 12-h ephemeris dataset
    "ntnconnect.engine.stages.handover",
    "ntnconnect.engine.stages.compute_stages",   # workload/offload/scheduling/cache/node_state/ho_classify/rrm(kpi)
    "ntnconnect.engine.stages.metrics",
    "ntnconnect.engine.stages.constraints",
    "ntnconnect.engine.stages.mobility",          # mobility (per-node position resolution)
    "ntnconnect.engine.stages.clustering",    # per-tick Earth-cell recomputation
    "ntnconnect.engine.stages.topology",          # topo_feeder, topo_backhaul
    "ntnconnect.engine.stages.routing",           # net_build_graph, net_route_isl, net_route_e2e, net_load_balance
    "ntnconnect.engine.stages.radio",             # consolidated RRM: csi|stochastic|dt branches
    "ntnconnect.engine.stages.rrm",               # rrm_mcs, rrm_allocate_rb, acc_admission, chan_beamforming, chan_power_control
    "ntnconnect.engine.stages.power_algorithms",   # rrm_power_equal, _waterfilling, _utility
    "ntnconnect.engine.stages.radio_link",         # scn_radio_link (combined BF + RRM + SINR)
    "ntnconnect.engine.stages.dt_channel",         # Digital-Twin SECOND channel pipeline
    "ntnconnect.engine.stages.channel_predict", # Digital-Twin predict stage
    "ntnconnect.engine.stages.compute",       # work_execute, state_energy, state_link
    "ntnconnect.engine.stages.finalize",          # post_aggregate, post_report, post_persist, post_summaries
]


# Modules whose import populates the radio-algorithm registry (self-register
# via @register_radio_algorithm); looked up by scn_radio_link at runtime.
DEFAULT_RADIO_ALGO_MODULES = [
    "ntnconnect.registry.agents.radio.min_power_socp",   # "cvx_mimo_min_power" (per-sat MU-MIMO SOCP)
    "ntnconnect.registry.agents.radio.link_budget_fast", # "link_budget_fast" - no CVX, ~10× faster
    # add new algorithms here, one line per file
]


# Channel-predictor modules (self-register before scn_channel_predict is built).
DEFAULT_CHANNEL_PREDICTOR_MODULES = [
    # add channel-predictor implementations here, one line per file
]


DEFAULT_HOOK_MODULES = [
    "ntnconnect.analysis.hooks._extras",
    "ntnconnect.analysis.hooks.plotting",
    "ntnconnect.analysis.hooks.audit",             # radio link audit (link_audit + power_alloc + beamforming_audit)
    "ntnconnect.analysis.hooks.audit_radio",       # rb_allocation + beam_budget + spectrum_usage
    "ntnconnect.analysis.hooks.audit_compute",     # compute_resource + task_audit + cache_audit
    "ntnconnect.analysis.hooks.validation",        # validation_checks
    "ntnconnect.analysis.hooks.scn_ho_counter",    # per-scenario HO counter
    "ntnconnect.analysis.hooks.users_per_tick",    # per-tick users frame with cluster_id
    "ntnconnect.analysis.hooks.scn_beam_log",      # per-tick (user, beam_id) when beam_tagged
    "ntnconnect.analysis.hooks.visibility_log",     # per-tick visibility + contact + SINR snapshot
    "ntnconnect.analysis.hooks.handover_events_log",# per-tick handover-event diff
    "ntnconnect.analysis.hooks.link_visibility_log",# remaining-visibility per (user, sat) tick
    "ntnconnect.analysis.hooks.scenario_snapshot",  # static scenario map at end-of-sim
    "ntnconnect.analysis.hooks.user_trace",         # per-user time-series trace at end-of-sim
]


class SimulationEngine:
    def __init__(self, cfg: SimulationConfig, sim_dir: Path | None = None):
        self.cfg = cfg
        self.sim_dir = Path(sim_dir) if sim_dir else \
            Path(cfg.storage.data_dir) / f"run_{cfg.storage.run_id:03d}"
        self.orchestrator = SimulationOrchestrator(cfg)
        self.kpi_results: dict = {}      # KPI name -> List[pd.DataFrame]
        # channel-twin store, mirrored into state.data by scn_channel_predict so
        # multiprocessing can pickle it; created on demand (None until first predict).
        self.channel_twin = None
        # persistent per-sat Satellite objects (own channel/history/predictor/DTChannel)
        self.satellites: dict[int, "Satellite"] = {}        # type: ignore
        # Agent registries; empty falls back to the vectorised dataframe path.
        self.users:    dict[int, "User"]    = {}            # type: ignore
        self.gateways: dict[int, "Gateway"] = {}            # type: ignore
        self._sat_id_to_op_id: dict[int, int] = {}
        self._is_setup = False

    def _resolve_operator_for_sat(self, sat_id: int):
        """Resolve a sat_id to its Operator via each operator's sat-id range."""
        cached = self._sat_id_to_op_id.get(int(sat_id))
        if cached is not None:
            for op in self.cfg.operators:
                if op.id == cached: return op
        for op in self.cfg.operators:
            lo, hi = getattr(op, "sat_id_range", (0, 0))
            if int(lo) <= int(sat_id) <= int(hi):
                self._sat_id_to_op_id[int(sat_id)] = int(op.id)
                return op
        return self.cfg.operators[0] if self.cfg.operators else None

    def get_or_create_satellite(self, sat_id: int):
        """Lazy per-sat Satellite object factory. Reuses across ticks."""
        sid = int(sat_id)
        sat = self.satellites.get(sid)
        if sat is not None: return sat
        from ntnconnect.engine.satellite import Satellite
        op = self._resolve_operator_for_sat(sid)
        if op is None: return None
        rng = getattr(self, "rng_predictor", np.random.default_rng())
        sat = Satellite(sid, op, self.cfg, rng=rng)
        self.satellites[sid] = sat
        return sat

    def _auto_batch_outputs(self):
        """Derive per-batch output-flush cadence from the dataset manifest so phase 2
        never buffers the whole run; an explicit batch_log_every_n_ticks wins."""
        try:
            if int(getattr(self.cfg.outputs, "batch_log_every_n_ticks", 0) or 0) > 0:
                return                                   # explicit setting wins
            import json
            run = Path(self.cfg.storage.data_dir) / f"run_{self.cfg.storage.run_id:03d}"
            mpath = run / "manifest.json"
            if not mpath.is_file():
                return
            m = json.loads(mpath.read_text())
            bs = float(m.get("batch_size_s", 0) or 0)
            step = float(self.cfg.env.step_size_s or 0)
            if bs > 0 and step > 0:
                n = max(1, round(bs / step))
                self.cfg.outputs.batch_log_every_n_ticks = n
                _log.info("auto per-batch output flush: every %d ticks "
                          "(dataset batch %.0fs / step %.3gs)", n, bs, step)
        except Exception as e:
            _log.warning("auto batch-output derivation skipped: %s", e)

    def setup(self):
        if self._is_setup: return
        self._init_rng()      # before any stochastic subsystem is instantiated
        if getattr(self.cfg.experiment, "strict_config", False):
            from ntnconnect.experiment.strict import assert_no_missing
            assert_no_missing(self.cfg, allowed_empty=[
                # optional fields: None/empty == "no preference"
                "experiment.notes",
                "experiment.global_seed",                 # optional
                "users.preferences.preferred_operator_id",
                "users.max_users_per_cluster",
                "outputs.formats",
                "constraints.geometry.interpolation_step_s",
                "defaults_channel.access_models",          # empty == built-in defaults
            ])
            _log.info("Strict config mode: all required fields present.")
        # range checks always run: catch div-by-zero / log-of-zero pathologies
        try:
            from ntnconnect.experiment.strict import validate_value_ranges
            validate_value_ranges(self.cfg)
        except Exception as e:
            _log.error("Configuration value-range validation failed: %s", e)
            raise
        self._auto_batch_outputs()
        self._import_modules(DEFAULT_STAGE_MODULES)
        # register radio algorithms + channel predictors before groups are built
        self._import_modules(DEFAULT_RADIO_ALGO_MODULES)
        self._import_modules(DEFAULT_CHANNEL_PREDICTOR_MODULES)
        self._build_groups()
        self._build_agent_registries()
        self._import_hook_modules()
        self.orchestrator.validate()
        _log.info("Engine ready: %d groups, %d hooks",
                  len(self.orchestrator.groups),
                  len(self.orchestrator.hooks))
        self._is_setup = True

    def _init_rng(self):
        """Deterministic per-subsystem RNG plumbing (independent child streams)."""
        import numpy as _np, random as _py_rand
        seed = getattr(self.cfg.experiment, "global_seed", None)
        if seed is None:
            seed = getattr(self.cfg.experiment, "random_seed", None)
        if seed is None:
            seed = _np.random.SeedSequence().entropy
            _log.warning("No global_seed (or random_seed) set; using "
                         "ephemeral master seed=%d. Reproducibility OFF.", seed)
        self.master_seed = int(seed)
        ss = _np.random.SeedSequence(self.master_seed)
        ch = ss.spawn(8)      # spare streams for future subsystems
        self.rng_channel   = _np.random.default_rng(ch[0])
        self.rng_predictor = _np.random.default_rng(ch[1])
        self.rng_mc        = _np.random.default_rng(ch[2])
        self.rng_handover  = _np.random.default_rng(ch[3])
        self.rng_misc      = _np.random.default_rng(ch[4])
        # seed legacy global RNGs too, for ported code not yet on self.rng_*
        _np.random.seed(self.master_seed & 0xFFFFFFFF)
        _py_rand.seed(self.master_seed)
        _log.info("Engine RNG initialised: master_seed=%d", self.master_seed)

    def _build_agent_registries(self):
        """Build engine.users (from cfg.users.explicit_user_list) and
        engine.gateways (from each operator's ground_stations); empty when absent."""
        from ntnconnect.engine.user    import User
        from ntnconnect.engine.gateway import Gateway
        # Users
        explicit = getattr(getattr(self.cfg, "users", None),
                            "explicit_user_list", None)
        if explicit:
            for u in explicit:
                uid = int(u.get("user_id", u.get("id", -1)))
                if uid < 0: continue
                self.users[uid] = User(
                    user_id=uid,
                    lat=float(u.get("lat", 0.0)),
                    lon=float(u.get("lon", 0.0)),
                    alt_m=float(u.get("alt_m", 0.0)),
                    cfg=self.cfg, engine=self)
        # Gateways (from every operator's GroundStationFleet)
        for op in getattr(self.cfg, "operators", []) or []:
            gs = getattr(op, "ground_stations", None)
            if gs is None: continue
            positions = list(getattr(gs, "positions", []) or [])
            base_id, _ = (gs.id_range if getattr(gs, "id_range", None) else (0, 0))
            for i, p in enumerate(positions):
                gid = int(base_id) + i
                if gid in self.gateways: continue
                self.gateways[gid] = Gateway(
                    gw_id=gid,
                    name=str(getattr(p, "name", f"GS_{gid}")),
                    operator_id=int(op.id),
                    lat=float(getattr(p, "lat", 0.0)),
                    lon=float(getattr(p, "lon", 0.0)),
                    alt_m=float(getattr(p, "alt_m", 0.0)),
                    source="cfg.ground_stations",
                    cfg=self.cfg, engine=self)
        if self.users or self.gateways:
            _log.info("Agent registries built: %d users, %d gateways",
                       len(self.users), len(self.gateways))

    def _import_modules(self, modules: List[str]):
        for m in modules:
            try:
                importlib.import_module(m)
            except ModuleNotFoundError as e:
                # optional module not installed: skip (documented extension point)
                _log.warning("Optional module not installed (%s): %s", m, e)
            except Exception as e:
                # a real error inside a module is a bug: refuse to start
                _log.error("Module %s failed to import: %s", m, e)
                raise

    def _import_hook_modules(self):
        for m in DEFAULT_HOOK_MODULES:
            try:
                importlib.import_module(m)
            except Exception as e:
                _log.warning("Skipping hook module %s: %s", m, e)
        # attach hooks the config asks for
        from ntnconnect.analysis.hooks.registry import list_hooks
        hooks_cfg = (getattr(self.cfg.outputs, "hooks", None)
                     or getattr(self.cfg.outputs, "toggles", {}) or {})
        for name, builder in list_hooks().items():
            if not hooks_cfg.get(name, True):
                continue
            try:
                self.orchestrator.add_hook(builder(self))
            except Exception as e:
                _log.warning("Hook '%s' build failed: %s", name, e)

    def _build_groups(self):
        """Read cfg.pipeline.groups into StageGroup objects."""
        groups_cfg = getattr(self.cfg, "pipeline", None)
        if groups_cfg is None or not getattr(groups_cfg, "groups", None):
            return

        registered = all_stages()      # across main + dt + compute registries
        for gspec in self.cfg.pipeline.groups:
            stages = []
            for sname in gspec.stages:
                if sname not in registered:
                    _log.warning("Stage '%s' not registered; skipped.", sname)
                    continue
                cls = registered[sname]
                inst = cls(self.cfg)
                inst.engine = self          # share cross-tick state
                stages.append(stage_callable(inst))
            if stages:
                self.orchestrator.add_group(StageGroup(
                    name=gspec.name,
                    stages=stages,
                    parallel=bool(gspec.parallel),
                    max_workers=int(getattr(gspec, "max_workers", 4)),
                ))

    def run(self, timestamps):
        if not self._is_setup:
            self.setup()
        scheduler = SimulationScheduler(timestamps, self.cfg.env.step_size_s)
        # profile wall/CPU/peak memory as a first-class run artifact
        from ntnconnect.engine.resource_profile import RunProfiler
        _prof = RunProfiler().start()
        result = self.orchestrator.run(scheduler)
        _prof.stop()
        try:
            _prof.write(self.sim_dir, extra={
                "n_intervals": len(timestamps) if timestamps is not None else None,
                "step_size_s": float(self.cfg.env.step_size_s),
                "master_seed": getattr(self, "master_seed", None),
            })
        except Exception as e:
            _log.warning("Could not write resource_profile.json: %s", e)
        # per-stage timing (txt + json)
        try:
            report = self.orchestrator.timing_report()
            _log.info(report)
            (self.sim_dir / "stage_timing.txt").write_text(report, encoding="utf-8")
        except Exception as e:
            _log.warning("Could not write stage_timing.txt: %s", e)
        # JSON view so analysis need not parse the text report
        try:
            import json
            timings = getattr(self.orchestrator, "stage_timings", None)
            if timings is not None:
                serialisable = {
                    str(k): (float(v) if hasattr(v, "__float__") else v)
                    for k, v in dict(timings).items()
                }
                (self.sim_dir / "stage_timing.json").write_text(
                    json.dumps({"master_seed": getattr(self, "master_seed", None),
                                "timings": serialisable}, indent=2),
                    encoding="utf-8")
        except Exception as e:
            _log.warning("Could not write stage_timing.json: %s", e)
        return result
