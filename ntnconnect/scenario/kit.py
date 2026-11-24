# Scenario kit: the ONE place scenarios resolve config/TLEs and run the phases,
# so every scenario folder is thin and identical in shape:
#
#   scenarios/<name>/
#     tles/          TLE files      (empty -> default TLEs kept on the operators)
#     configs/       YAML config    (empty -> the default configs/master.yaml)
#     scripts/       01_config, 02_dataset(phase1), 03_simulate(phase2),
#                    04_metrics(phase3), run_all, plots, analysis.ipynb
#     scenario.py    single-file PROGRAMMATIC config + run (no YAML, no many files)
#
# Fallbacks: no scenario config -> configs/master.yaml; no scenario TLEs -> the
# operators' own/default TLE paths. A scenario may also build its config purely in
# code and pass it to Scenario(programmatic_cfg=...).

from __future__ import annotations
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from ntnconnect.experiment.config import SimulationConfig, load_simulation_config

# repo root = .../ntnconnect/scenario/kit.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER_CONFIG = _REPO_ROOT / "configs" / "master.yaml"


def resolve_config(scenario_dir, config_name: Optional[str] = None) -> SimulationConfig:
    """Load the scenario's YAML config: configs/<config_name> if given, else the
    first *.yaml/*.yml in configs/, else the default master.yaml."""
    cdir = Path(scenario_dir) / "configs"
    if config_name:
        p = cdir / config_name
        if p.exists():
            return load_simulation_config(p)
    if cdir.is_dir():
        ys = sorted(cdir.glob("*.yaml")) + sorted(cdir.glob("*.yml"))
        if ys:
            return load_simulation_config(ys[0])
    return load_simulation_config(DEFAULT_MASTER_CONFIG)     # fallback: master


def resolve_tles(scenario_dir, cfg: SimulationConfig) -> SimulationConfig:
    """Point each operator at a matching TLE in the scenario's tles/ folder. If
    the folder is empty/absent, operators keep their own (default) TLE path."""
    tdir = Path(scenario_dir) / "tles"
    tles = (sorted(tdir.glob("*.tle")) + sorted(tdir.glob("*.txt"))
            if tdir.is_dir() else [])
    if not tles:
        return cfg                                            # default TLEs
    for op in cfg.operators:
        match = next((t for t in tles if op.name.lower() in t.stem.lower()), tles[0])
        op.satellites.tle_file_path = str(match)
    return cfg


class Scenario:
    """Thin driver every scenario shares. Resolves config + TLEs, then runs the
    three phases against one output directory."""

    def __init__(self, scenario_dir, config_name: Optional[str] = None,
                 programmatic_cfg: Optional[SimulationConfig] = None):
        self.dir = Path(scenario_dir).resolve()
        self.cfg = programmatic_cfg or resolve_config(self.dir, config_name)
        resolve_tles(self.dir, self.cfg)

    @property
    def out_dir(self) -> Path:
        return Path(self.cfg.storage.data_dir) / f"run_{self.cfg.storage.run_id:03d}"

    @property
    def run_dir(self) -> Path:
        """Where Phase 2 runs and Phase 3 analyses. The fine STUDY dataset is the
        primary and lives at out_dir root (the geometry dataset is in
        out_dir/geometry/), so the run directory is simply out_dir."""
        return self.out_dir

    def _timestamps(self):
        # Phase 2 runs on the fine STUDY timing (1 s) when datasets are built;
        # otherwise the env window (quick runs).
        from ntnconnect.scenario.phase1 import _build_timestamps
        if bool(getattr(self.cfg.experiment, "generate_coverage", True)):
            t = self.cfg.study_timing
            return _build_timestamps(t.utc, t.duration_s, t.step_s)
        return _build_timestamps(self.cfg.env.start_utc, self.cfg.env.duration_s,
                                 self.cfg.env.step_size_s)

    # --- the phases -----------------------------------------------------------
    def phase1(self) -> Path:
        """Phase 1: geometry / dataset (honours the coverage flag + idempotent skip)."""
        from ntnconnect.scenario.phase1 import run_phase1
        return run_phase1(self.cfg)

    def phase2(self) -> Path:
        """Phase 2: per-tick simulation over the STUDY dataset (run_dir)."""
        from ntnconnect.engine.engine import SimulationEngine
        eng = SimulationEngine(self.cfg, sim_dir=self.run_dir)
        eng.run(self._timestamps())
        return self.run_dir

    def phase3(self, **kwargs):
        """Phase 3: the ONE standard analysis for every scenario. Reads the
        standard per-run tables written by Phase 2 (same layout for all studies,
        no per-study manifest or path) and writes <out_dir>/analysis/. Returns
        the summary dict."""
        from ntnconnect.analysis.standard import run_standard_analysis
        return run_standard_analysis(self.run_dir)

    def run_all(self, **phase3_kwargs):
        self.phase1()
        self.phase2()
        return self.phase3(**phase3_kwargs)
