# SkyNTNPro v1 - Phase 1 driver: run the per-operator, per-node-type generators
# and write the partitioned tables under outputs/run_<id>/.

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from ntnconnect.experiment.config import SimulationConfig, DatasetTimingConfig
from ntnconnect.common.utils.partition_writer import write_partitioned
from ntnconnect.scenario.frames import lla_to_ecef
from ntnconnect.scenario.sat_generator import SatelliteGenerator
from ntnconnect.scenario.fixed_generators import (
    HAPSGenerator, UAVGenerator,
    GroundStationGenerator, BaseStationGenerator,
)
from ntnconnect.scenario.user_generator import generate_users


GENERATORS = {
    "satellite":      SatelliteGenerator,
    "haps":           HAPSGenerator,
    "uav":            UAVGenerator,
    "ground_station": GroundStationGenerator,
    "base_station":   BaseStationGenerator,
}


def _build_timestamps(start_utc, duration_s=None, step_s=None) -> List[datetime]:
    """Build the tick timestamps. Two call styles:
      * _build_timestamps(env)                          - an EnvConfig (legacy)
      * _build_timestamps(start_utc, duration_s, step_s) - explicit values
    """
    if duration_s is None and step_s is None and hasattr(start_utc, "start_utc"):
        env = start_utc
        start_utc, duration_s, step_s = env.start_utc, env.duration_s, env.step_size_s
    start = datetime.fromisoformat(str(start_utc).replace("Z", "+00:00"))
    n = max(1, int(float(duration_s) // float(step_s)))   # float math for fractional steps
    return [start + timedelta(seconds=i * float(step_s)) for i in range(n)]


_log = logging.getLogger(__name__)


def _ground_ecef(lat, lon, alt=None):
    """ECEF (N,3) for arrays of lat/lon[/alt_m]. None/empty -> None."""
    lat = np.asarray(lat, dtype=float)
    if lat.size == 0:
        return None
    lon = np.asarray(lon, dtype=float)
    alt = np.zeros_like(lat) if alt is None else np.asarray(alt, dtype=float)
    x, y, z = lla_to_ecef(lat, lon, alt)
    return np.column_stack([x, y, z])


def _visible_sat_ids(sat_df, groups) -> set:
    """Keep-set: sats reaching the elevation mask for any ground point in any group
    at any coarse tick (union rule: users at user-mask OR gateways at gw-mask).
    groups: list of (ground_ecef (M,3), mask_deg)."""
    if sat_df.empty:
        return set()
    prepped = []
    for gecef, mask in groups:
        if gecef is None or len(gecef) == 0:
            continue
        gunit = gecef / np.linalg.norm(gecef, axis=1, keepdims=True)
        prepped.append((gecef, gunit, float(mask)))
    if not prepped:                                     # no ground refs -> keep all
        return {int(x) for x in sat_df["node_id"].unique()}
    keep = set()
    for nid, g in sat_df.groupby("node_id", sort=False):
        sx, sy, sz = lla_to_ecef(g["lat"].to_numpy(), g["lon"].to_numpy(),
                                 g["alt_m"].to_numpy())
        sat_ecef = np.column_stack([sx, sy, sz])
        for gecef, gunit, mask in prepped:
            los = sat_ecef[:, None, :] - gecef[None, :, :]     # (N_t, M, 3)
            rng = np.linalg.norm(los, axis=2)
            up  = np.sum(los * gunit[None, :, :], axis=2)
            elev = np.degrees(np.arcsin(np.clip(up / rng, -1.0, 1.0)))
            if (elev >= mask).any():
                keep.add(int(nid))
                break
    return keep


def _batch_groups(fine_ts, batch_s):
    """Partition fine timestamps into ordered (batch_idx -> [timestamps]) preserving
    global order, so a per-batch build can restore global t_idx."""
    start = fine_ts[0]
    order: dict = {}
    for ts in fine_ts:
        b = int((ts - start).total_seconds() // batch_s)
        order.setdefault(b, []).append(ts)
    return order


def _generate_satellites_batched(cfg, op, gen, fine_ts, t, coarse_step, users_t0):
    """Offline satellites for a fine dataset: per batch, coarse-propagate all sats,
    keep the visible set, then fine-propagate only those; global t_idx restored."""
    user_ecef = _ground_ecef(users_t0["lat"].to_numpy(), users_t0["lon"].to_numpy(),
                             users_t0.get("alt_m")) if not users_t0.empty else None
    gws = list(getattr(op.ground_stations, "positions", []) or [])
    gw_ecef = _ground_ecef([g.lat for g in gws], [g.lon for g in gws],
                           [float(getattr(g, "alt_m", 0.0)) for g in gws]) if gws else None
    user_mask = float(getattr(cfg.constraints, "min_elevation_deg", None)
                      or getattr(cfg.handover, "min_elevation_deg", 0.0) or 0.0)
    gw_mask = float(getattr(cfg.constraints, "gateway_min_elevation_deg", 10.0))
    groups = [(user_ecef, user_mask)]
    if gw_ecef is not None:
        groups.append((gw_ecef, gw_mask))

    batch_s = float(t.batch_size_s)
    frames = []
    goff = 0
    for b, b_fine in _batch_groups(fine_ts, batch_s).items():
        b0, b1 = b_fine[0], b_fine[-1]
        span = (b1 - b0).total_seconds()
        n_coarse = int(span // coarse_step) + 1
        coarse_ts = [b0 + timedelta(seconds=i * coarse_step) for i in range(n_coarse)]
        keep = _visible_sat_ids(gen.generate(coarse_ts), groups)
        _log.info("  [%s batch %02d] pre-pass: %d visible", op.name, b, len(keep))
        if keep:
            sat_f = gen.generate(b_fine, keep_node_ids=keep)
            if not sat_f.empty:
                sat_f = sat_f.copy()
                sat_f["t_idx"] = sat_f["t_idx"].to_numpy() + goff
                frames.append(sat_f)
        goff += len(b_fine)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dataset_manifest(cfg, name, t) -> dict:
    """Identity of one dataset; a matching manifest skips regeneration (idempotent)."""
    return {
        "dataset":       name,
        "hours":         round(float(t.duration_s) / 3600.0, 6),
        "step_s":        float(t.step_s),
        "batch_size_s":  float(t.batch_size_s),
        "start_utc":     t.utc,
        "operators":     sorted(op.name for op in cfg.operators),
        "region":        cfg.env.region_name,
        "min_elev_deg":  float(cfg.handover.min_elevation_deg),
    }


def _write_batch_file(df, folder: Path, stem: str, b: int, goff: int,
                      compression: str) -> int:
    """Write one batch as <stem>__batch_NN.parquet, shifting local t_idx to global.
    Returns rows written (0 if empty)."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    if "t_idx" in df.columns:
        df["t_idx"] = df["t_idx"].to_numpy() + goff
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / f"{stem}__batch_{b:02d}.parquet",
                  compression=compression, index=False)
    return len(df)


def _generate_dataset(cfg, sub: Path, name: str, t, all_satellites: bool = False) -> Path:
    """Generate one dataset into `sub`, streaming to disk one batch at a time (never
    the whole timeline in RAM). Own manifest + idempotent skip. Online-mobility sats
    are init-only (t=0); everything else is the full offline timeline.
    all_satellites=True builds the coverage layer (no band/visibility filter);
    False builds the study layer (band-filtered + per-batch visibility keep-set)."""
    sub.mkdir(parents=True, exist_ok=True)
    manifest = _dataset_manifest(cfg, name, t)
    mpath = sub / "manifest.json"
    if mpath.exists():
        try:
            if json.loads(mpath.read_text()) == manifest:
                return sub                       # idempotent skip
        except (ValueError, OSError):
            pass

    comp = cfg.storage.parquet_compression
    fine_ts = _build_timestamps(t.utc, t.duration_s, t.step_s)
    batch_s = float(t.batch_size_s)
    coarse_step = float(getattr(t, "coarse_step_s", 0.0) or 0.0)
    _ONLINE = ("dataset", "static", "hover", "", None)

    # user t=0 + gateway positions drive the coarse visibility keep-set (union rule)
    users_t0 = generate_users(cfg, fine_ts[:1])
    user_ecef = (_ground_ecef(users_t0["lat"].to_numpy(), users_t0["lon"].to_numpy(),
                              users_t0.get("alt_m")) if not users_t0.empty else None)
    user_mask = float(getattr(cfg.constraints, "min_elevation_deg", None)
                      or getattr(cfg.handover, "min_elevation_deg", 0.0) or 0.0)
    gw_mask = float(getattr(cfg.constraints, "gateway_min_elevation_deg", 10.0))

    op_ctx = []   # (op, online_sat, groups)
    for op in cfg.operators:
        gws = list(getattr(op.ground_stations, "positions", []) or [])
        gw_ecef = (_ground_ecef([g.lat for g in gws], [g.lon for g in gws],
                                [float(getattr(g, "alt_m", 0.0)) for g in gws])
                   if gws else None)
        groups = [(user_ecef, user_mask)] + ([(gw_ecef, gw_mask)] if gw_ecef is not None else [])
        op_ctx.append((op, op.satellites.mobility.model not in _ONLINE, groups))

    goff = 0
    for b, b_fine in _batch_groups(fine_ts, batch_s).items():
        # users (shared across operators)
        _write_batch_file(generate_users(cfg, b_fine), sub / "users", "users",
                          b, goff, comp)
        for op, online_sat, groups in op_ctx:
            for ntype, Cls in GENERATORS.items():
                gen = Cls(cfg, op)
                if ntype == "satellite" and online_sat:
                    df = (gen.generate(b_fine[:1], apply_bands=not all_satellites)
                          if b == 0 else None)         # init-only, first batch
                elif ntype == "satellite" and all_satellites:
                    df = gen.generate(b_fine, apply_bands=False)
                elif ntype == "satellite" and coarse_step > 0:
                    span = (b_fine[-1] - b_fine[0]).total_seconds()
                    n_coarse = int(span // coarse_step) + 1
                    coarse_ts = [b_fine[0] + timedelta(seconds=i * coarse_step)
                                 for i in range(n_coarse)]
                    keep = _visible_sat_ids(gen.generate(coarse_ts), groups)
                    df = gen.generate(b_fine, keep_node_ids=keep) if keep else None
                elif ntype == "satellite":
                    df = gen.generate(b_fine, apply_bands=not all_satellites)
                else:
                    df = gen.generate(b_fine)
                _write_batch_file(df, sub / f"{ntype}s", op.name, b, goff, comp)
        goff += len(b_fine)

    mpath.write_text(json.dumps(manifest, indent=2))
    return sub


def run_phase1(cfg: SimulationConfig) -> Path:
    """Phase 1: build the dataset(s). By default writes a study layer (fine,
    band-filtered + visibility keep-set, mirrored at out_dir root for the engine)
    and a coverage layer (coarse, all sats, for visualisation).
    generate_coverage=False writes only the study layer from the env window."""
    out_dir = Path(cfg.storage.data_dir) / f"run_{cfg.storage.run_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_used(out_dir / "config_used.yaml")

    if not bool(getattr(cfg.experiment, "generate_coverage", True)):
        # quick single dataset from the env window
        env_t = DatasetTimingConfig(utc=cfg.env.start_utc,
                                    duration_s=float(cfg.env.duration_s),
                                    step_s=float(cfg.env.step_size_s),
                                    batch_size_s=float(cfg.env.step_size_s))
        _generate_dataset(cfg, out_dir, "study", env_t)
        return out_dir

    # coverage layer (coarse, all sats)
    _generate_dataset(cfg, out_dir / "coverage", "coverage", cfg.geometry_timing,
                      all_satellites=True)
    # study layer (fine, filtered) -> out_dir root, read by the engine
    _generate_dataset(cfg, out_dir, "study", cfg.study_timing)
    return out_dir


def make_online(cfg: SimulationConfig, sat_model: str = "auto") -> SimulationConfig:
    """Online preset: no precomputed ephemeris, sats propagated live in Phase 2.
    sat_model 'auto' picks skyfield when a TLE is present, else kepler."""
    for op in cfg.operators:
        m = sat_model
        if m == "auto":
            prop = getattr(op.satellites, "propagator", "auto")
            m = "skyfield" if (prop in ("skyfield", "skyfield_auto_tle",
                                        "sgp4", "sgp4_auto_tle")
                               or op.satellites.tle_file_path) else "kepler"
        op.satellites.mobility.model = m
    return cfg
