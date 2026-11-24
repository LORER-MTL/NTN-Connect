# Satellite generator with real SGP4 propagation (Skyfield) when available,
# falling back to a Kepler approximation otherwise.

from __future__ import annotations
import logging
import math
from typing import Any, List
import numpy as np
import pandas as pd

from ntnconnect.scenario.base_generator import BaseNodeGenerator
from ntnconnect.scenario.frames import lla_to_ecef

_log = logging.getLogger(__name__)

try:
    import skyfield.api  # noqa: F401
    _HAS_SKYFIELD = True
except Exception:
    _HAS_SKYFIELD = False
    _log.info("Skyfield not importable - falling back to Kepler approximation.")

EARTH_R = 6_371_000.0
EARTH_MU = 3.986004418e14


def _with_ecef(df: pd.DataFrame) -> pd.DataFrame:
    """Add WGS-84 ECEF x/y/z (metres) from the geodetic lat/lon/alt columns, so
    the standard satellite table carries an ECEF position too. Same convention as
    the original build_ephemeris: x,y,z = lla_to_ecef(lat, lon, alt)."""
    if df.empty or not {"lat", "lon", "alt_m"} <= set(df.columns):
        return df
    x, y, z = lla_to_ecef(df["lat"].to_numpy(), df["lon"].to_numpy(),
                          df["alt_m"].to_numpy())
    df["x"] = x.astype(float)
    df["y"] = y.astype(float)
    df["z"] = z.astype(float)
    return df


def _synth_walker(n_sats: int, n_planes: int, alt_km: float, incl_deg: float,
                  catalog_id_start: int, name_prefix: str):
    """Synthesize a Walker constellation as (sat_id, plane, slot, RAAN, M0)."""
    sats_per_plane = max(1, n_sats // max(1, n_planes))
    a_m = EARTH_R + alt_km * 1000.0
    n_rad = math.sqrt(EARTH_MU / a_m**3)         # mean motion rad/s
    out = []
    sid = catalog_id_start
    for p in range(n_planes):
        raan = 2 * math.pi * p / n_planes
        for s in range(sats_per_plane):
            m0 = 2 * math.pi * s / sats_per_plane
            out.append({
                "sat_id": sid,
                "plane": p,
                "slot": s,
                "raan_rad": raan,
                "m0_rad":   m0,
                "alt_km":   alt_km,
                "incl_rad": math.radians(incl_deg),
                "n_rad":    n_rad,
                "name":     f"{name_prefix}_{sid}",
            })
            sid += 1
    return out


# Walker-Delta -> TLE synthesis (inline, no skyntn dependency).
# Produces a TLE file consumable by Skyfield's load.tle_file().
# Math matches the legacy skyntn.generators.tle_generator.TLEGenerator.
WGS84_A_KM       = 6378.137
EARTH_MU_KM3_S2  = 398600.4418


def _tle_checksum(line: str) -> int:
    return (sum(int(c) for c in line if c.isdigit()) + line.count("-")) % 10


def _alpha5(satnum: int) -> str:
    """Encode satnum >= 100000 with the standard 'Alpha-5' scheme so it
    fits the 5-column TLE satnum field. Skyfield/SGP4 understands this."""
    if satnum < 100000:
        return f"{satnum:05d}"
    # Replace leading digit with a letter, skipping I and O.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # 24 letters
    lead   = (satnum - 100000) // 10000
    rest   = satnum - 100000 - lead * 10000
    if not (0 <= lead < len(alphabet)):
        raise ValueError(f"satnum {satnum} exceeds Alpha-5 range.")
    return f"{alphabet[lead]}{rest:04d}"


def _tle_epoch_str(dt) -> str:
    """TLE epoch field: 'YY DDD.DDDDDDDD' (14 chars including space)."""
    from datetime import datetime, timezone
    if dt is None:
        dt = datetime.now(timezone.utc)
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    year = dt.year % 100
    day_of_year = dt.timetuple().tm_yday + (
        dt.hour * 3600 + dt.minute * 60 + dt.second
        + dt.microsecond * 1e-6) / 86400.0
    return f"{year:02d}{day_of_year:012.8f}"


def _synth_walker_tle(sat_cfg, op, epoch, out_path) -> None:
    epoch_str = _tle_epoch_str(epoch)
    sid       = int(sat_cfg.catalog_id_start)
    lines: list[str] = []
    for shell in sat_cfg.shells:
        if shell.num_satellites % shell.num_planes != 0:
            raise ValueError(f"shell '{shell.name}': num_satellites must "
                             f"be divisible by num_planes.")
        sats_per_plane = shell.num_satellites // shell.num_planes
        a_km   = WGS84_A_KM + float(shell.altitude_km)
        n_rad  = math.sqrt(EARTH_MU_KM3_S2 / a_km**3)
        mm     = n_rad * 86400.0 / (2.0 * math.pi)   # rev/day
        F      = int(shell.phasing_factor)
        for p in range(shell.num_planes):
            for s in range(sats_per_plane):
                raan = (360.0 * p / shell.num_planes) % 360.0
                ma   = ((s * (360.0 / sats_per_plane)
                       + p * (360.0 * F / shell.num_satellites)) % 360.0)
                sat5 = _alpha5(sid)
                l1_no_cs = (f"1 {sat5}U 25001A   {epoch_str}"
                            f"  .00000000  00000-0  00000-0 0  9990")
                l1 = f"{l1_no_cs}{_tle_checksum(l1_no_cs)}"
                l2_no_cs = (f"2 {sat5} {float(shell.inclination_deg):8.4f} "
                            f"{raan:8.4f} 0001000   0.0000 "
                            f"{ma:8.4f} {mm:11.8f}    00")
                l2 = f"{l2_no_cs}{_tle_checksum(l2_no_cs)}"
                lines.append(f"{op.name}-{shell.name}-{sid}\n{l1}\n{l2}")
                sid += 1
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _propagate_kepler(sat, dt_s):
    """Closed-form circular Kepler. Returns (lat, lon, alt_m, v_eci)."""
    M = sat["m0_rad"] + sat["n_rad"] * dt_s
    a = EARTH_R + sat["alt_km"] * 1000.0
    i = sat["incl_rad"]
    raan = sat["raan_rad"]
    # Position in orbital plane (z_op == 0 for a coplanar orbit)
    x_op = a * math.cos(M); y_op = a * math.sin(M)
    # Rotate by inclination then RAAN -> ECI
    x1 = x_op
    y1 = y_op * math.cos(i)
    z1 = y_op * math.sin(i)
    x_eci = x1 * math.cos(raan) - y1 * math.sin(raan)
    y_eci = x1 * math.sin(raan) + y1 * math.cos(raan)
    z_eci = z1
    # Velocity (perpendicular to position in orbital plane; vz_op == 0)
    vmag = math.sqrt(EARTH_MU / a)
    vx_op = -vmag * math.sin(M); vy_op = vmag * math.cos(M)
    vx1 = vx_op
    vy1 = vy_op * math.cos(i)
    vz1 = vy_op * math.sin(i)
    vx_eci = vx1 * math.cos(raan) - vy1 * math.sin(raan)
    vy_eci = vx1 * math.sin(raan) + vy1 * math.cos(raan)
    vz_eci = vz1
    # ECI -> approximate lat/lon (Greenwich sidereal angle ignored = ECEF~ECI
    # for the purposes of a research stub; full ECEF needs ERA correction).
    r = math.sqrt(x_eci**2 + y_eci**2 + z_eci**2)
    lat = math.degrees(math.asin(z_eci / r))
    lon = math.degrees(math.atan2(y_eci, x_eci))
    alt = r - EARTH_R
    return lat, lon, alt, (vx_eci, vy_eci, vz_eci)


class SatelliteGenerator(BaseNodeGenerator):
    node_type = "satellite"

    def generate(self, timestamps: List[Any], keep_node_ids=None,
                 apply_bands: bool = True) -> pd.DataFrame:
        """Propagate the operator's satellites over `timestamps`.

        keep_node_ids - optional iterable of node_ids to restrict propagation to
        (the surviving set from a coarse visibility pre-pass). None = all sats.
        Propagating only the kept sats is what makes the fine study dataset light.
        apply_bands - when False, the inc/alt band filter is skipped so EVERY
        satellite in the TLE is emitted (the coverage/all-sats layer for viz).
        """
        sat_cfg = self.op.satellites
        if not sat_cfg.shells and not sat_cfg.tle_file_path:
            return pd.DataFrame()

        keep = None if keep_node_ids is None else {int(x) for x in keep_node_ids}
        self._apply_bands = bool(apply_bands)
        cap  = self._capability_cols()
        mode = str(getattr(sat_cfg, "propagator", "auto") or "auto").lower()
        # Canonical name is "skyfield" (Skyfield's SGP4). "sgp4"/"sgp4_auto_tle"
        # are kept as back-compat aliases.
        _ALIAS = {"sgp4": "skyfield", "sgp4_auto_tle": "skyfield_auto_tle"}
        mode = _ALIAS.get(mode, mode)
        _log.info("SatelliteGenerator: propagator=%s", mode)

        # Dispatch
        if mode == "kepler":
            return self._kepler_path(sat_cfg, timestamps, cap, t0=timestamps[0], keep=keep)

        if mode == "skyfield":
            if not _HAS_SKYFIELD:
                raise RuntimeError("propagator='skyfield' requires the 'skyfield' "
                                   "package (not importable).")
            tle_path = self._resolve_tle_path(sat_cfg, auto_generate=False,
                                              timestamps=timestamps)
            return self._propagate_sgp4(tle_path, timestamps, cap, keep=keep)

        if mode == "skyfield_auto_tle":
            if not _HAS_SKYFIELD:
                raise RuntimeError("propagator='skyfield_auto_tle' requires "
                                   "the 'skyfield' package.")
            tle_path = self._resolve_tle_path(sat_cfg, auto_generate=True,
                                              timestamps=timestamps)
            return self._propagate_sgp4(tle_path, timestamps, cap, keep=keep)

        # mode == "auto" (default): Skyfield SGP4 if a TLE is available, else Kepler.
        if _HAS_SKYFIELD and sat_cfg.tle_file_path:
            try:
                from pathlib import Path
                tle_path = Path(sat_cfg.tle_file_path)
                if tle_path.exists():
                    return self._propagate_sgp4(tle_path, timestamps, cap, keep=keep)
            except Exception as e:
                _log.warning("SGP4 propagation failed (%s); falling back to Kepler.", e)

        # Auto fallback: Kepler
        return self._kepler_path(sat_cfg, timestamps, cap, t0=timestamps[0], keep=keep)

    def _resolve_tle_path(self, sat_cfg, auto_generate: bool, timestamps):
        """Return the TLE path to feed into SGP4.

        - If sat_cfg.tle_file_path is set and exists -> use it.
        - Else, when auto_generate=True, synthesise one from sat_cfg.shells
          via the v0 TLEGenerator and cache it under sat_cfg.tle_cache_dir.
        - Else raise.
        """
        from pathlib import Path
        if sat_cfg.tle_file_path:
            p = Path(sat_cfg.tle_file_path)
            if p.exists():
                return p
            if not auto_generate:
                raise FileNotFoundError(
                    f"propagator='skyfield' but tle_file_path '{p}' does not exist. "
                    f"Use propagator='skyfield_auto_tle' to synthesise one.")
        if not auto_generate:
            raise FileNotFoundError(
                "propagator='skyfield' needs tle_file_path. Switch to "
                "'skyfield_auto_tle' to synthesise one from shells.")
        if not sat_cfg.shells:
            raise ValueError(
                "propagator='skyfield_auto_tle' requires at least one shell.")
        # Cache name encodes the shell signature so a config change invalidates it.
        cache_dir = Path(sat_cfg.tle_cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        sig = "_".join(f"{s.name}-{s.num_satellites}-{s.num_planes}-"
                       f"{s.altitude_km:.0f}-{s.inclination_deg:.1f}"
                       for s in sat_cfg.shells)
        out = cache_dir / f"{self.op.name}_{sig}.tle"
        if out.exists():
            _log.info("Using cached auto-TLE %s", out)
            return out
        epoch = timestamps[0] if timestamps else None
        _synth_walker_tle(sat_cfg, self.op, epoch, out)
        _log.info("Synthesised auto-TLE -> %s", out)
        return out

    def _kepler_path(self, sat_cfg, timestamps, cap, t0, keep=None) -> pd.DataFrame:
        """Closed-form circular two-body propagation from Walker shells."""
        catalog = []
        sid = sat_cfg.catalog_id_start
        for shell in sat_cfg.shells:
            shell_cat = _synth_walker(
                n_sats=shell.num_satellites, n_planes=shell.num_planes,
                alt_km=shell.altitude_km, incl_deg=shell.inclination_deg,
                catalog_id_start=sid, name_prefix=f"{self.op.name}-{shell.name}",
            )
            catalog.extend(shell_cat)
            sid += len(shell_cat)
        if keep is not None:
            catalog = [s for s in catalog if int(s["sat_id"]) in keep]
        if not catalog: return pd.DataFrame()

        t0 = timestamps[0] if timestamps else None
        rows = []
        for t_idx, ts in enumerate(timestamps):
            dt_s = (ts - t0).total_seconds() if t0 is not None else 0.0
            for sat in catalog:
                lat, lon, alt, v = _propagate_kepler(sat, dt_s)
                rows.append({
                    "timestamp":   ts,
                    "t_idx":       t_idx,
                    "node_id":     int(sat["sat_id"]),
                    "node_type":   self.node_type,
                    "operator_id": int(self.op.id),
                    "name":        sat["name"],
                    "lat":         float(lat),
                    "lon":         float(lon),
                    "alt_m":       float(alt),
                    "vx_mps":      float(v[0]),
                    "vy_mps":      float(v[1]),
                    "vz_mps":      float(v[2]),
                    **cap,
                })
        return _with_ecef(pd.DataFrame(rows))

    def _propagate_sgp4(self, tle_path, timestamps, cap, keep=None) -> pd.DataFrame:
        """Real SGP4 propagation through Skyfield."""
        from skyfield.api import load, EarthSatellite, wgs84
        ts_scale = load.timescale()
        sats = load.tle_file(str(tle_path))
        if not sats: return pd.DataFrame()

        # OPTIONAL band filter (general Phase-1 feature; empty = no filter). Keep
        # only sats whose inclination [deg] / altitude [km] fall in a configured
        # band - isolates a specific shell (e.g. Starlink 53 deg / 550 km) from a
        # full catalog. Ported from the WISEE build_ephemeris inc/alt-band logic.
        _satcfg = self.op.satellites
        inc_bands = [tuple(b) for b in (getattr(_satcfg, "inc_bands", None) or [])]
        alt_bands = [tuple(b) for b in (getattr(_satcfg, "alt_bands", None) or [])]
        if not getattr(self, "_apply_bands", True):
            inc_bands, alt_bands = [], []            # coverage: keep ALL sats
        if inc_bands or alt_bands:
            import math as _m
            # Inclination band: TLE inclination (build_ephemeris uses the same).
            if inc_bands:
                cand = [s for s in sats
                        if any(lo <= _m.degrees(float(getattr(s.model, "inclo", -1.0))) <= hi
                               for lo, hi in inc_bands)]
            else:
                cand = list(sats)
            # Altitude band: MEDIAN PROPAGATED subpoint altitude over the window
            # (np.nanmedian of the SGP4 geodetic altitude), NOT the mean-motion
            # semi-major axis - the latter mis-drops band-edge sats (e.g. Starlink
            # 51142: a-R_E=498 km would drop it, but its median alt is 500.8 km).
            # Matches build_ephemeris med_alt. A subsample keeps it cheap; the
            # median is a stable orbital property so coarse/fine calls agree.
            if alt_bands and cand:
                idx = np.unique(np.linspace(0, len(timestamps) - 1,
                                            min(len(timestamps), 32)).astype(int))
                t_samp = [ts_scale.from_datetime(timestamps[i]) for i in idx]
                kept = []
                for s in cand:
                    alt_km = np.array([wgs84.subpoint(s.at(t)).elevation.km
                                       for t in t_samp])
                    if not np.isfinite(alt_km).all():
                        continue
                    med = float(np.nanmedian(alt_km))
                    if any(lo <= med <= hi for lo, hi in alt_bands):
                        kept.append(s)
                cand = kept
            if cand:
                sats = cand

        def _resolve_sid(s):
            return int(getattr(s.model, "satnum", abs(hash(s.name)) % 1_000_000))

        # Limit to the operator's declared catalog range ONLY when it is a real
        # range (hi > lo). The default (0, 0) means "unset" - filtering on it
        # would keep just satnum 0 (one satellite). Matches the hi>lo convention
        # used by the config's per-operator id-range map.
        lo, hi = self.op.sat_id_range or (0, 0)
        if hi > lo:
            filtered = [s for s in sats if lo <= _resolve_sid(s) <= hi]
            if filtered:
                sats = filtered

        # Coarse-pre-pass keep-set: propagate only sats visible during this batch.
        if keep is not None:
            sats = [s for s in sats if _resolve_sid(s) in keep]
            if not sats:
                return pd.DataFrame()

        rows = []
        for t_idx, ts in enumerate(timestamps):
            # Skyfield expects a Time object (UTC)
            t = ts_scale.from_datetime(ts)
            for s in sats:
                geo = s.at(t)
                sub = wgs84.subpoint(geo)
                vel = geo.velocity.m_per_s
                rows.append({
                    "timestamp":   ts,
                    "t_idx":       t_idx,
                    "node_id":     _resolve_sid(s),
                    "node_type":   self.node_type,
                    "operator_id": int(self.op.id),
                    "name":        s.name.strip(),
                    "lat":         float(sub.latitude.degrees),
                    "lon":         float(sub.longitude.degrees),
                    "alt_m":       float(sub.elevation.m),
                    "vx_mps":      float(vel[0]),
                    "vy_mps":      float(vel[1]),
                    "vz_mps":      float(vel[2]),
                    **cap,
                })
        return _with_ecef(pd.DataFrame(rows))
