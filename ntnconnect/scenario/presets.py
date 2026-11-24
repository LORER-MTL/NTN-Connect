# Scenario presets: reusable building blocks for self-contained scenario configs.
# Constellations synthesise their own Walker TLEs from shell parameters, so a
# scenario runs with no external TLE files (drop real TLEs in tles/ to override).

from __future__ import annotations
from ntnconnect.experiment.config import (
    Operator, SatelliteFleet, ShellConfig, RadioConfig, ChannelConfig,
    GroundStationFleet, FixedNodeEntry, PipelineConfig, StageGroupSpec,
)


def wisee_handover_pipeline() -> PipelineConfig:
    """WISEE handover-study Phase-2 pipeline (geometry-first):
      read -> geometry -> handover -> channel -> external_interference_const
      -> radio -> sinr_calc -> ho_classify -> metrics -> finalize.
    external_interference_const (i_ext_w) is what the min-power RRM sizing consumes,
    not the `interference` stage (interference_w)."""
    groups = [
        ("read",        ["read"]),
        ("geometry",    ["geometry"]),
        ("handover",    ["handover"]),
        ("channel",     ["channel"]),
        ("interference",["external_interference_const"]),
        ("radio",       ["radio"]),
        ("sinr",        ["sinr_calc"]),
        ("classify",    ["ho_classify"]),
        ("metrics",     ["metrics"]),
        ("finalize",    ["finalize"]),
    ]
    return PipelineConfig(groups=[StageGroupSpec(n, stages=s) for n, s in groups])


def standard_pipeline(with_radio: bool = True, with_clustering: bool = False,
                      with_classify: bool = True) -> PipelineConfig:
    """Current-pipeline chain; each scenario includes only the stages it needs:
      control_plane -> read -> dimensions -> mobility -> [clustering] -> geometry
      -> measurement -> handover -> channel -> [interference -> radio] -> sinr_calc
      -> [ho_classify] -> metrics -> finalize.
    with_radio adds interference + radio RRM; with_clustering adds the Earth-cell
    stage; with_classify adds ho_classify."""
    groups = [
        ("setup",        ["control_plane"]),
        ("read",         ["read"]),
        ("dimensions",   ["dimensions"]),
        ("mobility",     ["mobility"]),
    ]
    if with_clustering:
        groups.append(("clustering", ["clustering"]))
    groups += [
        ("geometry",     ["geometry"]),
        ("measurement",  ["measurement"]),
        ("handover",     ["handover"]),
        ("channel",      ["channel"]),
    ]
    if with_radio:
        groups += [("interference", ["interference"]), ("radio", ["radio"])]
    groups += [("sinr", ["sinr_calc"])]
    if with_classify:
        groups.append(("classify", ["ho_classify"]))
    groups += [("metrics", ["metrics"]), ("finalize", ["finalize"])]
    return PipelineConfig(groups=[StageGroupSpec(n, stages=s) for n, s in groups])

# Approximate published shells, reduced-density for light demo runs (scale up for a
# full study). num_satellites is the total and must be divisible by num_planes (Walker).
_SHELLS = {
    "oneweb":   ShellConfig(name="OneWeb",   altitude_km=1200.0, inclination_deg=87.4,
                            num_planes=12, num_satellites=48),   # 4/plane
    "starlink": ShellConfig(name="Starlink", altitude_km=550.0,  inclination_deg=53.0,
                            num_planes=24, num_satellites=48),   # 2/plane
    "kuiper":   ShellConfig(name="Kuiper",   altitude_km=630.0,  inclination_deg=51.9,
                            num_planes=17, num_satellites=34),   # 2/plane
}
_OP_ID = {"oneweb": 1, "starlink": 2, "kuiper": 3}
_CARRIER_HZ = 20.0e9        # Ku/Ka default [Hz]


def constellation_operator(name: str, op_id: int | None = None,
                           carrier_hz: float = _CARRIER_HZ) -> Operator:
    """Operator for a named constellation (oneweb/starlink/kuiper) via Walker TLEs."""
    key = name.lower()
    if key not in _SHELLS:
        raise ValueError(f"unknown constellation '{name}'; have {sorted(_SHELLS)}")
    sats = SatelliteFleet(generate_tle=True, propagator="skyfield_auto_tle",
                          shells=[_SHELLS[key]])
    return Operator(
        id=op_id if op_id is not None else _OP_ID[key],
        name=key,
        radio=RadioConfig(carrier_freq_hz=carrier_hz),
        channel_model=ChannelConfig(model_type="3GPP"),
        satellites=sats,
    )


def constellation_operators(names, carrier_hz: float = _CARRIER_HZ):
    """Several constellation operators at once."""
    return [constellation_operator(n, carrier_hz=carrier_hz) for n in names]


def gateway_fleet(op_id: int, sites) -> GroundStationFleet:
    """Gateway fleet from (name, lat, lon) tuples."""
    return GroundStationFleet(
        id_range=(op_id * 1000, op_id * 1000 + len(sites)),
        positions=[FixedNodeEntry(name=n, lat=la, lon=lo, alt_m=0.0)
                   for (n, la, lo) in sites],
    )
