"""Digital-twin foundation (first stage). Owns activation and per-tier ownership.
The parallel twin lane is built later; this only sets up the twin's home and
activation, gated so nothing exists when the DT is off (physical pipeline
untouched). See ntnconnect/DIGITAL_TWIN_PLAN.md."""
from __future__ import annotations

_TIERS = ("node", "operator", "network")


class DigitalTwin:
    """Created by the control_plane stage when cfg.dt.enabled. Holds the tier
    scaffolding (node / operator / network) the parallel lane will populate.
    Off -> not created at all."""

    def __init__(self, cfg):
        dt = cfg.dt
        self.enabled = bool(dt.enabled)
        self.tiers = [t for t in dt.tiers if t in _TIERS]
        self.aging = {
            "elev_std_deg": float(dt.aged_elev_std_deg),
            "az_std_deg":   float(dt.aged_az_std_deg),
            "corr_time_s":  float(dt.aged_corr_time_s),
        }
        self.seed = int(cfg.experiment.random_seed)
        # tier ownership: the homes the future twin lane populates.
        self.node_tier = "node" in self.tiers
        self.operator_ids = [op.id for op in cfg.operators] if "operator" in self.tiers else []
        self.network = "network" in self.tiers
        self.state: dict = {}     # reserved twin.* namespace (future lane)

    def active(self, tier: str) -> bool:
        return self.enabled and tier in self.tiers

    def __repr__(self):
        return (f"DigitalTwin(enabled={self.enabled}, tiers={self.tiers}, "
                f"operators={self.operator_ids})")
