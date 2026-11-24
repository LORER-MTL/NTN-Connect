# Stage: DT channel pipeline (parallel to `channel`). Real -> h_per_antenna,
# nominal (aged geometry) -> h_nom_per_antenna; both consumed by scn_channel_predict.
# Enable per-operator via op.dt_channel_model; None => no-op.
from __future__ import annotations
import logging
from types import SimpleNamespace
import numpy as np
import pandas as pd

from ntnconnect.engine.stage_base import Stage
from ntnconnect.registry.registry import register_dt_stage
from ntnconnect.registry.channels.pipeline import ChannelModelFactory

_log = logging.getLogger(__name__)

# Aged-geometry defaults - fallback only if the DT cfg doesn't define
# the fields. With proper ChannelConfig now exposing them, these are
# essentially dead code, but kept for back-compat with hand-rolled cfgs.
_DEFAULT_AGED_ELEV_STD_DEG = 0.2
_DEFAULT_AGED_AZ_STD_DEG   = 0.2


def _flatten_dt_channel_cfg(op):
    """Merge op.radio + op.dt_channel_model into one namespace.

    Mirrors stages.channel._flatten_channel_cfg but pulls from
    dt_channel_model instead of channel_model.
    """
    out = SimpleNamespace()
    for src in (op.radio, op.dt_channel_model):
        if src is None:
            continue
        for k, v in vars(src).items():
            setattr(out, k, v)
    return out


@register_dt_stage("dt_channel")
class DTChannelStage(Stage):
    """Run the second (DT) ChannelPipeline."""

    reads  = ["links"]
    writes = []                              # writes state.data only
    thread_safe = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pipelines = {}
        for op in cfg.operators:
            dt_cfg = getattr(op, "dt_channel_model", None)
            if dt_cfg is None:
                continue
            try:
                self._pipelines[op.id] = ChannelModelFactory.build(
                    _flatten_dt_channel_cfg(op))
                _log.info("DT channel pipeline built for op=%s "
                          "(model=%s, components=%s)",
                          op.name, dt_cfg.model_type,
                          [c.name for c in self._pipelines[op.id].components])
            except Exception as e:
                _log.warning("op=%s DT channel pipeline build failed: %s",
                             op.name, e)

    def run(self, state):
        if not self._pipelines:
            return                            # nothing to do, no-op stage
        links = state.data.get("links")
        if not isinstance(links, pd.DataFrame) or links.empty:
            return

        # Aged-geometry RNG - uses the engine's channel stream so the
        # noisy view is reproducible under the master seed.
        rng = getattr(self.engine, "rng_channel", np.random.default_rng())

        h_nom_per_link: dict[int, np.ndarray] = {}

        for op in self.cfg.operators:
            pipe = self._pipelines.get(op.id)
            if pipe is None:
                continue
            mask = (links["operator_id"] == op.id).values
            if not mask.any():
                continue
            sub = links.loc[mask]
            n   = len(sub)

            # Build the DT view of geometry. We START with the real
            # elev/az/dist from geometry and add Gaussian noise
            # whose std comes from the DT config (defaults match Colab).
            elev_std = float(getattr(op.dt_channel_model,
                                     "geo_nom_aged_elev_std_deg",
                                     _DEFAULT_AGED_ELEV_STD_DEG))
            az_std   = float(getattr(op.dt_channel_model,
                                     "geo_nom_aged_az_std_deg",
                                     _DEFAULT_AGED_AZ_STD_DEG))
            elev_real = sub.get("elev_deg", pd.Series(np.zeros(n))).values
            az_real   = sub.get("azimuth_deg",
                                sub.get("azimuth", pd.Series(np.zeros(n)))).values
            dist_real = np.maximum(sub.get("dist_m", pd.Series(np.ones(n))).values,
                                    1.0)
            elev_dt = np.clip(elev_real + rng.normal(0.0, elev_std, n),
                              0.0, 90.0) if elev_std > 0 else elev_real
            az_dt   = (az_real + rng.normal(0.0, az_std, n)
                       if az_std > 0 else az_real)
            # The DT doesn't know the satellite's micro-perturbation -
            # but slant range is well-estimated from ephemeris, so dist
            # is left unperturbed (Colab convention).

            # Range-rate same - the DT estimates this from ephemeris.
            rate_mps = (sub.get("range_rate_km_s",
                                pd.Series(np.zeros(n))).values * 1000.0)

            try:
                loss, h, doppler, extras = pipe.execute(
                    dist_m=dist_real, elev_deg=elev_dt,
                    azimuth_deg=az_dt,
                    range_rate_m_s=rate_mps,
                    dt_sec=float(self.cfg.env.step_size_s),
                    pos_m=None,
                )
                h_arr = np.asarray(h)
                if h_arr.ndim < 2:
                    # DT pipeline produced scalar h - that's fine, but
                    # scn_channel_predict expects per-antenna. We leave
                    # h_nom_per_link entries blank for those rows and
                    # scn_channel_predict will fall back.
                    _log.info("DT pipeline produced scalar h; enable "
                              "steering_upa on the DT model for per-antenna.")
                    continue
                row_idx = np.where(mask)[0]
                for j, ri in enumerate(row_idx):
                    h_nom_per_link[int(ri)] = h_arr[j]
            except Exception as e:
                _log.warning("DT channel pipeline run failed for op=%s: %s",
                             op.name, e)

        state.data["h_nom_per_antenna"] = h_nom_per_link
