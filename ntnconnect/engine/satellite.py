# Satellite runtime object: owns its ChannelHistoryStore, predictor, DTChannel,
# and real/DT ChannelPipelines (private per-sat state). release_user() drops history.
from __future__ import annotations
from typing import Any, Optional
import logging
import numpy as np

from ntnconnect.registry.channels._history    import ChannelHistoryStore
from ntnconnect.registry.channels.predictor    import get_channel_predictor, BaseChannelPredictor
from ntnconnect.registry.channels.dt_channel   import DTChannel
from ntnconnect.registry.channels.pipeline     import ChannelModelFactory

_log = logging.getLogger(__name__)


def _flatten_op_to_channel_cfg(op, which: str):
    """Build a flat namespace from op.radio + op.<which>.

    `which` is 'channel_model' for the real pipeline, 'dt_channel_model'
    for the DT pipeline. Returns None if the source is missing.
    """
    from types import SimpleNamespace
    src = getattr(op, which, None)
    if src is None:
        return None
    out = SimpleNamespace()
    for s in (op.radio, src):
        if s is None:
            continue
        for k, v in vars(s).items():
            setattr(out, k, v)
    return out


class Satellite:
    """First-class persistent satellite object."""

    def __init__(self, sat_id: int, operator, cfg, rng: np.random.Generator):
        self.sat_id      = int(sat_id)
        self.operator_id = int(operator.id)
        self.operator    = operator
        self.cfg         = cfg
        self._rng        = rng

        # Antenna / geometry knobs (resolved from operator config).
        self.n_tx_x = int(operator.channel_model.n_tx_x)
        self.n_tx_y = int(operator.channel_model.n_tx_y)
        self.n_tx   = self.n_tx_x * self.n_tx_y
        self.window_size = int(getattr(cfg.handover,
                                       "channel_predictor_window", 20))

        # Real channel pipeline (per-sat instance, OP-config-driven)
        real_cfg = _flatten_op_to_channel_cfg(operator, "channel_model")
        try:
            self.real_pipeline = ChannelModelFactory.build(real_cfg)
        except Exception as e:
            _log.warning("Satellite %s real-pipeline build failed: %s",
                         sat_id, e)
            self.real_pipeline = None

        # DT channel pipeline (per-sat instance, optional)
        dt_cfg = _flatten_op_to_channel_cfg(operator, "dt_channel_model")
        if dt_cfg is not None:
            try:
                self.dt_pipeline = ChannelModelFactory.build(dt_cfg)
            except Exception as e:
                _log.warning("Satellite %s dt-pipeline build failed: %s",
                             sat_id, e)
                self.dt_pipeline = None
        else:
            self.dt_pipeline = None

        # History + predictor + DTChannel (per-sat state)
        self.channel_twin = ChannelHistoryStore(
            window_size=self.window_size, n_tx=self.n_tx)
        try:
            self.predictor = get_channel_predictor(cfg)        # may be None
        except Exception as e:
            _log.warning("Satellite %s predictor build failed: %s",
                         sat_id, e)
            self.predictor = None
        if self.predictor is not None:
            self.dt_channel = DTChannel(
                predictor      = self.predictor,
                n_tx           = self.n_tx,
                window_size    = self.window_size,
                history_store  = self.channel_twin,
                rng            = self._rng,
            )
        else:
            self.dt_channel = None

    # Lifecycle helpers
    def release_user(self, user_id: int) -> None:
        """Drop a user's history (and posterior) - used on handover so
        the user cold-starts on the next satellite that serves them."""
        uid = int(user_id)
        # ChannelHistoryStore stores per-(sat, user) - for this sat that's
        # just the (self.sat_id, uid) entry.
        if hasattr(self.channel_twin, "_meas"):
            self.channel_twin._meas.pop((self.sat_id, uid), None)
            self.channel_twin._nom.pop((self.sat_id, uid), None)
        if self.dt_channel is not None:
            self.dt_channel._posterior.pop((self.sat_id, uid), None)

    def __repr__(self):
        return (f"Satellite(sat_id={self.sat_id}, op={self.operator_id}, "
                f"n_tx={self.n_tx}, users_in_history="
                f"{sum(1 for k in (self.channel_twin._meas or {}).keys() if k[0]==self.sat_id)})")
