# SkyNTNPro v1 - Strict configuration mode
#
# Enforces: no implicit defaults at runtime.
# - Loader marks all dataclass fields not present in YAML as "missing".
# - Strict mode raises on missing fields at setup.
# - require(cfg, "env.duration_s") is the canonical accessor: it raises
#  loudly if the field wasn't set by the user.

from __future__ import annotations
import logging
from dataclasses import fields, is_dataclass
from typing import Any, List

_log = logging.getLogger(__name__)


class MissingConfigError(KeyError):
    """Raised when a required config field is not present in YAML."""


def require(cfg: Any, dotted_path: str) -> Any:
    """Strict get. Walks `dotted_path` on `cfg` and raises MissingConfigError
    if any segment is None or missing.

    Example:
        v = require(cfg, "env.duration_s")
        op_radio = require(operator, "radio.carrier_freq_hz")
    """
    cur = cfg
    walked: List[str] = []
    for part in dotted_path.split("."):
        walked.append(part)
        if not hasattr(cur, part):
            raise MissingConfigError(
                f"Required config field missing: {'.'.join(walked)}"
            )
        cur = getattr(cur, part)
        if cur is None:
            raise MissingConfigError(
                f"Required config field is None: {'.'.join(walked)}"
            )
    return cur


def collect_missing(cfg: Any, prefix: str = "") -> List[str]:
    """Recursively walk a dataclass and return dotted paths that are None or
    empty containers. Used by the strict validator at setup time."""
    missing: List[str] = []
    if not is_dataclass(cfg): return missing
    for f in fields(cfg):
        val = getattr(cfg, f.name, None)
        full = f"{prefix}.{f.name}" if prefix else f.name
        if val is None:
            missing.append(full)
            continue
        if is_dataclass(val):
            missing.extend(collect_missing(val, full))
        elif isinstance(val, list) and not val:
            # Empty list = either truly empty (operators=[]) or missing
            # We only flag if the parent expected at least one entry; the
            # caller (validator) decides.
            pass
    return missing


# Fields that are legitimately optional everywhere: None/empty == "no preference"
# or "use the built-in default" (e.g. model selectors: None == off). Always
# allowed, so every caller (engine + tests) gets consistent strict behaviour.
DEFAULT_ALLOWED_EMPTY: List[str] = [
    "experiment.notes",
    "experiment.global_seed",
    "users.preferences.preferred_operator_id",
    "users.max_users_per_cluster",
    "users.mobility.type",                 # legacy alias of mobility.model
    "outputs.formats",
    "constraints.geometry.interpolation_step_s",
    "defaults_channel.access_models",      # node-type loss map; empty == defaults
    "defaults_channel.rain_model",         # optional model selectors; None == off
    "defaults_channel.gas_model",
    "defaults_channel.shadowing_model",
    "defaults_channel.kfactor_model",
]


def assert_no_missing(cfg: Any, allowed_empty: List[str] = ()) -> None:
    """Raise MissingConfigError if any required field is None. The canonical
    DEFAULT_ALLOWED_EMPTY optional fields are always permitted; `allowed_empty`
    adds caller-specific ones."""
    allow = list(DEFAULT_ALLOWED_EMPTY) + list(allowed_empty)
    missing = [m for m in collect_missing(cfg)
               if not any(m.startswith(a) for a in allow)]
    if missing:
        raise MissingConfigError(
            "Strict config mode: the following fields are not set in YAML:\n  "
            + "\n  ".join(missing)
        )


# value-range validation
class InvalidConfigError(ValueError):
    """Raised when a config field is present but out of its valid range."""


def validate_value_ranges(cfg: Any) -> None:
    """Reject configurations that would put solvers in pathological regimes.

    These checks intentionally catch *only* hard scientific errors (division
    by zero, log of zero, infeasible-by-construction); they do not impose
    cosmetic limits. New checks should follow the same standard.
    """
    h = getattr(cfg, "handover", None)
    if h is not None:
        eps = float(getattr(h, "dro_epsilon", 0.05))
        if not (0.0 < eps < 1.0):
            raise InvalidConfigError(
                f"handover.dro_epsilon must be in (0,1); got {eps}.")
        n_mc  = int(getattr(h, "mc_channel_samples", 1))
        if n_mc < 0:
            raise InvalidConfigError(
                f"handover.mc_channel_samples must be >= 0; got {n_mc}.")
        win   = int(getattr(h, "channel_predictor_window", 20))
        if win < 0:
            raise InvalidConfigError(
                f"handover.channel_predictor_window must be >= 0; got {win}.")
        tc    = float(getattr(h, "mc_coherence_time_s", 0.0))
        if tc < 0:
            raise InvalidConfigError(
                f"handover.mc_coherence_time_s must be >= 0; got {tc}.")

    env = getattr(cfg, "env", None)
    if env is not None:
        step = float(getattr(env, "step_size_s", 0.0))
        if step <= 0:
            raise InvalidConfigError(
                f"env.step_size_s must be > 0; got {step}.")
        dur  = float(getattr(env, "duration_s", 0.0))
        if dur <= 0:
            raise InvalidConfigError(
                f"env.duration_s must be > 0; got {dur}.")

    for op in (getattr(cfg, "operators", []) or []):
        radio = getattr(op, "radio", None)
        if radio is None: continue
        if float(getattr(radio, "power_max_w", 0.0)) <= 0:
            raise InvalidConfigError(
                f"operators[*].radio.power_max_w must be > 0.")
        if float(getattr(radio, "bandwidth_hz", 0.0)) <= 0:
            raise InvalidConfigError(
                f"operators[*].radio.bandwidth_hz must be > 0.")
        if float(getattr(radio, "carrier_freq_hz", 0.0)) <= 0:
            raise InvalidConfigError(
                f"operators[*].radio.carrier_freq_hz must be > 0.")
