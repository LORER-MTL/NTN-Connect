"""ntnconnect.common.validation - pluggable physical-sanity VALIDATOR registry.

Each physical check is a *registered, named* validator. A scenario chooses
which validators to run on its outputs (or registers its own), so the simulator
stays general - no scenario is forced into a fixed validation profile.

    from ntnconnect.common.validation import register_validator, run_validators, list_validators

    @register_validator("my_check", category="custom")
    def _my(df, exp):
        return ok("my_check", bool(...), {"value": ...})

    report = run_validators(["noise_floor", "sinr_range"], df, expected)

`expected` is a plain dict the caller fills (so validators are decoupled from any
config schema): e.g. {"noise_dbw": -127, "gain_db": 30, "pl_range": (120,210),
"sinr_range": (-30,45)}.
"""
from __future__ import annotations
from ntnconnect.common.validation.registry import (
    CheckResult, ok, register_validator, list_validators, get_validator,
    run_validators, catalog as validator_catalog,
)

# Importing this package registers the built-in validators.
from ntnconnect.common.validation import builtin_validators  # noqa: F401

__all__ = [
    "CheckResult", "ok", "register_validator", "list_validators",
    "get_validator", "run_validators", "validator_catalog",
]
