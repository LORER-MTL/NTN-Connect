"""Validator registry core - register / list / run named physical-sanity checks.

A validator is ``fn(df, expected) -> CheckResult``:
  * ``df``        a pandas DataFrame of served-link snapshots (one row / link-step)
  * ``expected``  a plain dict of reference values the caller supplies
Validators never raise to the caller - ``run_validators`` traps exceptions and
records them as a failed check, so a bad validator cannot crash a scenario.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
import logging

_log = logging.getLogger("ntnconnect.common.validation")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


def ok(name: str, passed, detail: Optional[dict] = None) -> CheckResult:
    """Build a CheckResult, coercing to a *python* bool (numpy bool_ breaks
    isinstance-based gating and json.dumps - the exact audit bug)."""
    return CheckResult(name, bool(passed), detail or {})


# name -> (fn, category, doc)
_VALIDATORS: dict[str, tuple[Callable, str, str]] = {}


def register_validator(name: str, *, category: str = "general"):
    """Decorator: register a validator under ``name``. Re-registering overrides
    (so a scenario can swap a built-in for its own)."""
    def deco(fn: Callable) -> Callable:
        _VALIDATORS[name] = (fn, category, (fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


def get_validator(name: str) -> Optional[Callable]:
    rec = _VALIDATORS.get(name)
    return rec[0] if rec else None


def list_validators(category: Optional[str] = None) -> list[str]:
    return sorted(n for n, (_, c, _d) in _VALIDATORS.items()
                  if category is None or c == category)


def catalog() -> dict:
    """{name: {category, doc}} - what validators a scenario can pick from."""
    return {n: {"category": c, "doc": d} for n, (_, c, d) in sorted(_VALIDATORS.items())}


def run_validators(names: Optional[Iterable[str]], df, expected: Optional[dict] = None) -> dict:
    """Run the selected validators (or all if ``names`` is None) and return a
    JSON-serialisable report ``{"PASS": bool, "checks": {name: {...}}}``."""
    expected = expected or {}
    sel = list(names) if names is not None else list_validators()
    checks: dict[str, dict] = {}
    for n in sel:
        fn = get_validator(n)
        if fn is None:
            checks[n] = {"passed": False, "error": "unknown validator"}
            _log.warning("validation: unknown validator %r (skipped)", n)
            continue
        try:
            r = fn(df, expected)
            if not isinstance(r, CheckResult):
                r = ok(n, r)
        except Exception as e:  # a buggy validator must not crash the scenario
            _log.warning("validation: %r raised %s", n, e)
            r = CheckResult(n, False, {"error": str(e)})
        checks[n] = {"passed": bool(r.passed), **{k: _jsonable(v) for k, v in r.detail.items()}}
    passed = all(c.get("passed") for c in checks.values()) if checks else False
    return {"PASS": bool(passed), "checks": checks}


def _jsonable(v):
    """Coerce numpy scalars to native python so json.dumps never fails."""
    try:
        import numpy as np
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    return v
